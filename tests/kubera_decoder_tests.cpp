#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <iced_x86/decoder.hpp>
#include <iced_x86/instruction.hpp>
#include <iced_x86/op_code_info.hpp>
#include <iced_x86/op_code_operand_kind.hpp>

#include "seven/executor.hpp"
#include "seven/handler_helpers.hpp"
#include "seven/memory.hpp"

// The x86 decoder here is a hand-written C++ port of the Rust iced_x86 crate, and the emulator
// makes real decisions from what it reports: which register a SIMD operand names, whether an
// operand is memory, which Code a byte sequence is. A handler that fills in the wrong register or
// leaves an operand slot untouched does not fail loudly -- OpKind::REGISTER and Register::NONE are
// both zero, so an unwritten slot reads back as a plausible register operand. These pin the cases
// that were getting it wrong.

namespace {

[[nodiscard]] auto try_decode(const char* hex) {
  const auto raw = seven::parse_hex_bytes(hex);
  iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(raw.data(), raw.size()), 0x1000);
  return decoder.decode();
}

[[nodiscard]] iced_x86::Instruction decode(const char* hex) {
  const auto decoded = try_decode(hex);
  EXPECT_TRUE(decoded.has_value()) << hex;
  return decoded.has_value() ? decoded.value() : iced_x86::Instruction{};
}

}  // namespace

// EVEX carries the r/m register's high bits in two places: B is bit 3 of the index and X is bit 4.
// The decoder packs both into extra_base_register_base_evex and B alone into
// extra_base_register_base (which the SIB path wants), and the EVEX handlers were adding both, so
// B counted twice. Anything with B set named a register 8 slots too high, and past 31 that walks
// straight out of the vector file into whatever enum values follow it.
TEST(KuberaDecoder, EvexRegisterIndexDoesNotCountTheBBitTwice) {
  struct Case { const char* name; const char* bytes; std::uint32_t operand; iced_x86::Register expected; };
  const Case cases[] = {
      // vmovups xmm0, xmm9   -- B=1, X=0
      {"xmm9", "62 D1 7C 08 10 C1", 1, iced_x86::Register::XMM9},
      // vmovups xmm0, xmm25  -- B=1, X=1
      {"xmm25", "62 91 7C 08 10 C1", 1, iced_x86::Register::XMM25},
      // vmovups zmm0, zmm31
      {"zmm31 src", "62 91 7C 48 10 C7", 1, iced_x86::Register::ZMM31},
      // vmovups zmm31, zmm0  -- store direction, r/m is the destination
      {"zmm31 dst", "62 91 7C 48 11 C7", 0, iced_x86::Register::ZMM31},
  };
  for (const auto& c : cases) {
    const auto instr = decode(c.bytes);
    EXPECT_EQ(instr.op_register(c.operand), c.expected) << c.name << " (" << c.bytes << ")";
  }
}

// The same double count reached the memory base, which was missed because the fix above only
// touched the register operands. A memory base is four bits wide, rm plus EVEX.B; EVEX.X extends
// the SIB index and nothing else, and read_sib already applies it there. Adding
// extra_base_register_base_evex (which packs B and X together) on top of extra_base_register_base
// counted B twice and folded X in on top, so vmovups zmm0, [r8] named EIP as its base and every
// AVX-512 access through r8-r15 addressed off the wrong register. Past RIP the index walks on into
// the segment registers and the vector file.
TEST(KuberaDecoder, EvexMemoryBaseDoesNotCountTheBBitTwice) {
  struct Case { const char* name; const char* bytes; iced_x86::Register expected; };
  const Case cases[] = {
      // vmovups zmm0, [rcx] -- nothing extended
      {"[rcx]", "62 F1 7C 48 10 01", iced_x86::Register::RCX},
      // vmovups zmm0, [r9] -- B extends the base
      {"[r9] via B", "62 D1 7C 48 10 01", iced_x86::Register::R9},
      // X is only the SIB index's high bit, so with no SIB byte it must not touch the base
      {"[rcx] with X set", "62 B1 7C 48 10 01", iced_x86::Register::RCX},
      // the disp8 and disp32 forms compute the base separately and had the same fault
      {"[r8+0x10]", "62 D1 7C 48 10 40 10", iced_x86::Register::R8},
      {"[r8+0x10000]", "62 D1 7C 48 10 80 00 00 01 00", iced_x86::Register::R8},
      // the SIB path takes B through read_sib instead and must stay where it is
      {"[r12]", "62 D1 7C 48 10 04 24", iced_x86::Register::R12},
  };
  for (const auto& c : cases) {
    const auto instr = decode(c.bytes);
    EXPECT_EQ(instr.op1_kind(), iced_x86::OpKind::MEMORY) << c.name << " (" << c.bytes << ")";
    EXPECT_EQ(instr.memory_base(), c.expected) << c.name << " (" << c.bytes << ")";
  }
}

// 0F 20/22 are the control-register moves and 0F 21/23 the debug-register ones, but both pairs land
// in the same two handlers and those handlers hardcoded Register::CR0 as the base. So MOV r64, DR2
// reported CR2, and resolve_debug_index rejected it -- the DR0-DR3 watchpoint emulation could not
// be reached through MOV DR at all.
TEST(KuberaDecoder, DebugRegisterMovesNameDebugRegisters) {
  const auto read = decode("0F 21 11");  // mov rcx, dr2
  EXPECT_EQ(read.code(), iced_x86::Code::MOV_R64_DR);
  EXPECT_EQ(read.op1_register(), iced_x86::Register::DR2);

  const auto write = decode("0F 23 11");  // mov dr2, rcx
  EXPECT_EQ(write.code(), iced_x86::Code::MOV_DR_R64);
  EXPECT_EQ(write.op0_register(), iced_x86::Register::DR2);

  // The control-register forms next door must keep working.
  const auto cr = decode("0F 20 11");  // mov rcx, cr2
  EXPECT_EQ(cr.op1_register(), iced_x86::Register::CR2);
}

// MOVNTQ declares two operands and the handler only ever wrote the first. Reading operand 1 then
// came back as (REGISTER, NONE), which passes an op_kind check and turns into a register index of
// zero or an underflow, depending on what the reader subtracts.
TEST(KuberaDecoder, MovntqNamesItsSourceRegister) {
  const auto instr = decode("0F E7 11");  // movntq [rcx], mm2
  ASSERT_EQ(instr.op_count(), 2u);
  EXPECT_EQ(instr.op0_kind(), iced_x86::OpKind::MEMORY);
  EXPECT_EQ(instr.op1_kind(), iced_x86::OpKind::REGISTER);
  EXPECT_EQ(instr.op1_register(), iced_x86::Register::MM2);
}

// These opcodes mean one instruction with a register operand and a different one with a memory
// operand. The handler carries both Codes but only ever set the first, so a memory-form encoding
// came back wearing the register form's Code: the operand set says memory while the Code says
// otherwise, and anything dispatching on Code runs the wrong semantics.
TEST(KuberaDecoder, TheMemoryFormsOfMovlpsAndMovhpsGetTheirOwnCode) {
  const auto movlps = decode("0F 12 11");  // movlps xmm2, [rcx]
  EXPECT_EQ(movlps.op1_kind(), iced_x86::OpKind::MEMORY);
  EXPECT_EQ(movlps.code(), iced_x86::Code::MOVLPS_XMM_M64);

  const auto movhps = decode("0F 16 11");  // movhps xmm2, [rcx]
  EXPECT_EQ(movhps.op1_kind(), iced_x86::OpKind::MEMORY);
  EXPECT_EQ(movhps.code(), iced_x86::Code::MOVHPS_XMM_M64);

  // The register forms these were being confused with still decode as themselves.
  EXPECT_EQ(decode("0F 12 D1").code(), iced_x86::Code::MOVHLPS_XMM_XMM);
  EXPECT_EQ(decode("0F 16 D1").code(), iced_x86::Code::MOVLHPS_XMM_XMM);
}

// PUSH/POP FS and GS, ENTER, and the descriptor-table stores all default to a 64-bit operand size
// in long mode -- there is no 32-bit encoding for them there, only 16 via a 66 prefix. The handlers
// indexed their code array by state().operand_size, which reflects prefixes alone and reads "32"
// when there are none, so every unprefixed encoding picked the 32-bit form. That is not cosmetic:
// the emulator takes the number of bytes it pushes straight from the Code.
TEST(KuberaDecoder, DefaultSixtyFourBitOperandsPickTheSixtyFourBitCode) {
  struct Case { const char* name; const char* bytes; iced_x86::Code expected; };
  const Case cases[] = {
      {"push fs", "0F A0", iced_x86::Code::PUSHQ_FS},
      {"pop fs", "0F A1", iced_x86::Code::POPQ_FS},
      {"push gs", "0F A8", iced_x86::Code::PUSHQ_GS},
      {"pop gs", "0F A9", iced_x86::Code::POPQ_GS},
      {"enter", "C8 11 22 33", iced_x86::Code::ENTERQ_IMM16_IMM8},
      {"sgdt", "0F 01 00", iced_x86::Code::SGDT_M1664},
      {"sidt", "0F 01 08", iced_x86::Code::SIDT_M1664},
      {"lgdt", "0F 01 10", iced_x86::Code::LGDT_M1664},
      {"lidt", "0F 01 18", iced_x86::Code::LIDT_M1664},
      // A 66 prefix is the only way to ask for the narrow form, and it must still work.
      {"push fs, opsize 16", "66 0F A0", iced_x86::Code::PUSHW_FS},
      {"enter, opsize 16", "66 C8 11 22 33", iced_x86::Code::ENTERW_IMM16_IMM8},
      // iced gives the descriptor-table stores a separate Code for a 16-bit operand size rather
      // than reusing the 32-bit one, so this is SGDT_M1632_16 and not SGDT_M1632.
      {"sgdt, opsize 16", "66 0F 01 00", iced_x86::Code::SGDT_M1632_16},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(decode(c.bytes).code(), c.expected) << c.name << " (" << c.bytes << ")";
  }
}

// VEX.vvvv is four bits wide in long mode but the mask register file only has eight entries, so a
// mask operand taken from it has to be narrowed. Most of the mask handlers here do that; the two
// three-operand ones did not, and K0 + 15 lands on CR3. A decoded operand naming a register outside
// its own class is how an index walks out of the array it is about to be used to subscript.
TEST(KuberaDecoder, MaskOperandsStayInTheMaskRegisterFile) {
  struct Case { const char* name; const char* bytes; std::uint32_t operand; };
  const Case cases[] = {
      // kandw k1, k?, k3 -- vvvv counts up past the end of the mask file
      {"kandw vvvv=8", "C5 BC 41 CB", 1},
      {"kandw vvvv=15", "C5 84 41 CB", 1},
      // kortestw and the GPR-destination form reach vvvv the same way
      {"kandnw vvvv=15", "C5 84 42 CB", 1},
      {"kxorw vvvv=15", "C5 84 47 CB", 1},
  };
  // Rejecting the encoding outright and narrowing the index are both fine. Handing back a
  // register from some other file is not.
  for (const auto& c : cases) {
    const auto instr = try_decode(c.bytes);
    if (!instr.has_value() || instr->code() == iced_x86::Code::INVALID) {
      continue;
    }
    const auto reg = instr->op_register(c.operand);
    EXPECT_GE(reg, iced_x86::Register::K0) << c.name << " (" << c.bytes << ")";
    EXPECT_LE(reg, iced_x86::Register::K7) << c.name << " (" << c.bytes << ")";
  }

  // The in-range encodings still name what they should.
  const auto ok = decode("C5 EC 41 CB");  // kandw k1, k2, k3
  EXPECT_EQ(ok.op0_register(), iced_x86::Register::K1);
  EXPECT_EQ(ok.op1_register(), iced_x86::Register::K2);
  EXPECT_EQ(ok.op2_register(), iced_x86::Register::K3);
}

// VEX.L is ignored by the scalar instructions (the LIG group), and the tables already dispatch the
// length-sensitive ones to a separate handler per L before the operands are read. get_vec_reg threw
// the handler's own register class away and re-derived it from the vector length, so vmovss with
// L set came back naming YMM registers for an instruction that only ever touches the low 32 bits.
// The length-dispatched cases are here too, since they are what the re-derivation was there for.
TEST(KuberaDecoder, VectorLengthDoesNotPromoteScalarOperands) {
  struct Case { const char* name; const char* bytes; iced_x86::Register op0; };
  const Case cases[] = {
      {"vmovups xmm", "C5 F8 10 C1", iced_x86::Register::XMM0},
      {"vmovups ymm", "C5 FC 10 C1", iced_x86::Register::YMM0},
      {"vaddps xmm", "C5 F0 58 C2", iced_x86::Register::XMM0},
      {"vaddps ymm", "C5 F4 58 C2", iced_x86::Register::YMM0},
      {"evex vmovups xmm", "62 F1 7C 08 10 C1", iced_x86::Register::XMM0},
      {"evex vmovups ymm", "62 F1 7C 28 10 C1", iced_x86::Register::YMM0},
      {"evex vmovups zmm", "62 F1 7C 48 10 C1", iced_x86::Register::ZMM0},
      // A spread of handler shapes, since removing the re-derivation had to leave every
      // length-dispatched form exactly where it was.
      {"vmovups ymm store", "C5 FC 11 C1", iced_x86::Register::YMM1},
      {"vmovups ymm load", "C5 FC 10 01", iced_x86::Register::YMM0},
      {"vroundpd ymm", "C4 E3 7D 09 C1 05", iced_x86::Register::YMM0},
      {"vblendps ymm", "C4 E3 75 0C C2 0F", iced_x86::Register::YMM0},
      {"vpermilps ymm", "C4 E2 7D 0C C1", iced_x86::Register::YMM0},
      {"evex vaddps zmm", "62 F1 74 48 58 C2", iced_x86::Register::ZMM0},
      {"evex vmovups zmm store", "62 F1 7C 48 11 C1", iced_x86::Register::ZMM1},
      // L is set on both of these and must be ignored: they are scalar.
      {"vmovss, L set", "C5 86 10 C0", iced_x86::Register::XMM0},
      {"vmovsd, L set", "C5 87 10 C0", iced_x86::Register::XMM0},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(decode(c.bytes).op0_register(), c.op0) << c.name << " (" << c.bytes << ")";
  }
}

// iced carries a per-Code operand table describing how each Code is encoded, and the decoder is
// supposed to agree with it. Where a Code's table says an operand is register-only -- either a
// register field outright, or the r/m field restricted to mod == 3 -- that operand must never come
// back as memory. When it does, the decoder handed out the wrong Code for the encoding, and the
// handler that Code dispatches to reads a register slot the decoder never filled in.
//
// Two ways that happened here. The serialized tables store one Code per group and count the rest
// off from it, which lands on an unrelated instruction wherever the enum is not contiguous; and a
// register-only handler read a memory operand instead of rejecting mod != 3.
namespace {

[[nodiscard]] bool operand_kind_is_register_only(iced_x86::OpCodeOperandKind kind) {
  using K = iced_x86::OpCodeOperandKind;
  switch (kind) {
    case K::R8_REG: case K::R16_REG: case K::R32_REG: case K::R64_REG:
    case K::R16_RM: case K::R32_RM: case K::R64_RM:
    case K::SEG_REG: case K::K_REG: case K::KP1_REG: case K::K_RM:
    case K::MM_REG: case K::MM_RM:
    case K::XMM_REG: case K::XMM_RM:
    case K::YMM_REG: case K::YMM_RM:
    case K::ZMM_REG: case K::ZMM_RM:
    case K::CR_REG: case K::DR_REG: case K::TR_REG: case K::BND_REG:
    case K::TMM_REG: case K::TMM_RM:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool operand_is_memory(iced_x86::OpKind kind) {
  // The memory OpKind family is contiguous, MEMORY_SEG_SI through MEMORY.
  return kind >= iced_x86::OpKind::MEMORY_SEG_SI && kind <= iced_x86::OpKind::MEMORY;
}

void expect_operands_match_table(const std::vector<std::uint8_t>& bytes) {
  iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(bytes.data(), bytes.size()), 0x1000);
  const auto decoded = decoder.decode();
  if (!decoded.has_value() || decoded->code() == iced_x86::Code::INVALID) {
    return;
  }
  const auto& info = iced_x86::OpCodeInfo::get(decoded->code());
  const auto count = decoded->op_count();
  for (std::uint32_t i = 0; i < count && i < 5; ++i) {
    if (!operand_is_memory(decoded->op_kind(i))) continue;
    ASSERT_FALSE(operand_kind_is_register_only(info.op_kind(i)))
        << "code " << static_cast<unsigned>(decoded->code()) << " operand " << i
        << " decoded as memory but its own table says register-only";
  }
}

}  // namespace

TEST(KuberaDecoder, DerivedCodesAgreeWithTheirOwnOperandTables) {
  // The two encodings that were getting it wrong, pinned by name.
  // vmovhps xmm1, xmm3, [rbx] -- was landing on the EVEX register-only form
  EXPECT_EQ(decode("C5 E0 16 0B").code(), iced_x86::Code::VEX_VMOVHPS_XMM_XMM_M64);
  // and the register form it sits next to must not have moved
  EXPECT_EQ(decode("C5 E0 16 CB").code(), iced_x86::Code::VEX_VMOVLHPS_XMM_XMM_XMM);
  // vpbroadcastb takes its source from a GPR; there is no memory form at this opcode
  EXPECT_FALSE(try_decode("62 F2 7D 08 7A 00").has_value());
  EXPECT_EQ(decode("62 F2 7D 08 7A C0").code(), iced_x86::Code::EVEX_VPBROADCASTB_XMM_K1Z_R32);

  const std::vector<std::vector<std::uint8_t>> prefixes = {
      {}, {0x0F}, {0x0F, 0x38}, {0x0F, 0x3A},
      {0x66, 0x0F}, {0x66, 0x0F, 0x38}, {0x66, 0x0F, 0x3A},
      {0xF2, 0x0F}, {0xF3, 0x0F},
      {0x48, 0x0F}, {0x66, 0x48, 0x0F}, {0x4C, 0x0F},
      {0xC5, 0xF8}, {0xC5, 0xFC}, {0xC5, 0xE0}, {0xC5, 0xE1}, {0xC5, 0xE4},
      {0xC4, 0xE1, 0x78}, {0xC4, 0xE1, 0xF8}, {0xC4, 0xE2, 0x79}, {0xC4, 0xE3, 0x79},
      {0xC4, 0xE2, 0x7D}, {0xC4, 0xE3, 0x7D}, {0xC4, 0xE2, 0xF9}, {0xC4, 0xE3, 0xF9},
      {0x62, 0xF1, 0x7C, 0x08}, {0x62, 0xF1, 0x7C, 0x48}, {0x62, 0xF1, 0xFD, 0x08},
      {0x62, 0xF2, 0x7D, 0x08}, {0x62, 0xF2, 0xFD, 0x48}, {0x62, 0xF3, 0x7D, 0x08},
      {0x62, 0xF2, 0x7D, 0x48}, {0x62, 0xF3, 0xFD, 0x08},
  };
  // Both mod forms of every opcode on each map, across enough r/m values to reach the SIB and
  // displacement paths.
  for (const auto& prefix : prefixes) {
    for (unsigned opcode = 0; opcode < 256; ++opcode) {
      for (unsigned modrm : {0x00u, 0x04u, 0x05u, 0x0Bu, 0x48u, 0x8Bu, 0xC0u, 0xCBu}) {
        std::vector<std::uint8_t> bytes = prefix;
        bytes.push_back(static_cast<std::uint8_t>(opcode));
        bytes.push_back(static_cast<std::uint8_t>(modrm));
        for (int i = 0; i < 8; ++i) bytes.push_back(0x11);
        expect_operands_match_table(bytes);
        if (::testing::Test::HasFatalFailure()) return;
      }
    }
  }
}

// A 0x67 prefix in 64-bit mode makes a mod=0 rm=101 operand EIP-relative rather than RIP-relative:
// the target is computed in 32 bits and wraps at 4G. The decoder chose between the two off its
// bitness, so every 0x67 form took the RIP path and the EIP branch next to it was unreachable.
TEST(KuberaDecoder, AddressSizePrefixMakesAnIpRelativeOperandEipRelative) {
  // mov eax, [eip-0x2000] decoded at 0x1000, so the target wraps from 0x1007 - 0x2000.
  const auto narrow = decode("67 8B 05 00 E0 FF FF");
  EXPECT_EQ(narrow.memory_base(), iced_x86::Register::EIP);
  EXPECT_EQ(narrow.ip_rel_memory_address(), 0xFFFFF007ull);

  const auto wide = decode("48 8B 05 00 E0 FF FF");
  EXPECT_EQ(wide.memory_base(), iced_x86::Register::RIP);
  EXPECT_EQ(wide.ip_rel_memory_address(), 0xFFFFFFFFFFFFF007ull);
}

// [disp32] with neither a base nor an index leaves no register behind for a consumer to read the
// address size off, so the displacement has to be truncated at decode time.
TEST(KuberaDecoder, AddressSizePrefixTruncatesABareDisplacement) {
  const auto narrow = decode("67 8B 04 25 00 E0 FF FF");
  EXPECT_EQ(narrow.memory_base(), iced_x86::Register::NONE);
  EXPECT_EQ(narrow.memory_index(), iced_x86::Register::NONE);
  EXPECT_EQ(narrow.memory_displacement64(), 0xFFFFE000ull);

  const auto wide = decode("8B 04 25 00 E0 FF FF");
  EXPECT_EQ(wide.memory_displacement64(), 0xFFFFFFFFFFFFE000ull);
}

// The effective address wraps within the address size before any segment base is added, so
// [eax-0x10] with eax=8 lands just under 4G instead of at a sign-extended -8.
TEST(KuberaDecoder, ThirtyTwoBitEffectiveAddressWrapsAtFourGigabytes) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  memory.map(0xFFFFF000, 0x1000);
  const std::uint32_t marker = 0x00C0FFEEu;
  ASSERT_TRUE(memory.write(0xFFFFFFF8, &marker, sizeof(marker)));

  const std::uint8_t code[] = {0x67, 0x8B, 0x40, 0xF0};
  ASSERT_TRUE(memory.write(0x1000, code, sizeof(code)));

  state.rip = 0x1000;
  state.gpr[0] = 8;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], marker);
}

// A segment override on a RIP-relative operand was dropped: that path returned the ip-relative
// target directly, before the FS/GS base was ever added.
TEST(KuberaDecoder, ASegmentOverrideStillAppliesToARipRelativeOperand) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  memory.map(0x51000, 0x1000);
  state.fs_base = 0x50000;

  // mov rax, fs:[rip+0], eight bytes long, so the operand resolves to fs_base + 0x1008.
  const std::uint8_t code[] = {0x64, 0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00};
  ASSERT_TRUE(memory.write(0x1000, code, sizeof(code)));
  const std::uint64_t marker = 0x1122334455667788ull;
  ASSERT_TRUE(memory.write(state.fs_base + 0x1008, &marker, sizeof(marker)));

  state.rip = 0x1000;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], marker);
}

// Outside 64-bit mode, C4, C5 and 62 are only a VEX or EVEX prefix when the byte after them has
// mod == 3. With mod != 3 they are LES, LDS and BOUND. The three prefix handlers each carry a
// handler_mem for exactly that case and all three ignored it and decoded as a prefix regardless,
// so those instructions were undecodable in the two modes that have them -- BOUND's handler was
// sitting there unreachable -- and the reported length was wrong on top of it, which desynchronises
// everything walking the stream afterwards.
TEST(KuberaDecoder, TheLegacyFormsOfTheVexAndEvexOpcodesStillDecode) {
  struct Case { const char* name; std::uint8_t op; iced_x86::Code in16; iced_x86::Code in32; };
  const Case cases[] = {
      {"bound", 0x62, iced_x86::Code::BOUND_R16_M1616, iced_x86::Code::BOUND_R32_M3232},
      {"les", 0xC4, iced_x86::Code::LES_R16_M1616, iced_x86::Code::LES_R32_M1632},
      {"lds", 0xC5, iced_x86::Code::LDS_R16_M1616, iced_x86::Code::LDS_R32_M1632},
  };

  for (const auto& c : cases) {
    // modrm 0x03 is mod=00 rm=011, so a memory operand: the legacy form, not a prefix.
    const std::vector<std::uint8_t> bytes = {c.op, 0x03, 0x10, 0x20, 0x30, 0x40, 0x50};
    for (const std::uint32_t bitness : {16u, 32u}) {
      iced_x86::Decoder dec(bitness, std::span<const std::uint8_t>(bytes.data(), bytes.size()), 0);
      const auto decoded = dec.decode();
      ASSERT_TRUE(decoded.has_value()) << c.name << " at bitness " << bitness;
      EXPECT_EQ(decoded->code(), bitness == 16 ? c.in16 : c.in32) << c.name << " at bitness " << bitness;
      EXPECT_EQ(decoded->length(), 2u) << c.name << " consumes only the opcode and its modrm";
    }

    // In long mode there is no legacy form: all three are always a prefix, and this encoding is
    // not a valid one, so it must still be rejected rather than silently becoming an instruction.
    iced_x86::Decoder dec64(64, std::span<const std::uint8_t>(bytes.data(), bytes.size()), 0);
    const auto decoded64 = dec64.decode();
    EXPECT_FALSE(decoded64.has_value() && decoded64->code() == c.in32) << c.name << " has no legacy form in long mode";
  }

  // mod == 3 still selects the prefix in 32-bit mode, which is the half that already worked.
  const std::vector<std::uint8_t> vex = {0xC5, 0xF9, 0x6E, 0xC1};  // vmovd xmm0, ecx
  iced_x86::Decoder dec(32, std::span<const std::uint8_t>(vex.data(), vex.size()), 0);
  const auto decoded = dec.decode();
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->code(), iced_x86::Code::VEX_VMOVD_XMM_RM32);
}

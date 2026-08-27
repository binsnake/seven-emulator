#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <iced_x86/decoder.hpp>
#include <iced_x86/instruction.hpp>

#include "seven/handler_helpers.hpp"

// The x86 decoder here is a hand-written C++ port of the Rust iced_x86 crate, and the emulator
// makes real decisions from what it reports: which register a SIMD operand names, whether an
// operand is memory, which Code a byte sequence is. A handler that fills in the wrong register or
// leaves an operand slot untouched does not fail loudly -- OpKind::REGISTER and Register::NONE are
// both zero, so an unwritten slot reads back as a plausible register operand. These pin the cases
// that were getting it wrong.

namespace {

[[nodiscard]] iced_x86::Instruction decode(const char* hex) {
  const auto raw = seven::parse_hex_bytes(hex);
  iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(raw.data(), raw.size()), 0x1000);
  const auto decoded = decoder.decode();
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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <iced_x86/code.hpp>
#include <iced_x86/instruction_create.hpp>
#include <iced_x86/register.hpp>

#include "kubera_test_support.hpp"

namespace {

using namespace kubera::test;

TEST(KuberaSimd, VpAndUsesExplicitSources) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with3(
      iced_x86::Code::VEX_VPAND_XMM_XMM_XMMM128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1,
      iced_x86::Register::XMM2);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vpand xmm0, xmm1, xmm2"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 0, 0xFFFF'FFFF'FFFF'FFFFull, 0xFFFF'FFFF'FFFF'FFFFull);
               set_xmm_u64(state, 1, 0xFF00'FF00'0F0F'0F0Full, 0xAAAA'5555'1234'5678ull);
               set_xmm_u64(state, 2, 0x0F0F'F0F0'FFFF'0000ull, 0xFFFF'0000'FFFF'0000ull);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x0F00'F000'0F0F'0000ull);
               EXPECT_EQ(xmm_u64(state, 0, 1), 0xAAAA'0000'1234'0000ull);
             });
}

TEST(KuberaSimd, VaddssPreservesUpperLanesFromSrc1) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with3(
      iced_x86::Code::VEX_VADDSS_XMM_XMM_XMMM32,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1,
      iced_x86::Register::XMM2);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vaddss xmm0, xmm1, xmm2"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u32x4(state,
                             0,
                             std::bit_cast<std::uint32_t>(100.0f),
                             0xDEAD'BEEFu,
                             0xCAFEBABEu,
                             0x1357'2468u);
               set_xmm_u32x4(state,
                             1,
                             std::bit_cast<std::uint32_t>(1.5f),
                             0x1122'3344u,
                             0x5566'7788u,
                             0x99AA'BBCCu);
               set_xmm_u32x4(state,
                             2,
                             std::bit_cast<std::uint32_t>(2.25f),
                             0x0102'0304u,
                             0x0506'0708u,
                             0x090A'0B0Cu);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_FLOAT_EQ(xmm_scalar<float>(state, 0), 3.75f);
               EXPECT_EQ(xmm_u32(state, 0, 1), 0x1122'3344u);
               EXPECT_EQ(xmm_u32(state, 0, 2), 0x5566'7788u);
               EXPECT_EQ(xmm_u32(state, 0, 3), 0x99AA'BBCCu);
             });
}

TEST(KuberaSimd, SseCvtsi2ssConvertsAndPreservesUpperLanes) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::CVTSI2SS_XMM_RM64,
      iced_x86::Register::XMM0,
      iced_x86::Register::RAX);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvtsi2ss xmm0, rax"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 42;
               set_xmm_u32x4(state, 0, 0xDEAD'BEEFu, 0x1122'3344u, 0x5566'7788u, 0x99AA'BBCCu);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_FLOAT_EQ(xmm_scalar<float>(state, 0), 42.0f);
               EXPECT_EQ(xmm_u32(state, 0, 1), 0x1122'3344u);
               EXPECT_EQ(xmm_u32(state, 0, 2), 0x5566'7788u);
               EXPECT_EQ(xmm_u32(state, 0, 3), 0x99AA'BBCCu);
             });
}


TEST(KuberaSimd, Sse2Cvttsd2siTruncates) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::CVTTSD2SI_R64_XMMM64,
      iced_x86::Register::RAX,
      iced_x86::Register::XMM1);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "cvttsd2si rax, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_scalar<double>(state, 1, 3.75);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 3u);
             });
}

TEST(KuberaSimd, Sse2PextrwExtractsSelectedWord) {
  std::vector<std::uint8_t> bytes{0x66, 0x0F, 0xC5, 0xC1, 0x01};

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFF'FFFF'FFFF'FFFFull;
               set_xmm_u32x4(state, 1, 0xABCD'5678u, 0, 0, 0);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0xABCDu);
             });
}


TEST(KuberaSimd, Sse3HaddpsProducesHorizontalSums) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::HADDPS_XMM_XMMM128,
      iced_x86::Register::XMM0,
      iced_x86::Register::XMM1);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "haddps xmm0, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u32x4(state,
                             0,
                             std::bit_cast<std::uint32_t>(1.0f),
                             std::bit_cast<std::uint32_t>(2.0f),
                             std::bit_cast<std::uint32_t>(10.0f),
                             std::bit_cast<std::uint32_t>(20.0f));
               set_xmm_u32x4(state,
                             1,
                             std::bit_cast<std::uint32_t>(3.0f),
                             std::bit_cast<std::uint32_t>(4.0f),
                             std::bit_cast<std::uint32_t>(30.0f),
                             std::bit_cast<std::uint32_t>(40.0f));
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_FLOAT_EQ(xmm_scalar<float>(state, 0), 3.0f);
               EXPECT_FLOAT_EQ(std::bit_cast<float>(xmm_u32(state, 0, 1)), 30.0f);
               EXPECT_FLOAT_EQ(std::bit_cast<float>(xmm_u32(state, 0, 2)), 7.0f);
               EXPECT_FLOAT_EQ(std::bit_cast<float>(xmm_u32(state, 0, 3)), 70.0f);
             });
}

TEST(KuberaSimd, LegacyPandFaultsOnMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::PAND_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "pand xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0xFF);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8004);  // 16-byte alignment requires the low nibble to be zero
  set_xmm_u64(state, 0, 0x1111'1111'1111'1111ull, 0x2222'2222'2222'2222ull);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, LegacyPandAllowsAlignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::PAND_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "pand xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16);
  std::fill(src.begin(), src.begin() + 8, std::uint8_t{0x0F});
  std::fill(src.begin() + 8, src.end(), std::uint8_t{0xF0});
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8000);  // already 16-byte aligned
  set_xmm_u64(state, 0, 0xFFFF'FFFF'FFFF'FFFFull, 0xFFFF'FFFF'FFFF'FFFFull);

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 0, 0), 0x0F0F'0F0F'0F0F'0F0Full);
  EXPECT_EQ(xmm_u64(state, 0, 1), 0xF0F0'F0F0'F0F0'F0F0ull);
}

TEST(KuberaSimd, VexVpandAllowsMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with3(
      iced_x86::Code::VEX_VPAND_XMM_XMM_XMMM128, iced_x86::Register::XMM0, iced_x86::Register::XMM1, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vpand xmm0, xmm1, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0xFF);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8004);  // VEX form never requires alignment
  set_xmm_u64(state, 1, 0x1111'1111'1111'1111ull, 0x2222'2222'2222'2222ull);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
}

TEST(KuberaSimd, MovsldupFaultsOnMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::MOVSLDUP_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "movsldup xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0xAB);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  // MOVSLDUP's memory operand is m128 per the SDM, and DOES require 16-byte alignment despite
  // the "duplicate" naming similarity to MOVDDUP (which is genuinely unaligned-safe, m64 operand)
  // -- confirmed on real hardware via a standalone probe, not assumed.
  set_reg(state, iced_x86::Register::RAX, 0x8004);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, LegacyMovapsFaultsOnMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::MOVAPS_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "movaps xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0xFF);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8004);  // MOVAPS ("Aligned") requires 16-byte alignment

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, LegacyMovupsAllowsMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::MOVUPS_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "movups xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0xFF);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8004);  // MOVUPS ("Unaligned") never requires alignment

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
}

TEST(KuberaSimd, VexVmovapsAllowsMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::VEX_VMOVAPS_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vmovaps xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0xFF);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8004);  // VEX form never requires alignment, "Aligned" name notwithstanding

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
}

TEST(KuberaSimd, LegacyMovntpsFaultsOnMisalignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::MOVNTPS_M128_XMM, mem, iced_x86::Register::XMM0);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "movntps [rax], xmm0"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  set_reg(state, iced_x86::Register::RAX, 0x8004);  // non-temporal stores require 16-byte alignment
  set_xmm_u64(state, 0, 0x1111'1111'1111'1111ull, 0x2222'2222'2222'2222ull);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, VexVshufpsSelectsCorrectLanesFromBothSources) {
  // Regression for a real orphaned-handler gap: this handler existed and was correct, but was
  // registered under the wrong name (missing the trailing _IMM8 the real Code enum value has),
  // so real VSHUFPS/VEX_VSHUFPS_..._IMM8 hit unsupported_instruction until it was renamed and
  // wired into handled_codes.def.
  //
  // Hand-encoded (InstructionFactory can't produce an IMMEDIATE8 operand here -- iced_x86's
  // get_immediate_op_kind() is a stub that always returns UNKNOWN_OP_KIND, so every with3/with4
  // int32_t-immediate overload falls back to IMMEDIATE32, which the real encoder then rejects
  // for an instruction whose Code value expects IMMEDIATE8; see iced_x86_rflags_stub notes).
  // vshufps xmm0, xmm1, xmm2, 0xE4 -- VEX.128.0F.WIG C6 /r ib.
  const auto bytes = seven::parse_hex_bytes("C5 F0 C6 C2 E4");

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u32x4(state, 1, 0x1111'1111u, 0x2222'2222u, 0x3333'3333u, 0x4444'4444u);
               set_xmm_u32x4(state, 2, 0x5555'5555u, 0x6666'6666u, 0x7777'7777u, 0x8888'8888u);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u32(state, 0, 0), 0x1111'1111u);  // lhs[0]
               EXPECT_EQ(xmm_u32(state, 0, 1), 0x2222'2222u);  // lhs[1]
               EXPECT_EQ(xmm_u32(state, 0, 2), 0x7777'7777u);  // rhs[2]
               EXPECT_EQ(xmm_u32(state, 0, 3), 0x8888'8888u);  // rhs[3]
             });
}

TEST(KuberaSimd, VexVpslldYmmShiftsBothLanesByXmmSourcedCount) {
  // Regression for the same class of gap, plus the operand-width mismatch that made it: the real
  // Code enum's count operand is XMM-width (VEX_VPSLLD_YMM_YMM_XMMM128), not YMM-width as the
  // handler's original (wrong) name implied -- the SDM specifies the variable shift count for
  // PSLL/PSRL/PSRA always comes from the low 64 bits of a 128-bit source even for the 256-bit
  // destination form. Also verifies both 128-bit lanes of the YMM destination get shifted
  // independently by the same count, per AVX's per-lane semantics for this instruction.
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with3(
      iced_x86::Code::VEX_VPSLLD_YMM_YMM_XMMM128,
      iced_x86::Register::YMM0,
      iced_x86::Register::YMM1,
      iced_x86::Register::XMM2);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "vpslld ymm0, ymm1, xmm2"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               const seven::SimdUint lane0 = seven::SimdUint(0x00000001u) | (seven::SimdUint(0x00000002u) << 32) |
                                             (seven::SimdUint(0x00000003u) << 64) | (seven::SimdUint(0x00000004u) << 96);
               const seven::SimdUint lane1 = seven::SimdUint(0x00000005u) | (seven::SimdUint(0x00000006u) << 32) |
                                             (seven::SimdUint(0x00000007u) << 64) | (seven::SimdUint(0x00000008u) << 96);
               state.vectors[1].value = lane0 | (lane1 << 128);
               set_xmm_u64(state, 2, 4, 0);  // shift count = 4, only the low 64 bits matter
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u32(state, 0, 0), 0x00000010u);
               EXPECT_EQ(xmm_u32(state, 0, 1), 0x00000020u);
               EXPECT_EQ(xmm_u32(state, 0, 2), 0x00000030u);
               EXPECT_EQ(xmm_u32(state, 0, 3), 0x00000040u);
               EXPECT_EQ(static_cast<std::uint32_t>((state.vectors[0].value >> 128) & seven::SimdUint(0xFFFFFFFFu)), 0x00000050u);
               EXPECT_EQ(static_cast<std::uint32_t>((state.vectors[0].value >> 160) & seven::SimdUint(0xFFFFFFFFu)), 0x00000060u);
               EXPECT_EQ(static_cast<std::uint32_t>((state.vectors[0].value >> 192) & seven::SimdUint(0xFFFFFFFFu)), 0x00000070u);
               EXPECT_EQ(static_cast<std::uint32_t>((state.vectors[0].value >> 224) & seven::SimdUint(0xFFFFFFFFu)), 0x00000080u);
             });
}

TEST(KuberaSimd, Sse42Crc32MatchesCastagnoliReference) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::CRC32_R64_RM64,
      iced_x86::Register::RAX,
      iced_x86::Register::RBX);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "crc32 rax, rbx"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x1234'5678u;
               state.gpr[3] = 0x0123'4567'89AB'CDEFu;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               const auto expected = static_cast<std::uint64_t>(crc32c_update(0x1234'5678u, 0x0123'4567'89AB'CDEFu, 8));
               EXPECT_EQ(state.gpr[0], expected);
             });
}

}  // namespace

TEST(KuberaSimd, EvexBroadcastRepeatsTheElementNotTheWholeOperand) {
  constexpr std::uint64_t kData = 0x4000;

  // 62 F1 75 58 FE 07 -- vpaddd zmm0, zmm1, dword ptr [rdi]{1to16}. The broadcast element is the
  // packed operand's ELEMENT size, which lives in the raw MemorySize's element_size. The handler
  // took memory_size() (already collapsed to the full 64-byte operand width) and ran that byte
  // count back through get_size, reindexing the table to PACKED128_UINT8 and getting 16. So it
  // read 16 bytes of guest memory instead of 4 and repeated that block 4 times instead of
  // broadcasting one dword across all 16 lanes.
  run_single(seven::parse_hex_bytes("62 F1 75 58 FE 07"),
             [kData](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[7] = kData;
               state.vectors[1].value = 0;
               // Only the first dword may be read. The next twelve bytes differ so an over-read
               // shows up in the result rather than blending in.
               static constexpr std::uint32_t kWords[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
               ASSERT_TRUE(memory.write(kData, kWords, sizeof(kWords)));
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               for (std::size_t lane = 0; lane < 8; ++lane) {
                 EXPECT_EQ(xmm_u64(state, 0, lane), 0x1111'1111'1111'1111ull)
                     << "every dword lane is the broadcast element, lane pair " << lane;
               }
             });
}

TEST(KuberaSimd, PackedArithmeticRequiresAnAlignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::ADDPS_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "addps xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8004);
  set_xmm_u64(state, 0, 0, 0);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, PackedSqrtRequiresAnAlignedMemoryOperand) {
  std::vector<std::uint8_t> bytes;
  const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
  const auto instr = iced_x86::InstructionFactory::with2(iced_x86::Code::SQRTPD_XMM_XMMM128, iced_x86::Register::XMM0, mem);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "sqrtpd xmm0, [rax]"));

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  memory.map(0x8000, 0x1000);
  std::vector<std::uint8_t> src(16, 0);
  ASSERT_TRUE(memory.write(0x8000, src.data(), src.size()));
  set_reg(state, iced_x86::Register::RAX, 0x8008);
  set_xmm_u64(state, 0, 0, 0);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

// x86 MIN/MAX are defined as one compare with a fixed tie-break -- (SRC1 < SRC2) ? SRC1 : SRC2 --
// so SRC2 wins whenever the comparison is false: either operand a NaN, or both operands zero of
// whatever sign. std::fmin/std::fmax, which every MIN/MAX handler used, return the operand that is
// NOT a NaN and pick a zero by value, so a guest reading a NaN back out of MINPS got the wrong lane.

TEST(KuberaSimd, MinpsFollowsX86OperandOrderForNanAndSignedZero) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::MINPS_XMM_XMMM128, iced_x86::Register::XMM0, iced_x86::Register::XMM1);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "minps xmm0, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               // lanes, low to high: 1.0 vs 3.0 | 2.0 vs QNaN | QNaN vs 5.0 | +0.0 vs -0.0
               set_xmm_u64(state, 0, 0x40000000'3F800000ull, 0x00000000'7FC00000ull);
               set_xmm_u64(state, 1, 0x7FC00000'40400000ull, 0x80000000'40A00000ull);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x7FC00000'3F800000ull)
                   << "lane 0 takes the smaller SRC1; lane 1 must take SRC2's NaN";
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x80000000'40A00000ull)
                   << "a NaN in SRC1 yields SRC2, and +0.0 against -0.0 yields SRC2";
             });
}

TEST(KuberaSimd, MaxpsFollowsX86OperandOrderForNanAndSignedZero) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::MAXPS_XMM_XMMM128, iced_x86::Register::XMM0, iced_x86::Register::XMM1);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "maxps xmm0, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               // lanes, low to high: 9.0 vs 3.0 | 2.0 vs QNaN | QNaN vs 5.0 | +0.0 vs -0.0
               set_xmm_u64(state, 0, 0x40000000'41100000ull, 0x00000000'7FC00000ull);
               set_xmm_u64(state, 1, 0x7FC00000'40400000ull, 0x80000000'40A00000ull);
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x7FC00000'41100000ull)
                   << "lane 0 takes the larger SRC1; lane 1 must take SRC2's NaN";
               EXPECT_EQ(xmm_u64(state, 0, 1), 0x80000000'40A00000ull)
                   << "a NaN in SRC1 yields SRC2, and +0.0 against -0.0 yields SRC2 even for MAX";
             });
}

TEST(KuberaSimd, ScalarMinsdAlsoTakesSrc2WhenEitherOperandIsNan) {
  std::vector<std::uint8_t> bytes;
  const auto instr = iced_x86::InstructionFactory::with2(
      iced_x86::Code::MINSD_XMM_XMMM64, iced_x86::Register::XMM0, iced_x86::Register::XMM1);
  ASSERT_TRUE(encode_to_bytes(instr, bytes, "minsd xmm0, xmm1"));

  run_single(bytes,
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 0, 0x4000000000000000ull, 0);  // 2.0
               set_xmm_u64(state, 1, 0x7FF8000000000000ull, 0);  // QNaN
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(xmm_u64(state, 0, 0), 0x7FF8000000000000ull);
             });
}

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <span>

#include <iced_x86/code.hpp>
#include <iced_x86/decoder.hpp>
#include <iced_x86/encoding_kind.hpp>
#include <iced_x86/instruction_info.hpp>
#include <iced_x86/instruction_create.hpp>
#include <iced_x86/register.hpp>

#include "kubera_test_support.hpp"

namespace {

using namespace kubera::test;

// CMake ships three SIMD profiles and Executor::simd_profile_allows refuses any encoding or vector
// width the configured one turned off. A test written around EVEX or a 256-bit operand has a
// different correct answer in each, so the ones that cannot hold everywhere say so rather than
// failing two of the three builds we ship.
constexpr bool kProfileHasAvx512 = SEVEN_ENABLE_AVX512 != 0;
constexpr std::size_t kProfileVectorBytes = SEVEN_MAX_VECTOR_BYTES;

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

// "The VEX form never requires alignment" is true of exception classes 2 and 4 -- ADDPS, PAND and
// the rest of the arithmetic families -- but not of class 1, which is where the explicitly-aligned
// moves live. VMOVAPS keeps the requirement in every encoding, scaled to the operand width, which
// is the whole reason a compiler still picks VMOVUPS when it cannot prove alignment even though
// the two have run at the same speed since Sandy Bridge.
TEST(KuberaSimd, VexVmovapsFaultsOnMisalignedMemoryOperand) {
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
  set_reg(state, iced_x86::Register::RAX, 0x8004);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
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
  // Regression for an orphaned handler: correct, but registered without the trailing _IMM8 the real
  // Code has, so VSHUFPS hit unsupported_instruction until it was wired into handled_codes.def.
  // Hand-encoded because get_immediate_op_kind() is a stub, so the factory produces an IMMEDIATE32
  // the encoder then rejects. vshufps xmm0, xmm1, xmm2, 0xE4 -- VEX.128.0F.WIG C6 /r ib.
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
  if (kProfileVectorBytes < 32) GTEST_SKIP() << "256-bit operands are off in this SIMD profile";
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

// run_single asserts the step succeeded, so a refusal needs its own runner.
[[nodiscard]] seven::StopReason run_for_stop_reason(std::span<const std::uint8_t> bytes) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  return executor.step(state, memory).reason;
}

}  // namespace

// The gate itself, asserted from whichever side the build we are in happens to be on. This is the
// only test of simd_profile_allows that means something in all three profiles, and the only one
// that would have caught the gate being dead code back when the encoding lookup always answered
// LEGACY.
TEST(KuberaSimd, TheProfileGateRefusesExactlyTheEncodingsTheBuildDisabled) {
  const auto evex_zmm = seven::parse_hex_bytes("62 F1 75 48 FE C2");  // vpaddd zmm0, zmm1, zmm2
  const auto vex_ymm = seven::parse_hex_bytes("C5 F5 DB C2");         // vpand ymm0, ymm1, ymm2
  const auto vex_xmm = seven::parse_hex_bytes("C5 F1 DB C2");         // vpand xmm0, xmm1, xmm2

  const auto expected = [](bool allowed) {
    return allowed ? seven::StopReason::none : seven::StopReason::unsupported_instruction;
  };

  EXPECT_EQ(run_for_stop_reason(evex_zmm), expected(kProfileHasAvx512))
      << "EVEX follows SEVEN_ENABLE_AVX512";
  EXPECT_EQ(run_for_stop_reason(vex_ymm), expected(kProfileVectorBytes >= 32))
      << "a 256-bit operand follows SEVEN_MAX_VECTOR_BYTES";
  EXPECT_EQ(run_for_stop_reason(vex_xmm), seven::StopReason::none)
      << "128-bit VEX is in every profile we ship";
}

TEST(KuberaSimd, EvexBroadcastRepeatsTheElementNotTheWholeOperand) {
  if (!kProfileHasAvx512) GTEST_SKIP() << "EVEX is off in this SIMD profile";
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

// Executor::simd_profile_allows gates the AVX and AVX-512 build profiles on
// InstructionExtensions::encoding(). That used to return LEGACY for everything, which silently
// turned both gates into dead code: a build configured with AVX-512 off still ran EVEX
// instructions. It reads the generated ENC_FLAGS3 table now, and this pins that it keeps doing so,
// because nothing else in the suite would notice it regressing to a constant.

TEST(KuberaSimd, EncodingKindIsReadFromTheTableNotAssumedLegacy) {
  const auto encoding_of = [](const char* hex) {
    const auto bytes = seven::parse_hex_bytes(hex);
    iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(bytes.data(), bytes.size()), 0x1000,
                              iced_x86::DecoderOptions::NO_INVALID_CHECK);
    const auto decoded = decoder.decode();
    EXPECT_TRUE(decoded.has_value()) << hex;
    return iced_x86::InstructionExtensions::encoding(decoded.value());
  };

  EXPECT_EQ(encoding_of("0F 58 C1"), iced_x86::EncodingKind::LEGACY) << "addps xmm0, xmm1";
  EXPECT_EQ(encoding_of("C5 FC 58 C1"), iced_x86::EncodingKind::VEX) << "vaddps ymm0, ymm0, ymm1";
  EXPECT_EQ(encoding_of("62 F1 7C 48 58 C1"), iced_x86::EncodingKind::EVEX) << "vaddps zmm0, zmm0, zmm1";
}

// AVX-512 writemasking is implemented in exactly one file. The EVEX moves and the EVEX pack family
// route through shared helpers that had no notion of it, so `vmovapd zmm0{k1}, zmm1` wrote every
// lane no matter what k1 held. Stopping cleanly is what the EVEX forms with no handler at all
// already do, and it is a great deal better than a silently wrong register.
TEST(KuberaSimd, AnEvexMoveWithAMaskRegisterStopsInsteadOfIgnoringIt) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // vmovapd zmm0{k1}, zmm1
  write_bytes(memory, kBase, seven::parse_hex_bytes("62 F1 FD 49 28 C1"));

  state.opmask[1] = 0x1;                      // only lane 0 active
  state.vectors[0].value = 0;
  state.vectors[1].value = seven::SimdUint(0xFFFFFFFFFFFFFFFFull);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::unsupported_instruction);
  EXPECT_EQ(state.vectors[0].value, seven::SimdUint(0))
      << "the destination must not have been written at all";
}

TEST(KuberaSimd, TheSameEvexMoveWithNoMaskStillWorks) {
  if (!kProfileHasAvx512) GTEST_SKIP() << "EVEX is off in this SIMD profile";
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // vmovapd zmm0, zmm1  -- same instruction, no mask register named
  write_bytes(memory, kBase, seven::parse_hex_bytes("62 F1 FD 48 28 C1"));

  state.vectors[0].value = 0;
  state.vectors[1].value = seven::SimdUint(0x1234567890ABCDEFull);

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.vectors[0].value, seven::SimdUint(0x1234567890ABCDEFull));
}

// The half-register moves come in a two-operand legacy form and a three-operand VEX form, and the
// handler table gave the legacy ones the VEX shape. MOVHPS xmm1, [rbx] then read operand slot 2,
// which the instruction does not have -- an unwritten slot reads back as (REGISTER, NONE), which
// resolves to register index zero -- so it merged RAX into the high half and never touched the
// memory operand at all. Not even a page fault for an unmapped address.
TEST(KuberaSimd, LegacyMovhpsMergesTheMemoryOperandIntoTheHighHalf) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 16 0B"));  // movhps xmm1, [rbx]

  state.gpr[0] = 0xAAAA'AAAA'AAAA'AAAAull;  // rax, which the wrong slot resolved to
  state.gpr[3] = kBase + 0x200;             // rbx
  set_xmm_u64(state, 0, 0x1111'1111'1111'1111ull, 0x1111'1111'1111'1111ull);
  set_xmm_u64(state, 1, 0x2222'2222'2222'2222ull, 0x3333'3333'3333'3333ull);
  const std::uint64_t loaded = 0xDEAD'BEEF'CAFE'BABEull;
  ASSERT_TRUE(memory.write(kBase + 0x200, &loaded, sizeof(loaded)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 1, 0), 0x2222'2222'2222'2222ull) << "low half must be preserved";
  EXPECT_EQ(xmm_u64(state, 1, 1), loaded) << "high half must come from memory";
}

// Same shape, and it must still fault when the memory operand is unreachable.
TEST(KuberaSimd, LegacyMovhpsFaultsOnAnUnmappedSource) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 16 0B"));  // movhps xmm1, [rbx]
  state.gpr[3] = 0x9000'0000ull;  // nothing mapped there

  EXPECT_NE(executor.step(state, memory).reason, seven::StopReason::none);
}

// VMOVLPS loads the LOW half from memory and keeps the merge source's high half. The table had it
// on the high-merge helper, so it wrote the m64 into the wrong half of the destination.
TEST(KuberaSimd, VmovlpsLoadsTheLowHalf) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C5 E0 12 0B"));  // vmovlps xmm1, xmm3, [rbx]

  state.gpr[3] = kBase + 0x200;  // rbx
  set_xmm_u64(state, 3, 0x4444'4444'4444'4444ull, 0x3333'3333'3333'3333ull);
  const std::uint64_t loaded = 0xDEAD'BEEF'CAFE'BABEull;
  ASSERT_TRUE(memory.write(kBase + 0x200, &loaded, sizeof(loaded)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 1, 0), loaded) << "low half must come from memory";
  EXPECT_EQ(xmm_u64(state, 1, 1), 0x3333'3333'3333'3333ull) << "high half from the merge source";
}

// This encoding is a three-operand VMOVHPS: high half from memory, low half from the merge
// source. The decoder used to hand it the EVEX VMOVLHPS Code -- a register-only form -- because
// the table derives the memory-form Code by counting one on from the register form and the two
// are not adjacent in this enum. The handler then read a register out of a slot the decoder had
// marked as memory, which resolves to XMM0, and the memory operand went untouched.
TEST(KuberaSimd, VmovhpsWithAMemorySourceIsNotTheRegisterOnlyForm) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C5 E0 16 0B"));  // vmovhps xmm1, xmm3, [rbx]

  state.gpr[3] = kBase + 0x200;  // rbx
  set_xmm_u64(state, 0, 0x9999'9999'9999'9999ull, 0x9999'9999'9999'9999ull);
  set_xmm_u64(state, 1, 0, 0);
  set_xmm_u64(state, 3, 0x4444'4444'4444'4444ull, 0x5555'5555'5555'5555ull);
  const std::uint64_t loaded = 0xDEAD'BEEF'CAFE'BABEull;
  ASSERT_TRUE(memory.write(kBase + 0x200, &loaded, sizeof(loaded)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 1, 0), 0x4444'4444'4444'4444ull) << "low half from the merge source";
  EXPECT_EQ(xmm_u64(state, 1, 1), loaded) << "high half from memory";
}

// The register-only form at the same opcode still decodes and runs as itself.
TEST(KuberaSimd, VmovlhpsStillMovesTheLowHalvesUp) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C5 E0 16 CB"));  // vmovlhps xmm1, xmm3, xmm3

  set_xmm_u64(state, 1, 0, 0);
  set_xmm_u64(state, 3, 0x4444'4444'4444'4444ull, 0x5555'5555'5555'5555ull);

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 1, 0), 0x4444'4444'4444'4444ull);
  EXPECT_EQ(xmm_u64(state, 1, 1), 0x4444'4444'4444'4444ull);
}

// vex_pack refuses a mask it cannot honour; vex_unpack, sitting right next to it in the same file
// and backing the twelve reachable EVEX VPUNPCK codes, did not. `vpunpcklbw xmm0{k1}, xmm1, xmm2`
// wrote all sixteen lanes whatever k1 held, which is a wrong register rather than a missing feature.
TEST(KuberaSimd, AnEvexUnpackWithAMaskRegisterStopsInsteadOfIgnoringIt) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // vpunpcklbw xmm0{k1}, xmm1, xmm2
  write_bytes(memory, kBase, seven::parse_hex_bytes("62 F1 75 09 60 C2"));

  state.opmask[1] = 0x1;  // only lane 0 active
  set_xmm_u64(state, 0, 0, 0);
  set_xmm_u64(state, 1, 0x0706050403020100ull, 0);
  set_xmm_u64(state, 2, 0x1716151413121110ull, 0);

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::unsupported_instruction);
  EXPECT_EQ(state.vectors[0].value, seven::SimdUint(0))
      << "the destination must not have been written at all";
}

TEST(KuberaSimd, TheSameEvexUnpackWithNoMaskStillWorks) {
  if (!kProfileHasAvx512) GTEST_SKIP() << "EVEX is off in this SIMD profile";
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // vpunpcklbw xmm0, xmm1, xmm2 -- same instruction, no mask register named
  write_bytes(memory, kBase, seven::parse_hex_bytes("62 F1 75 08 60 C2"));

  set_xmm_u64(state, 0, 0, 0);
  set_xmm_u64(state, 1, 0x0706050403020100ull, 0);
  set_xmm_u64(state, 2, 0x1716151413121110ull, 0);

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 0, 0), 0x1303120211011000ull);
  EXPECT_EQ(xmm_u64(state, 0, 1), 0x1707160615051404ull);
}

// The legacy-SSE 16-byte rule reached the arithmetic, logic, shift, pack and shuffle families but
// stopped short of three more legacy m128 forms: CMPPD, the packed double-to-integer converts, and
// the PCMPxSTRx string compares. All three are exception class 2 or 4, so real hardware raises
// #GP(0) on a misaligned memory operand exactly like ADDPS does.
TEST(KuberaSimd, PackedCompareRequiresAnAlignedMemoryOperand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("66 0F C2 00 00"));  // cmppd xmm0, [rax], 0
  memory.map(0x8000, 0x1000);
  set_reg(state, iced_x86::Register::RAX, 0x8004);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, PackedDoubleToIntConvertRequiresAnAlignedMemoryOperand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("F2 0F E6 00"));  // cvtpd2dq xmm0, [rax]
  memory.map(0x8000, 0x1000);
  set_reg(state, iced_x86::Register::RAX, 0x8008);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, PcmpistriRequiresAnAlignedMemoryOperand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("66 0F 3A 63 00 00"));  // pcmpistri xmm0, [rax], 0
  memory.map(0x8000, 0x1000);
  set_reg(state, iced_x86::Register::RAX, 0x8001);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

// The unaligned twin at the neighbouring opcode must stay unaligned-safe.
TEST(KuberaSimd, VexMovupsAcceptsAMisalignedMemoryOperand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C5 F8 10 00"));  // vmovups xmm0, [rax]
  memory.map(0x8000, 0x1000);
  const std::uint64_t loaded = 0xDEAD'BEEF'CAFE'BABEull;
  ASSERT_TRUE(memory.write(0x8004, &loaded, sizeof(loaded)));
  set_reg(state, iced_x86::Register::RAX, 0x8004);

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(xmm_u64(state, 0, 0), loaded);
}

// The required alignment is the operand width, not a flat 16 bytes, so a 32-byte-aligned address
// still faults at zmm width.
TEST(KuberaSimd, EvexMovapdRequiresTheFullVectorWidthOfAlignment) {
  if (!kProfileHasAvx512) GTEST_SKIP() << "EVEX is off in this SIMD profile";
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("62 F1 FD 48 28 00"));  // vmovapd zmm0, [rax]
  memory.map(0x8000, 0x1000);
  set_reg(state, iced_x86::Register::RAX, 0x8020);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

TEST(KuberaSimd, VexNonTemporalStoreRequiresAnAlignedMemoryOperand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C5 F9 E7 00"));  // vmovntdq [rax], xmm0
  memory.map(0x8000, 0x1000);
  set_reg(state, iced_x86::Register::RAX, 0x8008);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
}

// Every EVEX form in simd_shuffle.cpp and simd_pack.cpp was written, compiled and then never
// reachable: each carried a function name that no Code enum value matches (a spurious source
// operand on the three duplicating moves, a missing B32/B64 broadcast suffix on the rest), so none
// of them was ever registered and every one of these instructions stopped as
// unsupported_instruction with a working handler sitting behind the wrong name.
TEST(KuberaSimd, TheEvexShuffleAndPackFormsAreReachableAndNotJustImplemented) {
  if (!kProfileHasAvx512) GTEST_SKIP() << "EVEX is off in this SIMD profile";

  // 62 F1 6C 08 C6 CB 1B -- vshufps xmm1, xmm2, xmm3, 0x1B.
  // imm 0x1B selects lhs[3], lhs[2], rhs[1], rhs[0].
  run_single(seven::parse_hex_bytes("62 F1 6C 08 C6 CB 1B"),
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 2, 0x2222'2222'1111'1111ull, 0x4444'4444'3333'3333ull);
               set_xmm_u64(state, 3, 0xBBBB'BBBB'AAAA'AAAAull, 0xDDDD'DDDD'CCCC'CCCCull);
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(xmm_u64(state, 1, 0), 0x3333'3333'4444'4444ull);
               EXPECT_EQ(xmm_u64(state, 1, 1), 0xAAAA'AAAA'BBBB'BBBBull);
             });

  // 62 F1 6C 08 14 CB -- vunpcklps xmm1, xmm2, xmm3: interleaves the low two dwords of each.
  run_single(seven::parse_hex_bytes("62 F1 6C 08 14 CB"),
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 2, 0x2222'2222'1111'1111ull, 0x4444'4444'3333'3333ull);
               set_xmm_u64(state, 3, 0xBBBB'BBBB'AAAA'AAAAull, 0xDDDD'DDDD'CCCC'CCCCull);
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(xmm_u64(state, 1, 0), 0xAAAA'AAAA'1111'1111ull);
               EXPECT_EQ(xmm_u64(state, 1, 1), 0xBBBB'BBBB'2222'2222ull);
             });

  // 62 F1 7E 08 12 CA -- vmovsldup xmm1, xmm2. The two-operand form whose handler used to be named
  // as if it took three.
  run_single(seven::parse_hex_bytes("62 F1 7E 08 12 CA"),
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 2, 0x2222'2222'1111'1111ull, 0x4444'4444'3333'3333ull);
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(xmm_u64(state, 1, 0), 0x1111'1111'1111'1111ull);
               EXPECT_EQ(xmm_u64(state, 1, 1), 0x3333'3333'3333'3333ull);
             });

  // 62 F1 6D 08 62 CB -- vpunpckldq xmm1, xmm2, xmm3, the same shape over in simd_pack.cpp.
  run_single(seven::parse_hex_bytes("62 F1 6D 08 62 CB"),
             [](seven::CpuState& state, seven::Memory&) {
               set_xmm_u64(state, 2, 0x2222'2222'1111'1111ull, 0x4444'4444'3333'3333ull);
               set_xmm_u64(state, 3, 0xBBBB'BBBB'AAAA'AAAAull, 0xDDDD'DDDD'CCCC'CCCCull);
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(xmm_u64(state, 1, 0), 0xAAAA'AAAA'1111'1111ull);
               EXPECT_EQ(xmm_u64(state, 1, 1), 0xBBBB'BBBB'2222'2222ull);
             });
}

// The other half of why these were held back: both files read a memory operand without ever asking
// whether EVEX.b was set, so a {1toN} source read the full operand width instead of one element.
TEST(KuberaSimd, TheEvexShuffleBroadcastReadsOneElementNotTheWholeOperand) {
  if (!kProfileHasAvx512) GTEST_SKIP() << "EVEX is off in this SIMD profile";
  constexpr std::uint64_t kData = 0x4000;

  // 62 F1 6C 18 C6 8F 00 00 00 00 90 -- vshufps xmm1, xmm2, dword ptr [rdi]{1to4}, 0x90.
  // imm 0x90 takes lhs[0], lhs[0], rhs[1], rhs[2]. Reading one dword and repeating it makes both
  // rhs lanes the first dword; reading all sixteen bytes would take the second and third instead,
  // which is what the four distinct values in memory are there to expose.
  run_single(seven::parse_hex_bytes("62 F1 6C 18 C6 8F 00 00 00 00 90"),
             [kData](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[7] = kData;
               set_xmm_u64(state, 2, 0x2222'2222'1111'1111ull, 0x4444'4444'3333'3333ull);
               static constexpr std::uint32_t kWords[4] = {0xAAAA'AAAAu, 0xBBBB'BBBBu, 0xCCCC'CCCCu, 0xDDDD'DDDDu};
               ASSERT_TRUE(memory.write(kData, kWords, sizeof(kWords)));
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(xmm_u64(state, 1, 0), 0x1111'1111'1111'1111ull);
               EXPECT_EQ(xmm_u64(state, 1, 1), 0xAAAA'AAAA'AAAA'AAAAull)
                   << "both lanes are the broadcast element, not the second and third dwords";
             });
}

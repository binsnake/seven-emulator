#include <gtest/gtest.h>

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

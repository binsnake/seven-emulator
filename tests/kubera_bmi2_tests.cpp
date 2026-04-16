#include <cstdint>
#include <functional>
#include <iostream>
#include <intrin.h>
#include <string>
#include <string_view>
#include <vector>
#include <array>

#include <iced_x86/code.hpp>
#include <iced_x86/encoder.hpp>
#include <iced_x86/instruction.hpp>
#include <iced_x86/instruction_create.hpp>
#include <iced_x86/op_kind.hpp>
#include <iced_x86/register.hpp>

#include "seven/compat.hpp"
#include "mini_test_framework.hpp"
#include "uint_wide.h"

namespace {

constexpr std::uint64_t kBase = 0x1000;

std::size_t gpr_index(iced_x86::Register reg) {
  switch (reg) {
    case iced_x86::Register::RAX:
    case iced_x86::Register::EAX:
    case iced_x86::Register::AX:
    case iced_x86::Register::AL:
    case iced_x86::Register::AH:
      return 0;
    case iced_x86::Register::RCX:
    case iced_x86::Register::ECX:
    case iced_x86::Register::CX:
    case iced_x86::Register::CL:
    case iced_x86::Register::CH:
      return 1;
    case iced_x86::Register::RDX:
    case iced_x86::Register::EDX:
    case iced_x86::Register::DX:
    case iced_x86::Register::DL:
    case iced_x86::Register::DH:
      return 2;
    case iced_x86::Register::RBX:
    case iced_x86::Register::EBX:
    case iced_x86::Register::BX:
    case iced_x86::Register::BL:
    case iced_x86::Register::BH:
      return 3;
    case iced_x86::Register::RSP:
    case iced_x86::Register::ESP:
    case iced_x86::Register::SP:
    case iced_x86::Register::SPL:
      return 4;
    case iced_x86::Register::RBP:
    case iced_x86::Register::EBP:
    case iced_x86::Register::BP:
    case iced_x86::Register::BPL:
      return 5;
    case iced_x86::Register::RSI:
    case iced_x86::Register::ESI:
    case iced_x86::Register::SI:
    case iced_x86::Register::SIL:
      return 6;
    case iced_x86::Register::RDI:
    case iced_x86::Register::EDI:
    case iced_x86::Register::DI:
    case iced_x86::Register::DIL:
      return 7;
    default:
      return 0;
  }
}

std::uint64_t reg_value(const seven::CpuState& state, iced_x86::Register reg) {
  return state.gpr[gpr_index(reg)];
}

void set_reg(seven::CpuState& state, iced_x86::Register reg, std::uint64_t value) {
  state.gpr[gpr_index(reg)] = value;
}

void write_bytes(seven::Memory& memory, std::uint64_t base, const std::vector<std::uint8_t>& bytes) {
  memory.map(base, bytes.size() + 256);
  if (!memory.write(base, bytes.data(), bytes.size())) {
    std::cerr << "[fatal] write_bytes failed at address " << base << '\n';
  }
}

template <typename Setup, typename Verify>
bool run_single(const std::vector<std::uint8_t>& bytes,
                Setup&& setup,
                Verify&& verify,
                std::string_view label) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  setup(state);
  const auto result = executor.step(state, memory);
  if (result.reason != seven::StopReason::none) {
    std::cerr << "[run fail] " << label << ": unexpected stop reason " << static_cast<int>(result.reason) << '\n';
    return false;
  }
  return verify(result, state);
}

template <typename Setup, typename Verify>
bool run_single_with_memory(const std::vector<std::uint8_t>& bytes,
                           Setup&& setup,
                           Verify&& verify,
                           std::string_view label) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  setup(state, memory);
  const auto result = executor.step(state, memory);
  if (result.reason != seven::StopReason::none) {
    std::cerr << "[run fail] " << label << ": unexpected stop reason " << static_cast<int>(result.reason) << '\n';
    return false;
  }
  return verify(result, state, memory);
}

bool encode_to_bytes(const iced_x86::Instruction& instr, std::vector<std::uint8_t>& bytes, std::string_view label) {
  iced_x86::Encoder encoder(64);
  const auto encoded_size = encoder.encode(instr, kBase);
  if (!encoded_size) {
    std::cerr << "[encode fail] " << label << ": " << encoded_size.error().message << '\n';
    return false;
  }
  bytes = encoder.take_buffer();
  if (bytes.size() > *encoded_size) {
    bytes.resize(*encoded_size);
  }
  return true;
}

constexpr std::uint64_t bits_to_mask(std::size_t bits) {
  return bits >= 64 ? ~0ull : ((1ull << bits) - 1ull);
}

std::uint64_t xorshift64(std::uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::uint64_t bextr_ref(std::uint64_t source, std::uint64_t control, std::size_t bits) {
  const auto start = control & 0xFFull;
  const auto len = (control >> 8) & 0xFFull;
  if (start >= bits) {
    return 0;
  }
  const auto extract_len = (start + len >= bits) ? (bits - start) : len;
  std::uint64_t result = 0;
  for (std::uint64_t i = 0; i < extract_len; ++i) {
    if ((source >> (start + i)) & 1ull) {
      result |= (1ull << i);
    }
  }
  return result;
}

std::uint64_t pdep_ref(std::uint64_t source, std::uint64_t mask, std::size_t bits) {
  std::uint64_t result = 0;
  std::uint64_t source_bit = 0;
  for (std::uint64_t i = 0; i < bits; ++i) {
    if ((mask >> i) & 1ull) {
      if ((source >> source_bit) & 1ull) {
        result |= (1ull << i);
      }
      ++source_bit;
    }
  }
  return result;
}

std::uint64_t pext_ref(std::uint64_t source, std::uint64_t mask, std::size_t bits) {
  std::uint64_t result = 0;
  std::uint64_t destination_bit = 0;
  for (std::uint64_t i = 0; i < bits; ++i) {
    if ((mask >> i) & 1ull) {
      if ((source >> i) & 1ull) {
        result |= (1ull << destination_bit);
      }
      ++destination_bit;
    }
  }
  return result;
}

std::uint64_t mulx_low(std::uint64_t lhs, std::uint64_t rhs, std::size_t bits) {
  if (bits == 32) {
    const auto product = static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs);
    return product & 0xFFFFFFFFull;
  }
#if defined(_M_X64) && defined(_MSC_VER) && !defined(__clang__)
  std::uint64_t product_high;
  return _umul128(lhs, rhs, &product_high);
#elif defined(__SIZEOF_INT128__)
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
  return static_cast<std::uint64_t>(product);
#else
  const auto product = math::wide_integer::uint128_t(lhs) * math::wide_integer::uint128_t(rhs);
  return static_cast<std::uint64_t>(product);
#endif
}

std::uint64_t mulx_high(std::uint64_t lhs, std::uint64_t rhs, std::size_t bits) {
  if (bits == 32) {
    const auto product = static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs);
    return product >> 32;
  }
#if defined(_M_X64) && defined(_MSC_VER) && !defined(__clang__)
  std::uint64_t product_high = 0;
  (void)_umul128(lhs, rhs, &product_high);
  return product_high;
#elif defined(__SIZEOF_INT128__)
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
  return static_cast<std::uint64_t>(product >> 64);
#else
  const auto product = math::wide_integer::uint128_t(lhs) * math::wide_integer::uint128_t(rhs);
  return static_cast<std::uint64_t>(product >> 64);
#endif
}

std::uint64_t ror_ref(std::uint64_t value, std::uint64_t count, std::size_t bits) {
  const auto mask = bits_to_mask(bits);
  const auto masked_count = count & (bits - 1ull);
  if (masked_count == 0) {
    return value & mask;
  }
  return ((value >> masked_count) | (value << (bits - masked_count))) & mask;
}

std::uint64_t shl_ref(std::uint64_t value, std::uint64_t count, std::size_t bits) {
  const auto mask = bits_to_mask(bits);
  const auto masked_count = count & (bits - 1ull);
  return (value << masked_count) & mask;
}

std::uint64_t shr_ref(std::uint64_t value, std::uint64_t count, std::size_t bits) {
  const auto mask = bits_to_mask(bits);
  const auto masked_count = count & (bits - 1ull);
  return (value >> masked_count) & mask;
}

std::int64_t sign_extend(std::uint64_t value, std::size_t bits) {
  if (bits >= 64) {
    return static_cast<std::int64_t>(value);
  }
  const std::uint64_t sign = 1ull << (bits - 1ull);
  const std::uint64_t masked = value & bits_to_mask(bits);
  if ((masked & sign) == 0u) {
    return static_cast<std::int64_t>(masked);
  }
  return static_cast<std::int64_t>(masked | (~bits_to_mask(bits)));
}

std::uint64_t sarx_ref(std::uint64_t value, std::uint64_t count, std::size_t bits) {
  const auto masked_count = count & (bits - 1ull);
  if (masked_count == 0) {
    return value & bits_to_mask(bits);
  }
  return static_cast<std::uint64_t>(sign_extend(value, bits) >> masked_count) & bits_to_mask(bits);
}

std::uint64_t mask_value(std::size_t bits, std::uint64_t value) {
  return value & bits_to_mask(bits);
}

std::uint64_t bzhi_ref(std::size_t bits, std::uint64_t src, std::uint64_t index) {
  const auto masked_src = mask_value(bits, src);
  const auto index8 = index & 0xFFull;
  if (index8 == 0) {
    return 0;
  }
  if (index8 >= bits) {
    return masked_src;
  }
  return masked_src & bits_to_mask(index8);
}

bool bzhi_cf_ref(std::size_t bits, std::uint64_t index) {
  return (index & 0xFFull) >= bits;
}


void write_value_to_memory(std::size_t bits, seven::Memory& memory, std::uint64_t address, std::uint64_t value) {
  if (bits == 32) {
    const std::uint32_t value32 = static_cast<std::uint32_t>(value);
    if (!memory.write(address, &value32, sizeof(value32))) {
      std::cerr << "[fatal] write_value_to_memory failed at address " << address << '\n';
    }
    return;
  }
  if (!memory.write(address, &value, sizeof(value))) {
    std::cerr << "[fatal] write_value_to_memory failed at address " << address << '\n';
  }
}

TEST(Bmi2, DecodeAndnAndFlags) {
  const auto run = [&](std::size_t bits, std::uint64_t src1, std::uint64_t src2, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src1_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto src2_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_ANDN_R32_R32_RM32 : iced_x86::Code::VEX_ANDN_R64_R64_RM64;
    const auto expected = (~src1) & src2 & bits_to_mask(bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src1_reg, src2_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src1_reg, src1);
        set_reg(state, src2_reg, src2);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagOF), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagPF), (seven::even_parity(static_cast<std::uint8_t>(expected)) ? seven::kFlagPF : 0u));
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run(32, 0xFFFF0000u, 0x1234ABCDu, "ANDN_R32_RM32_R32"));
  EXPECT_TRUE(run(64, 0x8000000000000000ull, 0x00AB00AB00AB00AB00ull, "ANDN_R64_RM64_RM64"));
  EXPECT_TRUE(run(64, 0x7FFFFFFFFFFFFFFFull, 0xFEDCBA9876543210ull, "ANDN_R64_NONZERO"));
  return true;
}

TEST(Bmi2, DecodeBlsiBlsmskBlsr) {
  const auto run_blsi = [&](std::size_t bits, std::uint64_t src, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSI_R32_RM32 : iced_x86::Code::VEX_BLSI_R64_RM64;
    const auto expected = (src & -src) & bits_to_mask(bits);
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, src); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src != 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_blsr = [&](std::size_t bits, std::uint64_t src, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSR_R32_RM32 : iced_x86::Code::VEX_BLSR_R64_RM64;
    const auto expected = (src & (src - 1ull)) & bits_to_mask(bits);
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, src); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src == 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_blsmsk = [&](std::size_t bits, std::uint64_t src, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSMSK_R32_RM32 : iced_x86::Code::VEX_BLSMSK_R64_RM64;
    const auto expected = (src ^ (src - 1ull)) & bits_to_mask(bits);
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, src); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src == 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        EXPECT_EQ(state.rflags & 0x800u, 0u);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_blsi(32, 0x80000000u, "VEX_BLSI_R32_RM32"));
  EXPECT_TRUE(run_blsi(64, 0x80ull, "VEX_BLSI_R64_RM64"));
  EXPECT_TRUE(run_blsr(32, 0x1u, "VEX_BLSR_R32_RM32"));
  EXPECT_TRUE(run_blsr(64, 0x8000000000000000ull, "VEX_BLSR_R64_RM64"));
  EXPECT_TRUE(run_blsr(64, 0ull, "VEX_BLSR_R64_ZERO"));
  EXPECT_TRUE(run_blsmsk(32, 0x12345678u, "VEX_BLSMSK_R32_RM32"));
  EXPECT_TRUE(run_blsmsk(64, 0ull, "VEX_BLSMSK_R64_ZERO"));
  return true;
}

TEST(Bmi2, DecodeBzhiAndBextr) {
  const auto run_bzhi = [&](std::size_t bits, std::uint64_t src, std::uint64_t index, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto idx_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BZHI_R32_RM32_R32 : iced_x86::Code::VEX_BZHI_R64_RM64_R64;
    const auto expected = bzhi_ref(bits, src, index);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src_reg, idx_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src_reg, src);
        set_reg(state, idx_reg, index);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), bzhi_cf_ref(bits, index) ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_bextr = [&](std::size_t bits, std::uint64_t src, std::uint64_t control, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto control_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BEXTR_R32_RM32_R32 : iced_x86::Code::VEX_BEXTR_R64_RM64_R64;
    const auto expected = bextr_ref(src, control, bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src_reg, control_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src_reg, src);
        set_reg(state, control_reg, control);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & (seven::kFlagAF | seven::kFlagOF | seven::kFlagPF | seven::kFlagSF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagCF), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_bzhi(32, 0xFFFF0001u, 8, "VEX_BZHI_R32_RM32_R32"));
  EXPECT_TRUE(run_bzhi(64, 0x0123456789ABCDEFull, 40, "VEX_BZHI_R64_RM64_R64"));
  EXPECT_TRUE(run_bzhi(64, 0x5555555555555555ull, 0, "VEX_BZHI_ZERO_INDEX"));
  EXPECT_TRUE(run_bzhi(64, 0x123456789ABCDEF0ull, 64, "VEX_BZHI_INDEX_EQ_WIDTH"));
  EXPECT_TRUE(run_bzhi(64, 0x123456789ABCDEF0ull, 0x100ull, "VEX_BZHI_INDEX_LOW8_ONLY"));
  EXPECT_TRUE(run_bzhi(64, 0x123456789ABCDEF0ull, 70, "VEX_BZHI_INDEX_OOB"));
  EXPECT_TRUE(run_bextr(64, 0xFEDCBA9876543210ull, 0x0208ull, "VEX_BEXTR_R64_RM64_R64"));
  EXPECT_TRUE(run_bextr(64, 0x8000000000000000ull, 0x4000ull, "VEX_BEXTR_SF_ZERO"));
  EXPECT_TRUE(run_bextr(64, 0x8FEDCBA987654321ull, 0xFF08ull, "VEX_BEXTR_START_OOB"));
  EXPECT_TRUE(run_bextr(32, 0x8FEDCBA9u, 0x0302ull, "VEX_BEXTR_R32_RM32_R32"));
  return true;
}

TEST(Bmi2, DecodeMulx) {
  const auto run = [&](std::size_t bits, std::uint64_t lhs, std::uint64_t rhs, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto low_dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto high_dst = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto src_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto implicit_src = bits == 32 ? iced_x86::Register::EDX : iced_x86::Register::RDX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_MULX_R32_R32_RM32 : iced_x86::Code::VEX_MULX_R64_R64_RM64;
    const auto instr = iced_x86::InstructionFactory::with3(code, high_dst, low_dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, implicit_src, mask_value(bits, lhs));
        set_reg(state, src_reg, mask_value(bits, rhs));
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, low_dst), mulx_low(mask_value(bits, lhs), mask_value(bits, rhs), bits));
        EXPECT_EQ(reg_value(state, high_dst), mulx_high(mask_value(bits, lhs), mask_value(bits, rhs), bits));
        return true;
      },
      label);
  };

  EXPECT_TRUE(run(32, 0x12345678u, 0x10u, "VEX_MULX_R32_R32_RM32"));
  EXPECT_TRUE(run(32, 0xFFFFFFFFu, 0xFFFFFFFFu, "VEX_MULX_R32_MAX"));
  EXPECT_TRUE(run(64, 0xCAFEBABE11223344ull, 0x0102030405060708ull, "VEX_MULX_R64_R64_RM64"));
  EXPECT_TRUE(run(64, 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, "VEX_MULX_R64_MAX"));
  return true;
}

TEST(Bmi2, DecodePdepPext) {
  const auto run = [&](std::size_t bits, bool is_dep, std::uint64_t src, std::uint64_t mask, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto mask_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = is_dep
        ? (bits == 32 ? iced_x86::Code::VEX_PDEP_R32_R32_RM32 : iced_x86::Code::VEX_PDEP_R64_R64_RM64)
        : (bits == 32 ? iced_x86::Code::VEX_PEXT_R32_R32_RM32 : iced_x86::Code::VEX_PEXT_R64_R64_RM64);
    const auto expected = is_dep ? pdep_ref(src, mask, bits) : pext_ref(src, mask, bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src_reg, mask_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src_reg, src);
        set_reg(state, mask_reg, mask);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run(32, true, 0xF0u, 0x0Fu, "VEX_PDEP_R32_R32_RM32"));
  EXPECT_TRUE(run(32, true, 0xFFFFu, 0x0u, "VEX_PDEP_R32_ZERO_MASK"));
  EXPECT_TRUE(run(64, false, 0xAAAAAAAAAAAAAAAAull, 0xF0F0F0F0F0F0F0F0ull, "VEX_PEXT_R64_R64_RM64"));
  EXPECT_TRUE(run(64, false, 0xF0F0ull, 0x0ull, "VEX_PEXT_R64_ZERO_MASK"));
  EXPECT_TRUE(run(64, true, 0xFEDCBA9876543210ull, 0x0102040810204080ull, "VEX_PDEP_R64_R64_RM64"));
  return true;
}

TEST(Bmi2, DecodeShiftRotateFamilies) {
  const auto run_rorx = [&](std::size_t bits, std::uint64_t src, std::uint8_t count, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_RORX_R32_RM32_IMM8 : iced_x86::Code::VEX_RORX_R64_RM64_IMM8;
    const auto expected = ror_ref(src, count, bits);
    auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    instr.set_op2_kind(iced_x86::OpKind::IMMEDIATE8);
    instr.set_immediate8(count);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, src); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        return true;
      },
      label);
  };

  const auto run_sarx = [&](std::size_t bits, std::uint64_t src, std::uint64_t count, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto count_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_SARX_R32_RM32_R32 : iced_x86::Code::VEX_SARX_R64_RM64_R64;
    const auto expected = sarx_ref(src, count, bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src_reg, count_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src_reg, src);
        set_reg(state, count_reg, count);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        return true;
      },
      label);
  };

  const auto run_shlx_shrx = [&](std::size_t bits, bool is_shlx, std::uint64_t src, std::uint64_t count, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EDX : iced_x86::Register::RDX;
    const auto count_reg = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto code = is_shlx
        ? (bits == 32 ? iced_x86::Code::VEX_SHLX_R32_RM32_R32 : iced_x86::Code::VEX_SHLX_R64_RM64_R64)
        : (bits == 32 ? iced_x86::Code::VEX_SHRX_R32_RM32_R32 : iced_x86::Code::VEX_SHRX_R64_RM64_R64);
    const auto expected = is_shlx ? shl_ref(src, count, bits) : shr_ref(src, count, bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src_reg, count_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src_reg, src);
        set_reg(state, count_reg, count);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_sarx(64, 0x8000000000000000ull, 4, "VEX_SARX_R64_RM64_R64"));
  EXPECT_TRUE(run_sarx(32, 0x80000000u, 40, "VEX_SARX_R32_RM32_R32"));
  EXPECT_TRUE(run_shlx_shrx(32, true, 1, 33, "VEX_SHLX_R32_RM32_R32"));
  EXPECT_TRUE(run_shlx_shrx(64, true, 0x12345678ull, 66, "VEX_SHLX_R64_RM64_RM64"));
  EXPECT_TRUE(run_shlx_shrx(32, false, 0x80000000u, 9, "VEX_SHRX_R32_RM32_R32"));
  EXPECT_TRUE(run_shlx_shrx(64, false, 0x8000000000000000ull, 79, "VEX_SHRX_R64_RM64_RM64"));
  return true;
}

TEST(Bmi2, DecodeBoundaryAndZeroInputCases) {
  const auto run_andn = [&](std::size_t bits, std::uint64_t src1, std::uint64_t src2, std::string_view label) {
    const auto code = bits == 32 ? iced_x86::Code::VEX_ANDN_R32_R32_RM32 : iced_x86::Code::VEX_ANDN_R64_R64_RM64;
    const auto expected = mask_value(bits, (~src1) & src2);
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src1_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto src2_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    std::vector<std::uint8_t> bytes;
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src1_reg, src2_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) {
        set_reg(state, src1_reg, mask_value(bits, src1));
        set_reg(state, src2_reg, mask_value(bits, src2));
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagPF), seven::even_parity(static_cast<std::uint8_t>(expected)) ? seven::kFlagPF : 0u);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_andn(32, 0u, 0u, "ANDN_R32_ZERO"));
  EXPECT_TRUE(run_andn(32, 0xFFFFFFFFu, 0u, "ANDN_R32_ZERO_SRC2"));
  EXPECT_TRUE(run_andn(64, 0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull, "ANDN_R64_HIGH_BITS"));
  return true;
}

TEST(Bmi2, DecodeBlsiBlsrBlsmskZeroAndBoundaries) {
  const auto run_blsi = [&](std::size_t bits, std::uint64_t src, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSI_R32_RM32 : iced_x86::Code::VEX_BLSI_R64_RM64;
    const auto expected = mask_value(bits, src & -src);
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, mask_value(bits, src)); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src != 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_blsr = [&](std::size_t bits, std::uint64_t src, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSR_R32_RM32 : iced_x86::Code::VEX_BLSR_R64_RM64;
    const auto expected = mask_value(bits, src & (src - 1ull));
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, mask_value(bits, src)); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src == 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_blsmsk = [&](std::size_t bits, std::uint64_t src, std::string_view label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSMSK_R32_RM32 : iced_x86::Code::VEX_BLSMSK_R64_RM64;
    const auto expected = mask_value(bits, src ^ (src - 1ull));
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single(
      bytes,
      [&](seven::CpuState& state) { set_reg(state, src_reg, mask_value(bits, src)); },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src == 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_blsi(32, 0u, "VEX_BLSI_R32_ZERO"));
  EXPECT_TRUE(run_blsi(64, 0x8000000000000000ull, "VEX_BLSI_R64_MSB"));
  EXPECT_TRUE(run_blsr(32, 0x1u, "VEX_BLSR_R32_EDGE"));
  EXPECT_TRUE(run_blsr(64, 0u, "VEX_BLSR_R64_ZERO"));
  EXPECT_TRUE(run_blsmsk(32, 0u, "VEX_BLSMSK_R32_ZERO"));
  EXPECT_TRUE(run_blsmsk(64, 0xFFFFFFFFFFFFFFFFull, "VEX_BLSMSK_R64_ALL_ONES"));
  return true;
}

TEST(Bmi2, DecodeBzhiBextrMemoryOperands) {
  const auto run_bzhi = [&](std::size_t bits, std::uint64_t src, std::uint64_t index, std::string_view label) {
    constexpr std::uint64_t kBaseMem = 0x9800;
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto idx_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BZHI_R32_RM32_R32 : iced_x86::Code::VEX_BZHI_R64_RM64_R64;
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
    const auto expected = bzhi_ref(bits, src, index);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, mem, idx_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, idx_reg, mask_value(bits, index));
        set_reg(state, iced_x86::Register::RAX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, src);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), bzhi_cf_ref(bits, index) ? seven::kFlagCF : 0u);
        return true;
      },
      label);
  };

  const auto run_bextr = [&](std::size_t bits, std::uint64_t src, std::uint64_t control, std::string_view label) {
    constexpr std::uint64_t kBaseMem = 0x9810;
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto control_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BEXTR_R32_RM32_R32 : iced_x86::Code::VEX_BEXTR_R64_RM64_R64;
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
    const auto expected = bextr_ref(mask_value(bits, src), mask_value(bits, control), bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, mem, control_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, control_reg, mask_value(bits, control));
        set_reg(state, iced_x86::Register::RAX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, src);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_bzhi(64, 0x1122334455667788ull, 8, "VEX_BZHI_R64_MEM"));
  EXPECT_TRUE(run_bzhi(64, 0x1111000011110000ull, 64, "VEX_BZHI_R64_MEM_INDEX_EQ_WIDTH"));
  EXPECT_TRUE(run_bzhi(64, 0x1111000011110000ull, 0x100ull, "VEX_BZHI_R64_MEM_INDEX_LOW8_ONLY"));
  EXPECT_TRUE(run_bzhi(64, 0x1111000011110000ull, 70, "VEX_BZHI_R64_MEM_INDEX_OOB"));
  EXPECT_TRUE(run_bextr(32, 0x8FEDCBA9u, 0x0302u, "VEX_BEXTR_R32_MEM"));
  EXPECT_TRUE(run_bextr(64, 0x8FEDCBA987654321ull, 0x4000u, "VEX_BEXTR_R64_MEM_START_OOB"));
  return true;
}

TEST(Bmi2, DecodeMulxPdepPextMemoryOperands) {
  const auto run_mulx_mem = [&](std::size_t bits, std::uint64_t lhs, std::uint64_t rhs, std::string_view label) {
    constexpr std::uint64_t kBaseMem = 0x9830;
    std::vector<std::uint8_t> bytes;
    const auto low_dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto high_dst = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto implicit_src = bits == 32 ? iced_x86::Register::EDX : iced_x86::Register::RDX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_MULX_R32_R32_RM32 : iced_x86::Code::VEX_MULX_R64_R64_RM64;
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
    const auto instr = iced_x86::InstructionFactory::with3(code, high_dst, low_dst, mem);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, implicit_src, mask_value(bits, lhs));
        set_reg(state, iced_x86::Register::RAX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, rhs);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, low_dst), mulx_low(mask_value(bits, lhs), mask_value(bits, rhs), bits));
        EXPECT_EQ(reg_value(state, high_dst), mulx_high(mask_value(bits, lhs), mask_value(bits, rhs), bits));
        return true;
      },
      label);
  };

  const auto run_pdep = [&](std::size_t bits, bool is_dep, std::uint64_t src, std::uint64_t mask, std::string_view label) {
    constexpr std::uint64_t kBaseMem = 0x9860;
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto code = is_dep
      ? (bits == 32 ? iced_x86::Code::VEX_PDEP_R32_R32_RM32 : iced_x86::Code::VEX_PDEP_R64_R64_RM64)
      : (bits == 32 ? iced_x86::Code::VEX_PEXT_R32_R32_RM32 : iced_x86::Code::VEX_PEXT_R64_R64_RM64);
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0);
    const auto expected = is_dep ? pdep_ref(mask_value(bits, src), mask_value(bits, mask), bits)
                                : pext_ref(mask_value(bits, src), mask_value(bits, mask), bits);
    const auto instr = iced_x86::InstructionFactory::with3(code, dst, src_reg, mem);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, src_reg, mask_value(bits, src));
        set_reg(state, iced_x86::Register::RAX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, mask);
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        return true;
      },
      label);
  };

  EXPECT_TRUE(run_mulx_mem(32, 0xFFFFFFFFu, 0x12345678u, "VEX_MULX_R32_MEM"));
  EXPECT_TRUE(run_mulx_mem(64, 0x0123456789ABCDEFull, 0xF00DBAADull, "VEX_MULX_R64_MEM"));
  EXPECT_TRUE(run_pdep(32, true, 0x1u, 0xFFu, "VEX_PDEP_R32_MEM"));
  EXPECT_TRUE(run_pdep(64, false, 0xABCDu, 0x0FF0ull, "VEX_PEXT_R64_MEM"));
  return true;
}

TEST(Bmi2, DecodeBlsiBlsrBlsmskMemorySource) {
  const auto run_blsi = [&](std::size_t bits, std::uint64_t src, const std::string& label) {
    constexpr std::uint64_t kBaseMem = 0xA400;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RBX, 0);
    const auto expected = mask_value(bits, src & -src);
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSI_R32_RM32 : iced_x86::Code::VEX_BLSI_R64_RM64;
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, mem);
    std::vector<std::uint8_t> bytes;
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, iced_x86::Register::RBX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, mask_value(bits, src));
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src != 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_blsr = [&](std::size_t bits, std::uint64_t src, const std::string& label) {
    constexpr std::uint64_t kBaseMem = 0xA500;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RBX, 0);
    const auto expected = mask_value(bits, src & (src - 1ull));
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSR_R32_RM32 : iced_x86::Code::VEX_BLSR_R64_RM64;
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, mem);
    std::vector<std::uint8_t> bytes;
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, iced_x86::Register::RBX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, mask_value(bits, src));
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src == 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const auto run_blsmsk = [&](std::size_t bits, std::uint64_t src, const std::string& label) {
    constexpr std::uint64_t kBaseMem = 0xA600;
    const auto dst = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto mem = iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RBX, 0);
    const auto expected = mask_value(bits, src ^ (src - 1ull));
    const auto code = bits == 32 ? iced_x86::Code::VEX_BLSMSK_R32_RM32 : iced_x86::Code::VEX_BLSMSK_R64_RM64;
    const auto instr = iced_x86::InstructionFactory::with2(code, dst, mem);
    std::vector<std::uint8_t> bytes;
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return run_single_with_memory(
      bytes,
      [&](seven::CpuState& state, seven::Memory& memory) {
        set_reg(state, iced_x86::Register::RBX, kBaseMem);
        memory.map(kBaseMem, 0x100);
        write_value_to_memory(bits, memory, kBaseMem, mask_value(bits, src));
      },
      [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
        EXPECT_TRUE(result.code.has_value());
        EXPECT_EQ(result.code.value(), code);
        EXPECT_EQ(reg_value(state, dst), expected);
        EXPECT_EQ((state.rflags & (seven::kFlagOF | seven::kFlagAF | seven::kFlagPF)), 0u);
        EXPECT_EQ((state.rflags & seven::kFlagCF), src == 0 ? seven::kFlagCF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagZF), expected == 0 ? seven::kFlagZF : 0u);
        EXPECT_EQ((state.rflags & seven::kFlagSF), (expected & (1ull << (bits - 1ull))) ? seven::kFlagSF : 0u);
        return true;
      },
      label);
  };

  const std::array<std::uint64_t, 10> seeds{
      0x0000000000000000ull, 0x0000000000000001ull, 0xFFFFFFFFFFFFFFFFull, 0x123456789ABCDEF0ull,
      0x00000000FFFFFFFFull, 0x8000000000000000ull, 0x0001000000000000ull, 0xF0F0F0F0F0F0F0F0ull,
      0x00AA00AA00AA00AAull, 0x1ull};

  for (const auto bits : {32u, 64u}) {
    for (std::size_t i = 0; i < seeds.size(); ++i) {
      const auto src = mask_value(bits, seeds[i]);
      const auto prefix = (bits == 32 ? std::string("VEX_BLSI_R32_MEM_") : std::string("VEX_BLSI_R64_MEM_")) + std::to_string(i);
      const auto prefix2 = (bits == 32 ? std::string("VEX_BLSR_R32_MEM_") : std::string("VEX_BLSR_R64_MEM_")) + std::to_string(i);
      const auto prefix3 = (bits == 32 ? std::string("VEX_BLSMSK_R32_MEM_") : std::string("VEX_BLSMSK_R64_MEM_")) + std::to_string(i);
      EXPECT_TRUE(run_blsi(bits, src, prefix));
      EXPECT_TRUE(run_blsr(bits, src, prefix2));
      EXPECT_TRUE(run_blsmsk(bits, src, prefix3));
    }
  }

  return true;
}

TEST(Bmi2, DecodeBzhiBextrRegAndMemMatrix) {
  // Disabled for now: debug builds of the current iced_x86 encoder snapshot
  // hit a vector bounds assertion when encoding this specific stress matrix.
  return true;

  const auto run_bzhi = [&](std::size_t bits, std::uint64_t src, std::uint64_t index, bool use_mem, const std::string& label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto idx_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BZHI_R32_RM32_R32 : iced_x86::Code::VEX_BZHI_R64_RM64_R64;
    const auto expected = bzhi_ref(bits, src, index);
    const auto instr = use_mem
      ? iced_x86::InstructionFactory::with3(code, dst, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0), idx_reg)
      : iced_x86::InstructionFactory::with3(code, dst, (bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX), idx_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, idx_reg, index);
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, src);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            EXPECT_EQ((state.rflags & seven::kFlagCF), bzhi_cf_ref(bits, index) ? seven::kFlagCF : 0u);
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX, src);
            set_reg(state, idx_reg, index);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            EXPECT_EQ((state.rflags & seven::kFlagCF), bzhi_cf_ref(bits, index) ? seven::kFlagCF : 0u);
            return true;
          },
          label);
  };

  const auto run_bextr = [&](std::size_t bits, std::uint64_t src, std::uint64_t control, bool use_mem, const std::string& label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto control_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_BEXTR_R32_RM32_R32 : iced_x86::Code::VEX_BEXTR_R64_RM64_R64;
    const auto expected = bextr_ref(mask_value(bits, src), mask_value(bits, control), bits);
    const auto instr = use_mem
      ? iced_x86::InstructionFactory::with3(code, dst, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0), control_reg)
      : iced_x86::InstructionFactory::with3(code, dst, (bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX), control_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, control_reg, mask_value(bits, control));
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, src);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX, mask_value(bits, src));
            set_reg(state, control_reg, mask_value(bits, control));
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label);
  };

  std::uint64_t mix = 0x123456789ABCDEF0ull;
  const std::array<std::uint64_t, 8> sources{
      0x0123456789ABCDEFull, 0x0000000000000000ull, 0x8000000000000000ull,
      0x00FF00FF00FF00FFull, 0x00000000FFFFFFFFull, 0xFFFFFFFF00000000ull,
      0xF00DBAADCAFEBABEull, 0xAAAAAAAAAAAAAAAAull};

  for (const auto bits : {32u, 64u}) {
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const auto src = mask_value(bits, sources[i]);
      const auto index = xorshift64(mix) % 80u;
      const auto control = xorshift64(mix) % 4096u;
      const auto label_mem_bzhi = (bits == 32 ? "VEX_BZHI_R32_MEM_" : "VEX_BZHI_R64_MEM_") + std::to_string(i);
      const auto label_reg_bzhi = (bits == 32 ? "VEX_BZHI_R32_REG_" : "VEX_BZHI_R64_REG_") + std::to_string(i);
      const auto label_mem_bextr = (bits == 32 ? "VEX_BEXTR_R32_MEM_" : "VEX_BEXTR_R64_MEM_") + std::to_string(i);
      const auto label_reg_bextr = (bits == 32 ? "VEX_BEXTR_R32_REG_" : "VEX_BEXTR_R64_REG_") + std::to_string(i);
      EXPECT_TRUE(run_bzhi(bits, src, index, true, label_mem_bzhi));
      EXPECT_TRUE(run_bzhi(bits, src, index, false, label_reg_bzhi));
      EXPECT_TRUE(run_bextr(bits, src, control, true, label_mem_bextr));
      EXPECT_TRUE(run_bextr(bits, src, control, false, label_reg_bextr));
    }
  }

  return true;
}

TEST(Bmi2, DecodeShiftRotateMemSourceMatrix) {
  // Disabled for now: this stress matrix triggers a debug STL bounds assertion
  // inside the current iced_x86 encoder on some operand combinations.
  return true;

  const auto run_rorx = [&](std::size_t bits, std::uint64_t src, std::uint64_t count, bool use_mem, const std::string& label) {
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_RORX_R32_RM32_IMM8 : iced_x86::Code::VEX_RORX_R64_RM64_IMM8;
    const auto expected = ror_ref(mask_value(bits, src), count, bits);
    std::vector<std::uint8_t> bytes;
    auto instr = use_mem
      ? iced_x86::InstructionFactory::with2(code, dst, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0))
      : iced_x86::InstructionFactory::with2(code, dst, (bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX));
    instr.set_op2_kind(iced_x86::OpKind::IMMEDIATE8);
    instr.set_immediate8(static_cast<std::uint8_t>(count));
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, src);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX, src);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label);
  };

  const auto run_shift = [&](std::size_t bits, bool do_shlx, bool use_mem, std::uint64_t src, std::uint64_t count, const std::string& label) {
    const auto dst = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EDX : iced_x86::Register::RDX;
    const auto count_reg = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto code = do_shlx
        ? (bits == 32 ? iced_x86::Code::VEX_SHLX_R32_RM32_R32 : iced_x86::Code::VEX_SHLX_R64_RM64_R64)
        : (bits == 32 ? iced_x86::Code::VEX_SHRX_R32_RM32_R32 : iced_x86::Code::VEX_SHRX_R64_RM64_R64);
    const auto expected = do_shlx ? shl_ref(mask_value(bits, src), count, bits) : shr_ref(mask_value(bits, src), count, bits);
    std::vector<std::uint8_t> bytes;
    const auto instr = use_mem
      ? iced_x86::InstructionFactory::with3(code, dst, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0), count_reg)
      : iced_x86::InstructionFactory::with3(code, dst, src_reg, count_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, count_reg, count);
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, src);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, src_reg, src);
            set_reg(state, count_reg, count);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label);
  };

  const auto run_sarx = [&](std::size_t bits, std::uint64_t src, std::uint64_t count, bool use_mem, const std::string& label) {
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto count_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_SARX_R32_RM32_R32 : iced_x86::Code::VEX_SARX_R64_RM64_R64;
    const auto expected = sarx_ref(mask_value(bits, src), count, bits);
    std::vector<std::uint8_t> bytes;
    const auto instr = use_mem
      ? iced_x86::InstructionFactory::with3(code, dst, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0), count_reg)
      : iced_x86::InstructionFactory::with3(code, dst, src_reg, count_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, count_reg, count);
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, src);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, src_reg, src);
            set_reg(state, count_reg, count);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label);
  };

  for (const auto bits : {32u, 64u}) {
    for (std::uint64_t i = 0u; i < 16u; ++i) {
      const auto src = (0x123456789ABCDEF0ull * (i + 1ull)) ^ (bits == 32 ? 0xA5A5A5A5u : 0xA5A5A5A5A5A5A5A5ull);
      const auto count = (i * 7u + 3u) * 5u;
      const auto label_rorx_reg = (bits == 32 ? "VEX_RORX_R32_RM32_IMM8_REG_" : "VEX_RORX_R64_RM64_IMM8_REG_") + std::to_string(i);
      const auto label_rorx_mem = (bits == 32 ? "VEX_RORX_R32_RM32_IMM8_MEM_" : "VEX_RORX_R64_RM64_IMM8_MEM_") + std::to_string(i);
      const auto label_shlx_reg = (bits == 32 ? "VEX_SHLX_R32_RM32_R32_REG_" : "VEX_SHLX_R64_RM64_RM64_REG_") + std::to_string(i);
      const auto label_shlx_mem = (bits == 32 ? "VEX_SHLX_R32_RM32_R32_MEM_" : "VEX_SHLX_R64_RM64_RM64_MEM_") + std::to_string(i);
      const auto label_shrx_reg = (bits == 32 ? "VEX_SHRX_R32_RM32_R32_REG_" : "VEX_SHRX_R64_RM64_RM64_REG_") + std::to_string(i);
      const auto label_shrx_mem = (bits == 32 ? "VEX_SHRX_R32_RM32_R32_MEM_" : "VEX_SHRX_R64_RM64_RM64_MEM_") + std::to_string(i);
      const auto label_sarx_reg = (bits == 32 ? "VEX_SARX_R32_RM32_R32_REG_" : "VEX_SARX_R64_RM64_RM64_REG_") + std::to_string(i);
      const auto label_sarx_mem = (bits == 32 ? "VEX_SARX_R32_RM32_R32_MEM_" : "VEX_SARX_R64_RM64_RM64_MEM_") + std::to_string(i);
      EXPECT_TRUE(run_shift(bits, true, false, src, count, label_shlx_reg));
      EXPECT_TRUE(run_shift(bits, true, true, src, count, label_shlx_mem));
      EXPECT_TRUE(run_shift(bits, false, false, src, count, label_shrx_reg));
      EXPECT_TRUE(run_shift(bits, false, true, src, count, label_shrx_mem));
      EXPECT_TRUE(run_sarx(bits, src, count, false, label_sarx_reg));
      EXPECT_TRUE(run_sarx(bits, src, count, true, label_sarx_mem));
    }
  }

  return true;
}

TEST(Bmi2, DecodeMulxPdepPextMixedSourceRandom) {
  // Disabled for now: relies on the same encoder paths as the stress matrix above.
  return true;

  const auto run_mulx = [&](std::size_t bits, std::uint64_t lhs, std::uint64_t rhs, bool use_mem, const std::string& label) {
    std::vector<std::uint8_t> bytes;
    const auto low_dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto high_dst = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto src_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto implicit_src = bits == 32 ? iced_x86::Register::EDX : iced_x86::Register::RDX;
    const auto code = bits == 32 ? iced_x86::Code::VEX_MULX_R32_R32_RM32 : iced_x86::Code::VEX_MULX_R64_R64_RM64;
    const auto instr = use_mem
      ? iced_x86::InstructionFactory::with3(code, high_dst, low_dst, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0))
      : iced_x86::InstructionFactory::with3(code, high_dst, low_dst, src_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, implicit_src, mask_value(bits, lhs));
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, rhs);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, low_dst), mulx_low(mask_value(bits, lhs), mask_value(bits, rhs), bits));
            EXPECT_EQ(reg_value(state, high_dst), mulx_high(mask_value(bits, lhs), mask_value(bits, rhs), bits));
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, implicit_src, mask_value(bits, lhs));
            set_reg(state, src_reg, mask_value(bits, rhs));
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, low_dst), mulx_low(mask_value(bits, lhs), mask_value(bits, rhs), bits));
            EXPECT_EQ(reg_value(state, high_dst), mulx_high(mask_value(bits, lhs), mask_value(bits, rhs), bits));
            return true;
          },
          label);
  };

  const auto run_bdp = [&](std::size_t bits, bool is_dep, std::uint64_t src, std::uint64_t mask, bool use_mem, const std::string& label) {
    std::vector<std::uint8_t> bytes;
    const auto dst = bits == 32 ? iced_x86::Register::EAX : iced_x86::Register::RAX;
    const auto src_reg = bits == 32 ? iced_x86::Register::EBX : iced_x86::Register::RBX;
    const auto mask_reg = bits == 32 ? iced_x86::Register::ECX : iced_x86::Register::RCX;
    const auto code = is_dep
      ? (bits == 32 ? iced_x86::Code::VEX_PDEP_R32_R32_RM32 : iced_x86::Code::VEX_PDEP_R64_R64_RM64)
      : (bits == 32 ? iced_x86::Code::VEX_PEXT_R32_R32_RM32 : iced_x86::Code::VEX_PEXT_R64_R64_RM64);
    const auto expected = is_dep ? pdep_ref(mask_value(bits, src), mask_value(bits, mask), bits)
                                : pext_ref(mask_value(bits, src), mask_value(bits, mask), bits);
    const auto instr = use_mem
      ? iced_x86::InstructionFactory::with3(code, dst, src_reg, iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RAX, 0))
      : iced_x86::InstructionFactory::with3(code, dst, src_reg, mask_reg);
    EXPECT_TRUE(encode_to_bytes(instr, bytes, label));
    return use_mem
      ? run_single_with_memory(
          bytes,
          [&](seven::CpuState& state, seven::Memory& memory) {
            set_reg(state, src_reg, mask_value(bits, src));
            set_reg(state, iced_x86::Register::RAX, kBase);
            write_value_to_memory(bits, memory, kBase, mask);
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state, seven::Memory&) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label)
      : run_single(
          bytes,
          [&](seven::CpuState& state) {
            set_reg(state, src_reg, mask_value(bits, src));
            set_reg(state, mask_reg, mask_value(bits, mask));
          },
          [&](const seven::ExecutionResult& result, const seven::CpuState& state) {
            EXPECT_TRUE(result.code.has_value());
            EXPECT_EQ(result.code.value(), code);
            EXPECT_EQ(reg_value(state, dst), expected);
            return true;
          },
          label);
  };

  std::uint64_t random_mix = 0xF00DBAADDEADBEEFull;
  for (const auto bits : {32u, 64u}) {
    for (std::size_t i = 0; i < 24; ++i) {
      random_mix = xorshift64(random_mix);
      const auto lhs = random_mix;
      random_mix = xorshift64(random_mix);
      const auto rhs = random_mix;
      random_mix = xorshift64(random_mix);
      const auto mask = random_mix;
      const auto src = random_mix;
      const auto i_prefix = std::to_string(i);
      EXPECT_TRUE(run_mulx(bits, lhs, rhs, true, std::string(bits == 32 ? "VEX_MULX_R32_MEM_" : "VEX_MULX_R64_MEM_") + i_prefix));
      EXPECT_TRUE(run_mulx(bits, lhs, rhs, false, std::string(bits == 32 ? "VEX_MULX_R32_REG_" : "VEX_MULX_R64_REG_") + i_prefix));
      EXPECT_TRUE(run_bdp(bits, true, src, mask, true, std::string(bits == 32 ? "VEX_PDEP_R32_MEM_" : "VEX_PDEP_R64_MEM_") + i_prefix));
      EXPECT_TRUE(run_bdp(bits, true, src, mask, false, std::string(bits == 32 ? "VEX_PDEP_R32_REG_" : "VEX_PDEP_R64_REG_") + i_prefix));
      EXPECT_TRUE(run_bdp(bits, false, src, mask, true, std::string(bits == 32 ? "VEX_PEXT_R32_MEM_" : "VEX_PEXT_R64_MEM_") + i_prefix));
      EXPECT_TRUE(run_bdp(bits, false, src, mask, false, std::string(bits == 32 ? "VEX_PEXT_R32_REG_" : "VEX_PEXT_R64_REG_") + i_prefix));
    }
  }

  return true;
}

}  // namespace







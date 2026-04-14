#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <iced_x86/encoder.hpp>
#include <iced_x86/instruction.hpp>

#include "seven/executor.hpp"
#include "seven/memory.hpp"
#include "seven/types.hpp"

namespace kubera::test {

constexpr std::uint64_t kBase = 0x1000;

inline void write_bytes(seven::Memory& memory, std::uint64_t base, std::span<const std::uint8_t> bytes) {
  memory.map(base, bytes.size() + 0x100);
  ASSERT_TRUE(memory.write(base, bytes.data(), bytes.size()));
}

inline bool encode_to_bytes(const iced_x86::Instruction& instr, std::vector<std::uint8_t>& bytes, std::string_view label) {
  iced_x86::Encoder encoder(64);
  const auto encoded_size = encoder.encode(instr, kBase);
  if (!encoded_size) {
    ADD_FAILURE() << "[encode fail] " << label << ": " << encoded_size.error().message;
    return false;
  }
  bytes = encoder.take_buffer();
  if (bytes.size() > *encoded_size) {
    bytes.resize(*encoded_size);
  }
  return true;
}

inline std::size_t gpr_index(iced_x86::Register reg) {
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
    case iced_x86::Register::R8:
    case iced_x86::Register::R8_D:
    case iced_x86::Register::R8_W:
    case iced_x86::Register::R8_L:
      return 8;
    case iced_x86::Register::R9:
    case iced_x86::Register::R9_D:
    case iced_x86::Register::R9_W:
    case iced_x86::Register::R9_L:
      return 9;
    case iced_x86::Register::R10:
    case iced_x86::Register::R10_D:
    case iced_x86::Register::R10_W:
    case iced_x86::Register::R10_L:
      return 10;
    case iced_x86::Register::R11:
    case iced_x86::Register::R11_D:
    case iced_x86::Register::R11_W:
    case iced_x86::Register::R11_L:
      return 11;
    case iced_x86::Register::R12:
    case iced_x86::Register::R12_D:
    case iced_x86::Register::R12_W:
    case iced_x86::Register::R12_L:
      return 12;
    case iced_x86::Register::R13:
    case iced_x86::Register::R13_D:
    case iced_x86::Register::R13_W:
    case iced_x86::Register::R13_L:
      return 13;
    case iced_x86::Register::R14:
    case iced_x86::Register::R14_D:
    case iced_x86::Register::R14_W:
    case iced_x86::Register::R14_L:
      return 14;
    case iced_x86::Register::R15:
    case iced_x86::Register::R15_D:
    case iced_x86::Register::R15_W:
    case iced_x86::Register::R15_L:
      return 15;
    default:
      return 0;
  }
}

inline void set_reg(seven::CpuState& state, iced_x86::Register reg, std::uint64_t value) {
  state.gpr[gpr_index(reg)] = value;
}

inline std::uint64_t reg_value(const seven::CpuState& state, iced_x86::Register reg) {
  return state.gpr[gpr_index(reg)];
}

inline void set_xmm_u64(seven::CpuState& state, std::size_t index, std::uint64_t low, std::uint64_t high) {
  state.vectors[index].value = seven::SimdUint(low) | (seven::SimdUint(high) << 64);
}

inline std::uint64_t xmm_u64(const seven::CpuState& state, std::size_t index, std::size_t lane) {
  return static_cast<std::uint64_t>((state.vectors[index].value >> (lane * 64)) & seven::SimdUint(0xFFFFFFFFFFFFFFFFull));
}

inline void set_xmm_u32x4(seven::CpuState& state,
                          std::size_t index,
                          std::uint32_t lane0,
                          std::uint32_t lane1,
                          std::uint32_t lane2,
                          std::uint32_t lane3) {
  state.vectors[index].value = seven::SimdUint(lane0) |
                               (seven::SimdUint(lane1) << 32) |
                               (seven::SimdUint(lane2) << 64) |
                               (seven::SimdUint(lane3) << 96);
}

inline std::uint32_t xmm_u32(const seven::CpuState& state, std::size_t index, std::size_t lane) {
  return static_cast<std::uint32_t>((state.vectors[index].value >> (lane * 32)) & seven::SimdUint(0xFFFFFFFFull));
}

template <typename Float>
inline void set_xmm_scalar(seven::CpuState& state, std::size_t index, Float value) {
  if constexpr (std::is_same_v<Float, float>) {
    set_xmm_u32x4(state, index, std::bit_cast<std::uint32_t>(value), 0, 0, 0);
  } else {
    static_assert(std::is_same_v<Float, double>);
    set_xmm_u64(state, index, std::bit_cast<std::uint64_t>(value), 0);
  }
}

template <typename Float>
inline Float xmm_scalar(const seven::CpuState& state, std::size_t index) {
  if constexpr (std::is_same_v<Float, float>) {
    return std::bit_cast<float>(xmm_u32(state, index, 0));
  } else {
    static_assert(std::is_same_v<Float, double>);
    return std::bit_cast<double>(xmm_u64(state, index, 0));
  }
}

template <typename Setup, typename Verify>
inline void run_single(std::span<const std::uint8_t> bytes, Setup&& setup, Verify&& verify) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, bytes);
  setup(state, memory);
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  verify(result, state, memory);
}

inline std::uint32_t crc32c_update(std::uint32_t crc, std::uint64_t value, std::size_t width) {
  constexpr std::uint32_t kPoly = 0x82F63B78u;
  for (std::size_t i = 0; i < width; ++i) {
    crc ^= static_cast<std::uint8_t>((value >> (i * 8)) & 0xFFu);
    for (unsigned bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1u) != 0u ? kPoly : 0u);
    }
  }
  return crc;
}

}  // namespace kubera::test

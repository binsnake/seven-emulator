#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <iced_x86/code.hpp>

namespace seven {

using X87Scalar = boost::multiprecision::cpp_bin_float_50;

#ifndef SEVEN_VECTOR_BITS
#define SEVEN_VECTOR_BITS 512
#endif

#ifndef SEVEN_MAX_VECTOR_BYTES
#define SEVEN_MAX_VECTOR_BYTES (SEVEN_VECTOR_BITS / 8)
#endif

#if SEVEN_VECTOR_BITS == 128
using SimdUint = boost::multiprecision::uint128_t;
#elif SEVEN_VECTOR_BITS == 256
using SimdUint = boost::multiprecision::uint256_t;
#elif SEVEN_VECTOR_BITS == 512
using SimdUint = boost::multiprecision::uint512_t;
#else
#error "SEVEN_VECTOR_BITS must be one of 128, 256, or 512"
#endif

constexpr std::size_t kVectorBits = SEVEN_VECTOR_BITS;
constexpr std::size_t kVectorBytes = SEVEN_MAX_VECTOR_BYTES;

enum class ExecutionMode : std::uint8_t {
  real16,
  compat32,
  long64,
};

[[nodiscard]] constexpr std::uint32_t decoder_bitness(ExecutionMode mode) noexcept {
  switch (mode) {
    case ExecutionMode::real16:
      return 16;
    case ExecutionMode::compat32:
      return 32;
    case ExecutionMode::long64:
    default:
      return 64;
  }
}

[[nodiscard]] constexpr std::size_t instruction_pointer_width(ExecutionMode mode) noexcept {
  switch (mode) {
    case ExecutionMode::real16:
      return 2;
    case ExecutionMode::compat32:
      return 4;
    case ExecutionMode::long64:
    default:
      return 8;
  }
}

[[nodiscard]] constexpr std::size_t stack_pointer_width(ExecutionMode mode) noexcept {
  return instruction_pointer_width(mode);
}

enum class StopReason : std::uint8_t {
  none,
  halted,
  invalid_opcode,
  unsupported_instruction,
  floating_point_exception,
  page_fault,
  divide_error,
  general_protection,
  decode_error,
  execution_limit,
  stop_requested,
};

struct ExceptionInfo {
  StopReason reason = StopReason::none;
  std::uint64_t address = 0;
  std::uint32_t error_code = 0;
};

struct VectorRegister {
  SimdUint value = 0;
};

struct DescriptorTableRegister {
  std::uint64_t base = 0;
  std::uint16_t limit = 0;
};

struct CpuState {
  std::array<std::uint64_t, 16> gpr{};
  std::array<std::uint64_t, 8> mmx{};
  std::array<std::uint16_t, 6> sreg{};  // ES,CS,SS,DS,FS,GS selectors
  std::array<std::uint64_t, 16> cr{};
  std::array<std::uint64_t, 16> dr{};
  std::array<std::uint64_t, 8> tr{};
  std::uint64_t rip = 0;
  ExecutionMode mode = ExecutionMode::long64;
  std::uint64_t rflags = 0x202;
  std::uint64_t fs_base = 0;
  std::uint64_t gs_base = 0;
  DescriptorTableRegister gdtr{};
  DescriptorTableRegister idtr{};
  std::unordered_map<std::uint32_t, std::uint64_t> msr{};
  std::array<std::uint64_t, 2> xcr{3u, 0u};
  std::uint32_t mxcsr = 0x1F80;
  std::uint16_t x87_control_word = 0x037F;
  std::uint16_t x87_status_word = 0;
  std::uint8_t x87_top = 0;
  std::array<X87Scalar, 8> x87_stack{};
  std::array<std::uint8_t, 8> x87_tags{0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3};
  std::array<std::uint64_t, 8> opmask{};
  std::array<VectorRegister, 32> vectors{};
  std::uint8_t debug_suppression = 0;
  bool pending_single_step = false;
  std::uint64_t pending_debug_hit_bits = 0;

  [[nodiscard]] std::size_t x87_phys_index(std::size_t st_index) const noexcept {
    return static_cast<std::size_t>((x87_top + static_cast<std::uint8_t>(st_index)) & 0x7);
  }

  [[nodiscard]] std::size_t mmx_phys_index(std::size_t mm_index) const noexcept {
    return mm_index & 0x7;
  }

  [[nodiscard]] bool x87_is_empty(std::size_t st_index) const noexcept {
    return x87_tags[x87_phys_index(st_index)] == 0x3;
  }

  [[nodiscard]] X87Scalar x87_get(std::size_t st_index) const noexcept {
    return x87_stack[x87_phys_index(st_index)];
  }

  void x87_set(std::size_t st_index, X87Scalar value) noexcept {
    const auto idx = x87_phys_index(st_index);
    x87_stack[idx] = value;
    x87_tags[idx] = (value == 0) ? 0x1 : 0x0;
  }

  void x87_mark_empty(std::size_t st_index) noexcept {
    x87_tags[x87_phys_index(st_index)] = 0x3;
  }

  void x87_reset() noexcept {
    x87_stack.fill(0);
    x87_tags.fill(0x3);
    x87_top = 0;
    x87_status_word = 0;
    x87_control_word = 0x037F;
  }

  [[nodiscard]] std::uint16_t get_x87_control_word() const noexcept {
    return x87_control_word;
  }

  void set_x87_control_word(std::uint16_t value) noexcept {
    x87_control_word = value;
  }

  [[nodiscard]] std::uint16_t get_x87_status_word() const noexcept {
    return x87_status_word;
  }

  void set_x87_status_word(std::uint16_t value) noexcept {
    x87_status_word = value;
    x87_top = static_cast<std::uint8_t>((x87_status_word >> 11) & 0x7);
  }

  [[nodiscard]] std::uint8_t get_x87_top() const noexcept {
    return x87_top;
  }

  void set_x87_top(std::uint8_t top) noexcept {
    x87_top = static_cast<std::uint8_t>(top & 0x7);
    x87_status_word = static_cast<std::uint16_t>((x87_status_word & ~std::uint16_t(0x3800)) | (std::uint16_t(x87_top) << 11));
  }

  bool x87_push(X87Scalar value) noexcept {
    const std::uint8_t new_top = static_cast<std::uint8_t>((x87_top + 7) & 0x7);
    if (x87_tags[new_top] != 0x3) {
      return false;
    }
    x87_top = new_top;
    x87_stack[new_top] = value;
    x87_tags[new_top] = (value == 0) ? 0x1 : 0x0;
    set_x87_top(x87_top);
    return true;
  }

  bool x87_pop() noexcept {
    if (x87_tags[x87_top] == 0x3) {
      return false;
    }
    x87_tags[x87_top] = 0x3;
    x87_top = static_cast<std::uint8_t>((x87_top + 1) & 0x7);
    set_x87_top(x87_top);
    return true;
  }

  void x87_swap(std::size_t st_index) noexcept {
    const auto i0 = x87_phys_index(0);
    const auto iN = x87_phys_index(st_index);
    std::swap(x87_stack[i0], x87_stack[iN]);
    std::swap(x87_tags[i0], x87_tags[iN]);
  }

  [[nodiscard]] std::uint64_t mmx_get(std::size_t mm_index) const noexcept {
    return mmx[mmx_phys_index(mm_index)];
  }

  void mmx_set(std::size_t mm_index, std::uint64_t value) noexcept {
    const auto idx = mmx_phys_index(mm_index);
    mmx[idx] = value;
    x87_tags[idx] = 0x0;
  }
};

[[nodiscard]] constexpr std::uint64_t mode_address_mask(ExecutionMode mode) noexcept {
  const auto width = instruction_pointer_width(mode);
  return width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
}

[[nodiscard]] constexpr std::uint64_t mask_instruction_pointer(const CpuState& state, std::uint64_t value) noexcept {
  return value & mode_address_mask(state.mode);
}

[[nodiscard]] constexpr std::uint64_t mask_stack_pointer(const CpuState& state, std::uint64_t value) noexcept {
  const auto width = stack_pointer_width(state.mode);
  const auto mask = width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
  return value & mask;
}

[[nodiscard]] constexpr std::uint64_t mask_linear_address(const CpuState& state, std::uint64_t value) noexcept {
  return value & mode_address_mask(state.mode);
}

struct ExecutionResult {
  StopReason reason = StopReason::none;
  std::uint64_t retired = 0;
  std::optional<ExceptionInfo> exception;
  std::optional<iced_x86::Code> code;

  [[nodiscard]] bool ok() const noexcept { return reason == StopReason::none || reason == StopReason::halted; }
};

constexpr std::uint64_t kFlagCF = 1ull << 0;
constexpr std::uint64_t kFlagPF = 1ull << 2;
constexpr std::uint64_t kFlagAF = 1ull << 4;
constexpr std::uint64_t kFlagZF = 1ull << 6;
constexpr std::uint64_t kFlagSF = 1ull << 7;
constexpr std::uint64_t kFlagTF = 1ull << 8;
constexpr std::uint64_t kFlagIF = 1ull << 9;
constexpr std::uint64_t kFlagDF = 1ull << 10;
constexpr std::uint64_t kFlagOF = 1ull << 11;
constexpr std::uint64_t kFlagRF = 1ull << 16;

[[nodiscard]] constexpr std::uint64_t mask_for_width(std::size_t width) noexcept {
  return width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
}

[[nodiscard]] constexpr bool even_parity(std::uint8_t value) noexcept {
  value ^= value >> 4;
  value &= 0x0F;
  return ((0x6996 >> value) & 1u) == 0;
}

[[nodiscard]] constexpr std::uint64_t sign_bit_for_width(std::size_t width) noexcept {
  return 1ull << ((width * 8) - 1);
}

[[nodiscard]] constexpr std::uint64_t sign_extend(std::uint64_t value, std::size_t width) noexcept {
  const auto mask = mask_for_width(width);
  value &= mask;
  const auto sign = sign_bit_for_width(width);
  if ((value & sign) == 0) {
    return value;
  }
  return value | ~mask;
}

}  // namespace seven


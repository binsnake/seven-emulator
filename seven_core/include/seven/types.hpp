#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include <iced_x86/code.hpp>
#include "uint_wide.h"
#include "seven/float80.hpp"

namespace seven {

using X87Scalar = seven::Float80;

#ifndef SEVEN_VECTOR_BITS
#define SEVEN_VECTOR_BITS 512
#endif

#ifndef SEVEN_MAX_VECTOR_BYTES
#define SEVEN_MAX_VECTOR_BYTES (SEVEN_VECTOR_BITS / 8)
#endif

#if SEVEN_VECTOR_BITS == 128
using SimdUint = math::wide_integer::uint128_t;
#elif SEVEN_VECTOR_BITS == 256
using SimdUint = math::wide_integer::uint256_t;
#elif SEVEN_VECTOR_BITS == 512
using SimdUint = math::wide_integer::uint512_t;
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
  std::array<std::uint16_t, 6> sreg{};  // ES,CS,SS,DS,FS,GS selectors
  // cr[0]=CR0, cr[4]=CR4 -- Windows 10/11 x64 typical values
  std::array<std::uint64_t, 16> cr{0x80050033u, 0, 0, 0, 0x370678u};
  // dr[6]/dr[7] carry bits that read as 1 on real hardware no matter what is written; these are
  // their reset values. See kDr6ReservedOnes/kDr7ReservedOnes.
  std::array<std::uint64_t, 16> dr{0, 0, 0, 0, 0, 0, 0xFFFF0FF0ull, 0x400ull};
  std::array<std::uint64_t, 8> tr{};
  std::uint64_t rip = 0;
  ExecutionMode mode = ExecutionMode::long64;
  std::uint64_t rflags = 0x202;
  std::uint64_t fs_base = 0;
  std::uint64_t gs_base = 0;
  DescriptorTableRegister gdtr{};
  DescriptorTableRegister idtr{};
  // EFER (LME+LMA+NXE+SCE) and STAR (syscall CS/SS) -- Windows 10/11 x64 typical values
  std::unordered_map<std::uint32_t, std::uint64_t> msr{
    {0xC0000080u, 0x0000'0000'0000'0D01u},  // EFER
    {0xC0000081u, 0x0023'0010'0000'0000u},  // STAR
  };
  std::array<std::uint64_t, 2> xcr{3u, 0u};
  std::uint32_t mxcsr = 0x1F80;
  std::uint16_t x87_control_word = 0x037F;
  std::uint16_t x87_status_word = 0;
  std::uint8_t x87_top = 0;
  std::array<X87Scalar, 8> x87_stack{};
  std::array<std::uint8_t, 8> x87_tags{0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3};
  std::array<std::uint64_t, 8> opmask{};
  std::array<VectorRegister, 32> vectors{};
  // Per-guest, deliberately. This used to be a function-local static inside the RDTSC handler,
  // which meant every Executor in the process shared one counter: two guests that are supposed to
  // be isolated could watch each other's progress through it, and two Executors on separate
  // threads raced on the increment.
  std::uint64_t tsc = 0;
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
    // Every other x87_tags access in the tree resolves its index through x87_phys_index or a
    // literal bound; this was the one place that read x87_top raw and trusted every writer to have
    // masked it. Masking here makes the 0..7 invariant hold at the point of use instead of being a
    // contract each writer has to remember, which is a promise a deserializer has already broken.
    const auto phys = x87_phys_index(0);
    if (x87_tags[phys] == 0x3) {
      return false;
    }
    x87_tags[phys] = 0x3;
    x87_top = static_cast<std::uint8_t>((phys + 1) & 0x7);
    set_x87_top(x87_top);
    return true;
  }

  void x87_swap(std::size_t st_index) noexcept {
    const auto i0 = x87_phys_index(0);
    const auto iN = x87_phys_index(st_index);
    std::swap(x87_stack[i0], x87_stack[iN]);
    std::swap(x87_tags[i0], x87_tags[iN]);
  }

  // MM0-MM7 are not a register file of their own: they are the low 64 bits (the significand) of the
  // PHYSICAL x87 registers R0-R7, which is why they are indexed without TOP. They used to live in a
  // separate array, so FXSAVE/FNSAVE stored zeros for them and FXRSTOR/FRSTOR never brought them
  // back. Writing one fills the aliased register's exponent and sign with ones, which is what makes
  // the x87 side read the value back as a NaN.
  [[nodiscard]] std::uint64_t mmx_get(std::size_t mm_index) const noexcept {
    return x87_stack[mmx_phys_index(mm_index)].val.signif;
  }

  void mmx_set(std::size_t mm_index, std::uint64_t value) noexcept {
    const auto idx = mmx_phys_index(mm_index);
    x87_stack[idx].val.signif = value;
    x87_stack[idx].val.signExp = 0xFFFFu;
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

// The ALU status flags eligible for the block liveness pass's dead-write elimination (see
// seven/ir.hpp). Deliberately excludes control bits (TF/IF/DF/RF/...) -- those are never
// dead-code-eliminated, only ever set explicitly by the instructions that own them.
constexpr std::uint64_t kAluStatusFlagsMask = kFlagCF | kFlagPF | kFlagAF | kFlagZF | kFlagSF | kFlagOF;

// The rflags bits that actually exist. Bit 1 reads back as 1 no matter what is written; bits 3, 5,
// 15 and everything from 22 up read back as 0. POPF and IRET are the only two instructions that
// load rflags wholesale from guest memory, so they are the only two that can smuggle a reserved bit
// in, and both have to drop it on the way through.
constexpr std::uint64_t kRflagsWritableMask = 0x00000000003F7FD5ull;
constexpr std::uint64_t kRflagsReservedOnes = 0x0000000000000002ull;

// 4-level paging (48-bit virtual addresses): bits 63:47 must all equal bit 47. Anything else is a
// #GP(0) on real hardware, raised ahead of any page walk.
[[nodiscard]] constexpr bool is_canonical_address(std::uint64_t address) noexcept {
  constexpr int kShift = 16;  // 64 - 48
  return (static_cast<std::int64_t>(address << kShift) >> kShift) == static_cast<std::int64_t>(address);
}

// DR6 keeps B0-B3 plus BD/BS/BT; DR7 keeps the enable pairs, LE/GE, GD, and the R/W+LEN fields.
// Everything else in each reads back as a fixed 1 or 0 whatever the guest writes, and both are
// 32 bits of architectural state, so the upper half never sticks either.
constexpr std::uint64_t kDr6WritableMask = 0x0000E00Full;
constexpr std::uint64_t kDr6ReservedOnes = 0xFFFF0FF0ull;
constexpr std::uint64_t kDr7WritableMask = 0xFFFF23FFull;
constexpr std::uint64_t kDr7ReservedOnes = 0x00000400ull;

[[nodiscard]] constexpr std::uint64_t mask_for_width(std::size_t width) noexcept {
  return width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
}

[[nodiscard]] constexpr bool even_parity(std::uint8_t value) noexcept {
  value ^= value >> 4;
  value &= 0x0F;
  return ((0x6996 >> value) & 1u) == 0;
}

[[nodiscard]] constexpr std::uint64_t sign_bit_for_width(std::size_t width) noexcept {
  // Guarded the way mask_for_width above already is. Every caller today passes a literal, so the
  // shift is always in range -- but the two are used interchangeably on the same width and only one
  // of them survived a width of 0 or one past 8, which is the kind of asymmetry that bites later.
  return width == 0 ? 0ull : (width >= 8 ? (1ull << 63) : (1ull << ((width * 8) - 1)));
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


#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "seven/handler_helpers.hpp"
#include "seven/x87_encoding.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

constexpr std::uint16_t kX87ExceptionInvalid = 0x0001u;
constexpr std::uint16_t kX87ExceptionDenormal = 0x0002u;
constexpr std::uint16_t kX87ExceptionZeroDiv = 0x0004u;
constexpr std::uint16_t kX87ExceptionOverflow = 0x0008u;
constexpr std::uint16_t kX87ExceptionUnderflow = 0x0010u;
constexpr std::uint16_t kX87ExceptionPrecision = 0x0020u;
constexpr std::uint16_t kX87ExceptionStackFault = 0x0040u;
constexpr std::uint16_t kX87ExceptionMask = 0x003Fu;

// An enabled underflow trap is handed a result whose exponent has been biased back into range
// rather than the tiny one the masked path would store. IEEE 754-2019 7.5 and SDM 4.9.1.5 agree on
// 24576 for the 80-bit format.
constexpr int kX87UnderflowTrapBias = 24576;

inline bool x87_exceptions_masked(const CpuState& state, std::uint16_t exceptions);
inline ExecutionResult x87_exception(ExecutionContext& ctx, std::uint16_t exceptions);
inline ExecutionResult x87_stack_underflow(ExecutionContext& ctx);
inline ExecutionResult x87_stack_underflow_into(ExecutionContext& ctx, std::size_t dst_index);
inline ExecutionResult x87_stack_overflow(ExecutionContext& ctx);
inline std::uint16_t x87_classify_result(const X87Scalar& result, const X87Scalar& lhs, const X87Scalar& rhs);
inline std::uint16_t x87_operand_exceptions(const X87Scalar& lhs, const X87Scalar& rhs);
inline X87Scalar x87_round_to_control(const CpuState& state, X87Scalar value);

// The QNaN floating-point indefinite: sign 1, exponent all ones, significand 0xC000...0. This is
// what hardware leaves behind whenever a masked exception has no meaningful answer to give.
inline X87Scalar x87_indefinite() {
  extFloat80_t v;
  v.signExp = 0xFFFFu;
  v.signif = 0xC000000000000000ULL;
  return X87Scalar(v);
}

inline std::size_t x87_st_index(iced_x86::Register reg) {
  return static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::ST0));
}

// A masked stack underflow does not skip the instruction: hardware still retires the pop. x87_pop()
// deliberately refuses to move TOP while ST(0) is empty, which is right everywhere else, so the
// indefinite goes into ST(0) first and the ordinary pop then leaves exactly what hardware leaves --
// TOP advanced, the vacated register tagged empty.
inline void x87_underflow_pop(CpuState& state) {
  state.x87_set(0, x87_indefinite());
  (void)state.x87_pop();
}

inline uint_fast8_t x87_softfloat_rounding_mode(const CpuState& state) {
  switch ((state.get_x87_control_word() >> 10) & 0x3u) {
    case 0: return softfloat_round_near_even;
    case 1: return softfloat_round_min;
    case 2: return softfloat_round_max;
    default: return softfloat_round_minMag;
  }
}

// softfloat names the five IEEE exceptions and the x87 status word names six, in a different order.
// The sixth, #D, has no softfloat counterpart because no IEEE format needs one -- it is a question
// about the operands, which is why x87_operand_exceptions below asks it separately.
inline std::uint16_t x87_exceptions_from_softfloat(uint_fast8_t flags) {
  std::uint16_t exceptions = 0;
  if ((flags & softfloat_flag_invalid) != 0) exceptions |= kX87ExceptionInvalid;
  if ((flags & softfloat_flag_infinite) != 0) exceptions |= kX87ExceptionZeroDiv;
  if ((flags & softfloat_flag_overflow) != 0) exceptions |= kX87ExceptionOverflow;
  if ((flags & softfloat_flag_underflow) != 0) exceptions |= kX87ExceptionUnderflow;
  if ((flags & softfloat_flag_inexact) != 0) exceptions |= kX87ExceptionPrecision;
  return exceptions;
}

// A biased exponent of zero with a significand that is not zero, which covers pseudo-denormals as
// well as ordinary ones.
inline bool x87_is_denormal_operand(const X87Scalar& value) {
  return (value.val.signExp & 0x7FFFu) == 0u && value.val.signif != 0u;
}

// The exceptions that are decided by what walked in rather than by what came out: #D, and the #IA
// that an encoding contradicting its own exponent raises before any arithmetic happens.
inline std::uint16_t x87_operand_exceptions(const X87Scalar& lhs, const X87Scalar& rhs) {
  std::uint16_t exceptions = 0;
  if (seven::isunsupported(lhs) || seven::isunsupported(rhs)) {
    exceptions |= kX87ExceptionInvalid;
  }
  if (x87_is_denormal_operand(lhs) || x87_is_denormal_operand(rhs)) {
    exceptions |= kX87ExceptionDenormal;
  }
  return exceptions;
}

struct X87Computed {
  X87Scalar value;
  std::uint16_t exceptions;
};

// One x87 operation, run under the guest's rounding control, with the exceptions read off softfloat
// rather than inferred from the answer afterwards. SoftFloat takes both the rounding mode and the
// flags through globals, so each needs owning for the length of the call and putting back after.
// An unsupported operand never reaches the arithmetic: there is no value to compute with, and the
// masked answer is the indefinite.
template <typename Fn>
inline X87Computed x87_evaluate(const CpuState& state, const X87Scalar& lhs, const X87Scalar& rhs, Fn&& fn) {
  const std::uint16_t operands = x87_operand_exceptions(lhs, rhs);
  if ((operands & kX87ExceptionInvalid) != 0) {
    return {x87_indefinite(), operands};
  }
  seven::RoundingGuard rounding(x87_softfloat_rounding_mode(state));
  const auto computed = seven::with_exception_flags([&] { return fn(lhs, rhs); });
  return {computed.value,
          static_cast<std::uint16_t>(operands | x87_exceptions_from_softfloat(computed.flags))};
}

// Common tail for the forms whose destination is a stack register. A masked exception does not stop
// the write, and an unmasked one does -- except for #U, which both IEEE 754-2019 7.5 and SDM 4.9.1.5
// deliver to the handler as a result with the exponent biased back into range, rather than leaving
// the destination holding whatever was there before.
inline ExecutionResult x87_finish(ExecutionContext& ctx, std::size_t dst_index, const X87Computed& computed) {
  if (computed.exceptions != 0) {
    auto result = x87_exception(ctx, computed.exceptions);
    if (!result.ok()) {
      if ((computed.exceptions & kX87ExceptionUnderflow) != 0 &&
          !x87_exceptions_masked(ctx.state, kX87ExceptionUnderflow)) {
        ctx.state.x87_set(dst_index, seven::ldexp(computed.value, kX87UnderflowTrapBias));
      }
      return result;
    }
  }
  ctx.state.x87_set(dst_index, computed.value);
  return {};
}

// FST/FSTP m32 and m64 round an 80-bit value into a format that cannot hold it, and that rounding is
// where their #P, #O and #U come from. None of it used to be computed.
template <typename Narrow>
inline std::pair<Narrow, std::uint16_t> x87_narrow(const CpuState& state, const X87Scalar& value) {
  const bool unsupported = seven::isunsupported(value);
  const X87Scalar source = unsupported ? x87_indefinite() : value;
  seven::RoundingGuard rounding(x87_softfloat_rounding_mode(state));
  const auto narrowed = seven::with_exception_flags([&] { return static_cast<Narrow>(source); });
  auto exceptions = x87_exceptions_from_softfloat(narrowed.flags);
  if (unsupported) exceptions |= kX87ExceptionInvalid;
  return {narrowed.value, exceptions};
}

// FCHS and FABS are the only callers and both are pure sign-bit edits, which is why the SDM gives
// them #IS and nothing else -- not even the #IA a signalling NaN or an unsupported encoding raises
// everywhere else. Running them through the result classifier read a denormal operand back as an
// underflow.
template <typename Fn>
inline ExecutionResult x87_unary_st0(ExecutionContext& ctx, Fn&& fn) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  ctx.state.x87_set(0, fn(ctx.state.x87_get(0)));
  return {};
}

inline ExecutionResult x87_fxch(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(1) != iced_x86::OpKind::REGISTER) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto reg = ctx.instr.op_register(1);
  if (reg < iced_x86::Register::ST0 || reg > iced_x86::Register::ST7) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto idx = x87_st_index(reg);
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(idx)) {
    auto result = x87_stack_underflow(ctx);
    if (!result.ok()) return result;
    // Both registers are source and destination here, so a masked underflow fills whichever one is
    // empty with the indefinite and the exchange still happens.
    if (ctx.state.x87_is_empty(0)) ctx.state.x87_set(0, x87_indefinite());
    if (ctx.state.x87_is_empty(idx)) ctx.state.x87_set(idx, x87_indefinite());
  }
  ctx.state.x87_swap(idx);
  return {};
}

template <typename Fn>
inline ExecutionResult x87_binary_mem_st0(ExecutionContext& ctx, std::size_t width, Fn&& fn) {
  X87Scalar rhs = 0;
  if (width == 4) {
    float v = 0.0f;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else if (width == 8) {
    double v = 0.0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const auto lhs = ctx.state.x87_get(0);
  return x87_finish(ctx, 0, x87_evaluate(ctx.state, lhs, rhs, std::forward<Fn>(fn)));
}

template <typename Fn>
inline ExecutionResult x87_binary_mem_int_st0(ExecutionContext& ctx, std::size_t width, Fn&& fn) {
  X87Scalar rhs = 0;
  if (width == 2) {
    std::int16_t v = 0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 2)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else if (width == 4) {
    std::int32_t v = 0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else if (width == 8) {
    std::int64_t v = 0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const auto lhs = ctx.state.x87_get(0);
  return x87_finish(ctx, 0, x87_evaluate(ctx.state, lhs, rhs, std::forward<Fn>(fn)));
}

// iced's decoder value-initializes Instruction, and OpKind::REGISTER and Register::NONE are both
// zero, so an operand slot the decoder never wrote reads back as a REGISTER named NONE -- which
// sails straight through an `op_kind(i) != REGISTER` check. Several handlers below are reached by
// instructions carrying fewer operands than they ask for, and x87_st_index(NONE) then underflows to
// a huge value that x87_phys_index quietly masks down to ST(7). Ask the instruction how many
// operands it actually has before touching one.
[[nodiscard]] inline bool x87_operand_is_st(const ExecutionContext& ctx, std::uint32_t op) {
  if (op >= ctx.instr.op_count() || ctx.instr.op_kind(op) != iced_x86::OpKind::REGISTER) {
    return false;
  }
  const auto reg = ctx.instr.op_register(op);
  return reg >= iced_x86::Register::ST0 && reg <= iced_x86::Register::ST7;
}

template <typename Fn>
inline ExecutionResult x87_binary_st_indices(ExecutionContext& ctx, std::size_t dst_idx, std::size_t src_idx, Fn&& fn) {
  if (ctx.state.x87_is_empty(dst_idx) || ctx.state.x87_is_empty(src_idx)) return x87_stack_underflow_into(ctx, dst_idx);
  const auto lhs = ctx.state.x87_get(dst_idx);
  const auto rhs = ctx.state.x87_get(src_idx);
  return x87_finish(ctx, dst_idx, x87_evaluate(ctx.state, lhs, rhs, std::forward<Fn>(fn)));
}

template <typename Fn>
inline ExecutionResult x87_binary_st_regs(ExecutionContext& ctx, std::uint32_t dst_op, std::uint32_t src_op, Fn&& fn) {
  if (!x87_operand_is_st(ctx, dst_op) || !x87_operand_is_st(ctx, src_op)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return x87_binary_st_indices(ctx, x87_st_index(ctx.instr.op_register(dst_op)),
                               x87_st_index(ctx.instr.op_register(src_op)), std::forward<Fn>(fn));
}

inline ExecutionResult x87_store_mem(ExecutionContext& ctx, std::size_t width, bool pop) {
  bool underflowed = false;
  if (ctx.state.x87_is_empty(0)) {
    auto fault = x87_stack_underflow(ctx);
    if (!fault.ok()) return fault;
    underflowed = true;
  }
  // Masked, the store still happens; it just stores the indefinite. Narrowing it to m32 or m64 gives
  // that format's own indefinite, so there is nothing to special-case below.
  const auto value = underflowed ? x87_indefinite() : ctx.state.x87_get(0);
  ExecutionResult result = {};
  if (width == 4) {
    const auto [narrowed, exceptions] = x87_narrow<float>(ctx.state, value);
    if (exceptions != 0) {
      auto fault = x87_exception(ctx, exceptions);
      if (!fault.ok()) return fault;
    }
    result = detail::write_memory_checked(ctx, detail::memory_address(ctx), narrowed);
  } else if (width == 8) {
    const auto [narrowed, exceptions] = x87_narrow<double>(ctx.state, value);
    if (exceptions != 0) {
      auto fault = x87_exception(ctx, exceptions);
      if (!fault.ok()) return fault;
    }
    result = detail::write_memory_checked(ctx, detail::memory_address(ctx), narrowed);
  } else if (width == 10) {
    std::array<std::uint8_t, 16> raw{};
    x87_encoding::encode_ext80(value, raw.data());
    const auto base = detail::memory_address(ctx);
    for (std::size_t i = 0; i < 10; ++i) {
      if (!ctx.memory.write(base + i, &raw[i], 1)) return detail::memory_fault(ctx, base + i);
    }
    result = {};
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!result.ok()) return result;
  if (pop) {
    if (underflowed) x87_underflow_pop(ctx.state);
    else if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  }
  return {};
}

inline ExecutionResult x87_push_from_memory(ExecutionContext& ctx, std::size_t width) {
  X87Scalar value = 0;
  if (width == 10) {
    std::array<std::uint8_t, 16> raw{};
    if (!ctx.memory.read(detail::memory_address(ctx), raw.data(), 10)) {
      return detail::memory_fault(ctx, detail::memory_address(ctx));
    }
    value = x87_encoding::decode_ext80(raw.data());
  } else if (width == 4) {
    float v = 0.0f;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    value = static_cast<X87Scalar>(v);
  } else if (width == 8) {
    double v = 0.0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    value = static_cast<X87Scalar>(v);
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!ctx.state.x87_push(value)) {
    return x87_stack_overflow(ctx);
  }
  return {};
}

inline ExecutionResult x87_store_to_memory(ExecutionContext& ctx, std::size_t width, bool pop) {
  return x87_store_mem(ctx, width, pop);
}

inline ExecutionResult x87_store_integer(ExecutionContext& ctx, std::size_t width, bool pop, bool truncate_only) {
  X87Scalar lower = 0;
  X87Scalar upper = 0;
  if (width == 2) {
    lower = std::numeric_limits<std::int16_t>::min();
    upper = std::numeric_limits<std::int16_t>::max();
  } else if (width == 4) {
    lower = std::numeric_limits<std::int32_t>::min();
    upper = std::numeric_limits<std::int32_t>::max();
  } else if (width == 8) {
    lower = std::numeric_limits<std::int64_t>::min();
    upper = std::numeric_limits<std::int64_t>::max();
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }

  // These forms store the INTEGER indefinite, which is the most negative value of the destination
  // width and not the floating-point one -- and `lower` already is it.
  bool underflowed = false;
  if (ctx.state.x87_is_empty(0)) {
    auto fault = x87_stack_underflow(ctx);
    if (!fault.ok()) return fault;
    underflowed = true;
  }

  std::uint16_t exceptions = 0;
  X87Scalar stored = lower;
  if (!underflowed) {
    const auto value = ctx.state.x87_get(0);
    const auto rounded = truncate_only ? seven::trunc(value) : x87_round_to_control(ctx.state, value);
    if (rounded != value) {
      exceptions |= kX87ExceptionPrecision;
    }
    stored = rounded;
    if (seven::isnan(rounded) || seven::isinf(rounded) || rounded < lower || rounded > upper) {
      exceptions |= kX87ExceptionInvalid;
      stored = lower;
    }
    // An unmasked exception means the instruction faults, and a faulting instruction must not have
    // touched its destination. This used to write first and report afterwards.
    if (exceptions != 0) {
      auto result = x87_exception(ctx, exceptions);
      if (!result.ok()) return result;
    }
  }

  const std::int64_t out = static_cast<std::int64_t>(stored);
  if (width == 2) {
    const std::int16_t v = static_cast<std::int16_t>(out);
    if (!ctx.memory.write(detail::memory_address(ctx), &v, 2)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  } else if (width == 4) {
    const std::int32_t v = static_cast<std::int32_t>(out);
    if (!ctx.memory.write(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  } else {
    const std::int64_t v = out;
    if (!ctx.memory.write(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (pop) {
    if (underflowed) x87_underflow_pop(ctx.state);
    else if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  }
  return {};
}

inline ExecutionResult x87_load_integer(ExecutionContext& ctx, std::size_t width) {
  X87Scalar value = 0;
  if (width == 2) {
    std::int16_t v = 0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 2)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    value = static_cast<X87Scalar>(v);
  } else if (width == 4) {
    std::int32_t v = 0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    value = static_cast<X87Scalar>(v);
  } else if (width == 8) {
    std::int64_t v = 0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    value = static_cast<X87Scalar>(v);
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!ctx.state.x87_push(value)) return x87_stack_overflow(ctx);
  return {};
}

inline X87Scalar x87_round_half_even(X87Scalar value) {
  const X87Scalar flo = seven::floor(value);
  const X87Scalar frac = value - flo;
  const X87Scalar half = X87Scalar(0.5);
  if (frac < half) return flo;
  if (frac > half) return flo + 1;
  const X87Scalar parity = seven::fmod(seven::abs(flo), X87Scalar(2));
  return parity == 0 ? flo : flo + 1;
}

inline X87Scalar x87_round_to_control(const CpuState& state, X87Scalar value) {
  switch ((state.get_x87_control_word() >> 10) & 0x3u) {
    case 0: return x87_round_half_even(value);
    case 1: return seven::floor(value);
    case 2: return seven::ceil(value);
    default: return seven::trunc(value);
  }
}

inline int x87_cmp(X87Scalar a, X87Scalar b, bool quiet, std::uint16_t& exceptions) {
  if (seven::isnan(a) || seven::isnan(b)) {
    // "Quiet" is FUCOM, and it is only quiet about quiet NaNs. A signalling operand raises #IA on
    // both forms; the flag used to suppress it for either kind.
    if (!quiet || seven::issnan(a) || seven::issnan(b)) {
      exceptions |= kX87ExceptionInvalid;
    }
    return -2;
  }
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

inline void x87_set_cmp_flags(ExecutionContext& ctx, int relation) {
  auto sw = ctx.state.get_x87_status_word();
  sw &= static_cast<std::uint16_t>(~0x4700u);
  if (relation == -2) sw |= static_cast<std::uint16_t>(0x4500u);
  else if (relation < 0) sw |= static_cast<std::uint16_t>(0x0100u);
  else if (relation == 0) sw |= static_cast<std::uint16_t>(0x4000u);
  ctx.state.set_x87_status_word(sw);
}

inline void x87_set_c2(ExecutionContext& ctx, bool value) {
  auto sw = ctx.state.get_x87_status_word();
  sw = value ? static_cast<std::uint16_t>(sw | 0x0400u) : static_cast<std::uint16_t>(sw & ~0x0400u);
  ctx.state.set_x87_status_word(sw);
}

// FPREM/FPREM1 hand back the low three bits of the quotient, and not in register order: Q2 goes to
// C0, Q1 to C3, Q0 to C1. A guest reducing an argument by pi/4 reads them to work out which octant
// it landed in, so getting the pairing wrong is silently wrong trigonometry rather than a fault.
inline void x87_set_quotient_bits(ExecutionContext& ctx, std::uint64_t quotient) {
  auto sw = ctx.state.get_x87_status_word();
  sw &= static_cast<std::uint16_t>(~0x4300u);
  if ((quotient & 0x4u) != 0) sw |= static_cast<std::uint16_t>(0x0100u);
  if ((quotient & 0x2u) != 0) sw |= static_cast<std::uint16_t>(0x4000u);
  if ((quotient & 0x1u) != 0) sw |= static_cast<std::uint16_t>(0x0200u);
  ctx.state.set_x87_status_word(sw);
}

// FSIN/FCOS/FSINCOS/FPTAN only reduce arguments below 2^63. Past that hardware raises C2 and leaves
// the operand and the stack exactly as they were rather than returning a meaningless answer.
inline bool x87_trig_argument_out_of_range(const X87Scalar& value) {
  if (seven::isnan(value) || seven::isinf(value)) return false;
  return (value.val.signExp & 0x7FFFu) >= 0x403Eu;
}

inline void x87_set_eflags_cmp(ExecutionContext& ctx, int relation) {
  auto sw = ctx.state.get_x87_status_word();
  sw &= static_cast<std::uint16_t>(~0x0200u);
  ctx.state.set_x87_status_word(sw);
  detail::set_flag(ctx.state.rflags, kFlagCF, relation == -2 || relation < 0);
  detail::set_flag(ctx.state.rflags, kFlagPF, relation == -2);
  detail::set_flag(ctx.state.rflags, kFlagZF, relation == -2 || relation == 0);
}

inline ExecutionResult x87_move_if(ExecutionContext& ctx, bool take) {
  if (!take) return {};
  if (ctx.instr.op_kind(1) != iced_x86::OpKind::REGISTER) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto src = ctx.instr.op_register(1);
  if (src < iced_x86::Register::ST0 || src > iced_x86::Register::ST7) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto src_idx = x87_st_index(src);
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(src_idx)) return x87_stack_underflow_into(ctx, 0);
  ctx.state.x87_set(0, ctx.state.x87_get(src_idx));
  return {};
}

inline ExecutionResult x87_compare_mem(ExecutionContext& ctx, std::size_t width, bool pop, bool eflags, bool quiet = false) {
  X87Scalar rhs = 0;
  std::uint16_t exceptions = 0;
  if (width == 4) {
    float v = 0.0f;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else if (width == 8) {
    double v = 0.0;
    if (!ctx.memory.read(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
    rhs = static_cast<X87Scalar>(v);
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (ctx.state.x87_is_empty(0)) {
    exceptions |= kX87ExceptionInvalid;
    if (eflags) x87_set_eflags_cmp(ctx, -2);
    else x87_set_cmp_flags(ctx, -2);
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  const int rel = x87_cmp(ctx.state.x87_get(0), rhs, quiet, exceptions);
  if (eflags) x87_set_eflags_cmp(ctx, rel);
  else x87_set_cmp_flags(ctx, rel);
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_compare_indices(ExecutionContext& ctx, std::size_t lhs_idx, std::size_t rhs_idx, bool pop_lhs, bool pop_rhs, bool eflags, bool quiet = false) {
  std::uint16_t exceptions = 0;
  if (ctx.state.x87_is_empty(lhs_idx) || ctx.state.x87_is_empty(rhs_idx)) {
    exceptions |= kX87ExceptionInvalid;
    if (eflags) x87_set_eflags_cmp(ctx, -2);
    else x87_set_cmp_flags(ctx, -2);
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  const int rel = x87_cmp(ctx.state.x87_get(lhs_idx), ctx.state.x87_get(rhs_idx), quiet, exceptions);
  if (eflags) x87_set_eflags_cmp(ctx, rel);
  else x87_set_cmp_flags(ctx, rel);
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  if (pop_rhs && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  if (pop_lhs && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_compare_regs(ExecutionContext& ctx, std::uint32_t lhs_op, std::uint32_t rhs_op, bool pop_lhs, bool pop_rhs, bool eflags, bool quiet = false) {
  if (!x87_operand_is_st(ctx, lhs_op) || !x87_operand_is_st(ctx, rhs_op)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return x87_compare_indices(ctx, x87_st_index(ctx.instr.op_register(lhs_op)),
                             x87_st_index(ctx.instr.op_register(rhs_op)), pop_lhs, pop_rhs, eflags, quiet);
}

inline ExecutionResult x87_compare_st0_sti(ExecutionContext& ctx, std::uint32_t src_op, bool eflags, bool pop_st0, bool quiet = false) {
  if (!x87_operand_is_st(ctx, src_op)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto src_reg = ctx.instr.op_register(src_op);
  const auto src_idx = x87_st_index(src_reg);
  std::uint16_t exceptions = 0;
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(src_idx)) {
    exceptions |= kX87ExceptionInvalid;
    if (eflags) x87_set_eflags_cmp(ctx, -2);
    else x87_set_cmp_flags(ctx, -2);
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  const int rel = x87_cmp(ctx.state.x87_get(0), ctx.state.x87_get(src_idx), quiet, exceptions);
  if (eflags) x87_set_eflags_cmp(ctx, rel);
  else x87_set_cmp_flags(ctx, rel);
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  if (pop_st0 && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_load_bcd(ExecutionContext& ctx) {
  const auto base = detail::memory_address(ctx);
  std::uint64_t magnitude = 0;
  std::uint64_t mul = 1;
  for (std::size_t i = 0; i < 9; ++i) {
    std::uint8_t b = 0;
    if (!ctx.memory.read(base + i, &b, 1)) return detail::memory_fault(ctx, base + i);
    magnitude += static_cast<std::uint64_t>(b & 0x0F) * mul;
    mul *= 10;
    magnitude += static_cast<std::uint64_t>((b >> 4) & 0x0F) * mul;
    mul *= 10;
  }
  std::uint8_t sign = 0;
  if (!ctx.memory.read(base + 9, &sign, 1)) return detail::memory_fault(ctx, base + 9);
  X87Scalar value = static_cast<X87Scalar>(magnitude);
  if ((sign & 0x80u) != 0) value = -value;
  if (!ctx.state.x87_push(value)) return x87_stack_overflow(ctx);
  return {};
}

inline ExecutionResult x87_store_bcd(ExecutionContext& ctx) {
  const auto base = detail::memory_address(ctx);
  if (ctx.state.x87_is_empty(0)) {
    auto fault = x87_stack_underflow(ctx);
    if (!fault.ok()) return fault;
    // Packed decimal has an indefinite of its own, FFFFC000_00000000_0000, and it is not any of the
    // digit encodings this function can produce, so it is written out byte by byte.
    std::array<std::uint8_t, 10> raw{};
    raw[7] = 0xC0u;
    raw[8] = 0xFFu;
    raw[9] = 0xFFu;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      if (!ctx.memory.write(base + i, &raw[i], 1)) return detail::memory_fault(ctx, base + i);
    }
    // FBSTP is the only caller and it pops in its own handler, so leave ST(0) holding the indefinite
    // for that pop to consume rather than popping here.
    ctx.state.x87_set(0, x87_indefinite());
    return {};
  }
  X87Scalar value = ctx.state.x87_get(0);
  const bool neg = value < 0;
  if (neg) value = -value;
  std::uint64_t v = static_cast<std::uint64_t>(value);
  for (std::size_t i = 0; i < 9; ++i) {
    const std::uint8_t d0 = static_cast<std::uint8_t>(v % 10); v /= 10;
    const std::uint8_t d1 = static_cast<std::uint8_t>(v % 10); v /= 10;
    const std::uint8_t packed = static_cast<std::uint8_t>((d1 << 4) | d0);
    if (!ctx.memory.write(base + i, &packed, 1)) return detail::memory_fault(ctx, base + i);
  }
  const std::uint8_t sign = static_cast<std::uint8_t>(neg ? 0x80u : 0x00u);
  if (!ctx.memory.write(base + 9, &sign, 1)) return detail::memory_fault(ctx, base + 9);
  return {};
}

inline ExecutionResult x87_store_st0_to_sti(ExecutionContext& ctx, bool pop) {
  // FSTP ST(i) and friends carry exactly one operand, the destination. Reading operand 1 here got
  // Register::NONE for every one of them, so `fstp st(0)` -- the ordinary way to drop the top of
  // the stack -- wrote to ST(7) and left it tagged valid.
  if (!x87_operand_is_st(ctx, 0)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto idx = x87_st_index(ctx.instr.op_register(0));
  if (ctx.state.x87_is_empty(0)) {
    auto fault = x87_stack_underflow(ctx);
    if (!fault.ok()) return fault;
    ctx.state.x87_set(idx, x87_indefinite());
    if (pop) x87_underflow_pop(ctx.state);
    return {};
  }
  ctx.state.x87_set(idx, ctx.state.x87_get(0));
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_free_sti(ExecutionContext& ctx, bool pop) {
  if (!x87_operand_is_st(ctx, 0)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto reg = ctx.instr.op_register(0);
  ctx.state.x87_mark_empty(x87_st_index(reg));
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_fstp_m80fp(ExecutionContext& ctx) {
  const auto base = detail::memory_address(ctx);
  bool underflowed = false;
  if (ctx.state.x87_is_empty(0)) {
    auto fault = x87_stack_underflow(ctx);
    if (!fault.ok()) return fault;
    underflowed = true;
  }
  std::array<std::uint8_t, 10> raw{};
  x87_encoding::encode_ext80(underflowed ? x87_indefinite() : ctx.state.x87_get(0), raw.data());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (!ctx.memory.write(base + i, &raw[i], 1)) return detail::memory_fault(ctx, base + i);
  }
  if (underflowed) {
    x87_underflow_pop(ctx.state);
  } else if (!ctx.state.x87_pop()) {
    return x87_stack_underflow(ctx);
  }
  return {};
}

inline std::uint16_t x87_fxam_class_bits(const X87Scalar& value, bool empty) {
  if (empty) {
    return static_cast<std::uint16_t>(0x4100u);
  }
  // C3 = C2 = C0 = 0 is the "unsupported" answer. It has to come first: an unnormal reads as an
  // ordinary normal by value, and a pseudo-NaN reads as a NaN, so either would be claimed below.
  if (seven::isunsupported(value)) {
    return static_cast<std::uint16_t>(0x0000u);
  }
  if (seven::isnan(value)) {
    return static_cast<std::uint16_t>(0x0100u);
  }
  if (seven::isinf(value)) {
    return static_cast<std::uint16_t>(0x0500u);
  }
  if (value == 0) {
    return static_cast<std::uint16_t>(0x4000u);
  }
  const X87Scalar abs_value = seven::abs(value);
  const X87Scalar min_normal = std::numeric_limits<X87Scalar>::min();
  if (abs_value < min_normal) {
    return static_cast<std::uint16_t>(0x4400u);
  }
  return static_cast<std::uint16_t>(0x0400u);
}

inline bool x87_exceptions_masked(const CpuState& state, std::uint16_t exceptions) {
  return (exceptions & ~state.get_x87_control_word() & kX87ExceptionMask) == 0;
}

// The last resort, for the transcendentals only. FSIN and its neighbours are computed by narrowing
// to a host double and calling libm, so softfloat never sees the operation and has no flags to
// report; everything else now reads them out of x87_evaluate instead of guessing here. The one
// guess left is the underflow below: a transcendental landing in the denormal range is inexact in
// every case that matters, so tininess stands in for tininess-and-inexactness.
inline std::uint16_t x87_classify_result(const X87Scalar& result, const X87Scalar& lhs, const X87Scalar& rhs) {
  std::uint16_t exceptions = x87_operand_exceptions(lhs, rhs);
  if (seven::isnan(result)) {
    // A NaN that walked in as an operand propagates quietly: QNaN + 1 is a QNaN and raises nothing.
    // #IA belongs to the operations that MADE one -- a signalling operand being quieted, or a form
    // with no answer at all like inf - inf, whose operands are not NaNs to begin with.
    const bool propagated = seven::isnan(lhs) || seven::isnan(rhs);
    if (!propagated || seven::issnan(lhs) || seven::issnan(rhs)) {
      exceptions |= kX87ExceptionInvalid;
    }
  }
  // Overflow means the exact answer was finite and too large to represent. An infinity handed in as
  // an operand is not that (inf + 1 is inf, quietly), and neither is the infinity a division by zero
  // produces -- that is #Z, which the divide helper reports itself. A zero operand is what separates
  // the second case from a real overflow, since none of add, sub or mul can overflow through one.
  if (seven::isinf(result) && !seven::isinf(lhs) && !seven::isinf(rhs) && lhs != 0 && rhs != 0) {
    exceptions |= kX87ExceptionOverflow;
  }
  const X87Scalar abs_result = seven::abs(result);
  const X87Scalar min_normal = std::numeric_limits<X87Scalar>::min();
  if (result != 0 && abs_result < min_normal) {
    exceptions |= kX87ExceptionUnderflow;
  }
  return exceptions;
}

inline ExecutionResult x87_exception(ExecutionContext& ctx, std::uint16_t exceptions) {
  auto sw = ctx.state.get_x87_status_word();
  sw |= static_cast<std::uint16_t>(exceptions & kX87ExceptionMask);
  sw |= static_cast<std::uint16_t>(exceptions & kX87ExceptionStackFault);
  const bool masked = x87_exceptions_masked(ctx.state, exceptions);
  if (!masked) {
    sw |= static_cast<std::uint16_t>(0x0080u);
  }
  ctx.state.set_x87_status_word(sw);
  if (masked) {
    return {};
  }
  return {StopReason::floating_point_exception, 0, ExceptionInfo{StopReason::floating_point_exception, ctx.state.rip, exceptions}, ctx.instr.code()};
}

inline ExecutionResult x87_stack_underflow(ExecutionContext& ctx) {
  auto sw = ctx.state.get_x87_status_word();
  sw &= static_cast<std::uint16_t>(~0x0200u);
  ctx.state.set_x87_status_word(sw);
  return x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionInvalid | kX87ExceptionStackFault));
}

// Same fault, but for the instructions whose destination is a stack register. A masked stack
// underflow is not a no-op on hardware: the instruction still completes and leaves the indefinite in
// its destination. Only the unmasked path stops before writing anything.
inline ExecutionResult x87_stack_underflow_into(ExecutionContext& ctx, std::size_t dst_index) {
  auto result = x87_stack_underflow(ctx);
  if (!result.ok()) return result;
  ctx.state.x87_set(dst_index, x87_indefinite());
  return {};
}

inline ExecutionResult x87_stack_overflow(ExecutionContext& ctx) {
  auto sw = ctx.state.get_x87_status_word();
  sw |= static_cast<std::uint16_t>(0x0200u);
  ctx.state.set_x87_status_word(sw);
  auto result = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionInvalid | kX87ExceptionStackFault));
  if (!result.ok()) return result;
  // Masked: hardware still decrements TOP and drops the indefinite into the new top. Leaving TOP
  // where it was is not a smaller error than writing the wrong value -- every later ST(i) then
  // resolves to a different physical register than the guest is addressing.
  const auto top = static_cast<std::uint8_t>((ctx.state.get_x87_top() + 7) & 0x7);
  ctx.state.set_x87_top(top);
  ctx.state.x87_stack[top] = x87_indefinite();
  ctx.state.x87_tags[top] = 0x0;
  return {};
}

inline void x87_set_fxam_flags(ExecutionContext& ctx, const X87Scalar& value, bool empty) {
  auto sw = ctx.state.get_x87_status_word();
  sw &= static_cast<std::uint16_t>(~0x4700u);
  sw |= x87_fxam_class_bits(value, empty);
  // C1 is the register's sign bit whatever the class, empty included: FXAM reports on the register,
  // not on a value. An empty caller has no value to hand over, so read the sign out of the physical
  // register the instruction is looking at.
  const bool negative = empty ? seven::signbit(ctx.state.x87_stack[ctx.state.x87_phys_index(0)])
                              : seven::signbit(value);
  if (negative) {
    sw |= static_cast<std::uint16_t>(0x0200u);
  }
  ctx.state.set_x87_status_word(sw);
}

}  // namespace seven::handlers




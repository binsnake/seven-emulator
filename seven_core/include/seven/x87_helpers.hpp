#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

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

inline bool x87_exceptions_masked(const CpuState& state, std::uint16_t exceptions);
inline ExecutionResult x87_exception(ExecutionContext& ctx, std::uint16_t exceptions);
inline ExecutionResult x87_stack_underflow(ExecutionContext& ctx);
inline ExecutionResult x87_stack_overflow(ExecutionContext& ctx);
inline ExecutionResult x87_precision_if_changed(ExecutionContext& ctx, const X87Scalar& before, const X87Scalar& after);
inline std::uint16_t x87_classify_result(const X87Scalar& result, const X87Scalar& lhs, const X87Scalar& rhs);
inline std::uint16_t x87_precision_from_binary(const X87Scalar& lhs, const X87Scalar& rhs, const X87Scalar& result);
inline X87Scalar x87_round_to_control(const CpuState& state, X87Scalar value);

inline std::size_t x87_st_index(iced_x86::Register reg) {
  return static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::ST0));
}

template <typename Fn>
inline ExecutionResult x87_unary_st0(ExecutionContext& ctx, Fn&& fn) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto input = ctx.state.x87_get(0);
  const auto value = fn(input);
  const auto exceptions = x87_classify_result(value, input, 0);
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, value);
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
    return x87_stack_underflow(ctx);
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
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto lhs = ctx.state.x87_get(0);
  const auto value = fn(lhs, rhs);
  const auto exceptions = static_cast<std::uint16_t>(x87_classify_result(value, lhs, rhs) | x87_precision_from_binary(lhs, rhs, value));
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, value);
  return {};
}

template <typename Fn>
inline ExecutionResult x87_binary_mem_st0_with_status(ExecutionContext& ctx, std::size_t width, Fn&& fn) {
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
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto [value, exceptions] = fn(ctx.state.x87_get(0), rhs);
  const auto lhs = ctx.state.x87_get(0);
  const auto classified = static_cast<std::uint16_t>(x87_classify_result(value, lhs, rhs) | x87_precision_from_binary(lhs, rhs, value));
  if ((exceptions | classified) != 0) {
    auto result = x87_exception(ctx, static_cast<std::uint16_t>(exceptions | classified));
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, value);
  return {};
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
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto lhs = ctx.state.x87_get(0);
  const auto value = fn(lhs, rhs);
  const auto exceptions = static_cast<std::uint16_t>(x87_classify_result(value, lhs, rhs) | x87_precision_from_binary(lhs, rhs, value));
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, value);
  return {};
}

template <typename Fn>
inline ExecutionResult x87_binary_mem_int_st0_with_status(ExecutionContext& ctx, std::size_t width, Fn&& fn) {
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
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto [value, exceptions] = fn(ctx.state.x87_get(0), rhs);
  const auto lhs = ctx.state.x87_get(0);
  const auto classified = static_cast<std::uint16_t>(x87_classify_result(value, lhs, rhs) | x87_precision_from_binary(lhs, rhs, value));
  if ((exceptions | classified) != 0) {
    auto result = x87_exception(ctx, static_cast<std::uint16_t>(exceptions | classified));
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, value);
  return {};
}

template <typename Fn>
inline ExecutionResult x87_binary_st_regs(ExecutionContext& ctx, std::uint32_t dst_op, std::uint32_t src_op, Fn&& fn) {
  if (ctx.instr.op_kind(dst_op) != iced_x86::OpKind::REGISTER || ctx.instr.op_kind(src_op) != iced_x86::OpKind::REGISTER) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst = ctx.instr.op_register(dst_op);
  const auto src = ctx.instr.op_register(src_op);
  if (dst < iced_x86::Register::ST0 || dst > iced_x86::Register::ST7 || src < iced_x86::Register::ST0 || src > iced_x86::Register::ST7) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst_idx = x87_st_index(dst);
  const auto src_idx = x87_st_index(src);
  if (ctx.state.x87_is_empty(dst_idx) || ctx.state.x87_is_empty(src_idx)) return x87_stack_underflow(ctx);
  const auto lhs = ctx.state.x87_get(dst_idx);
  const auto rhs = ctx.state.x87_get(src_idx);
  const auto value = fn(lhs, rhs);
  const auto exceptions = static_cast<std::uint16_t>(x87_classify_result(value, lhs, rhs) | x87_precision_from_binary(lhs, rhs, value));
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(dst_idx, value);
  return {};
}

template <typename Fn>
inline ExecutionResult x87_binary_st_regs_with_status(ExecutionContext& ctx, std::uint32_t dst_op, std::uint32_t src_op, Fn&& fn) {
  if (ctx.instr.op_kind(dst_op) != iced_x86::OpKind::REGISTER || ctx.instr.op_kind(src_op) != iced_x86::OpKind::REGISTER) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst = ctx.instr.op_register(dst_op);
  const auto src = ctx.instr.op_register(src_op);
  if (dst < iced_x86::Register::ST0 || dst > iced_x86::Register::ST7 || src < iced_x86::Register::ST0 || src > iced_x86::Register::ST7) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst_idx = x87_st_index(dst);
  const auto src_idx = x87_st_index(src);
  if (ctx.state.x87_is_empty(dst_idx) || ctx.state.x87_is_empty(src_idx)) return x87_stack_underflow(ctx);
  const auto lhs = ctx.state.x87_get(dst_idx);
  const auto rhs = ctx.state.x87_get(src_idx);
  const auto [value, exceptions] = fn(lhs, rhs);
  const auto classified = static_cast<std::uint16_t>(x87_classify_result(value, lhs, rhs) | x87_precision_from_binary(lhs, rhs, value));
  if ((exceptions | classified) != 0) {
    auto result = x87_exception(ctx, static_cast<std::uint16_t>(exceptions | classified));
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(dst_idx, value);
  return {};
}

inline ExecutionResult x87_store_mem(ExecutionContext& ctx, std::size_t width, bool pop) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto value = ctx.state.x87_get(0);
  ExecutionResult result = {};
  if (width == 4) {
    result = detail::write_memory_checked(ctx, detail::memory_address(ctx), static_cast<float>(value));
  } else if (width == 8) {
    result = detail::write_memory_checked(ctx, detail::memory_address(ctx), static_cast<double>(value));
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
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
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
    return x87_exception(ctx, kX87ExceptionInvalid);
  }
  return {};
}

inline ExecutionResult x87_store_to_memory(ExecutionContext& ctx, std::size_t width, bool pop) {
  return x87_store_mem(ctx, width, pop);
}

inline ExecutionResult x87_store_integer(ExecutionContext& ctx, std::size_t width, bool pop, bool truncate_only) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const auto value = ctx.state.x87_get(0);
  const auto rounded = truncate_only ? boost::multiprecision::trunc(value) : x87_round_to_control(ctx.state, value);
  std::uint16_t exceptions = 0;
  if (rounded != value) {
    exceptions |= kX87ExceptionPrecision;
  }
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
  X87Scalar stored = rounded;
  if (boost::multiprecision::isnan(rounded) || boost::multiprecision::isinf(rounded) || rounded < lower || rounded > upper) {
    exceptions |= kX87ExceptionInvalid;
    stored = lower;
  }
  const std::int64_t out = static_cast<std::int64_t>(stored);
  if (width == 2) {
    const std::int16_t v = static_cast<std::int16_t>(out);
    if (!ctx.memory.write(detail::memory_address(ctx), &v, 2)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  } else if (width == 4) {
    const std::int32_t v = static_cast<std::int32_t>(out);
    if (!ctx.memory.write(detail::memory_address(ctx), &v, 4)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  } else if (width == 8) {
    const std::int64_t v = static_cast<std::int64_t>(out);
    if (!ctx.memory.write(detail::memory_address(ctx), &v, 8)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  } else {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
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
  const X87Scalar flo = boost::multiprecision::floor(value);
  const X87Scalar frac = value - flo;
  const X87Scalar half = X87Scalar(0.5);
  if (frac < half) return flo;
  if (frac > half) return flo + 1;
  const X87Scalar parity = boost::multiprecision::fmod(boost::multiprecision::abs(flo), 2);
  return parity == 0 ? flo : flo + 1;
}

inline X87Scalar x87_round_to_control(const CpuState& state, X87Scalar value) {
  switch ((state.get_x87_control_word() >> 10) & 0x3u) {
    case 0: return x87_round_half_even(value);
    case 1: return boost::multiprecision::floor(value);
    case 2: return boost::multiprecision::ceil(value);
    default: return boost::multiprecision::trunc(value);
  }
}

inline ExecutionResult x87_reg_move(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, bool pop) {
  if (ctx.instr.op_kind(dst) != iced_x86::OpKind::REGISTER || ctx.instr.op_kind(src) != iced_x86::OpKind::REGISTER) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst_reg = ctx.instr.op_register(dst);
  const auto src_reg = ctx.instr.op_register(src);
  if (ctx.state.x87_is_empty(x87_st_index(src_reg))) return x87_stack_underflow(ctx);
  ctx.state.x87_set(x87_st_index(dst_reg), ctx.state.x87_get(x87_st_index(src_reg)));
  if (pop) ctx.state.x87_mark_empty(x87_st_index(src_reg));
  return {};
}

inline int x87_cmp(X87Scalar a, X87Scalar b, bool quiet, std::uint16_t& exceptions) {
  if (boost::multiprecision::isnan(a) || boost::multiprecision::isnan(b)) {
    if (!quiet) {
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
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(src_idx)) return x87_stack_underflow(ctx);
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

inline ExecutionResult x87_compare_regs(ExecutionContext& ctx, std::uint32_t lhs_op, std::uint32_t rhs_op, bool pop_lhs, bool pop_rhs, bool eflags, bool quiet = false) {
  if (ctx.instr.op_kind(lhs_op) != iced_x86::OpKind::REGISTER || ctx.instr.op_kind(rhs_op) != iced_x86::OpKind::REGISTER) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto lhs_reg = ctx.instr.op_register(lhs_op);
  const auto rhs_reg = ctx.instr.op_register(rhs_op);
  const auto lhs_idx = x87_st_index(lhs_reg);
  const auto rhs_idx = x87_st_index(rhs_reg);
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

inline ExecutionResult x87_compare_st0_sti(ExecutionContext& ctx, std::uint32_t src_op, bool eflags, bool pop_st0, bool quiet = false) {
  if (ctx.instr.op_kind(src_op) != iced_x86::OpKind::REGISTER) return detail::memory_fault(ctx, detail::memory_address(ctx));
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
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
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
  if (ctx.instr.op_kind(1) != iced_x86::OpKind::REGISTER) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto dst = ctx.instr.op_register(1);
  const auto idx = x87_st_index(dst);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  ctx.state.x87_set(idx, ctx.state.x87_get(0));
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_free_sti(ExecutionContext& ctx, bool pop) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto reg = ctx.instr.op_register(0);
  ctx.state.x87_mark_empty(x87_st_index(reg));
  if (pop && !ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline ExecutionResult x87_fstp_m80fp(ExecutionContext& ctx) {
  const auto base = detail::memory_address(ctx);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  std::array<std::uint8_t, 10> raw{};
  x87_encoding::encode_ext80(ctx.state.x87_get(0), raw.data());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (!ctx.memory.write(base + i, &raw[i], 1)) return detail::memory_fault(ctx, base + i);
  }
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

inline std::uint16_t x87_fxam_class_bits(const X87Scalar& value, bool empty) {
  if (empty) {
    return static_cast<std::uint16_t>(0x4100u);
  }
  if (boost::multiprecision::isnan(value)) {
    return static_cast<std::uint16_t>(0x0100u);
  }
  if (boost::multiprecision::isinf(value)) {
    return static_cast<std::uint16_t>(0x0500u);
  }
  if (value == 0) {
    return static_cast<std::uint16_t>(0x4000u);
  }
  const X87Scalar abs_value = boost::multiprecision::abs(value);
  const X87Scalar min_normal = std::numeric_limits<X87Scalar>::min();
  if (abs_value < min_normal) {
    return static_cast<std::uint16_t>(0x4400u);
  }
  return static_cast<std::uint16_t>(0x0400u);
}

inline bool x87_exceptions_masked(const CpuState& state, std::uint16_t exceptions) {
  return (exceptions & ~state.get_x87_control_word() & kX87ExceptionMask) == 0;
}

inline std::uint16_t x87_classify_result(const X87Scalar& result, const X87Scalar& lhs, const X87Scalar& rhs) {
  std::uint16_t exceptions = 0;
  if (boost::multiprecision::isnan(result)) {
    exceptions |= kX87ExceptionInvalid;
  }
  if (boost::multiprecision::isinf(result)) {
    exceptions |= kX87ExceptionOverflow;
  }
  const X87Scalar abs_result = boost::multiprecision::abs(result);
  const X87Scalar min_normal = std::numeric_limits<X87Scalar>::min();
  if (result != 0 && abs_result < min_normal) {
    exceptions |= kX87ExceptionUnderflow;
  }
  if ((lhs != 0 && boost::multiprecision::abs(lhs) < min_normal) || (rhs != 0 && boost::multiprecision::abs(rhs) < min_normal) || (result != 0 && abs_result < min_normal)) {
    exceptions |= kX87ExceptionDenormal;
  }
  return exceptions;
}

inline std::uint16_t x87_precision_from_binary(const X87Scalar& lhs, const X87Scalar& rhs, const X87Scalar& result) {
  if (boost::multiprecision::isnan(result) || boost::multiprecision::isinf(result)) {
    return 0;
  }
  const bool add_exact = (result - lhs == rhs) || (result - rhs == lhs);
  const bool sub_exact = (lhs - result == rhs) || (result + rhs == lhs);
  const bool mul_exact = (lhs == 0 || rhs == 0) || (result / lhs == rhs) || (result / rhs == lhs);
  const bool div_exact = (rhs != 0 && result * rhs == lhs) || (lhs != 0 && result * lhs == rhs);
  if (add_exact || sub_exact || mul_exact || div_exact) {
    return 0;
  }
  if (result == 0 && lhs != 0 && rhs != 0) {
    return static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision);
  }
  return kX87ExceptionPrecision;
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

inline ExecutionResult x87_stack_overflow(ExecutionContext& ctx) {
  auto sw = ctx.state.get_x87_status_word();
  sw |= static_cast<std::uint16_t>(0x0200u);
  ctx.state.set_x87_status_word(sw);
  return x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionInvalid | kX87ExceptionStackFault));
}

inline ExecutionResult x87_precision_if_changed(ExecutionContext& ctx, const X87Scalar& before, const X87Scalar& after) {
  if (before == after) {
    return {};
  }
  return x87_exception(ctx, kX87ExceptionPrecision);
}

inline void x87_set_fxam_flags(ExecutionContext& ctx, const X87Scalar& value, bool empty) {
  auto sw = ctx.state.get_x87_status_word();
  sw &= static_cast<std::uint16_t>(~0x4700u);
  sw |= x87_fxam_class_bits(value, empty);
  if (boost::multiprecision::signbit(value)) {
    sw |= static_cast<std::uint16_t>(0x0200u);
  }
  ctx.state.set_x87_status_word(sw);
}

}  // namespace seven::handlers




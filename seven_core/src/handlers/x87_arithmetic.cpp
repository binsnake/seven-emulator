#include <cmath>
#include <limits>

#include "seven/handler_helpers.hpp"
#include "seven/x87_helpers.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

namespace {

inline std::pair<X87Scalar, std::uint16_t> x87_div_pair(X87Scalar lhs, X87Scalar rhs, bool reverse) {
  const X87Scalar dividend = reverse ? rhs : lhs;
  const X87Scalar divisor = reverse ? lhs : rhs;
  if (divisor == 0) {
    if (dividend == 0) {
      return {std::numeric_limits<X87Scalar>::quiet_NaN(), kX87ExceptionInvalid};
    }
    const X87Scalar inf = std::numeric_limits<X87Scalar>::infinity();
    const bool negative = seven::signbit(dividend) ^ seven::signbit(divisor);
    return {negative ? -inf : inf, kX87ExceptionZeroDiv};
  }
  const X87Scalar value = reverse ? rhs / lhs : lhs / rhs;
  std::uint16_t exceptions = 0;
  if (value == 0 && dividend != 0) {
    exceptions |= static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision);
  }
  if (value * divisor != dividend) {
    exceptions |= kX87ExceptionPrecision;
  }
  return {value, exceptions};
}

// FPREM and FPREM1 are partial remainders: one execution is only guaranteed to make progress when
// the two exponents are within 64 of each other. Past that the instruction reduces by an
// implementation-chosen amount, raises C2 to say it is not finished, and expects the guest to run it
// again. seven computed the whole remainder in one shot and never touched a condition code, so a
// guest looping on C2 spun forever and one reading the quotient bits read stale flags. Both take
// their operands implicitly from ST(0) and ST(1).
ExecutionResult x87_partial_remainder(ExecutionContext& ctx, bool round_to_nearest) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) {
    auto result = x87_stack_underflow_into(ctx, 0);
    if (!result.ok()) return result;
    x87_set_c2(ctx, false);
    return {};
  }
  const X87Scalar a = ctx.state.x87_get(0);
  const X87Scalar b = ctx.state.x87_get(1);

  if (seven::isnan(a) || seven::isnan(b) || seven::isinf(a) || b == X87Scalar(0)) {
    x87_set_c2(ctx, false);
    auto result = x87_exception(ctx, kX87ExceptionInvalid);
    if (!result.ok()) return result;
    ctx.state.x87_set(0, x87_indefinite());
    return {};
  }
  if (seven::isinf(b) || a == X87Scalar(0)) {
    x87_set_c2(ctx, false);
    x87_set_quotient_bits(ctx, 0);
    return {};
  }

  int exponent_a = 0;
  int exponent_b = 0;
  (void)seven::frexp(a, &exponent_a);
  (void)seven::frexp(b, &exponent_b);
  const int difference = exponent_a - exponent_b;

  if (difference >= 64) {
    // The SDM leaves the reduction size open between 32 and 63 bits of quotient. Scaling the divisor
    // up by the leftover exponent and taking an ordinary remainder against that is exact, and pulls
    // the difference down by at least 63 each time, so the guest's loop terminates.
    constexpr int kQuotientBits = 63;
    ctx.state.x87_set(0, seven::fmod(a, seven::ldexp(b, difference - kQuotientBits)));
    x87_set_c2(ctx, true);
    return {};
  }

  const X87Scalar quotient = round_to_nearest ? x87_round_half_even(a / b) : seven::trunc(a / b);
  const X87Scalar remainder = round_to_nearest ? seven::remainder(a, b) : seven::fmod(a, b);
  if (const auto exceptions = x87_classify_result(remainder, a, b); exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, remainder);
  x87_set_c2(ctx, false);
  x87_set_quotient_bits(ctx, static_cast<std::uint64_t>(seven::abs(quotient)));
  return {};
}

}  // namespace

ExecutionResult handle_code_FADD_M32FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return a + b; }); }
ExecutionResult handle_code_FSUBR_M32FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return b - a; }); }
ExecutionResult handle_code_FDIVR_M32FP(ExecutionContext& ctx) { return x87_binary_mem_st0_with_status(ctx, 4, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); }); }
ExecutionResult handle_code_FMUL_M32FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return a * b; }); }
ExecutionResult handle_code_FSUB_M32FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return a - b; }); }
ExecutionResult handle_code_FDIV_M32FP(ExecutionContext& ctx) { return x87_binary_mem_st0_with_status(ctx, 4, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); }); }
ExecutionResult handle_code_FADD_ST0_STI(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a + b; }); }
ExecutionResult handle_code_FMUL_ST0_STI(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a * b; }); }
ExecutionResult handle_code_FSUB_ST0_STI(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a - b; }); }
ExecutionResult handle_code_FSUBR_ST0_STI(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return b - a; }); }
ExecutionResult handle_code_FDIV_ST0_STI(ExecutionContext& ctx) { return x87_binary_st_regs_with_status(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); }); }
ExecutionResult handle_code_FDIVR_ST0_STI(ExecutionContext& ctx) { return x87_binary_st_regs_with_status(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); }); }
ExecutionResult handle_code_FADD_M64FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 8, [](X87Scalar a, X87Scalar b) { return a + b; }); }
ExecutionResult handle_code_FSUBR_M64FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 8, [](X87Scalar a, X87Scalar b) { return b - a; }); }
ExecutionResult handle_code_FDIVR_M64FP(ExecutionContext& ctx) { return x87_binary_mem_st0_with_status(ctx, 8, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); }); }
ExecutionResult handle_code_FMUL_M64FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 8, [](X87Scalar a, X87Scalar b) { return a * b; }); }
ExecutionResult handle_code_FSUB_M64FP(ExecutionContext& ctx) { return x87_binary_mem_st0(ctx, 8, [](X87Scalar a, X87Scalar b) { return a - b; }); }
ExecutionResult handle_code_FDIV_M64FP(ExecutionContext& ctx) { return x87_binary_mem_st0_with_status(ctx, 8, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); }); }
ExecutionResult handle_code_FADD_STI_ST0(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a + b; }); }
ExecutionResult handle_code_FMUL_STI_ST0(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a * b; }); }
ExecutionResult handle_code_FSUBR_STI_ST0(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return b - a; }); }
ExecutionResult handle_code_FSUB_STI_ST0(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a - b; }); }
ExecutionResult handle_code_FDIVR_STI_ST0(ExecutionContext& ctx) { return x87_binary_st_regs_with_status(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); }); }
ExecutionResult handle_code_FDIV_STI_ST0(ExecutionContext& ctx) { return x87_binary_st_regs_with_status(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); }); }
ExecutionResult handle_code_FADDP_STI_ST0(ExecutionContext& ctx) {
  const auto result = x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a + b; });
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}
ExecutionResult handle_code_FMULP_STI_ST0(ExecutionContext& ctx) {
  const auto result = x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a * b; });
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}
ExecutionResult handle_code_FSUBRP_STI_ST0(ExecutionContext& ctx) {
  const auto result = x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return b - a; });
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}
ExecutionResult handle_code_FSUBP_STI_ST0(ExecutionContext& ctx) {
  const auto result = x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return a - b; });
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}
ExecutionResult handle_code_FDIVRP_STI_ST0(ExecutionContext& ctx) {
  const auto result = x87_binary_st_regs_with_status(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); });
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}
ExecutionResult handle_code_FDIVP_STI_ST0(ExecutionContext& ctx) {
  const auto result = x87_binary_st_regs_with_status(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); });
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}
ExecutionResult handle_code_FSTP_M32FP(ExecutionContext& ctx) { return x87_store_mem(ctx, 4, true); }
ExecutionResult handle_code_FSTP_M64FP(ExecutionContext& ctx) { return x87_store_mem(ctx, 8, true); }
ExecutionResult handle_code_FIADD_M16INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 2, [](X87Scalar a, X87Scalar b) { return a + b; }); }
ExecutionResult handle_code_FIADD_M32INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return a + b; }); }
ExecutionResult handle_code_FIMUL_M16INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 2, [](X87Scalar a, X87Scalar b) { return a * b; }); }
ExecutionResult handle_code_FIMUL_M32INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return a * b; }); }
ExecutionResult handle_code_FISUB_M16INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 2, [](X87Scalar a, X87Scalar b) { return a - b; }); }
ExecutionResult handle_code_FISUB_M32INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return a - b; }); }
ExecutionResult handle_code_FISUBR_M16INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 2, [](X87Scalar a, X87Scalar b) { return b - a; }); }
ExecutionResult handle_code_FISUBR_M32INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0(ctx, 4, [](X87Scalar a, X87Scalar b) { return b - a; }); }
ExecutionResult handle_code_FIDIV_M16INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0_with_status(ctx, 2, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); }); }
ExecutionResult handle_code_FIDIV_M32INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0_with_status(ctx, 4, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, false); }); }
ExecutionResult handle_code_FIDIVR_M16INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0_with_status(ctx, 2, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); }); }
ExecutionResult handle_code_FIDIVR_M32INT(ExecutionContext& ctx) { return x87_binary_mem_int_st0_with_status(ctx, 4, [](X87Scalar a, X87Scalar b) { return x87_div_pair(a, b, true); }); }
// FPREM/FPREM1 encode no operands at all -- ST(0) and ST(1) are implicit. Going through the
// operand-reading form asked for two registers that were never there and faulted.
ExecutionResult handle_code_FPREM(ExecutionContext& ctx) { return x87_partial_remainder(ctx, false); }
ExecutionResult handle_code_FPREM1(ExecutionContext& ctx) { return x87_partial_remainder(ctx, true); }

}  // namespace seven::handlers




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
    const bool negative = boost::multiprecision::signbit(dividend) ^ boost::multiprecision::signbit(divisor);
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
ExecutionResult handle_code_FPREM(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return boost::multiprecision::fmod(a, b); }); }
ExecutionResult handle_code_FPREM1(ExecutionContext& ctx) { return x87_binary_st_regs(ctx, 0, 1, [](X87Scalar a, X87Scalar b) { return boost::multiprecision::remainder(a, b); }); }

}  // namespace seven::handlers




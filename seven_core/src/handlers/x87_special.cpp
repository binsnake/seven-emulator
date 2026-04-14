#include <array>
#include <cmath>

#include "seven/handler_helpers.hpp"
#include "seven/x87_helpers.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

ExecutionResult handle_code_FNOP(ExecutionContext&) {
  return {};
}

ExecutionResult handle_code_FDECSTP(ExecutionContext& ctx) {
  ctx.state.set_x87_top(static_cast<std::uint8_t>((ctx.state.get_x87_top() + 7) & 0x7));
  return {};
}

ExecutionResult handle_code_FINCSTP(ExecutionContext& ctx) {
  ctx.state.set_x87_top(static_cast<std::uint8_t>((ctx.state.get_x87_top() + 1) & 0x7));
  return {};
}

ExecutionResult handle_code_FXAM(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) {
    x87_set_fxam_flags(ctx, 0, true);
    return {};
  }
  x87_set_fxam_flags(ctx, ctx.state.x87_get(0), false);
  return {};
}

ExecutionResult handle_code_FBLD_M80BCD(ExecutionContext& ctx) {
  return x87_load_bcd(ctx);
}

ExecutionResult handle_code_FBSTP_M80BCD(ExecutionContext& ctx) {
  const auto result = x87_store_bcd(ctx);
  if (!result.ok()) return result;
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_FSTP_M80FP(ExecutionContext& ctx) {
  return x87_fstp_m80fp(ctx);
}

ExecutionResult handle_code_FXCH_ST0_STI(ExecutionContext& ctx) {
  return x87_fxch(ctx);
}

ExecutionResult handle_code_FXCH_ST0_STI_DDC8(ExecutionContext& ctx) {
  return x87_fxch(ctx);
}

ExecutionResult handle_code_FXCH_ST0_STI_DFC8(ExecutionContext& ctx) {
  return x87_fxch(ctx);
}

ExecutionResult handle_code_FSTP_STI(ExecutionContext& ctx) {
  return x87_store_st0_to_sti(ctx, true);
}

ExecutionResult handle_code_FSTP_STI_DFD0(ExecutionContext& ctx) {
  return x87_store_st0_to_sti(ctx, true);
}

ExecutionResult handle_code_FSTP_STI_DFD8(ExecutionContext& ctx) {
  return x87_store_st0_to_sti(ctx, true);
}

ExecutionResult handle_code_FSTPNCE_STI(ExecutionContext& ctx) {
  return x87_store_st0_to_sti(ctx, true);
}

ExecutionResult handle_code_FFREE_STI(ExecutionContext& ctx) {
  return x87_free_sti(ctx, false);
}

ExecutionResult handle_code_FFREEP_STI(ExecutionContext& ctx) {
  return x87_free_sti(ctx, true);
}

}  // namespace seven::handlers




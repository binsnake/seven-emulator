#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

constexpr std::uint32_t kMsrKernelGsBase = 0xC0000102u;

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

[[nodiscard]] ExecutionResult gp_fault(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
}

}  // namespace

// SWAPGS is CPL0-only on real hardware (#GP(0) otherwise) -- letting any privilege level swap in
// kernel_gs_base lets ordinary code point GS at whatever another privilege level last stashed
// there and read/write through it via any ordinary GS-relative memory operand.
ExecutionResult handle_code_SWAPGS(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto user_gs = ctx.state.gs_base;
  const auto kernel_gs = detail::read_msr(ctx.state, kMsrKernelGsBase);
  ctx.state.gs_base = kernel_gs;
  detail::write_msr(ctx.state, kMsrKernelGsBase, user_gs);
  return {};
}

}  // namespace seven::handlers


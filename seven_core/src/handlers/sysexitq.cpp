#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

[[nodiscard]] ExecutionResult gp_fault(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
}

}  // namespace

// SYSEXIT is CPL0-only (#GP(0) otherwise), same as SYSRET.
ExecutionResult handle_code_SYSEXITQ(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto rip = detail::read_register(ctx.state, iced_x86::Register::RCX);
  const auto rsp = detail::read_register(ctx.state, iced_x86::Register::RDX);
  ctx.state.mode = ExecutionMode::long64;
  ctx.state.rip = rip;
  detail::write_register(ctx.state, iced_x86::Register::RSP, rsp, 8);
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers

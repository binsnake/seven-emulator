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

// CPL0-only, and it drops to compat mode, so ring 3 must not reach it.
ExecutionResult handle_code_SYSEXITD(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto eip = detail::read_register(ctx.state, iced_x86::Register::ECX) & 0xFFFFFFFFull;
  const auto esp = detail::read_register(ctx.state, iced_x86::Register::EDX) & 0xFFFFFFFFull;
  ctx.state.mode = ExecutionMode::compat32;
  detail::write_register(ctx.state, iced_x86::Register::ESP, esp, 4);
  ctx.state.rip = eip;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers

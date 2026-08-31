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

// CPL0-only, same as the 64-bit form next door. This one also drops to compat mode, so without the
// check ring 3 could switch the execution mode out from under itself.
ExecutionResult handle_code_SYSRETD(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto restore_rip = detail::read_register(ctx.state, iced_x86::Register::ECX) & 0xFFFFFFFFull;
  const auto restore_flags = detail::read_register(ctx.state, iced_x86::Register::R11) & 0xFFFFFFFFull;
  ctx.state.mode = ExecutionMode::compat32;
  detail::write_register(ctx.state, iced_x86::Register::ESP, detail::read_register(ctx.state, iced_x86::Register::EDX), 4);
  ctx.state.rip = restore_rip;
  ctx.state.rflags = (restore_flags & 0x3C7FD7ull) | 0x2ull;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers

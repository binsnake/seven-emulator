#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_SYSRETQ(ExecutionContext& ctx) {
  const auto restore_rip = detail::read_register(ctx.state, iced_x86::Register::RCX);
  const auto restore_flags = detail::read_register(ctx.state, iced_x86::Register::R11);
  ctx.state.mode = ExecutionMode::long64;
  ctx.state.rip = restore_rip;
  ctx.state.rflags = restore_flags;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_SYSRETD(ExecutionContext& ctx) {
  const auto restore_rip = detail::read_register(ctx.state, iced_x86::Register::ECX) & 0xFFFFFFFFull;
  const auto restore_flags = detail::read_register(ctx.state, iced_x86::Register::R11) & 0xFFFFFFFFull;
  ctx.state.mode = ExecutionMode::compat32;
  detail::write_register(ctx.state, iced_x86::Register::EAX, restore_rip, 4);
  detail::write_register(ctx.state, iced_x86::Register::ESP, detail::read_register(ctx.state, iced_x86::Register::EDX), 4);
  ctx.state.rip = restore_rip;
  ctx.state.rflags = restore_flags;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


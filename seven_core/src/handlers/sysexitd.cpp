#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_SYSEXITD(ExecutionContext& ctx) {
  const auto eip = detail::read_register(ctx.state, iced_x86::Register::ECX) & 0xFFFFFFFFull;
  const auto esp = detail::read_register(ctx.state, iced_x86::Register::EDX) & 0xFFFFFFFFull;
  ctx.state.mode = ExecutionMode::compat32;
  detail::write_register(ctx.state, iced_x86::Register::EAX, eip, 4);
  detail::write_register(ctx.state, iced_x86::Register::ESP, esp, 4);
  ctx.state.rip = eip;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


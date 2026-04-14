#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_SYSEXITQ(ExecutionContext& ctx) {
  const auto rip = detail::read_register(ctx.state, iced_x86::Register::RCX);
  const auto rsp = detail::read_register(ctx.state, iced_x86::Register::RDX);
  ctx.state.mode = ExecutionMode::long64;
  ctx.state.rip = rip;
  detail::write_register(ctx.state, iced_x86::Register::RSP, rsp, 8);
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


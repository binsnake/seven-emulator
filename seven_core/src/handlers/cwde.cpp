#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CWDE(ExecutionContext& ctx) {
  const auto ax = detail::read_register(ctx.state, iced_x86::Register::AX);
  detail::write_register(ctx.state, iced_x86::Register::EAX, detail::sign_extend(ax, 2), 4);
  return {};
}

}  // namespace seven::handlers



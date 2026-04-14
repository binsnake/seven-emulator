#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CBW(ExecutionContext& ctx) {
  const auto al = detail::read_register(ctx.state, iced_x86::Register::AL);
  detail::write_register(ctx.state, iced_x86::Register::AX, detail::sign_extend(al, 1), 2);
  return {};
}

}  // namespace seven::handlers



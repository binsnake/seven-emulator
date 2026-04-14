#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CDQE(ExecutionContext& ctx) {
  const auto eax = detail::read_register(ctx.state, iced_x86::Register::EAX);
  detail::write_register(ctx.state, iced_x86::Register::RAX, detail::sign_extend(eax, 4), 8);
  return {};
}

}  // namespace seven::handlers



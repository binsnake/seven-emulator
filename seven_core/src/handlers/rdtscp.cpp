#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_RDTSCP(ExecutionContext& ctx) {
  detail::write_register(ctx.state, iced_x86::Register::EAX, 0ull, 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, 0ull, 4);
  detail::write_register(ctx.state, iced_x86::Register::ECX, 0ull, 4);
  return {};
}

}  // namespace seven::handlers



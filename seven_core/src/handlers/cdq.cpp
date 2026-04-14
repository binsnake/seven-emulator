#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CDQ(ExecutionContext& ctx) {
  const auto eax = detail::read_register(ctx.state, iced_x86::Register::EAX);
  const auto sign = (eax & 0x80000000ull) != 0;
  detail::write_register(ctx.state, iced_x86::Register::EDX, sign ? 0xFFFFFFFFull : 0ull, 4);
  return {};
}

}  // namespace seven::handlers



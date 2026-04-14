#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CQO(ExecutionContext& ctx) {
  const auto rax = detail::read_register(ctx.state, iced_x86::Register::RAX);
  const auto sign = (rax & 0x8000000000000000ull) != 0;
  detail::write_register(ctx.state, iced_x86::Register::RDX, sign ? ~0ull : 0ull, 8);
  return {};
}

}  // namespace seven::handlers



#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CWD(ExecutionContext& ctx) {
  const auto ax = detail::read_register(ctx.state, iced_x86::Register::AX);
  const auto sign = (ax & 0x8000ull) != 0;
  detail::write_register(ctx.state, iced_x86::Register::DX, sign ? 0xFFFFull : 0ull, 2);
  return {};
}

}  // namespace seven::handlers



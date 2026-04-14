#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_XLAT_M8(ExecutionContext& ctx) {
  const auto base = detail::read_register(ctx.state, iced_x86::Register::RBX);
  const auto index = detail::read_register(ctx.state, iced_x86::Register::AL);
  const auto address = base + index;
  std::uint8_t value = 0;
  if (!ctx.memory.read(address, &value, 1)) {
    return {StopReason::page_fault, 0,
            ExceptionInfo{StopReason::page_fault, address, 0},
            ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::AL, value, 1);
  return {};
}

}  // namespace seven::handlers


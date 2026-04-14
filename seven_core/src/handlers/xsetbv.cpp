#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_XSETBV(ExecutionContext& ctx) {
  const auto xcr_index = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::ECX));
  if (xcr_index > 1u) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  const auto low = detail::read_register(ctx.state, iced_x86::Register::EAX) & 0xFFFFFFFFull;
  const auto high = detail::read_register(ctx.state, iced_x86::Register::EDX) & 0xFFFFFFFFull;
  detail::write_xcr(ctx.state, xcr_index, (high << 32) | low);
  return {};
}

}  // namespace seven::handlers


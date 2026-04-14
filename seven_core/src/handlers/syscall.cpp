#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

constexpr std::uint32_t kMsrLStar = 0xC0000082u;
constexpr std::uint32_t kMsrSFMASK = 0xC0000084u;

}

ExecutionResult handle_code_SYSCALL(ExecutionContext& ctx) {
  detail::write_register(ctx.state, iced_x86::Register::RCX, ctx.next_rip, 8);
  detail::write_register(ctx.state, iced_x86::Register::R11, ctx.state.rflags, 8);
  const auto lstar = detail::read_msr(ctx.state, static_cast<std::uint32_t>(kMsrLStar));
  const auto sfmask = detail::read_msr(ctx.state, static_cast<std::uint32_t>(kMsrSFMASK));
  ctx.state.rflags &= ~sfmask;
  ctx.state.rip = lstar;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


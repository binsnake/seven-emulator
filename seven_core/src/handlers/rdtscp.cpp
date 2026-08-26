#include "seven/handler_helpers.hpp"

namespace seven::handlers {

// Reads the same counter RDTSC does. Returning a constant 0 here while RDTSC counted up meant a
// guest that used both saw time run backwards.
ExecutionResult handle_code_RDTSCP(ExecutionContext& ctx) {
  ++ctx.state.tsc;
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(ctx.state.tsc & 0xFFFFFFFFull), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(ctx.state.tsc >> 32), 4);
  detail::write_register(ctx.state, iced_x86::Register::ECX, detail::read_msr(ctx.state, 0xC0000103u), 4);
  return {};
}

}  // namespace seven::handlers



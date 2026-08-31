#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

}  // namespace

// XSETBV is CPL0-only on real hardware (#GP(0) otherwise, alongside the existing reserved-index
// check below) -- it was letting any privilege level rewrite XCR0.
ExecutionResult handle_code_XSETBV(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
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


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

constexpr std::uint32_t kMsrSysenterCS = 0x174u;
constexpr std::uint32_t kMsrSysenterEsp = 0x175u;
constexpr std::uint32_t kMsrSysenterEip = 0x176u;

}

ExecutionResult handle_code_SYSENTER(ExecutionContext& ctx) {
  detail::write_register(ctx.state, iced_x86::Register::ECX, ctx.next_rip, 4);
  const auto target = detail::read_msr(ctx.state, kMsrSysenterEip);
  const auto stack = detail::read_msr(ctx.state, kMsrSysenterEsp);
  const auto cs = detail::read_msr(ctx.state, kMsrSysenterCS);
  const auto cs_selector = static_cast<std::uint16_t>(cs & 0xFFFFu);
  if (cs_selector == 0u) {
    return {StopReason::unsupported_instruction, 0,
            ExceptionInfo{StopReason::unsupported_instruction, ctx.state.rip, 0},
            ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::CS, cs_selector, 2);
  if (stack != 0u) {
    detail::write_register(ctx.state, iced_x86::Register::RSP, stack, 8);
  }
  if (target == 0u) {
    return {StopReason::unsupported_instruction, 0,
            ExceptionInfo{StopReason::unsupported_instruction, ctx.state.rip, 0},
            ctx.instr.code()};
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_LEAVEW(ExecutionContext& ctx) {
  const auto bp = detail::read_register(ctx.state, iced_x86::Register::BP);
  auto sp = bp;
  std::uint16_t popped_bp = 0;
  if (!ctx.memory.read(sp, &popped_bp, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, sp, 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::SP, sp + 2, 2);
  detail::write_register(ctx.state, iced_x86::Register::BP, popped_bp, 2);
  return {};
}

ExecutionResult handle_code_LEAVED(ExecutionContext& ctx) {
  const auto bp = detail::read_register(ctx.state, iced_x86::Register::EBP);
  auto sp = bp;
  std::uint32_t popped_bp = 0;
  if (!ctx.memory.read(sp, &popped_bp, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, sp, 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::ESP, sp + 4, 4);
  detail::write_register(ctx.state, iced_x86::Register::EBP, popped_bp, 4);
  return {};
}

ExecutionResult handle_code_LEAVEQ(ExecutionContext& ctx) {
  const auto bp = detail::read_register(ctx.state, iced_x86::Register::RBP);
  auto sp = bp;
  std::uint64_t popped_bp = 0;
  if (!ctx.memory.read(sp, &popped_bp, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, sp, 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::RSP, sp + 8, 8);
  detail::write_register(ctx.state, iced_x86::Register::RBP, popped_bp, 8);
  return {};
}

}  // namespace seven::handlers



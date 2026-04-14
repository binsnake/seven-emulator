#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_BOUND_R16_M1616(ExecutionContext& ctx) {
  const auto destination = detail::read_register(ctx.state, ctx.instr.op_register(0));

  const auto address = detail::memory_address(ctx);
  std::uint16_t lower = 0;
  std::uint16_t upper = 0;
  if (!ctx.memory.read(address, &lower, 2) || !ctx.memory.read(address + 2, &upper, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }

  const auto destination_sign = static_cast<std::int16_t>(destination);
  const auto lower_sign = static_cast<std::int16_t>(lower);
  const auto upper_sign = static_cast<std::int16_t>(upper);
  if (destination_sign < lower_sign || destination_sign > upper_sign) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_BOUND_R32_M3232(ExecutionContext& ctx) {
  const auto destination = detail::read_register(ctx.state, ctx.instr.op_register(0));

  const auto address = detail::memory_address(ctx);
  std::uint32_t lower = 0;
  std::uint32_t upper = 0;
  if (!ctx.memory.read(address, &lower, 4) || !ctx.memory.read(address + 4, &upper, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }

  const auto destination_sign = static_cast<std::int32_t>(destination);
  const auto lower_sign = static_cast<std::int32_t>(lower);
  const auto upper_sign = static_cast<std::int32_t>(upper);
  if (destination_sign < lower_sign || destination_sign > upper_sign) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_AADD_M32_R32(ExecutionContext& ctx) {
  const auto address = detail::memory_address(ctx);
  if ((address & 0x3ull) != 0ull) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, address, 0}, ctx.instr.code()};
  }
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_AADD_M64_R64(ExecutionContext& ctx) {
  const auto address = detail::memory_address(ctx);
  if ((address & 0x7ull) != 0ull) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, address, 0}, ctx.instr.code()};
  }
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers


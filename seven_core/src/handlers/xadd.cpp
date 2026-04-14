#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_XADD_RM8_R8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  detail::set_add_flags(ctx.state, lhs, rhs, result, 1);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, lhs, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_XADD_RM16_R16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  detail::set_add_flags(ctx.state, lhs, rhs, result, 2);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, lhs, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_XADD_RM32_R32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  detail::set_add_flags(ctx.state, lhs, rhs, result, 4);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, lhs, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_XADD_RM64_R64(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  detail::set_add_flags(ctx.state, lhs, rhs, result, 8);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, lhs, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CMPXCHG486_RM8_R8(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::AL);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 1)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::AL, lhs, 1);
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG486_RM16_R16(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::AX);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 2)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::AX, lhs, 2);
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG486_RM32_R32(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::EAX);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 4)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::EAX, lhs, 4);
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  return {};
}

}  // namespace seven::handlers


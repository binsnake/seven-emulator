#include "seven/handler_helpers.hpp"

namespace seven::handlers {
namespace {

ExecutionResult xchg_rm_r(ExecutionContext& ctx, std::size_t width) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, width, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  if (!detail::write_operand(ctx, 0, rhs, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, lhs, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult xchg_r_r(ExecutionContext& ctx, std::size_t width) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, width, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 1, width, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 0, rhs, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, lhs, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_XCHG_RM8_R8(ExecutionContext& ctx) {
  return xchg_rm_r(ctx, 1);
}

ExecutionResult handle_code_XCHG_RM16_R16(ExecutionContext& ctx) {
  return xchg_rm_r(ctx, 2);
}

ExecutionResult handle_code_XCHG_RM32_R32(ExecutionContext& ctx) {
  return xchg_rm_r(ctx, 4);
}

ExecutionResult handle_code_XCHG_RM64_R64(ExecutionContext& ctx) {
  return xchg_rm_r(ctx, 8);
}

ExecutionResult handle_code_XCHG_R64_RAX(ExecutionContext& ctx) {
  return xchg_r_r(ctx, 8);
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_ARPL_R32M16_R32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto destination = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto source = detail::read_register(ctx.state, ctx.instr.op_register(1));

  const auto src_rpl = source & 0x3ull;
  const auto dst_rpl = destination & 0x3ull;
  const auto adjusted = (src_rpl > dst_rpl) ? ((destination & ~0x3ull) | src_rpl) : destination;
  const bool modified = adjusted != destination;
  if (!detail::write_operand(ctx, 0, adjusted, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, modified);
  return {};
}

ExecutionResult handle_code_ARPL_RM16_R16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto destination = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto source = detail::read_register(ctx.state, ctx.instr.op_register(1));

  const auto src_rpl = source & 0x3ull;
  const auto dst_rpl = destination & 0x3ull;
  const auto adjusted = (src_rpl > dst_rpl) ? ((destination & ~0x3ull) | src_rpl) : destination;
  const bool modified = adjusted != destination;
  if (!detail::write_operand(ctx, 0, adjusted, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, modified);
  return {};
}

}  // namespace seven::handlers


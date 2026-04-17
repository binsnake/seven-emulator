#include "seven/handler_helpers.hpp"

namespace seven::handlers {

// SMSW stores the low 16 bits of CR0 (the Machine Status Word).
// For register destinations: RM16 writes 16 bits, R32M16/R64M16 zero-extend.
// For memory destinations: all variants write exactly 16 bits.

static ExecutionResult smsw_impl(ExecutionContext& ctx, std::size_t reg_width) {
  const std::uint64_t msw = ctx.state.cr[0] & 0xFFFFu;
  const bool is_mem = ctx.instr.op_kind(0) == iced_x86::OpKind::MEMORY;
  const std::size_t width = is_mem ? 2 : reg_width;
  if (!detail::write_operand(ctx, 0, msw, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SMSW_RM16(ExecutionContext& ctx)   { return smsw_impl(ctx, 2); }
ExecutionResult handle_code_SMSW_R32M16(ExecutionContext& ctx) { return smsw_impl(ctx, 4); }
ExecutionResult handle_code_SMSW_R64M16(ExecutionContext& ctx) { return smsw_impl(ctx, 8); }

}  // namespace seven::handlers

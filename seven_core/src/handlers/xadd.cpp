#include "seven/handler_helpers.hpp"

namespace seven::handlers {
namespace {

// Intel writes the destination last, which only matters when both operands name the same register:
// writing the source afterwards puts the stale value back and the exchange vanishes. A memory
// destination cannot alias, and writing it first is what stops a faulting store from committing the
// source write, so the aliasing case skips the redundant source write instead of reordering.
ExecutionResult xadd_width(ExecutionContext& ctx, std::size_t width) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, width, &dst_ok);
  if (!dst_ok) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto result = lhs + rhs;
  if (!detail::write_operand(ctx, 0, result, width)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const bool same_register = ctx.instr.op0_kind() == iced_x86::OpKind::REGISTER &&
                             ctx.instr.op_register(0) == ctx.instr.op_register(1);
  if (!same_register && !detail::write_operand(ctx, 1, lhs, width)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  // Last, so a store that faults on a read-only destination leaves rflags alone. CMPXCHG and
  // BTS/BTR/BTC in this same family already order it this way.
  detail::set_add_flags(ctx.state, lhs, rhs, result, width);
  return {};
}

}  // namespace

ExecutionResult handle_code_XADD_RM8_R8(ExecutionContext& ctx) { return xadd_width(ctx, 1); }

ExecutionResult handle_code_XADD_RM16_R16(ExecutionContext& ctx) { return xadd_width(ctx, 2); }

ExecutionResult handle_code_XADD_RM32_R32(ExecutionContext& ctx) { return xadd_width(ctx, 4); }

ExecutionResult handle_code_XADD_RM64_R64(ExecutionContext& ctx) { return xadd_width(ctx, 8); }

}  // namespace seven::handlers

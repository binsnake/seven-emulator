#include "seven/handler_helpers.hpp"
#include <iced_x86/op_kind.hpp>

namespace seven::handlers {

namespace {
uint64_t read_bit_index(std::uint64_t value, std::size_t width) {
  return value & (8ull * width - 1ull);
}

ExecutionResult bts_rmw(ExecutionContext& ctx, std::size_t width, std::uint64_t bit_index) {
  const auto bit_span = 8ull * width;
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  std::uint64_t address = 0;

  if (ctx.instr.op0_kind() == iced_x86::OpKind::MEMORY) {
    const auto elem_index = bit_index / bit_span;
    address = detail::memory_address(ctx) + elem_index * width;
    bit = bit_index & (bit_span - 1ull);
    const auto rr = detail::read_memory_checked(ctx, address, &value, width);
    if (!rr.ok()) {
      return rr;
    }
  } else {
    bool lhs_ok = false;
    value = detail::read_operand(ctx, 0, width, &lhs_ok);
    if (!lhs_ok) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    bit = bit_index & (bit_span - 1ull);
  }

  const bool old_bit = ((value >> bit) & 1ull) != 0;
  value |= (1ull << bit);

  if (ctx.instr.op0_kind() == iced_x86::OpKind::MEMORY) {
    const auto wr = detail::write_memory_checked(ctx, address, &value, width);
    if (!wr.ok()) {
      return wr;
    }
  } else if (!detail::write_operand(ctx, 0, value, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }

  detail::set_flag(ctx.state.rflags, kFlagCF, old_bit);
  return {};
}
}

ExecutionResult handle_code_BTS_RM16_IMM8(ExecutionContext& ctx) {
  return bts_rmw(ctx, 2, read_bit_index(ctx.instr.immediate8(), 2));
}

ExecutionResult handle_code_BTS_RM16_R16(ExecutionContext& ctx) {
  return bts_rmw(ctx, 2, detail::read_register(ctx.state, ctx.instr.op_register(1)));
}

ExecutionResult handle_code_BTS_RM32_IMM8(ExecutionContext& ctx) {
  return bts_rmw(ctx, 4, read_bit_index(ctx.instr.immediate8(), 4));
}

ExecutionResult handle_code_BTS_RM32_R32(ExecutionContext& ctx) {
  return bts_rmw(ctx, 4, detail::read_register(ctx.state, ctx.instr.op_register(1)));
}

ExecutionResult handle_code_BTS_RM64_IMM8(ExecutionContext& ctx) {
  return bts_rmw(ctx, 8, read_bit_index(ctx.instr.immediate8(), 8));
}

ExecutionResult handle_code_BTS_RM64_R64(ExecutionContext& ctx) {
  return bts_rmw(ctx, 8, detail::read_register(ctx.state, ctx.instr.op_register(1)));
}

}  // namespace seven::handlers


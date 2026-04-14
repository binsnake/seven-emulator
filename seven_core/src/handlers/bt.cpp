#include "seven/handler_helpers.hpp"
#include <iced_x86/op_kind.hpp>

namespace seven::handlers {

namespace {
uint64_t read_bit_index_1(std::uint64_t value, std::size_t width) {
  return value & (8ull * width - 1ull);
}

ExecutionResult read_bt_base_value(ExecutionContext& ctx, std::size_t width, std::uint64_t bit_index, std::uint64_t& value_out,
                                   std::uint64_t& bit_out) {
  const auto bit_span = 8ull * width;
  if (ctx.instr.op0_kind() == iced_x86::OpKind::MEMORY) {
    const auto elem_index = bit_index / bit_span;
    const auto address = detail::memory_address(ctx) + elem_index * width;
    bit_out = bit_index & (bit_span - 1ull);
    return detail::read_memory_checked(ctx, address, &value_out, width);
  }

  bool ok = false;
  value_out = detail::read_operand(ctx, 0, width, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bit_out = bit_index & (bit_span - 1ull);
  return {};
}
}

ExecutionResult handle_code_BT_RM16_IMM8(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  const auto rr = read_bt_base_value(ctx, 2, read_bit_index_1(ctx.instr.immediate8(), 2), value, bit);
  if (!rr.ok()) {
    return rr;
  }
  const bool result = ((value >> bit) & 1ull) != 0;
  detail::set_flag(ctx.state.rflags, kFlagCF, result);
  return {};
}

ExecutionResult handle_code_BT_RM16_R16(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  const auto rr = read_bt_base_value(ctx, 2, detail::read_register(ctx.state, ctx.instr.op_register(1)), value, bit);
  if (!rr.ok()) {
    return rr;
  }
  const bool result = ((value >> bit) & 1ull) != 0;
  detail::set_flag(ctx.state.rflags, kFlagCF, result);
  return {};
}

ExecutionResult handle_code_BT_RM32_IMM8(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  const auto rr = read_bt_base_value(ctx, 4, read_bit_index_1(ctx.instr.immediate8(), 4), value, bit);
  if (!rr.ok()) {
    return rr;
  }
  const bool result = ((value >> bit) & 1ull) != 0;
  detail::set_flag(ctx.state.rflags, kFlagCF, result);
  return {};
}

ExecutionResult handle_code_BT_RM32_R32(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  const auto rr = read_bt_base_value(ctx, 4, detail::read_register(ctx.state, ctx.instr.op_register(1)), value, bit);
  if (!rr.ok()) {
    return rr;
  }
  const bool result = ((value >> bit) & 1ull) != 0;
  detail::set_flag(ctx.state.rflags, kFlagCF, result);
  return {};
}

ExecutionResult handle_code_BT_RM64_IMM8(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  const auto rr = read_bt_base_value(ctx, 8, read_bit_index_1(ctx.instr.immediate8(), 8), value, bit);
  if (!rr.ok()) {
    return rr;
  }
  const bool result = ((value >> bit) & 1ull) != 0;
  detail::set_flag(ctx.state.rflags, kFlagCF, result);
  return {};
}

ExecutionResult handle_code_BT_RM64_R64(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  std::uint64_t bit = 0;
  const auto rr = read_bt_base_value(ctx, 8, detail::read_register(ctx.state, ctx.instr.op_register(1)), value, bit);
  if (!rr.ok()) {
    return rr;
  }
  const bool result = ((value >> bit) & 1ull) != 0;
  detail::set_flag(ctx.state.rflags, kFlagCF, result);
  return {};
}

}  // namespace seven::handlers


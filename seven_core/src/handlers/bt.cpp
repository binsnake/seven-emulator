#include "seven/handler_helpers.hpp"
#include <bit>
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
    // bit_index is a signed two's-complement value (the caller sign-extends
    // register-sourced indices to the operand width; immediate-sourced ones
    // are already small and non-negative). Intel defines the effective
    // address as base + floor(bit_index / bit_span) elements -- an
    // arithmetic right shift by log2(bit_span) implements that floor
    // division exactly, including for negative indices. Unsigned truncating
    // division here previously sent register bit-indices to wildly
    // out-of-bounds addresses (seven-fuzzer finding).
    const auto shift = static_cast<unsigned>(std::countr_zero(bit_span));
    const auto elem_index = static_cast<std::int64_t>(bit_index) >> shift;
    // Unsigned throughout: the signed form overflows int64 for a base near the top of the address
    // space, which is undefined rather than the wraparound it looks like. Casting the (possibly
    // negative) element index to uint64 first gives the identical two's-complement result with
    // defined behaviour, and the range check downstream is what actually rejects a bad address.
    const auto address = detail::memory_address_with_displacement(
        ctx, static_cast<std::uint64_t>(elem_index) * static_cast<std::uint64_t>(width));
    bit_out = bit_index & (bit_span - 1ull);
    return detail::read_memory_checked(ctx, address, &value_out, width);
  }

  bool ok = false;
  value_out = detail::read_operand(ctx, 0, width, &ok);
  if (!ok) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
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
  const auto rr = read_bt_base_value(ctx, 2, detail::sign_extend(detail::read_register(ctx.state, ctx.instr.op_register(1)), 2), value, bit);
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
  const auto rr = read_bt_base_value(ctx, 4, detail::sign_extend(detail::read_register(ctx.state, ctx.instr.op_register(1)), 4), value, bit);
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


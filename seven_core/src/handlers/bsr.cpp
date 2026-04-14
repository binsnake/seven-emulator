#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_BSR_R16_RM16(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 2, value); !result.ok()) {
    return result;
  }
  if (value == 0) {
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  std::uint64_t index = 15;
  for (std::uint64_t bit = 15; bit > 0; --bit) {
    if ((value & (1ull << bit)) != 0) {
      index = bit;
      break;
    }
    if (bit == 1) {
      if ((value & 1ull) != 0) {
        index = 0;
      }
      break;
    }
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  detail::write_register(ctx.state, ctx.instr.op_register(0), index, 2);
  return {};
}

ExecutionResult handle_code_BSR_R32_RM32(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 4, value); !result.ok()) {
    return result;
  }
  if (value == 0) {
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  std::uint64_t index = 31;
  for (std::uint64_t bit = 31; bit > 0; --bit) {
    if ((value & (1ull << bit)) != 0) {
      index = bit;
      break;
    }
    if (bit == 1) {
      if ((value & 1ull) != 0) {
        index = 0;
      }
      break;
    }
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  detail::write_register(ctx.state, ctx.instr.op_register(0), index, 4);
  return {};
}

ExecutionResult handle_code_BSR_R64_RM64(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 8, value); !result.ok()) {
    return result;
  }
  if (value == 0) {
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  std::uint64_t index = 63;
  for (std::uint64_t bit = 63; bit > 0; --bit) {
    if ((value & (1ull << bit)) != 0) {
      index = bit;
      break;
    }
    if (bit == 1) {
      if ((value & 1ull) != 0) {
        index = 0;
      }
      break;
    }
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  detail::write_register(ctx.state, ctx.instr.op_register(0), index, 8);
  return {};
}

}  // namespace seven::handlers


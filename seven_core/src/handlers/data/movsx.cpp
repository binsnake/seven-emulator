#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_MOVSX_R16_RM8(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 1, value); !result.ok()) {
    return result;
  }
  value = detail::sign_extend(value, 1);
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 2);
  return {};
}

ExecutionResult handle_code_MOVSX_R32_RM8(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 1, value); !result.ok()) {
    return result;
  }
  value = detail::sign_extend(value, 1);
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOVSX_R64_RM8(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 1, value); !result.ok()) {
    return result;
  }
  value = detail::sign_extend(value, 1);
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 8);
  return {};
}

ExecutionResult handle_code_MOVSX_R16_RM16(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 2, value); !result.ok()) {
    return result;
  }
  value = detail::sign_extend(value, 2);
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 2);
  return {};
}

ExecutionResult handle_code_MOVSX_R32_RM16(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 2, value); !result.ok()) {
    return result;
  }
  value = detail::sign_extend(value, 2);
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOVSX_R64_RM16(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 1, 2, value); !result.ok()) {
    return result;
  }
  value = detail::sign_extend(value, 2);
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 8);
  return {};
}

}  // namespace seven::handlers


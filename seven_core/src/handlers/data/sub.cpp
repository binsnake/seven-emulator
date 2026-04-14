#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_SUB_RM8_R8(ExecutionContext& ctx) {
  bool dst_ok = false;
  bool src_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_operand(ctx, 1, 1, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 1);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_R8_RM8(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto rhs = detail::read_operand(ctx, 1, 1, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 1);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 1);

  return {};
}


ExecutionResult handle_code_SUB_RM8_IMM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const std::uint64_t rhs = ctx.instr.immediate8();
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 1);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_RM8_IMM8_82(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  std::uint64_t rhs = ctx.instr.immediate8();
  rhs = detail::sign_extend(rhs, 1);
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 1);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_AL_IMM8(ExecutionContext& ctx) {
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const std::uint64_t rhs = ctx.instr.immediate8();
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 1);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 1);

  return {};
}


ExecutionResult handle_code_SUB_AX_IMM16(ExecutionContext& ctx) {
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto rhs = ctx.instr.immediate16();
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 2);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 2);

  return {};
}


ExecutionResult handle_code_SUB_EAX_IMM32(ExecutionContext& ctx) {
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const std::uint64_t rhs = ctx.instr.immediate32();
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 4);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 4);

  return {};
}


ExecutionResult handle_code_SUB_RAX_IMM32(ExecutionContext& ctx) {
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  std::uint64_t rhs = ctx.instr.immediate32();
  rhs = detail::sign_extend(rhs, 4);
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 8);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 8);

  return {};
}


ExecutionResult handle_code_SUB_RM16_R16(ExecutionContext& ctx) {
  bool dst_ok = false;
  bool src_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 2);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto rhs = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 2);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 2);

  return {};
}


ExecutionResult handle_code_SUB_RM16_IMM16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = ctx.instr.immediate16();
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 2);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_RM16_IMM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  std::uint64_t rhs = ctx.instr.immediate8();
  rhs = detail::sign_extend(rhs, 1);
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 2);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_RM32_R32(ExecutionContext& ctx) {
  bool dst_ok = false;
  bool src_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 4);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto rhs = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 4);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 4);

  return {};
}


ExecutionResult handle_code_SUB_RM32_IMM32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const std::uint64_t rhs = ctx.instr.immediate32();
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 4);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_RM32_IMM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  std::uint64_t rhs = ctx.instr.immediate8();
  rhs = detail::sign_extend(rhs, 1);
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 4);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_RM64_R64(ExecutionContext& ctx) {
  bool dst_ok = false;
  bool src_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 8);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto lhs = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto rhs = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 8);
  detail::write_register(ctx.state, ctx.instr.op_register(0), result, 8);

  return {};
}


ExecutionResult handle_code_SUB_RM64_IMM32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  std::uint64_t rhs = ctx.instr.immediate32();
  rhs = detail::sign_extend(rhs, 4);
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 8);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SUB_RM64_IMM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  std::uint64_t rhs = ctx.instr.immediate8();
  rhs = detail::sign_extend(rhs, 1);
  const auto result = lhs - rhs;
  detail::set_sub_flags(ctx.state, lhs, rhs, result, 8);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers










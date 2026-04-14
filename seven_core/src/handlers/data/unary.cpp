#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_INC_RM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs + 1;
  detail::set_add_flags(ctx.state, lhs, 1ull, result, 1);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_INC_RM16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs + 1;
  detail::set_add_flags(ctx.state, lhs, 1ull, result, 2);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_INC_RM32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs + 1;
  detail::set_add_flags(ctx.state, lhs, 1ull, result, 4);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_INC_RM64(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs + 1;
  detail::set_add_flags(ctx.state, lhs, 1ull, result, 8);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_DEC_RM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs - 1;
  detail::set_sub_flags(ctx.state, lhs, 1ull, result, 1);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_DEC_RM16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs - 1;
  detail::set_sub_flags(ctx.state, lhs, 1ull, result, 2);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_DEC_RM32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs - 1;
  detail::set_sub_flags(ctx.state, lhs, 1ull, result, 4);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_DEC_RM64(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto old_cf = (ctx.state.rflags & kFlagCF) != 0ull;
  const auto result = lhs - 1;
  detail::set_sub_flags(ctx.state, lhs, 1ull, result, 8);
  detail::set_flag(ctx.state.rflags, kFlagCF, old_cf);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NEG_RM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(static_cast<std::uint8_t>(-lhs));
  detail::set_sub_flags(ctx.state, 0, lhs, result, 1);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NEG_RM16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(static_cast<std::uint16_t>(-lhs));
  detail::set_sub_flags(ctx.state, 0, lhs, result, 2);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NEG_RM32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(static_cast<std::uint32_t>(-lhs));
  detail::set_sub_flags(ctx.state, 0, lhs, result, 4);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NEG_RM64(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = 0ull - lhs;
  detail::set_sub_flags(ctx.state, 0, lhs, result, 8);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NOT_RM8(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(~lhs);
  if (!detail::write_operand(ctx, 0, result, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NOT_RM16(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(~lhs);
  if (!detail::write_operand(ctx, 0, result, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NOT_RM32(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(~lhs);
  if (!detail::write_operand(ctx, 0, result, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_NOT_RM64(ExecutionContext& ctx) {
  bool dst_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = static_cast<std::uint64_t>(~lhs);
  if (!detail::write_operand(ctx, 0, result, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers


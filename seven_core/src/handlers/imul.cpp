#include "seven/handler_helpers.hpp"

#include <limits>

namespace seven::handlers {

namespace {

template <typename OperandT, typename WideT>
bool imul_overflow(WideT product) {
  return product != static_cast<WideT>(static_cast<OperandT>(product));
}

template <typename OperandT, typename WideT>
ExecutionResult imul_reg_rm_imm(ExecutionContext& ctx, std::size_t width, std::size_t imm_width) {
  bool src_ok = false;
  const auto src = detail::sign_extend(detail::read_operand(ctx, 1, width, &src_ok), width);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto imm = detail::sign_extend(detail::immediate_value(ctx.instr, 2), imm_width);
  const auto product = static_cast<WideT>(static_cast<OperandT>(src)) * static_cast<WideT>(static_cast<OperandT>(imm));
  const auto truncated = static_cast<std::uint64_t>(static_cast<OperandT>(product));
  detail::write_register(ctx.state, ctx.instr.op_register(0), truncated, width);
  detail::set_multiply_flags(ctx.state, truncated, width, imul_overflow<OperandT>(product));
  return {};
}

}  // namespace


ExecutionResult handle_code_IMUL_R16_RM16(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto lhs = detail::sign_extend(detail::read_register(ctx.state, ctx.instr.op_register(0)), 2);
  const auto rhs = detail::sign_extend(detail::read_operand(ctx, 1, 2, &rhs_ok), 2);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto product = static_cast<std::int32_t>(static_cast<std::int16_t>(lhs)) *
                       static_cast<std::int32_t>(static_cast<std::int16_t>(rhs));
  const auto overflow = imul_overflow<std::int16_t>(product);
  detail::write_register(ctx.state, ctx.instr.op_register(0), static_cast<std::uint64_t>(static_cast<std::uint16_t>(product)), 2);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint16_t>(product), 2, overflow);
  return {};
}

ExecutionResult handle_code_IMUL_R16_RM16_IMM16(ExecutionContext& ctx) {
  return imul_reg_rm_imm<std::int16_t, std::int32_t>(ctx, 2, 2);
}

ExecutionResult handle_code_IMUL_R16_RM16_IMM8(ExecutionContext& ctx) {
  return imul_reg_rm_imm<std::int16_t, std::int32_t>(ctx, 2, 1);
}

ExecutionResult handle_code_IMUL_R32_RM32(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto lhs = detail::sign_extend(detail::read_register(ctx.state, ctx.instr.op_register(0)), 4);
  const auto rhs = detail::sign_extend(detail::read_operand(ctx, 1, 4, &rhs_ok), 4);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto product = static_cast<std::int64_t>(static_cast<std::int32_t>(lhs)) *
                       static_cast<std::int64_t>(static_cast<std::int32_t>(rhs));
  const auto overflow = imul_overflow<std::int32_t>(product);
  detail::write_register(ctx.state, ctx.instr.op_register(0), static_cast<std::uint64_t>(static_cast<std::uint32_t>(product)), 4);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint32_t>(product), 4, overflow);
  return {};
}

ExecutionResult handle_code_IMUL_R32_RM32_IMM32(ExecutionContext& ctx) {
  return imul_reg_rm_imm<std::int32_t, std::int64_t>(ctx, 4, 4);
}

ExecutionResult handle_code_IMUL_R32_RM32_IMM8(ExecutionContext& ctx) {
  return imul_reg_rm_imm<std::int32_t, std::int64_t>(ctx, 4, 1);
}

ExecutionResult handle_code_IMUL_R64_RM64(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto lhs = detail::sign_extend(detail::read_register(ctx.state, ctx.instr.op_register(0)), 8);
  const auto rhs = detail::sign_extend(detail::read_operand(ctx, 1, 8, &rhs_ok), 8);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto product = static_cast<boost::multiprecision::int128_t>(static_cast<std::int64_t>(lhs)) *
                       static_cast<boost::multiprecision::int128_t>(static_cast<std::int64_t>(rhs));
  const auto overflow = imul_overflow<std::int64_t>(product);
  detail::write_register(ctx.state, ctx.instr.op_register(0), static_cast<std::uint64_t>(static_cast<std::int64_t>(product)), 8);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint64_t>(static_cast<std::int64_t>(product)), 8, overflow);
  return {};
}

ExecutionResult handle_code_IMUL_R64_RM64_IMM32(ExecutionContext& ctx) {
  return imul_reg_rm_imm<std::int64_t, boost::multiprecision::int128_t>(ctx, 8, 4);
}

ExecutionResult handle_code_IMUL_R64_RM64_IMM8(ExecutionContext& ctx) {
  return imul_reg_rm_imm<std::int64_t, boost::multiprecision::int128_t>(ctx, 8, 1);
}

ExecutionResult handle_code_IMUL_RM8(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 1, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::AL);
  const auto product = static_cast<std::int16_t>(static_cast<std::int8_t>(lhs)) *
                       static_cast<std::int16_t>(static_cast<std::int8_t>(rhs));
  detail::write_register(ctx.state, iced_x86::Register::AX, static_cast<std::uint16_t>(product), 2);
  const bool overflow = imul_overflow<std::int8_t>(product);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint8_t>(product), 1, overflow);
  return {};
}

ExecutionResult handle_code_IMUL_RM16(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 2, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::AX);
  const auto product = static_cast<std::int32_t>(static_cast<std::int16_t>(lhs)) *
                       static_cast<std::int32_t>(static_cast<std::int16_t>(rhs));
  detail::write_register(ctx.state, iced_x86::Register::AX, static_cast<std::uint16_t>(product), 2);
  detail::write_register(ctx.state, iced_x86::Register::DX, static_cast<std::uint16_t>(product >> 16), 2);
  const auto overflow = imul_overflow<std::int16_t>(product);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint16_t>(product), 2, overflow);
  return {};
}

ExecutionResult handle_code_IMUL_RM32(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 4, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::EAX);
  const auto product = static_cast<std::int64_t>(static_cast<std::int32_t>(lhs)) *
                       static_cast<std::int64_t>(static_cast<std::int32_t>(rhs));
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(product), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(product >> 32), 4);
  const auto overflow = imul_overflow<std::int32_t>(product);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint32_t>(product), 4, overflow);
  return {};
}

ExecutionResult handle_code_IMUL_RM64(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 8, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::RAX);
  const auto product = static_cast<boost::multiprecision::int128_t>(static_cast<std::int64_t>(lhs)) *
                       static_cast<boost::multiprecision::int128_t>(static_cast<std::int64_t>(rhs));
  detail::write_register(ctx.state, iced_x86::Register::RAX,
                         static_cast<std::uint64_t>(static_cast<std::int64_t>(product)), 8);
  detail::write_register(ctx.state, iced_x86::Register::RDX,
                         static_cast<std::uint64_t>(static_cast<std::int64_t>(product >> 64u)), 8);
  const bool overflow = imul_overflow<std::int64_t>(product);
  detail::set_multiply_flags(ctx.state, static_cast<std::uint64_t>(static_cast<std::int64_t>(product)), 8, overflow);
  return {};
}

}  // namespace seven::handlers


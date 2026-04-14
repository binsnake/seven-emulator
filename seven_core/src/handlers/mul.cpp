#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_MUL_RM8(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 1, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::AL);
  const auto product = lhs * rhs;
  const auto high = (product >> 8) & 0xFFull;
  detail::write_register(ctx.state, iced_x86::Register::AX, product, 2);
  const auto overflow = high != 0;
  detail::set_multiply_flags(ctx.state, product, 1, overflow);
  return {};
}

ExecutionResult handle_code_MUL_RM16(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 2, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::AX);
  const auto product = lhs * rhs;
  detail::write_register(ctx.state, iced_x86::Register::AX, product, 2);
  detail::write_register(ctx.state, iced_x86::Register::DX, product >> 16, 2);
  const auto overflow = (product >> 16) != 0;
  detail::set_multiply_flags(ctx.state, static_cast<std::uint16_t>(product), 2, overflow);
  return {};
}

ExecutionResult handle_code_MUL_RM32(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 4, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::EAX);
  const std::uint64_t product = lhs * rhs;
  detail::write_register(ctx.state, iced_x86::Register::EAX, product, 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, product >> 32, 4);
  const auto overflow = (product >> 32) != 0;
  detail::set_multiply_flags(ctx.state, static_cast<std::uint32_t>(product), 4, overflow);
  return {};
}

ExecutionResult handle_code_MUL_RM64(ExecutionContext& ctx) {
  bool rhs_ok = false;
  const auto rhs = detail::read_operand(ctx, 0, 8, &rhs_ok);
  if (!rhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::RAX);
  const boost::multiprecision::uint128_t product =
      static_cast<boost::multiprecision::uint128_t>(lhs) * static_cast<boost::multiprecision::uint128_t>(rhs);
  const auto low = static_cast<std::uint64_t>(product);
  const auto high = static_cast<std::uint64_t>(product >> 64u);
  detail::write_register(ctx.state, iced_x86::Register::RAX, low, 8);
  detail::write_register(ctx.state, iced_x86::Register::RDX, high, 8);
  const auto overflow = high != 0;
  detail::set_multiply_flags(ctx.state, low, 8, overflow);
  return {};
}

}  // namespace seven::handlers



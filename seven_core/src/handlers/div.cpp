#include "seven/handler_helpers.hpp"

#include <limits>

namespace seven::handlers {

ExecutionResult handle_code_DIV_RM8(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 1, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto ax = detail::read_register(ctx.state, iced_x86::Register::AX);
  const auto dividend = static_cast<std::uint16_t>(ax);
  const auto quotient = dividend / divisor;
  const auto remainder = dividend % divisor;
  if (quotient > 0xFFull) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::AL, quotient, 1);
  detail::write_register(ctx.state, iced_x86::Register::AH, remainder, 1);
  return {};
}

ExecutionResult handle_code_DIV_RM16(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 2, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto low = detail::read_register(ctx.state, iced_x86::Register::AX);
  const auto high = detail::read_register(ctx.state, iced_x86::Register::DX);
  const auto dividend = (static_cast<std::uint32_t>(high) << 16) | static_cast<std::uint32_t>(low);
  const auto quotient = dividend / divisor;
  const auto remainder = dividend % divisor;
  if (quotient > 0xFFFFull) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::AX, quotient, 2);
  detail::write_register(ctx.state, iced_x86::Register::DX, remainder, 2);
  return {};
}

ExecutionResult handle_code_DIV_RM32(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 4, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto low = detail::read_register(ctx.state, iced_x86::Register::EAX);
  const auto high = detail::read_register(ctx.state, iced_x86::Register::EDX);
  const auto dividend = (static_cast<std::uint64_t>(high) << 32) | static_cast<std::uint64_t>(low);
  const auto quotient = dividend / divisor;
  const auto remainder = dividend % divisor;
  if (quotient > 0xFFFFFFFFull) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::EAX, quotient, 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, remainder, 4);
  return {};
}

ExecutionResult handle_code_DIV_RM64(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 8, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto low = detail::read_register(ctx.state, iced_x86::Register::RAX);
  const auto high = detail::read_register(ctx.state, iced_x86::Register::RDX);
  const math::wide_integer::uint128_t dividend =
      (static_cast<math::wide_integer::uint128_t>(high) << 64u) | static_cast<math::wide_integer::uint128_t>(low);
  const math::wide_integer::uint128_t quotient = dividend / static_cast<math::wide_integer::uint128_t>(divisor);
  const math::wide_integer::uint128_t remainder = dividend % static_cast<math::wide_integer::uint128_t>(divisor);
  if (quotient > std::numeric_limits<std::uint64_t>::max()) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::RAX, static_cast<std::uint64_t>(quotient), 8);
  detail::write_register(ctx.state, iced_x86::Register::RDX, static_cast<std::uint64_t>(remainder), 8);
  return {};
}

}  // namespace seven::handlers



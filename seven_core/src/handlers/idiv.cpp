#include "seven/handler_helpers.hpp"

#include <limits>

namespace seven::handlers {

ExecutionResult handle_code_IDIV_RM8(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 1, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto dividend = static_cast<std::int16_t>(detail::read_register(ctx.state, iced_x86::Register::AX));
  const auto rhs = static_cast<std::int8_t>(divisor);
  const auto quotient = dividend / rhs;
  const auto remainder = dividend % rhs;
  if (quotient < -128ll || quotient > 127ll) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::AL, static_cast<std::uint8_t>(static_cast<std::int8_t>(quotient)), 1);
  detail::write_register(ctx.state, iced_x86::Register::AH, static_cast<std::uint8_t>(static_cast<std::int8_t>(remainder)), 1);
  return {};
}

ExecutionResult handle_code_IDIV_RM16(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 2, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto low = detail::read_register(ctx.state, iced_x86::Register::AX);
  const auto high = detail::read_register(ctx.state, iced_x86::Register::DX);
  const auto dividend = (static_cast<std::int32_t>((static_cast<std::uint32_t>(high) << 16) | static_cast<std::uint32_t>(low)));
  const auto rhs = static_cast<std::int16_t>(divisor);
  if (rhs == -1 && dividend == std::numeric_limits<std::int32_t>::min()) {
    return detail::divide_fault(ctx);
  }
  const auto quotient = dividend / rhs;
  const auto remainder = dividend % rhs;
  if (quotient < -32768ll || quotient > 32767ll) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::AX, static_cast<std::uint16_t>(static_cast<std::int16_t>(quotient)), 2);
  detail::write_register(ctx.state, iced_x86::Register::DX, static_cast<std::uint16_t>(static_cast<std::int16_t>(remainder)), 2);
  return {};
}

ExecutionResult handle_code_IDIV_RM32(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 4, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto low = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::EAX));
  const auto high = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::EDX));
  const auto dividend = static_cast<std::int64_t>((static_cast<std::uint64_t>(high) << 32ull) | static_cast<std::uint64_t>(low));
  const auto rhs = static_cast<std::int32_t>(divisor);
  if (rhs == -1 && dividend == std::numeric_limits<std::int64_t>::min()) {
    return detail::divide_fault(ctx);
  }
  const auto quotient = dividend / rhs;
  const auto remainder = dividend % rhs;
  if (quotient < static_cast<std::int64_t>(-2147483648ll) || quotient > 2147483647ll) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(static_cast<std::int32_t>(quotient)), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(static_cast<std::int32_t>(remainder)), 4);
  return {};
}

ExecutionResult handle_code_IDIV_RM64(ExecutionContext& ctx) {
  std::uint64_t divisor = 0;
  if (const auto result = detail::read_divisor_checked(ctx, 8, divisor); result.reason != StopReason::none) {
    return result;
  }
  const auto low = detail::read_register(ctx.state, iced_x86::Register::RAX);
  const auto high = detail::read_register(ctx.state, iced_x86::Register::RDX);
  const auto dividend =
      (static_cast<boost::multiprecision::int128_t>(static_cast<std::int64_t>(high)) << 64u) |
      static_cast<boost::multiprecision::int128_t>(low);
  const auto rhs = static_cast<std::int64_t>(divisor);
  const auto quotient = dividend / static_cast<boost::multiprecision::int128_t>(rhs);
  const auto remainder = dividend % static_cast<boost::multiprecision::int128_t>(rhs);
  if (quotient < static_cast<boost::multiprecision::int128_t>(std::numeric_limits<std::int64_t>::min()) ||
      quotient > static_cast<boost::multiprecision::int128_t>(std::numeric_limits<std::int64_t>::max())) {
    return detail::divide_fault(ctx);
  }
  detail::write_register(ctx.state, iced_x86::Register::RAX, static_cast<std::uint64_t>(static_cast<std::int64_t>(quotient)), 8);
  detail::write_register(ctx.state, iced_x86::Register::RDX, static_cast<std::uint64_t>(static_cast<std::int64_t>(remainder)), 8);
  return {};
}

}  // namespace seven::handlers



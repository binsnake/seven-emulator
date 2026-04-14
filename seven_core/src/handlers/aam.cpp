#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_AAM_IMM8(ExecutionContext& ctx) {
  const auto base = ctx.instr.immediate8();
  if (base == 0) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  const auto al = detail::read_register(ctx.state, iced_x86::Register::AL);
  const auto quotient = static_cast<std::uint8_t>(al / base);
  const auto remainder = static_cast<std::uint8_t>(al % base);
  detail::write_register(ctx.state, iced_x86::Register::AL, remainder, 1);
  detail::write_register(ctx.state, iced_x86::Register::AH, quotient, 1);
  detail::set_flag(ctx.state.rflags, kFlagCF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagZF, remainder == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (remainder & 0x80ull) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(remainder));
  return {};
}

}  // namespace seven::handlers


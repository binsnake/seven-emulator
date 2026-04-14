#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_AAD_IMM8(ExecutionContext& ctx) {
  const auto base = ctx.instr.immediate8();
  if (base == 0) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  const auto al = detail::read_register(ctx.state, iced_x86::Register::AL);
  const auto ah = detail::read_register(ctx.state, iced_x86::Register::AH);
  const auto result = static_cast<std::uint8_t>(ah * base + al);
  detail::write_register(ctx.state, iced_x86::Register::AL, result, 1);
  detail::write_register(ctx.state, iced_x86::Register::AH, 0ull, 1);
  detail::set_flag(ctx.state.rflags, kFlagCF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagZF, result == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & 0x80ull) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(result));
  return {};
}

}  // namespace seven::handlers


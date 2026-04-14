#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_AAA(ExecutionContext& ctx) {
  auto al = detail::read_register(ctx.state, iced_x86::Register::AL);
  const bool af = (ctx.state.rflags & kFlagAF) != 0;
  const bool carry = af || ((al & 0x0Fu) > 9u);
  auto ah = detail::read_register(ctx.state, iced_x86::Register::AH);
  if (carry) {
    al += 6u;
    ah += 1u;
  }
  al &= 0x0Fu;
  detail::write_register(ctx.state, iced_x86::Register::AL, al, 1);
  detail::write_register(ctx.state, iced_x86::Register::AH, ah, 1);
  detail::set_flag(ctx.state.rflags, kFlagAF, carry);
  detail::set_flag(ctx.state.rflags, kFlagCF, carry);
  detail::set_flag(ctx.state.rflags, kFlagZF, (al & 0xFFull) == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (al & 0x80ull) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(static_cast<std::uint8_t>(al & 0xFFull)));
  return {};
}

}  // namespace seven::handlers



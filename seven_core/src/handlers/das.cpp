#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_DAS(ExecutionContext& ctx) {
  const auto old_al = detail::read_register(ctx.state, iced_x86::Register::AL);
  const bool old_cf = (ctx.state.rflags & kFlagCF) != 0;
  const bool old_af = (ctx.state.rflags & kFlagAF) != 0;
  auto al = old_al;
  const bool adjust = old_af || ((al & 0x0F) > 9u);
  // Unlike DAA, the second half of DAS has no else branch: a borrow out of AL - 6 sets CF and
  // nothing later clears it, so CF survives even when the high-nibble correction doesn't run.
  bool cf = old_cf;
  if (adjust) {
    cf = cf || (old_al < 6u);
    al -= 6u;
  }
  if (old_cf || (old_al > 0x99u)) {
    al -= 0x60u;
    cf = true;
  }
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  detail::set_flag(ctx.state.rflags, kFlagAF, adjust);
  detail::write_register(ctx.state, iced_x86::Register::AL, al, 1);
  detail::set_flag(ctx.state.rflags, kFlagZF, (al & 0xFFull) == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (al & 0x80ull) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(static_cast<std::uint8_t>(al & 0xFFull)));
  return {};
}

}  // namespace seven::handlers



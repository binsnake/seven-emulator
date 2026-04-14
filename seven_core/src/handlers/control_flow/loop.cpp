#include "control_flow_internal.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] constexpr std::uint64_t mask_for_width(std::size_t width) noexcept {
  return width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
}

[[nodiscard]] bool decrement_counter(ExecutionContext& ctx, iced_x86::Register reg, std::size_t width) {
  const auto mask = mask_for_width(width);
  const auto value = detail::read_register(ctx.state, reg) & mask;
  const auto next = (value - 1ull) & mask;
  detail::write_register(ctx.state, reg, next, width);
  return next != 0;
}

ExecutionResult loop_common(ExecutionContext& ctx, iced_x86::Register reg, std::size_t width, bool require_zf, bool require_nzf) {
  const auto non_zero = decrement_counter(ctx, reg, width);
  if (!non_zero) {
    return {};
  }
  const auto zf = (ctx.state.rflags & kFlagZF) != 0;
  if (require_zf && !zf) {
    return {};
  }
  if (require_nzf && zf) {
    return {};
  }
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult jcxz_common(ExecutionContext& ctx, iced_x86::Register reg, std::size_t width) {
  const auto value = detail::read_register(ctx.state, reg) & mask_for_width(width);
  if (value == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_LOOP_REL8_16_CX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::CX, 2, false, false); }
ExecutionResult handle_code_LOOP_REL8_32_CX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::CX, 2, false, false); }
ExecutionResult handle_code_LOOP_REL8_16_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, false, false); }
ExecutionResult handle_code_LOOP_REL8_32_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, false, false); }
ExecutionResult handle_code_LOOP_REL8_64_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, false, false); }
ExecutionResult handle_code_LOOP_REL8_16_RCX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::RCX, 8, false, false); }
ExecutionResult handle_code_LOOP_REL8_64_RCX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::RCX, 8, false, false); }

ExecutionResult handle_code_LOOPE_REL8_16_CX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::CX, 2, true, false); }
ExecutionResult handle_code_LOOPE_REL8_32_CX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::CX, 2, true, false); }
ExecutionResult handle_code_LOOPE_REL8_16_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, true, false); }
ExecutionResult handle_code_LOOPE_REL8_32_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, true, false); }
ExecutionResult handle_code_LOOPE_REL8_64_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, true, false); }
ExecutionResult handle_code_LOOPE_REL8_16_RCX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::RCX, 8, true, false); }
ExecutionResult handle_code_LOOPE_REL8_64_RCX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::RCX, 8, true, false); }

ExecutionResult handle_code_LOOPNE_REL8_16_CX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::CX, 2, false, true); }
ExecutionResult handle_code_LOOPNE_REL8_32_CX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::CX, 2, false, true); }
ExecutionResult handle_code_LOOPNE_REL8_16_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, false, true); }
ExecutionResult handle_code_LOOPNE_REL8_32_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, false, true); }
ExecutionResult handle_code_LOOPNE_REL8_64_ECX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::ECX, 4, false, true); }
ExecutionResult handle_code_LOOPNE_REL8_16_RCX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::RCX, 8, false, true); }
ExecutionResult handle_code_LOOPNE_REL8_64_RCX(ExecutionContext& ctx) { return loop_common(ctx, iced_x86::Register::RCX, 8, false, true); }

ExecutionResult handle_code_JECXZ_REL8_16(ExecutionContext& ctx) { return jcxz_common(ctx, iced_x86::Register::ECX, 4); }
ExecutionResult handle_code_JECXZ_REL8_32(ExecutionContext& ctx) { return jcxz_common(ctx, iced_x86::Register::ECX, 4); }
ExecutionResult handle_code_JECXZ_REL8_64(ExecutionContext& ctx) { return jcxz_common(ctx, iced_x86::Register::ECX, 4); }

ExecutionResult handle_code_JRCXZ_REL8_16(ExecutionContext& ctx) { return jcxz_common(ctx, iced_x86::Register::RCX, 8); }
ExecutionResult handle_code_JRCXZ_REL8_64(ExecutionContext& ctx) { return jcxz_common(ctx, iced_x86::Register::RCX, 8); }

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

ExecutionResult movs_impl(ExecutionContext& ctx, const std::size_t width) {
  const bool rep = ctx.instr.has_rep_prefix() || ctx.instr.has_repne_prefix();
  std::uint64_t count = rep ? ctx.state.gpr[1] : 1u;  // RCX
  if (count == 0) {
    return {};
  }

  const bool df = (ctx.state.rflags & kFlagDF) != 0;
  std::uint64_t rsi = ctx.state.gpr[6];
  std::uint64_t rdi = ctx.state.gpr[7];

  for (std::uint64_t i = 0; i < count; ++i) {
    const auto read_addr = rsi;
    const auto write_addr = rdi;
    std::uint64_t value = 0;
    if (!ctx.memory.read(read_addr, &value, width)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) {
        ctx.state.gpr[1] = count - i;
      }
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, read_addr, 0}, ctx.instr.code()};
    }
    value = detail::truncate(value, width);
    if (!ctx.memory.write(write_addr, &value, width)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) {
        ctx.state.gpr[1] = count - i;
      }
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, write_addr, 0}, ctx.instr.code()};
    }

    const auto hit_bits = detail::debug_data_breakpoint_hits(ctx.state, read_addr, width, true, false) |
                          detail::debug_data_breakpoint_hits(ctx.state, write_addr, width, false, true);

    if (df) {
      rsi -= width;
      rdi -= width;
    } else {
      rsi += width;
      rdi += width;
    }

    const auto remaining = count - i - 1;
    if (detail::note_debug_break(ctx, hit_bits, rep && remaining > 0)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) {
        ctx.state.gpr[1] = remaining;
      }
      return {};
    }
  }

  ctx.state.gpr[6] = rsi;
  ctx.state.gpr[7] = rdi;
  if (rep) {
    ctx.state.gpr[1] = 0;
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_MOVSB_M8_M8(ExecutionContext& ctx) {
  return movs_impl(ctx, 1);
}

ExecutionResult handle_code_MOVSW_M16_M16(ExecutionContext& ctx) {
  return movs_impl(ctx, 2);
}

ExecutionResult handle_code_MOVSD_M32_M32(ExecutionContext& ctx) {
  return movs_impl(ctx, 4);
}

ExecutionResult handle_code_MOVSQ_M64_M64(ExecutionContext& ctx) {
  return movs_impl(ctx, 8);
}

}  // namespace seven::handlers


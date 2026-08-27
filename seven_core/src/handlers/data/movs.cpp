#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

// A rep-prefixed string instruction's whole count executes inside one C++ loop, in one call to
// this handler, with no other opportunity for the caller (Executor::run(), a watchdog thread's
// Executor::request_stop(), a JitExecutor budget check) to interject -- ExecutionContext carries
// no reference back to the Executor at all, so a handler genuinely cannot poll for a cooperative
// stop request mid-loop. A guest setting RCX to a huge value before a single `rep movsb`-family
// instruction, with enough mapped memory to sustain it, can make step_impl's underlying call hang
// for as long as that span takes to walk -- with zero chance for any cooperative-cancellation
// mechanism elsewhere in this project (the exact contract JitExecutor::run() and seven-fuzzer's
// watchdog thread already depend on) to fire. Real hardware avoids this by checking for pending
// interrupts BETWEEN each iteration of a rep-prefixed instruction specifically so a long-running
// one stays preemptible -- capping iterations per call and yielding back to the caller (leaving
// rip pointing at this same instruction, so it re-enters and continues exactly where it left off)
// reproduces that same interruptibility without any guest-visible effect: RCX/RSI/RDI already
// reflect real partial progress, and nothing about the guest's own execution state changes.
constexpr std::uint64_t kMaxRepIterationsPerCall = 4096;

// A single-stepping guest sees a #DB after every iteration of a rep, not one after the whole loop.
// Dropping the cap to one while TF is set reuses the same yield as the budget above, and the
// executor already delivers TF at the end of whatever step() did.
[[nodiscard]] std::uint64_t rep_iterations_per_call(const ExecutionContext& ctx) noexcept {
  return (ctx.state.rflags & kFlagTF) != 0 ? 1u : kMaxRepIterationsPerCall;
}

ExecutionResult movs_impl(ExecutionContext& ctx, const std::size_t width) {
  const bool rep = ctx.instr.has_rep_prefix() || ctx.instr.has_repne_prefix();
  const auto addr_mask = detail::string_address_mask(ctx.instr);
  std::uint64_t count = rep ? (ctx.state.gpr[1] & addr_mask) : 1u;  // RCX
  if (count == 0) {
    return {};
  }

  const bool df = (ctx.state.rflags & kFlagDF) != 0;
  std::uint64_t rsi = ctx.state.gpr[6] & addr_mask;
  std::uint64_t rdi = ctx.state.gpr[7] & addr_mask;

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
      return detail::memory_fault(ctx, read_addr);
    }
    value = detail::truncate(value, width);
    if (!ctx.memory.write(write_addr, &value, width)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) {
        ctx.state.gpr[1] = count - i;
      }
      return detail::memory_fault(ctx, write_addr);
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
    rsi &= addr_mask;
    rdi &= addr_mask;

    const auto remaining = count - i - 1;
    if (detail::note_debug_break(ctx, hit_bits, rep && remaining > 0)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) {
        ctx.state.gpr[1] = remaining;
      }
      return {};
    }

    // See kMaxRepIterationsPerCall's comment above: yield back to the caller after a bounded
    // number of iterations rather than let a huge RCX run this whole loop uninterruptibly. rip
    // stays at this same instruction (control_flow_taken=true, state.rip left untouched) so the
    // next step() call simply continues the same rep from exactly where it left off.
    if (rep && remaining > 0 && (i + 1) >= rep_iterations_per_call(ctx)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      ctx.state.gpr[1] = remaining;
      ctx.push_rf_for_debug = true;
      ctx.control_flow_taken = true;
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


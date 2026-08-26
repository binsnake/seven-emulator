#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

[[nodiscard]] ExecutionResult gp_fault(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
}

}  // namespace

// CLTS is CPL0-only on real hardware (#GP(0) otherwise) -- it was unconditionally clearing
// CR0.TS regardless of privilege level.
ExecutionResult handle_code_CLTS(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  constexpr std::uint64_t kTaskSwitched = 1ull << 3;
  ctx.state.cr[0] &= ~kTaskSwitched;
  return {};
}

}  // namespace seven::handlers



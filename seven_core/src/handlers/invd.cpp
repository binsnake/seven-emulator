#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

}  // namespace

// The cache it would invalidate is not modelled, so the body stays empty, but the privilege check
// is still observable: at CPL 3 this faults instead of quietly succeeding.
ExecutionResult handle_code_INVD(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return {StopReason::general_protection, 0,
            ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers

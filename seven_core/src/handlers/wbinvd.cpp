#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

}  // namespace

// Same as INVD: nothing to write back, but the CPL0 requirement is real and a ring-3 guest must
// see #GP rather than silent success.
ExecutionResult handle_code_WBINVD(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return {StopReason::general_protection, 0,
            ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers

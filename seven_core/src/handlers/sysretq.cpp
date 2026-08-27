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

// SYSRET is the kernel's return path to user code and is CPL0-only (#GP(0) otherwise). Without the
// check, ring 3 could execute it directly -- and since rflags came wholesale out of R11, pick its
// own IOPL and undo the CPL<=IOPL gate on CLI/STI in two instructions.
ExecutionResult handle_code_SYSRETQ(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto restore_rip = detail::read_register(ctx.state, iced_x86::Register::RCX);
  const auto restore_flags = detail::read_register(ctx.state, iced_x86::Register::R11);
  ctx.state.mode = ExecutionMode::long64;
  ctx.state.rip = restore_rip;
  // RFLAGS <- (R11 & 3C7FD7h) | 2, which is what the hardware loads: reserved bits, VM and RF
  // never come from R11.
  ctx.state.rflags = (restore_flags & 0x3C7FD7ull) | 0x2ull;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers

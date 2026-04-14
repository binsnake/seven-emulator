#include "control_flow_internal.hpp"

namespace seven::handlers {

ExecutionResult handle_code_JGE_REL8_64(ExecutionContext& ctx) {
  if (((ctx.state.rflags & kFlagSF) != 0) == ((ctx.state.rflags & kFlagOF) != 0)) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JGE_REL32_64(ExecutionContext& ctx) {
  if (((ctx.state.rflags & kFlagSF) != 0) == ((ctx.state.rflags & kFlagOF) != 0)) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JGE_REL8_16(ExecutionContext& ctx) {
  if (((ctx.state.rflags & kFlagSF) != 0) == ((ctx.state.rflags & kFlagOF) != 0)) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JGE_REL8_32(ExecutionContext& ctx) {
  if (((ctx.state.rflags & kFlagSF) != 0) == ((ctx.state.rflags & kFlagOF) != 0)) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JGE_REL16(ExecutionContext& ctx) {
  if (((ctx.state.rflags & kFlagSF) != 0) == ((ctx.state.rflags & kFlagOF) != 0)) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JGE_REL32_32(ExecutionContext& ctx) {
  if (((ctx.state.rflags & kFlagSF) != 0) == ((ctx.state.rflags & kFlagOF) != 0)) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

}  // namespace seven::handlers




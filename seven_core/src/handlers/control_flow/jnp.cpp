#include "control_flow_internal.hpp"

namespace seven::handlers {

ExecutionResult handle_code_JNP_REL8_64(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagPF) == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JNP_REL32_64(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagPF) == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JNP_REL8_16(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagPF) == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JNP_REL8_32(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagPF) == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JNP_REL16(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagPF) == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

ExecutionResult handle_code_JNP_REL32_32(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagPF) == 0) {
    ctx.state.rip = ctx.instr.near_branch64();
    ctx.control_flow_taken = true;
  }
  return {};
}

}  // namespace seven::handlers



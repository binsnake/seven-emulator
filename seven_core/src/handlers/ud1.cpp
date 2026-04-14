#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_UD1_R16_RM16(ExecutionContext& ctx) {
  return {StopReason::invalid_opcode, 0,
          ExceptionInfo{StopReason::invalid_opcode, ctx.state.rip, 0},
          ctx.instr.code()};
}

ExecutionResult handle_code_UD1_R32_RM32(ExecutionContext& ctx) {
  return {StopReason::invalid_opcode, 0,
          ExceptionInfo{StopReason::invalid_opcode, ctx.state.rip, 0},
          ctx.instr.code()};
}

ExecutionResult handle_code_UD1_R64_RM64(ExecutionContext& ctx) {
  return {StopReason::invalid_opcode, 0,
          ExceptionInfo{StopReason::invalid_opcode, ctx.state.rip, 0},
          ctx.instr.code()};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_UD2(ExecutionContext& ctx) {
  return {StopReason::invalid_opcode, 0,
          ExceptionInfo{StopReason::invalid_opcode, ctx.state.rip, 0},
          ctx.instr.code()};
}

}  // namespace seven::handlers


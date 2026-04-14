#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_RSM(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0,
          ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0},
          ctx.instr.code()};
}

}  // namespace seven::handlers


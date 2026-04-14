#include "control_flow_internal.hpp"

namespace seven::handlers {

ExecutionResult handle_code_POP_R16(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 2);
}

ExecutionResult handle_code_POP_R32(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 4);
}

ExecutionResult handle_code_POP_RM16(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 2);
}

ExecutionResult handle_code_POP_RM32(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 4);
}
ExecutionResult handle_code_POPW_SS(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 2);
}


ExecutionResult handle_code_POP_R64(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 8);
}

ExecutionResult handle_code_POP_RM64(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 8);
}
ExecutionResult handle_code_POPD_SS(ExecutionContext& ctx) {
  return pop_operand_width(ctx, 4);
}


}  // namespace seven::handlers



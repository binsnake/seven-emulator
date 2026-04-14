#include "control_flow_internal.hpp"

namespace seven::handlers {

ExecutionResult handle_code_PUSH_R64(ExecutionContext& ctx) {
  return push_operand_width(ctx, 8);
}

ExecutionResult handle_code_PUSH_RM64(ExecutionContext& ctx) {
  return push_operand_width(ctx, 8);
}

ExecutionResult handle_code_PUSH_IMM8(ExecutionContext& ctx) {
  return push_imm_width(ctx, 1, 8, true);
}

ExecutionResult handle_code_PUSH_IMM32(ExecutionContext& ctx) {
  return push_imm_width(ctx, 4, 8, true);
}

ExecutionResult handle_code_PUSHQ_IMM8(ExecutionContext& ctx) {
  return push_imm_width(ctx, 1, 8, true);
}

ExecutionResult handle_code_PUSHQ_IMM32(ExecutionContext& ctx) {
  return push_imm_width(ctx, 4, 8, true);
}

ExecutionResult handle_code_PUSH_R16(ExecutionContext& ctx) {
  return push_operand_width(ctx, 2);
}

ExecutionResult handle_code_PUSH_R32(ExecutionContext& ctx) {
  return push_operand_width(ctx, 4);
}

ExecutionResult handle_code_PUSH_RM16(ExecutionContext& ctx) {
  return push_operand_width(ctx, 2);
}

ExecutionResult handle_code_PUSH_RM32(ExecutionContext& ctx) {
  return push_operand_width(ctx, 4);
}

ExecutionResult handle_code_PUSH_IMM16(ExecutionContext& ctx) {
  return push_imm_width(ctx, 2, 2, false);
}

ExecutionResult handle_code_PUSHW_IMM8(ExecutionContext& ctx) {
  return push_imm_width(ctx, 1, 2, true);
}

ExecutionResult handle_code_PUSHD_IMM8(ExecutionContext& ctx) {
  return push_imm_width(ctx, 1, 4, true);
}

ExecutionResult handle_code_PUSHD_IMM32(ExecutionContext& ctx) {
  return push_imm_width(ctx, 4, 4, true);
}

ExecutionResult handle_code_PUSHW_ES(ExecutionContext& ctx) {
  return push_segment(ctx, 0, 2);
}

ExecutionResult handle_code_PUSHW_CS(ExecutionContext& ctx) {
  return push_segment(ctx, 1, 2);
}

ExecutionResult handle_code_PUSHW_SS(ExecutionContext& ctx) {
  return push_segment(ctx, 2, 2);
}

ExecutionResult handle_code_PUSHW_DS(ExecutionContext& ctx) {
  return push_segment(ctx, 3, 2);
}

ExecutionResult handle_code_PUSHW_FS(ExecutionContext& ctx) {
  return push_segment(ctx, 4, 2);
}

ExecutionResult handle_code_PUSHW_GS(ExecutionContext& ctx) {
  return push_segment(ctx, 5, 2);
}

ExecutionResult handle_code_PUSHD_ES(ExecutionContext& ctx) {
  return push_segment(ctx, 0, 4);
}

ExecutionResult handle_code_PUSHD_CS(ExecutionContext& ctx) {
  return push_segment(ctx, 1, 4);
}

ExecutionResult handle_code_PUSHD_SS(ExecutionContext& ctx) {
  return push_segment(ctx, 2, 4);
}

ExecutionResult handle_code_PUSHD_DS(ExecutionContext& ctx) {
  return push_segment(ctx, 3, 4);
}

ExecutionResult handle_code_PUSHD_FS(ExecutionContext& ctx) {
  return push_segment(ctx, 4, 4);
}

ExecutionResult handle_code_PUSHD_GS(ExecutionContext& ctx) {
  return push_segment(ctx, 5, 4);
}

ExecutionResult handle_code_PUSHQ_FS(ExecutionContext& ctx) {
  return push_segment(ctx, 4, 8);
}

ExecutionResult handle_code_PUSHQ_GS(ExecutionContext& ctx) {
  return push_segment(ctx, 5, 8);
}

ExecutionResult handle_code_PUSHFW(ExecutionContext& ctx) {
  return push_flags_width(ctx, 2);
}

ExecutionResult handle_code_PUSHFD(ExecutionContext& ctx) {
  return push_flags_width(ctx, 4);
}

ExecutionResult handle_code_PUSHAW(ExecutionContext& ctx) {
  return push_all_width(ctx, 2);
}

ExecutionResult handle_code_PUSHAD(ExecutionContext& ctx) {
  return push_all_width(ctx, 4);
}

}  // namespace seven::handlers



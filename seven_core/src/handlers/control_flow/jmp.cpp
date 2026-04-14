#include "control_flow_internal.hpp"

namespace seven::handlers {

namespace {

ExecutionResult read_far_mem(ExecutionContext& ctx, std::size_t offset_width, std::uint64_t& offset, std::uint16_t& selector) {
  if (auto result = detail::read_operand_checked(ctx, 0, offset_width, offset); !result.ok()) {
    return result;
  }
  const auto address = detail::memory_address(ctx) + offset_width;
  if (auto result = detail::read_memory_checked(ctx, address, selector); !result.ok()) {
    return result;
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_JMP_RM64(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  if (auto result = read_near_target_width(ctx, 8, target); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_RM16(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  if (auto result = detail::read_operand_checked(ctx, 0, 2, target); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_RM32(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  // In long mode, near absolute indirect JMP is architecturally 64-bit.
  // Keep upper address bits even if decode reports an RM32 form.
  const auto width = (ctx.state.mode == ExecutionMode::long64) ? 8u : 4u;
  if (auto result = read_near_target_width(ctx, width, target); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_REL8_64(ExecutionContext& ctx) {
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_REL32_64(ExecutionContext& ctx) {
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_REL8_16(ExecutionContext& ctx) {
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_REL8_32(ExecutionContext& ctx) {
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_REL16(ExecutionContext& ctx) {
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_REL32_32(ExecutionContext& ctx) {
  ctx.state.rip = ctx.instr.near_branch64();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_M1616(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  std::uint16_t selector = 0;
  if (auto result = read_far_mem(ctx, 2, target, selector); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::real16;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_M1632(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  std::uint16_t selector = 0;
  if (auto result = read_far_mem(ctx, 4, target, selector); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::compat32;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_M1664(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  std::uint16_t selector = 0;
  if (auto result = read_far_mem(ctx, 8, target, selector); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::long64;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_PTR1616(ExecutionContext& ctx) {
  ctx.state.mode = ExecutionMode::real16;
  ctx.state.sreg[1] = ctx.instr.far_branch_selector();
  ctx.state.rip = ctx.instr.far_branch16();
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_JMP_PTR1632(ExecutionContext& ctx) {
  ctx.state.mode = ExecutionMode::compat32;
  ctx.state.sreg[1] = ctx.instr.far_branch_selector();
  ctx.state.rip = ctx.instr.far_branch32();
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


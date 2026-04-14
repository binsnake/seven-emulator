#include "control_flow_internal.hpp"

namespace seven::handlers {

namespace {

ExecutionResult push_far_return(ExecutionContext& ctx, std::uint16_t selector, std::uint64_t offset, std::size_t offset_width) {
  if (auto result = push_width(ctx, offset, offset_width); !result.ok()) {
    return result;
  }
  if (auto result = push_width(ctx, selector, 2); !result.ok()) {
    return result;
  }
  return {};
}

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

ExecutionResult handle_code_CALL_RM64(ExecutionContext& ctx) {
  return call_rm_width(ctx, 8);
}

ExecutionResult handle_code_CALL_REL32_64(ExecutionContext& ctx) {
  const auto target = ctx.instr.near_branch64();
  if (auto result = push_width(ctx, ctx.next_rip, 8); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_RM16(ExecutionContext& ctx) {
  return call_rm_width(ctx, 2);
}

ExecutionResult handle_code_CALL_RM32(ExecutionContext& ctx) {
  // In long mode, near absolute indirect CALL is architecturally 64-bit.
  // Some decode paths can still surface RM32 here; preserve 64-bit target semantics.
  if (ctx.state.mode == ExecutionMode::long64) {
    return call_rm_width(ctx, 8);
  }
  return call_rm_width(ctx, 4);
}

ExecutionResult handle_code_CALL_REL16(ExecutionContext& ctx) {
  const auto target = ctx.instr.near_branch64();
  if (auto result = push_width(ctx, ctx.next_rip, 2); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_REL32_32(ExecutionContext& ctx) {
  const auto target = ctx.instr.near_branch64();
  if (auto result = push_width(ctx, ctx.next_rip, 4); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_M1616(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  std::uint16_t selector = 0;
  if (auto result = read_far_mem(ctx, 2, target, selector); result.reason != StopReason::none) {
    return result;
  }
  if (auto result = push_far_return(ctx, selector, ctx.next_rip, 2); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::real16;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_M1632(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  std::uint16_t selector = 0;
  if (auto result = read_far_mem(ctx, 4, target, selector); result.reason != StopReason::none) {
    return result;
  }
  if (auto result = push_far_return(ctx, selector, ctx.next_rip, 4); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::compat32;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_M1664(ExecutionContext& ctx) {
  std::uint64_t target = 0;
  std::uint16_t selector = 0;
  if (auto result = read_far_mem(ctx, 8, target, selector); result.reason != StopReason::none) {
    return result;
  }
  if (auto result = push_far_return(ctx, selector, ctx.next_rip, 8); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::long64;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_PTR1616(ExecutionContext& ctx) {
  const std::uint16_t selector = ctx.instr.far_branch_selector();
  const std::uint16_t target = ctx.instr.far_branch16();
  if (auto result = push_far_return(ctx, selector, ctx.next_rip, 2); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::real16;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult handle_code_CALL_PTR1632(ExecutionContext& ctx) {
  const std::uint16_t selector = ctx.instr.far_branch_selector();
  const std::uint64_t target = ctx.instr.far_branch32();
  if (auto result = push_far_return(ctx, selector, ctx.next_rip, 4); result.reason != StopReason::none) {
    return result;
  }
  ctx.state.mode = ExecutionMode::compat32;
  ctx.state.sreg[1] = selector;
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace seven::handlers


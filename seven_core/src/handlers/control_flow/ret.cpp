#include "control_flow_internal.hpp"

namespace seven::handlers {
namespace {

ExecutionMode mode_for_far_width(std::size_t width) {
  switch (width) {
    case 2:
      return ExecutionMode::real16;
    case 4:
      return ExecutionMode::compat32;
    case 8:
    default:
      return ExecutionMode::long64;
  }
}

ExecutionResult retf_width(ExecutionContext& ctx, std::size_t offset_width, std::uint16_t imm16) {
  std::uint64_t target = 0;
  if (auto result = pop_width(ctx, target, offset_width); !result.ok()) {
    return result;
  }
  std::uint64_t selector = 0;
  if (auto result = pop_width(ctx, selector, 2); !result.ok()) {
    return result;
  }
  ctx.state.mode = mode_for_far_width(offset_width);
  ctx.state.sreg[1] = static_cast<std::uint16_t>(selector);
  ctx.state.rip = mask_instruction_pointer(ctx.state, target);
  ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] + imm16);
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult iret_width(ExecutionContext& ctx, std::size_t offset_width, std::size_t flags_width) {
  std::uint64_t target = 0;
  if (auto result = pop_width(ctx, target, offset_width); !result.ok()) {
    return result;
  }
  std::uint64_t selector = 0;
  if (auto result = pop_width(ctx, selector, 2); !result.ok()) {
    return result;
  }
  std::uint64_t flags = 0;
  if (auto result = pop_width(ctx, flags, flags_width); !result.ok()) {
    return result;
  }

  ctx.state.mode = mode_for_far_width(offset_width);
  ctx.state.sreg[1] = static_cast<std::uint16_t>(selector);
  ctx.state.rip = mask_instruction_pointer(ctx.state, target);
  if (flags_width == 8) {
    ctx.state.rflags = flags;
  } else if (flags_width == 4) {
    ctx.state.rflags = (ctx.state.rflags & ~0xFFFFFFFFull) | (flags & 0xFFFFFFFFull);
  } else {
    ctx.state.rflags = (ctx.state.rflags & ~0xFFFFull) | (flags & 0xFFFFull);
  }
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace

ExecutionResult handle_code_RETNQ(ExecutionContext& ctx) {
  return ret_width(ctx, 8, 0);
}

ExecutionResult handle_code_RETNQ_IMM16(ExecutionContext& ctx) {
  return ret_width(ctx, 8, ctx.instr.immediate16());
}

ExecutionResult handle_code_RETNW(ExecutionContext& ctx) {
  return ret_width(ctx, 2, 0);
}

ExecutionResult handle_code_RETNW_IMM16(ExecutionContext& ctx) {
  return ret_width(ctx, 2, ctx.instr.immediate16());
}

ExecutionResult handle_code_RETND(ExecutionContext& ctx) {
  return ret_width(ctx, 4, 0);
}

ExecutionResult handle_code_RETND_IMM16(ExecutionContext& ctx) {
  return ret_width(ctx, 4, ctx.instr.immediate16());
}

ExecutionResult handle_code_RETFW(ExecutionContext& ctx) {
  return retf_width(ctx, 2, 0);
}

ExecutionResult handle_code_RETFD(ExecutionContext& ctx) {
  return retf_width(ctx, 4, 0);
}

ExecutionResult handle_code_RETFQ(ExecutionContext& ctx) {
  return retf_width(ctx, 8, 0);
}

ExecutionResult handle_code_RETFW_IMM16(ExecutionContext& ctx) {
  return retf_width(ctx, 2, ctx.instr.immediate16());
}

ExecutionResult handle_code_RETFD_IMM16(ExecutionContext& ctx) {
  return retf_width(ctx, 4, ctx.instr.immediate16());
}

ExecutionResult handle_code_RETFQ_IMM16(ExecutionContext& ctx) {
  return retf_width(ctx, 8, ctx.instr.immediate16());
}

ExecutionResult handle_code_IRETW(ExecutionContext& ctx) {
  return iret_width(ctx, 2, 2);
}

ExecutionResult handle_code_IRETD(ExecutionContext& ctx) {
  return iret_width(ctx, 4, 4);
}

ExecutionResult handle_code_IRETQ(ExecutionContext& ctx) {
  return iret_width(ctx, 8, 8);
}

}  // namespace seven::handlers



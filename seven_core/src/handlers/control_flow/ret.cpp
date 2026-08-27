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
  const auto entry_sp = ctx.state.gpr[4];
  std::uint64_t target = 0;
  if (auto result = pop_width(ctx, target, offset_width); !result.ok()) {
    return result;
  }
  std::uint64_t selector = 0;
  if (auto result = pop_width(ctx, selector, 2); !result.ok()) {
    ctx.state.gpr[4] = entry_sp;
    return result;
  }
  if (!far_transfer_allowed(ctx.state, selector)) {
    ctx.state.gpr[4] = entry_sp;
    return far_transfer_fault(ctx);
  }
  if (offset_width == 8 && !is_canonical_address(target)) {
    ctx.state.gpr[4] = entry_sp;
    return branch_target_fault(ctx, target);
  }
  ctx.state.mode = mode_for_far_width(offset_width);
  ctx.state.sreg[1] = static_cast<std::uint16_t>(selector);
  ctx.state.rip = mask_instruction_pointer(ctx.state, target);
  ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] + imm16);
  ctx.control_flow_taken = true;
  return {};
}

ExecutionResult iret_width(ExecutionContext& ctx, std::size_t offset_width, std::size_t flags_width) {
  const auto entry_sp = ctx.state.gpr[4];
  std::uint64_t target = 0;
  if (auto result = pop_width(ctx, target, offset_width); !result.ok()) {
    return result;
  }
  std::uint64_t selector = 0;
  if (auto result = pop_width(ctx, selector, 2); !result.ok()) {
    ctx.state.gpr[4] = entry_sp;
    return result;
  }
  std::uint64_t flags = 0;
  if (auto result = pop_width(ctx, flags, flags_width); !result.ok()) {
    ctx.state.gpr[4] = entry_sp;
    return result;
  }

  if (!far_transfer_allowed(ctx.state, selector)) {
    ctx.state.gpr[4] = entry_sp;
    return far_transfer_fault(ctx);
  }
  if (offset_width == 8 && !is_canonical_address(target)) {
    ctx.state.gpr[4] = entry_sp;
    return branch_target_fault(ctx, target);
  }
  const auto cpl = ctx.state.sreg[1] & 0x3u;
  const auto iopl = (ctx.state.rflags >> 12) & 0x3u;
  ctx.state.mode = mode_for_far_width(offset_width);
  ctx.state.sreg[1] = static_cast<std::uint16_t>(selector);
  ctx.state.rip = mask_instruction_pointer(ctx.state, target);
  // IRET is the one instruction that can legitimately rewrite the system bits, but only from the
  // privilege level entitled to each: IOPL only at CPL 0, IF only at CPL <= IOPL. Below that they
  // stay as they were, which is the same thing POPFQ above already does unconditionally. Without
  // this, ring 3 could hand itself IOPL 3 and then CLI/STI, which cli_sti_allowed gates on exactly
  // this comparison. VM/VIF/VIP are preserved outright -- nothing here emulates virtual-8086.
  std::uint64_t protected_bits = (1ull << 17) | (1ull << 19) | (1ull << 20);
  if (cpl != 0) {
    protected_bits |= 3ull << 12;
  }
  if (cpl > iopl) {
    protected_bits |= kFlagIF;
  }
  std::uint64_t merged = ctx.state.rflags;
  if (flags_width == 8) {
    merged = flags;
  } else if (flags_width == 4) {
    merged = (ctx.state.rflags & ~0xFFFFFFFFull) | (flags & 0xFFFFFFFFull);
  } else {
    merged = (ctx.state.rflags & ~0xFFFFull) | (flags & 0xFFFFull);
  }
  // Same reserved-bit scrub POPFQ does. IRET is the other instruction that loads rflags straight out
  // of guest memory, and without this a guest could park values in bits 3/5/15 and 63:22, which read
  // back as fixed on hardware and cannot hold anything.
  merged &= kRflagsWritableMask;
  ctx.state.rflags = (merged & ~protected_bits) | (ctx.state.rflags & protected_bits) | kRflagsReservedOnes;
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



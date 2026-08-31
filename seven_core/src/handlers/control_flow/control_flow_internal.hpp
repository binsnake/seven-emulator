#pragma once

#include "seven/handlers_fwd.hpp"
#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] iced_x86::Register promote_gpr32_to_64(iced_x86::Register reg) {
  switch (reg) {
    case iced_x86::Register::EAX: return iced_x86::Register::RAX;
    case iced_x86::Register::ECX: return iced_x86::Register::RCX;
    case iced_x86::Register::EDX: return iced_x86::Register::RDX;
    case iced_x86::Register::EBX: return iced_x86::Register::RBX;
    case iced_x86::Register::ESP: return iced_x86::Register::RSP;
    case iced_x86::Register::EBP: return iced_x86::Register::RBP;
    case iced_x86::Register::ESI: return iced_x86::Register::RSI;
    case iced_x86::Register::EDI: return iced_x86::Register::RDI;
    case iced_x86::Register::R8_D: return iced_x86::Register::R8;
    case iced_x86::Register::R9_D: return iced_x86::Register::R9;
    case iced_x86::Register::R10_D: return iced_x86::Register::R10;
    case iced_x86::Register::R11_D: return iced_x86::Register::R11;
    case iced_x86::Register::R12_D: return iced_x86::Register::R12;
    case iced_x86::Register::R13_D: return iced_x86::Register::R13;
    case iced_x86::Register::R14_D: return iced_x86::Register::R14;
    case iced_x86::Register::R15_D: return iced_x86::Register::R15;
    default:
      return reg;
  }
}

[[nodiscard]] bool trace_call_rm_enabled() noexcept {
  static const bool enabled = [] {
    if (const char* v = std::getenv("SEVEN_TRACE_CALLRM")) {
      return v[0] != '\0' && v[0] != '0';
    }
    return false;
  }();
  return enabled;
}


ExecutionResult read_near_target_width(ExecutionContext& ctx, std::size_t width, std::uint64_t& target) {
  if (width == 8 && ctx.state.mode == ExecutionMode::long64 && ctx.instr.op_kind(0) == iced_x86::OpKind::REGISTER) {
    const auto reg = promote_gpr32_to_64(ctx.instr.op_register(0));
    target = detail::read_register(ctx.state, reg);
    return {};
  }
  return detail::read_operand_checked(ctx, 0, width, target);
}

ExecutionResult unsupported_instruction(ExecutionContext& ctx) {
  return {StopReason::unsupported_instruction, 0, ExceptionInfo{StopReason::unsupported_instruction, ctx.state.rip, 0}, ctx.instr.code()};
}

// RET, IRET and the indirect near branches take their target from guest data, the one way a
// non-canonical value reaches rip with no memory operand involved. Hardware #GP(0)s at the branch;
// deferring to the next step's fetch check blames the wrong instruction and retires the branch.
[[nodiscard]] ExecutionResult branch_target_fault(ExecutionContext& ctx, std::uint64_t target) {
  return {StopReason::general_protection, 0,
          ExceptionInfo{StopReason::general_protection, target, 0}, ctx.instr.code()};
}

// rsp only moves once the slot is written. A #PF is a fault, so hardware leaves rsp where it was and
// the retry works, which is what lets a guard page grow a stack. Committing the decrement first
// walked the stack pointer down one slot per retry.
ExecutionResult push_width(ExecutionContext& ctx, std::uint64_t value, std::size_t width) {
  const auto slot = mask_stack_pointer(ctx.state, ctx.state.gpr[4] - width);
  ExecutionResult result{};
  switch (width) {
    case 1: {
      const auto v = static_cast<std::uint8_t>(value);
      result = detail::write_memory_checked(ctx, slot, v);
      break;
    }
    case 2: {
      const auto v = static_cast<std::uint16_t>(value);
      result = detail::write_memory_checked(ctx, slot, v);
      break;
    }
    case 4: {
      const auto v = static_cast<std::uint32_t>(value);
      result = detail::write_memory_checked(ctx, slot, v);
      break;
    }
    case 8:
      result = detail::write_memory_checked(ctx, slot, value);
      break;
    default:
      return unsupported_instruction(ctx);
  }
  if (!result.ok()) {
    return result;
  }
  ctx.state.gpr[4] = slot;
  detail::note_stack_access(ctx, slot, width, true);
  return {};
}

ExecutionResult pop_width(ExecutionContext& ctx, std::uint64_t& value, std::size_t width) {
  const auto slot = ctx.state.gpr[4];
  switch (width) {
    case 1: {
      std::uint8_t v = 0;
      if (auto result = detail::read_memory_checked(ctx, ctx.state.gpr[4], v); !result.ok()) return result;
      value = v;
      break;
    }
    case 2: {
      std::uint16_t v = 0;
      if (auto result = detail::read_memory_checked(ctx, ctx.state.gpr[4], v); !result.ok()) return result;
      value = v;
      break;
    }
    case 4: {
      std::uint32_t v = 0;
      if (auto result = detail::read_memory_checked(ctx, ctx.state.gpr[4], v); !result.ok()) return result;
      value = v;
      break;
    }
    case 8: {
      std::uint64_t v = 0;
      if (auto result = detail::read_memory_checked(ctx, ctx.state.gpr[4], v); !result.ok()) return result;
      value = v;
      break;
    }
    default:
      return unsupported_instruction(ctx);
  }
  ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] + width);
  detail::note_stack_access(ctx, slot, width, false);
  return {};
}

ExecutionResult push_operand_width(ExecutionContext& ctx, std::size_t width) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 0, width, value); !result.ok()) {
    return result;
  }
  if (auto result = push_width(ctx, value, width); !result.ok()) {
    return result;
  }
  return {};
}

ExecutionResult push_imm_width(ExecutionContext& ctx, std::size_t imm_width, std::size_t stack_width, bool sign_extend_imm) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 0, imm_width, value); !result.ok()) {
    return result;
  }
  if (sign_extend_imm) {
    value = sign_extend(value, imm_width);
  }
  if (auto result = push_width(ctx, value, stack_width); !result.ok()) {
    return result;
  }
  return {};
}

ExecutionResult pop_operand_width(ExecutionContext& ctx, std::size_t width) {
  const auto old_sp = ctx.state.gpr[4];
  std::uint64_t value = 0;
  if (auto result = pop_width(ctx, value, width); !result.ok()) {
    return result;
  }
  if (auto result = detail::write_operand_checked(ctx, 0, value, width); !result.ok()) {
    // The destination's effective address is computed from the already-incremented rsp -- that part
    // is architectural, POP m64 really does address through the new rsp. Committing the increment
    // when the store then faults is not: the instruction is aborted and rsp goes back.
    ctx.state.gpr[4] = old_sp;
    return result;
  }
  if (ctx.instr.op_kind(0) == iced_x86::OpKind::REGISTER && ctx.instr.op_register(0) == iced_x86::Register::SS) {
    ctx.state.debug_suppression = 1;
    if ((ctx.state.rflags & kFlagTF) != 0) ctx.state.pending_single_step = true;
    ctx.debug_hit_bits |= detail::debug_data_breakpoint_hits(ctx.state, old_sp, width, true, false);
  }
  return {};
}

ExecutionResult push_segment(ExecutionContext& ctx, std::size_t index, std::size_t stack_width) {
  if (index >= ctx.state.sreg.size()) {
    return unsupported_instruction(ctx);
  }
  const std::uint64_t value = ctx.state.sreg[index];
  if (auto result = push_width(ctx, value, stack_width); !result.ok()) {
    return result;
  }
  return {};
}

ExecutionResult push_flags_width(ExecutionContext& ctx, std::size_t stack_width) {
  if (auto result = push_width(ctx, ctx.state.rflags, stack_width); !result.ok()) {
    return result;
  }
  return {};
}

ExecutionResult push_all_width(ExecutionContext& ctx, std::size_t width) {
  const std::uint64_t original_sp = ctx.state.gpr[4];
  const std::uint64_t regs[8] = {
      ctx.state.gpr[0], ctx.state.gpr[1], ctx.state.gpr[2], ctx.state.gpr[3],
      original_sp,      ctx.state.gpr[5], ctx.state.gpr[6], ctx.state.gpr[7]};
  for (const auto value : regs) {
    if (auto result = push_width(ctx, value, width); !result.ok()) {
      ctx.state.gpr[4] = original_sp;
      return result;
    }
  }
  return {};
}

ExecutionResult call_rm_width(ExecutionContext& ctx, std::size_t width) {
  std::uint64_t target = 0;
  if (auto result = read_near_target_width(ctx, width, target); !result.ok()) {
    return result;
  }
  if (trace_call_rm_enabled() && ctx.instr.op_kind(0) == iced_x86::OpKind::MEMORY) {
    std::fprintf(stderr,
                 "[seven-callrm] rip=0x%llx addr=0x%llx target=0x%llx next=0x%llx\n",
                 static_cast<unsigned long long>(ctx.state.rip),
                 static_cast<unsigned long long>(detail::memory_address(ctx)),
                 static_cast<unsigned long long>(target),
                 static_cast<unsigned long long>(ctx.next_rip));
  }
  // Ahead of the push, matching the SDM's order: a CALL to a non-canonical target faults with the
  // return address still unwritten and rsp untouched.
  if (!is_canonical_address(target)) {
    return branch_target_fault(ctx, target);
  }
  if (auto result = push_width(ctx, ctx.next_rip, width); !result.ok()) {
    return result;
  }
  ctx.state.rip = target;
  ctx.control_flow_taken = true;
  return {};
}

// sreg[1] is the only record of privilege level here, and the far branches load it straight from a
// guest-supplied selector, so one RETF or IRET would otherwise put the guest at CPL 0. Hardware lets
// a far transfer move outward but never inward, and gaining privilege needs a validated call gate.
// There are no descriptor tables here, so the selector's RPL is all there is to check.
[[nodiscard]] inline bool far_transfer_allowed(const CpuState& state, std::uint64_t selector) noexcept {
  return (selector & 0x3u) >= (state.sreg[1] & 0x3u);
}

[[nodiscard]] inline ExecutionResult far_transfer_fault(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0,
          ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
}

ExecutionResult ret_width(ExecutionContext& ctx, std::size_t width, std::uint16_t imm16) {
  const auto entry_sp = ctx.state.gpr[4];
  std::uint64_t target = 0;
  if (auto result = pop_width(ctx, target, width); !result.ok()) {
    return result;
  }
  target = mask_instruction_pointer(ctx.state, target);
  if (!is_canonical_address(target)) {
    ctx.state.gpr[4] = entry_sp;
    return branch_target_fault(ctx, target);
  }
  ctx.state.rip = target;
  ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] + imm16);
  ctx.control_flow_taken = true;
  return {};
}

}  // namespace

}  // namespace seven::handlers


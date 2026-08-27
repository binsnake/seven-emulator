#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

// ENTER's second operand used to be discarded. A non-zero nesting level copies that many pointers
// from the enclosing frames onto the new frame before the frame pointer moves, so ignoring it left
// rsp `level * width` slots too high and every display slot unwritten -- a silently wrong frame
// rather than a refused one, which is the worst of the three options for a guest that then indexes
// through it. Level is taken mod 32 by the instruction itself.
//
// The whole thing runs off locals and commits rsp/rbp only at the end, so a copy that faults
// partway leaves the registers where they were and the instruction restarts cleanly.
ExecutionResult enter_common(ExecutionContext& ctx, std::size_t width, iced_x86::Register sp_reg,
                             iced_x86::Register bp_reg) {
  const auto mask = mask_for_width(width);
  const auto frame_size = ctx.instr.immediate16();
  // iced gives ENTER's level operand kind IMMEDIATE8_2ND, so instr.immediate8() reads back as zero
  // for it. That is why discarding the level looked harmless.
  const auto nesting = static_cast<std::uint32_t>(detail::immediate_value(ctx.instr, 1) & 0xFFu) % 32u;

  auto frame_bp = detail::read_register(ctx.state, bp_reg) & mask;
  auto sp = detail::read_register(ctx.state, sp_reg) & mask;

  const auto push = [&](std::uint64_t value) -> ExecutionResult {
    sp = (sp - width) & mask;
    if (!ctx.memory.write(sp, &value, width)) {
      return detail::memory_fault(ctx, sp);
    }
    detail::note_stack_access(ctx, sp, width, true);
    return {};
  };

  if (auto result = push(frame_bp); !result.ok()) {
    return result;
  }
  const auto frame_temp = sp;

  for (std::uint32_t level = 1; level < nesting; ++level) {
    frame_bp = (frame_bp - width) & mask;
    std::uint64_t display = 0;
    if (!ctx.memory.read(frame_bp, &display, width)) {
      return detail::memory_fault(ctx, frame_bp);
    }
    if (auto result = push(display); !result.ok()) {
      return result;
    }
  }
  if (nesting > 0) {
    if (auto result = push(frame_temp); !result.ok()) {
      return result;
    }
  }

  // The local area is carved out below the display, not below the frame pointer, so this subtracts
  // from where the pushes actually stopped. Those coincide only at nesting level 0.
  detail::write_register(ctx.state, bp_reg, frame_temp, width);
  detail::write_register(ctx.state, sp_reg, (sp - frame_size) & mask, width);
  return {};
}

}  // namespace

ExecutionResult handle_code_ENTERW(ExecutionContext& ctx) {
  return enter_common(ctx, 2, iced_x86::Register::SP, iced_x86::Register::BP);
}

ExecutionResult handle_code_ENTERD(ExecutionContext& ctx) {
  return enter_common(ctx, 4, iced_x86::Register::ESP, iced_x86::Register::EBP);
}

ExecutionResult handle_code_ENTERQ(ExecutionContext& ctx) {
  return enter_common(ctx, 8, iced_x86::Register::RSP, iced_x86::Register::RBP);
}

ExecutionResult handle_code_ENTERW_IMM16_IMM8(ExecutionContext& ctx) {
  return handle_code_ENTERW(ctx);
}

ExecutionResult handle_code_ENTERD_IMM16_IMM8(ExecutionContext& ctx) {
  return handle_code_ENTERD(ctx);
}

ExecutionResult handle_code_ENTERQ_IMM16_IMM8(ExecutionContext& ctx) {
  return handle_code_ENTERQ(ctx);
}

}  // namespace seven::handlers


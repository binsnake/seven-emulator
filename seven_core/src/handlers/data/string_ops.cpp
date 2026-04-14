#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] std::uint64_t width_mask(std::size_t width) {
  if (width >= 8) {
    return ~0ull;
  }
  return (1ull << (width * 8)) - 1ull;
}

ExecutionResult cmps_impl(ExecutionContext& ctx, std::size_t width) {
  const bool rep = ctx.instr.has_rep_prefix() || ctx.instr.has_repne_prefix();
  const bool repne = ctx.instr.has_repne_prefix();
  std::uint64_t remaining = rep ? ctx.state.gpr[1] : 1ull;
  if (remaining == 0) {
    return {};
  }

  const bool df = (ctx.state.rflags & kFlagDF) != 0;
  std::uint64_t rsi = ctx.state.gpr[6];
  std::uint64_t rdi = ctx.state.gpr[7];

  while (remaining > 0) {
    const auto lhs_addr = rsi;
    const auto rhs_addr = rdi;
    std::uint64_t lhs = 0;
    if (!ctx.memory.read(lhs_addr, &lhs, width)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, lhs_addr, 0}, ctx.instr.code()};
    }
    std::uint64_t rhs = 0;
    if (!ctx.memory.read(rhs_addr, &rhs, width)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, rhs_addr, 0}, ctx.instr.code()};
    }
    lhs &= width_mask(width);
    rhs &= width_mask(width);
    const auto result = detail::truncate(lhs - rhs, width);
    detail::set_sub_flags(ctx.state, lhs, rhs, result, width, false);
    const auto hit_bits = detail::debug_data_breakpoint_hits(ctx.state, lhs_addr, width, true, false) |
                          detail::debug_data_breakpoint_hits(ctx.state, rhs_addr, width, true, false);

    if (df) {
      rsi -= width;
      rdi -= width;
    } else {
      rsi += width;
      rdi += width;
    }
    --remaining;

    bool continue_loop = false;
    if (rep && remaining > 0) {
      const bool zf = (ctx.state.rflags & kFlagZF) != 0;
      continue_loop = !(repne ? zf : !zf);
    }

    if (detail::note_debug_break(ctx, hit_bits, continue_loop)) {
      ctx.state.gpr[6] = rsi;
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {};
    }

    if (!rep || !continue_loop) {
      break;
    }
  }

  ctx.state.gpr[6] = rsi;
  ctx.state.gpr[7] = rdi;
  if (rep) ctx.state.gpr[1] = remaining;
  return {};
}

ExecutionResult scas_impl(ExecutionContext& ctx, std::size_t width) {
  const bool rep = ctx.instr.has_rep_prefix() || ctx.instr.has_repne_prefix();
  const bool repne = ctx.instr.has_repne_prefix();
  std::uint64_t remaining = rep ? ctx.state.gpr[1] : 1ull;
  if (remaining == 0) {
    return {};
  }

  const bool df = (ctx.state.rflags & kFlagDF) != 0;
  std::uint64_t rdi = ctx.state.gpr[7];
  const auto lhs = detail::read_register(ctx.state, iced_x86::Register::RAX) & width_mask(width);

  while (remaining > 0) {
    const auto rhs_addr = rdi;
    std::uint64_t rhs = 0;
    if (!ctx.memory.read(rhs_addr, &rhs, width)) {
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, rhs_addr, 0}, ctx.instr.code()};
    }
    rhs &= width_mask(width);
    const auto result = detail::truncate(lhs - rhs, width);
    detail::set_sub_flags(ctx.state, lhs, rhs, result, width, false);
    const auto hit_bits = detail::debug_data_breakpoint_hits(ctx.state, rhs_addr, width, true, false);

    if (df) {
      rdi -= width;
    } else {
      rdi += width;
    }
    --remaining;

    bool continue_loop = false;
    if (rep && remaining > 0) {
      const bool zf = (ctx.state.rflags & kFlagZF) != 0;
      continue_loop = !(repne ? zf : !zf);
    }

    if (detail::note_debug_break(ctx, hit_bits, continue_loop)) {
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {};
    }

    if (!rep || !continue_loop) {
      break;
    }
  }

  ctx.state.gpr[7] = rdi;
  if (rep) ctx.state.gpr[1] = remaining;
  return {};
}

ExecutionResult stos_impl(ExecutionContext& ctx, std::size_t width) {
  const bool rep = ctx.instr.has_rep_prefix() || ctx.instr.has_repne_prefix();
  std::uint64_t remaining = rep ? ctx.state.gpr[1] : 1ull;
  if (remaining == 0) {
    return {};
  }

  const bool df = (ctx.state.rflags & kFlagDF) != 0;
  std::uint64_t rdi = ctx.state.gpr[7];
  const auto value = detail::read_register(ctx.state, iced_x86::Register::RAX) & width_mask(width);

  while (remaining > 0) {
    const auto write_addr = rdi;
    if (!ctx.memory.write(write_addr, &value, width)) {
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, write_addr, 0}, ctx.instr.code()};
    }
    const auto hit_bits = detail::debug_data_breakpoint_hits(ctx.state, write_addr, width, false, true);
    if (df) {
      rdi -= width;
    } else {
      rdi += width;
    }
    --remaining;

    if (detail::note_debug_break(ctx, hit_bits, rep && remaining > 0)) {
      ctx.state.gpr[7] = rdi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {};
    }

    if (!rep) {
      break;
    }
  }

  ctx.state.gpr[7] = rdi;
  if (rep) ctx.state.gpr[1] = remaining;
  return {};
}

ExecutionResult lods_impl(ExecutionContext& ctx, std::size_t width) {
  const bool rep = ctx.instr.has_rep_prefix() || ctx.instr.has_repne_prefix();
  std::uint64_t remaining = rep ? ctx.state.gpr[1] : 1ull;
  if (remaining == 0) {
    return {};
  }

  const bool df = (ctx.state.rflags & kFlagDF) != 0;
  std::uint64_t rsi = ctx.state.gpr[6];

  while (remaining > 0) {
    const auto read_addr = rsi;
    std::uint64_t value = 0;
    if (!ctx.memory.read(read_addr, &value, width)) {
      ctx.state.gpr[6] = rsi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, read_addr, 0}, ctx.instr.code()};
    }
    detail::write_register(ctx.state, iced_x86::Register::RAX, value, width);
    const auto hit_bits = detail::debug_data_breakpoint_hits(ctx.state, read_addr, width, true, false);
    if (df) {
      rsi -= width;
    } else {
      rsi += width;
    }
    --remaining;

    if (detail::note_debug_break(ctx, hit_bits, rep && remaining > 0)) {
      ctx.state.gpr[6] = rsi;
      if (rep) ctx.state.gpr[1] = remaining;
      return {};
    }

    if (!rep) {
      break;
    }
  }

  ctx.state.gpr[6] = rsi;
  if (rep) ctx.state.gpr[1] = remaining;
  return {};
}

}  // namespace

ExecutionResult handle_code_CMPSB_M8_M8(ExecutionContext& ctx) { return cmps_impl(ctx, 1); }
ExecutionResult handle_code_CMPSW_M16_M16(ExecutionContext& ctx) { return cmps_impl(ctx, 2); }
ExecutionResult handle_code_CMPSD_M32_M32(ExecutionContext& ctx) { return cmps_impl(ctx, 4); }
ExecutionResult handle_code_CMPSQ_M64_M64(ExecutionContext& ctx) { return cmps_impl(ctx, 8); }

ExecutionResult handle_code_SCASB_AL_M8(ExecutionContext& ctx) { return scas_impl(ctx, 1); }
ExecutionResult handle_code_SCASW_AX_M16(ExecutionContext& ctx) { return scas_impl(ctx, 2); }
ExecutionResult handle_code_SCASD_EAX_M32(ExecutionContext& ctx) { return scas_impl(ctx, 4); }
ExecutionResult handle_code_SCASQ_RAX_M64(ExecutionContext& ctx) { return scas_impl(ctx, 8); }

ExecutionResult handle_code_STOSB_M8_AL(ExecutionContext& ctx) { return stos_impl(ctx, 1); }
ExecutionResult handle_code_STOSW_M16_AX(ExecutionContext& ctx) { return stos_impl(ctx, 2); }
ExecutionResult handle_code_STOSD_M32_EAX(ExecutionContext& ctx) { return stos_impl(ctx, 4); }
ExecutionResult handle_code_STOSQ_M64_RAX(ExecutionContext& ctx) { return stos_impl(ctx, 8); }

ExecutionResult handle_code_LODSB_AL_M8(ExecutionContext& ctx) { return lods_impl(ctx, 1); }
ExecutionResult handle_code_LODSW_AX_M16(ExecutionContext& ctx) { return lods_impl(ctx, 2); }
ExecutionResult handle_code_LODSD_EAX_M32(ExecutionContext& ctx) { return lods_impl(ctx, 4); }
ExecutionResult handle_code_LODSQ_RAX_M64(ExecutionContext& ctx) { return lods_impl(ctx, 8); }

}  // namespace seven::handlers


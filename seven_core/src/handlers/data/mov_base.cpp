#include "seven/handler_helpers.hpp"

#include <optional>

namespace seven::handlers {

namespace {

constexpr std::uint64_t kCr4DeBit = 1ull << 3;

[[nodiscard]] ExecutionResult gp_fault(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
}

[[nodiscard]] ExecutionResult ud_fault(ExecutionContext& ctx) {
  return {StopReason::invalid_opcode, 0, ExceptionInfo{StopReason::invalid_opcode, ctx.state.rip, 0}, ctx.instr.code()};
}

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

[[nodiscard]] std::optional<std::uint32_t> resolve_debug_index(const CpuState& state, iced_x86::Register reg) {
  if (reg >= iced_x86::Register::CR0 && reg <= iced_x86::Register::CR15) {
    reg = static_cast<iced_x86::Register>(
        static_cast<std::uint32_t>(iced_x86::Register::DR0) +
        (static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::CR0)));
  }
  if (reg < iced_x86::Register::DR0 || reg > iced_x86::Register::DR15) {
    return std::nullopt;
  }
  std::uint32_t index =
      static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::DR0);

  // SDM: DR4/DR5 alias DR6/DR7 only when CR4.DE=0, otherwise access is #UD.
  if (index == 4u || index == 5u) {
    if ((state.cr[4] & kCr4DeBit) != 0) {
      return std::nullopt;
    }
    index += 2u;
  }

  // x64 architectural debug registers are DR0..DR7.
  if (index > 7u) {
    return std::nullopt;
  }
  return index;
}

}  // namespace

ExecutionResult handle_code_MOV_RM8_R8(ExecutionContext& ctx) {
  bool dst_ok = false;
  bool src_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_operand(ctx, 1, 1, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 0, rhs, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return { };
}

ExecutionResult handle_code_MOV_RM16_R16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 0, value, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_RM32_R32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 0, value, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_RM64_R64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 0, value, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_AL_MOFFS8(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 1, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 1);
  return {};
}

ExecutionResult handle_code_MOV_AX_MOFFS16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 2);
  return {};
}

ExecutionResult handle_code_MOV_EAX_MOFFS32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOV_RAX_MOFFS64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 8);
  return {};
}

ExecutionResult handle_code_MOV_MOFFS8_AL(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 0, 1, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_MOFFS16_AX(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 0, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, value, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_MOFFS32_EAX(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 0, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, value, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_MOFFS64_RAX(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 0, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 1, value, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_R8_RM8(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 1, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 1);
  return {};
}

ExecutionResult handle_code_MOV_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 2);
  return {};
}

ExecutionResult handle_code_MOV_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOV_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 8);
  return {};
}

ExecutionResult handle_code_MOV_R8_IMM8(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate8();
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 1);
  return {};
}

ExecutionResult handle_code_MOV_R16_IMM16(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate16();
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 2);
  return {};
}

ExecutionResult handle_code_MOV_R32_IMM32(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate32();
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOV_R64_IMM64(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate64();
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 8);
  return {};
}

ExecutionResult handle_code_MOV_RM8_IMM8(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate8();
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_RM16_IMM16(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate16();
  if (!detail::write_operand(ctx, 0, value, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_RM32_IMM32(ExecutionContext& ctx) {
  const auto value = ctx.instr.immediate32();
  if (!detail::write_operand(ctx, 0, value, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_RM64_IMM32(ExecutionContext& ctx) {
  std::uint64_t value = ctx.instr.immediate32();
  value = detail::sign_extend(value, 4);
  if (!detail::write_operand(ctx, 0, value, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_RM16_SREG(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(1));
  if (!detail::write_operand(ctx, 0, value, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOV_R32M16_SREG(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(1));
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOV_R64M16_SREG(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(1));
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 8);
  return {};
}

ExecutionResult handle_code_MOV_SREG_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto dst_reg = ctx.instr.op_register(0);
  detail::write_register(ctx.state, dst_reg, value, 2);
  if (dst_reg == iced_x86::Register::SS) {
    ctx.state.debug_suppression = 1;
    if ((ctx.state.rflags & kFlagTF) != 0) ctx.state.pending_single_step = true;
    if (ctx.instr.op_kind(1) == iced_x86::OpKind::MEMORY) {
      ctx.debug_hit_bits |= detail::debug_data_breakpoint_hits(ctx.state, detail::memory_address(ctx), 2, true, false);
    }
  }
  return {};
}

ExecutionResult handle_code_MOV_SREG_R32M16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto dst_reg = ctx.instr.op_register(0);
  detail::write_register(ctx.state, dst_reg, value, 2);
  if (dst_reg == iced_x86::Register::SS) {
    ctx.state.debug_suppression = 1;
    if ((ctx.state.rflags & kFlagTF) != 0) ctx.state.pending_single_step = true;
    if (ctx.instr.op_kind(1) == iced_x86::OpKind::MEMORY) {
      ctx.debug_hit_bits |= detail::debug_data_breakpoint_hits(ctx.state, detail::memory_address(ctx), 4, true, false);
    }
  }
  return {};
}

ExecutionResult handle_code_MOV_SREG_R64M16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 2);
  return {};
}

ExecutionResult handle_code_MOV_R32_CR(ExecutionContext& ctx) {
  const auto as_cr = [](iced_x86::Register reg) {
    if (reg >= iced_x86::Register::DR0 && reg <= iced_x86::Register::DR15) {
      return static_cast<iced_x86::Register>(
          static_cast<std::uint32_t>(iced_x86::Register::CR0) +
          (static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::DR0)));
    }
    return reg;
  };
  const auto dst = ctx.instr.op_register(0);
  const auto src = as_cr(ctx.instr.op_register(1));
  const auto value = detail::read_register(ctx.state, src);
  detail::write_register(ctx.state, dst, value, 4);
  return {};
}

ExecutionResult handle_code_MOV_R64_CR(ExecutionContext& ctx) {
  const auto as_cr = [](iced_x86::Register reg) {
    if (reg >= iced_x86::Register::DR0 && reg <= iced_x86::Register::DR15) {
      return static_cast<iced_x86::Register>(
          static_cast<std::uint32_t>(iced_x86::Register::CR0) +
          (static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::DR0)));
    }
    return reg;
  };
  const auto dst = ctx.instr.op_register(0);
  const auto src = as_cr(ctx.instr.op_register(1));
  const auto value = detail::read_register(ctx.state, src);
  detail::write_register(ctx.state, dst, value, 8);
  return {};
}

ExecutionResult handle_code_MOV_R32_DR(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto dst = ctx.instr.op_register(0);
  const auto src_index = resolve_debug_index(ctx.state, ctx.instr.op_register(1));
  if (!src_index.has_value()) {
    return ud_fault(ctx);
  }
  const auto value = ctx.state.dr[src_index.value()];
  detail::write_register(ctx.state, dst, value, 4);
  return {};
}

ExecutionResult handle_code_MOV_R64_DR(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto dst = ctx.instr.op_register(0);
  const auto src_index = resolve_debug_index(ctx.state, ctx.instr.op_register(1));
  if (!src_index.has_value()) {
    return ud_fault(ctx);
  }
  const auto value = ctx.state.dr[src_index.value()];
  detail::write_register(ctx.state, dst, value, 8);
  return {};
}

ExecutionResult handle_code_MOV_CR_R32(ExecutionContext& ctx) {
  const auto as_cr = [](iced_x86::Register reg) {
    if (reg >= iced_x86::Register::DR0 && reg <= iced_x86::Register::DR15) {
      return static_cast<iced_x86::Register>(
          static_cast<std::uint32_t>(iced_x86::Register::CR0) +
          (static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::DR0)));
    }
    return reg;
  };
  const auto dst = as_cr(ctx.instr.op_register(0));
  const auto src = ctx.instr.op_register(1);
  const auto value = detail::truncate(detail::read_register(ctx.state, src), 4);
  detail::write_register(ctx.state, dst, value, 4);
  return {};
}

ExecutionResult handle_code_MOV_CR_R64(ExecutionContext& ctx) {
  const auto as_cr = [](iced_x86::Register reg) {
    if (reg >= iced_x86::Register::DR0 && reg <= iced_x86::Register::DR15) {
      return static_cast<iced_x86::Register>(
          static_cast<std::uint32_t>(iced_x86::Register::CR0) +
          (static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::DR0)));
    }
    return reg;
  };
  const auto dst = as_cr(ctx.instr.op_register(0));
  const auto src = ctx.instr.op_register(1);
  const auto value = detail::read_register(ctx.state, src);
  detail::write_register(ctx.state, dst, value, 8);
  return {};
}

ExecutionResult handle_code_MOV_DR_R32(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto dst_index = resolve_debug_index(ctx.state, ctx.instr.op_register(0));
  if (!dst_index.has_value()) {
    return ud_fault(ctx);
  }
  const auto src = ctx.instr.op_register(1);
  const auto value = detail::truncate(detail::read_register(ctx.state, src), 4);
  ctx.state.dr[dst_index.value()] = value;
  return {};
}

ExecutionResult handle_code_MOV_DR_R64(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto dst_index = resolve_debug_index(ctx.state, ctx.instr.op_register(0));
  if (!dst_index.has_value()) {
    return ud_fault(ctx);
  }
  const auto src = ctx.instr.op_register(1);
  const auto value = detail::read_register(ctx.state, src);
  ctx.state.dr[dst_index.value()] = value;
  return {};
}

ExecutionResult handle_code_MOV_R32_TR(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(1));
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_MOV_TR_R32(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(1));
  detail::write_register(ctx.state, ctx.instr.op_register(0), value, 4);
  return {};
}

ExecutionResult handle_code_LEA_R64_M(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), detail::memory_address(ctx), 8);
  return {};
}

ExecutionResult handle_code_LEA_R32_M(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), detail::memory_address(ctx), 4);
  return {};
}

ExecutionResult handle_code_LEA_R16_M(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), detail::memory_address(ctx), 2);
  return {};
}

}  // namespace seven::handlers


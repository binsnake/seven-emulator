#include "seven/handlers_fwd.hpp"
#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CLC(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagCF, false);
  return {};
}

ExecutionResult handle_code_STC(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagCF, true);
  return {};
}

ExecutionResult handle_code_CMC(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagCF, (ctx.state.rflags & kFlagCF) == 0);
  return {};
}

ExecutionResult handle_code_CLD(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagDF, false);
  return {};
}

ExecutionResult handle_code_STD(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagDF, true);
  return {};
}

ExecutionResult handle_code_CLI(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagIF, false);
  return {};
}

ExecutionResult handle_code_STI(ExecutionContext& ctx) {
  detail::set_flag(ctx.state.rflags, kFlagIF, true);
  return {};
}

ExecutionResult handle_code_PUSHFQ(ExecutionContext& ctx) {
  ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] - 8);
  if (!ctx.memory.write(ctx.state.gpr[4], &ctx.state.rflags, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, ctx.state.gpr[4], 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_POPFQ(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (!ctx.memory.read(ctx.state.gpr[4], &value, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, ctx.state.gpr[4], 0}, ctx.instr.code()};
  }
  ctx.state.rflags = value;
  ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] + 8);
  return {};
}

ExecutionResult handle_code_HLT(ExecutionContext& ctx) {
  (void)ctx;
  return {StopReason::halted, 0, std::nullopt, std::nullopt};
}

ExecutionResult handle_code_LAHF(ExecutionContext& ctx) {
  const auto ah = ((ctx.state.rflags & kFlagSF) ? 0x80u : 0u) |
                  ((ctx.state.rflags & kFlagZF) ? 0x40u : 0u) |
                  ((ctx.state.rflags & kFlagAF) ? 0x10u : 0u) |
                  ((ctx.state.rflags & kFlagPF) ? 0x04u : 0u) |
                  ((ctx.state.rflags & kFlagCF) ? 0x01u : 0u) |
                  0x02u;
  detail::write_register(ctx.state, iced_x86::Register::AH, ah);
  return {};
}

ExecutionResult handle_code_SAHF(ExecutionContext& ctx) {
  const auto ah = detail::read_register(ctx.state, iced_x86::Register::AH);
  detail::set_flag(ctx.state.rflags, kFlagSF, (ah & 0x80u) != 0);
  detail::set_flag(ctx.state.rflags, kFlagZF, (ah & 0x40u) != 0);
  detail::set_flag(ctx.state.rflags, kFlagAF, (ah & 0x10u) != 0);
  detail::set_flag(ctx.state.rflags, kFlagPF, (ah & 0x04u) != 0);
  detail::set_flag(ctx.state.rflags, kFlagCF, (ah & 0x01u) != 0);
  return {};
}

ExecutionResult handle_code_INT1(ExecutionContext& ctx) {
  return detail::dispatch_interrupt(ctx, 1u, ctx.next_rip);
}

ExecutionResult handle_code_INT3(ExecutionContext& ctx) {
  return detail::dispatch_interrupt(ctx, 3u, ctx.next_rip);
}

ExecutionResult handle_code_INT_IMM8(ExecutionContext& ctx) {
  const auto vector = static_cast<std::uint8_t>(ctx.instr.immediate8());
  return detail::dispatch_interrupt(ctx, vector, ctx.next_rip);
}

ExecutionResult handle_code_INTO(ExecutionContext& ctx) {
  if ((ctx.state.rflags & kFlagOF) == 0) {
    return {};
  }
  return detail::dispatch_interrupt(ctx, 4u, ctx.next_rip);
}

ExecutionResult handle_code_RDTSC(ExecutionContext& ctx) {
  static std::uint64_t tsc = 0;
  ++tsc;
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(tsc & 0xFFFFFFFFull), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(tsc >> 32), 4);
  return {};
}

ExecutionResult handle_code_CPUID(ExecutionContext& ctx) {
  const auto leaf = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::EAX));
  const auto subleaf = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::ECX));
  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
  if (leaf == 0) {
    eax = 1;
    ebx = 0x756E6547;
    edx = 0x49656E69;
    ecx = 0x6C65746E;
  } else if (leaf == 1) {
    eax = 0x000306A9;
    ebx = 0x00100800;
    ecx = 0;
    edx = 0x0183F3FF;
  } else if (leaf == 7 && subleaf == 0) {
    eax = 0;
    ebx = 0;
    ecx = 0;
    edx = 0;
  }
  detail::write_register(ctx.state, iced_x86::Register::EAX, eax, 4);
  detail::write_register(ctx.state, iced_x86::Register::EBX, ebx, 4);
  detail::write_register(ctx.state, iced_x86::Register::ECX, ecx, 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, edx, 4);
  return {};
}

ExecutionResult handle_code_ENDBR64(ExecutionContext&) {
  return {};
}

ExecutionResult handle_code_STMXCSR(ExecutionContext& ctx) {
  if (!detail::write_operand(ctx, 0, ctx.state.mxcsr, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_LDMXCSR(ExecutionContext& ctx) {
  bool ok = false;
  const auto value = detail::read_operand(ctx, 0, 4, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if ((value >> 16) != 0) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  ctx.state.mxcsr = static_cast<std::uint32_t>(value);
  return {};
}

ExecutionResult handle_code_STMXCSR_M32(ExecutionContext& ctx) {
  if (!detail::write_operand(ctx, 0, ctx.state.mxcsr, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_LDMXCSR_M32(ExecutionContext& ctx) {
  bool ok = false;
  const auto value = detail::read_operand(ctx, 0, 4, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if ((value >> 16) != 0) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  ctx.state.mxcsr = static_cast<std::uint32_t>(value);
  return {};
}

ExecutionResult handle_code_XGETBV(ExecutionContext& ctx) {
  const auto xcr = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::ECX));
  const std::uint64_t value = detail::read_xcr(ctx.state, xcr);
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(value), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(value >> 32), 4);
  return {};
}

ExecutionResult handle_code_RDSSPD_R32(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), 0, 4);
  return {};
}

ExecutionResult handle_code_RDSSPQ_R64(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), 0, 8);
  return {};
}


ExecutionResult handle_code_RDFSBASE_R64(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), ctx.state.fs_base, 8);
  return {};
}

ExecutionResult handle_code_RDGSBASE_R64(ExecutionContext& ctx) {
  detail::write_register(ctx.state, ctx.instr.op_register(0), ctx.state.gs_base, 8);
  return {};
}

ExecutionResult handle_code_WRFSBASE_R64(ExecutionContext& ctx) {
  ctx.state.fs_base = detail::read_register(ctx.state, ctx.instr.op_register(0));
  return {};
}

ExecutionResult handle_code_WRGSBASE_R64(ExecutionContext& ctx) {
  ctx.state.gs_base = detail::read_register(ctx.state, ctx.instr.op_register(0));
  return {};
}

}  // namespace seven::handlers


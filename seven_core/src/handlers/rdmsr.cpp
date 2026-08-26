#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

[[nodiscard]] bool cpl_is_zero(const CpuState& state) {
  return (state.sreg[1] & 0x3u) == 0;
}

[[nodiscard]] ExecutionResult gp_fault(ExecutionContext& ctx) {
  return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, ctx.state.rip, 0}, ctx.instr.code()};
}

}  // namespace

// RDMSR/RDMSRLIST are CPL0-only on real hardware by default (#GP(0) otherwise) -- any privilege
// level could otherwise read an arbitrary MSR another privilege level stored sensitive state in.
ExecutionResult handle_code_RDMSR(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto ecx = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::ECX));
  const auto value = detail::read_msr(ctx.state, ecx);
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(value), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(value >> 32), 4);
  return {};
}

ExecutionResult handle_code_RDMSRLIST(ExecutionContext& ctx) {
  if (!cpl_is_zero(ctx.state)) {
    return gp_fault(ctx);
  }
  const auto rdi = detail::read_register(ctx.state, iced_x86::Register::RDI);
  const auto rsi = detail::read_register(ctx.state, iced_x86::Register::RSI);
  if (((rdi | rsi) & 0x7ull) != 0ull) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }

  auto rcx = detail::read_register(ctx.state, iced_x86::Register::RCX);
  for (std::uint64_t bit = 0; bit < 64; ++bit) {
    const auto mask = 1ull << bit;
    if ((rcx & mask) == 0) {
      continue;
    }
    std::uint64_t msr_index = 0;
    const auto msr_address = rsi + (bit * 8);
    if (!ctx.memory.read(msr_address, &msr_index, 8)) {
      return detail::memory_fault(ctx, msr_address);
    }

    const auto value = detail::read_msr(ctx.state, static_cast<std::uint32_t>(msr_index));
    const auto value_address = rdi + (bit * 8);
    if (!ctx.memory.write(value_address, &value, 8)) {
      return detail::memory_fault(ctx, value_address);
    }

    rcx &= ~mask;
    detail::write_register(ctx.state, iced_x86::Register::RCX, rcx);
  }
  return {};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_WRMSR(ExecutionContext& ctx) {
  const auto ecx = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::ECX));
  const std::uint64_t eax = detail::read_register(ctx.state, iced_x86::Register::EAX);
  const std::uint64_t edx = detail::read_register(ctx.state, iced_x86::Register::EDX);
  detail::write_msr(ctx.state, ecx, (edx << 32) | (eax & 0xFFFFFFFFull));
  return {};
}

ExecutionResult handle_code_WRMSRNS(ExecutionContext& ctx) {
  bool msr_ok = false;
  const auto msr_index = static_cast<std::uint32_t>(detail::read_operand(ctx, 0, 4, &msr_ok));
  if (!msr_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const std::uint64_t eax = detail::read_register(ctx.state, iced_x86::Register::EAX);
  const std::uint64_t edx = detail::read_register(ctx.state, iced_x86::Register::EDX);
  detail::write_msr(ctx.state, msr_index, (edx << 32) | (eax & 0xFFFFFFFFull));
  return {};
}

ExecutionResult handle_code_WRMSRLIST(ExecutionContext& ctx) {
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
    std::uint64_t value = 0;
    const auto msr_address = rsi + (bit * 8);
    const auto value_address = rdi + (bit * 8);
    if (!ctx.memory.read(msr_address, &msr_index, 8)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, msr_address, 0}, ctx.instr.code()};
    }
    if (!ctx.memory.read(value_address, &value, 8)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, value_address, 0}, ctx.instr.code()};
    }
    detail::write_msr(ctx.state, static_cast<std::uint32_t>(msr_index), value);

    rcx &= ~mask;
    detail::write_register(ctx.state, iced_x86::Register::RCX, rcx);
  }
  return {};
}

}  // namespace seven::handlers


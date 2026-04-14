#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_RDMSR(ExecutionContext& ctx) {
  const auto ecx = static_cast<std::uint32_t>(detail::read_register(ctx.state, iced_x86::Register::ECX));
  const auto value = detail::read_msr(ctx.state, ecx);
  detail::write_register(ctx.state, iced_x86::Register::EAX, static_cast<std::uint32_t>(value), 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, static_cast<std::uint32_t>(value >> 32), 4);
  return {};
}

ExecutionResult handle_code_RDMSRLIST(ExecutionContext& ctx) {
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
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, msr_address, 0}, ctx.instr.code()};
    }

    const auto value = detail::read_msr(ctx.state, static_cast<std::uint32_t>(msr_index));
    const auto value_address = rdi + (bit * 8);
    if (!ctx.memory.write(value_address, &value, 8)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, value_address, 0}, ctx.instr.code()};
    }

    rcx &= ~mask;
    detail::write_register(ctx.state, iced_x86::Register::RCX, rcx);
  }
  return {};
}

}  // namespace seven::handlers


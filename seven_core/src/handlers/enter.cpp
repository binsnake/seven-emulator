#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_ENTERW(ExecutionContext& ctx) {
  const auto frame_size = ctx.instr.immediate16();
  const auto nesting = ctx.instr.immediate8();
  (void)nesting;

  const auto old_bp = detail::read_register(ctx.state, iced_x86::Register::BP);
  auto sp = detail::read_register(ctx.state, iced_x86::Register::SP);
  sp = (sp - 2ull) & 0xFFFFull;
  if (!ctx.memory.write(sp, &old_bp, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, sp, 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::SP, sp, 2);
  detail::write_register(ctx.state, iced_x86::Register::BP, sp, 2);
  sp = (sp - frame_size) & 0xFFFFull;
  detail::write_register(ctx.state, iced_x86::Register::SP, sp, 2);
  return {};
}

ExecutionResult handle_code_ENTERD(ExecutionContext& ctx) {
  const auto frame_size = ctx.instr.immediate16();
  const auto nesting = ctx.instr.immediate8();
  (void)nesting;

  const auto old_bp = detail::read_register(ctx.state, iced_x86::Register::EBP);
  auto sp = detail::read_register(ctx.state, iced_x86::Register::ESP);
  sp -= 4;
  if (!ctx.memory.write(sp, &old_bp, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, sp, 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::EBP, sp, 4);
  detail::write_register(ctx.state, iced_x86::Register::ESP, sp, 4);
  sp = (sp - frame_size) & 0xFFFFFFFFull;
  detail::write_register(ctx.state, iced_x86::Register::ESP, sp, 4);
  return {};
}

ExecutionResult handle_code_ENTERQ(ExecutionContext& ctx) {
  const auto frame_size = ctx.instr.immediate16();
  const auto nesting = ctx.instr.immediate8();
  (void)nesting;

  const auto old_bp = detail::read_register(ctx.state, iced_x86::Register::RBP);
  auto sp = detail::read_register(ctx.state, iced_x86::Register::RSP);
  sp -= 8;
  if (!ctx.memory.write(sp, &old_bp, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, sp, 0}, ctx.instr.code()};
  }
  detail::write_register(ctx.state, iced_x86::Register::RBP, sp, 8);
  detail::write_register(ctx.state, iced_x86::Register::RSP, sp, 8);
  sp -= frame_size;
  detail::write_register(ctx.state, iced_x86::Register::RSP, sp, 8);
  return {};
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


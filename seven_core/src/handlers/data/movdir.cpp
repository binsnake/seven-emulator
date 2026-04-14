#include <array>

#include "seven/handler_helpers.hpp"

namespace seven::handlers {
namespace {

ExecutionResult movdir64b_impl(ExecutionContext& ctx, std::size_t dest_width) {
  std::array<std::uint8_t, 64> bytes{};
  const auto src_address = detail::memory_address(ctx);
  if (!ctx.memory.read(src_address, bytes.data(), bytes.size())) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, src_address, 0}, ctx.instr.code()};
  }

  const auto dest_reg = ctx.instr.op_register(0);
  const auto dest_mask = mask_for_width(dest_width);
  const auto dest_address = detail::read_register(ctx.state, dest_reg) & dest_mask;
  if (!ctx.memory.write(dest_address, bytes.data(), bytes.size())) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, dest_address, 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_MOVNTI_M32_R32(ExecutionContext& ctx) {
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

ExecutionResult handle_code_MOVNTI_M64_R64(ExecutionContext& ctx) {
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

ExecutionResult handle_code_MOVDIRI_M32_R32(ExecutionContext& ctx) {
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

ExecutionResult handle_code_MOVDIRI_M64_R64(ExecutionContext& ctx) {
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

ExecutionResult handle_code_MOVDIR64B_R16_M512(ExecutionContext& ctx) {
  return movdir64b_impl(ctx, 2);
}

ExecutionResult handle_code_MOVDIR64B_R32_M512(ExecutionContext& ctx) {
  return movdir64b_impl(ctx, 4);
}

ExecutionResult handle_code_MOVDIR64B_R64_M512(ExecutionContext& ctx) {
  return movdir64b_impl(ctx, 8);
}

}  // namespace seven::handlers


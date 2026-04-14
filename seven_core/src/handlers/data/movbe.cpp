#include <bit>

#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_MOVBE_R16_M16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto swapped = std::byteswap(static_cast<std::uint16_t>(value));
  detail::write_register(ctx.state, ctx.instr.op_register(0), swapped, 2);
  return {};
}

ExecutionResult handle_code_MOVBE_R32_M32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto swapped = std::byteswap(static_cast<std::uint32_t>(value));
  detail::write_register(ctx.state, ctx.instr.op_register(0), swapped, 4);
  return {};
}

ExecutionResult handle_code_MOVBE_R64_M64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto swapped = std::byteswap(static_cast<std::uint64_t>(value));
  detail::write_register(ctx.state, ctx.instr.op_register(0), swapped, 8);
  return {};
}

ExecutionResult handle_code_MOVBE_M16_R16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto swapped = std::byteswap(static_cast<std::uint16_t>(value));
  if (!detail::write_operand(ctx, 0, swapped, 2)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOVBE_M32_R32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto swapped = std::byteswap(static_cast<std::uint32_t>(value));
  if (!detail::write_operand(ctx, 0, swapped, 4)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_MOVBE_M64_R64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto value = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto swapped = std::byteswap(static_cast<std::uint64_t>(value));
  if (!detail::write_operand(ctx, 0, swapped, 8)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers


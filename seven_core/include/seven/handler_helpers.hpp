#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <iced_x86/code.hpp>
#include <iced_x86/instruction.hpp>

#include "seven/executor.hpp"

namespace seven {
namespace detail {

void set_flag(std::uint64_t& rflags, std::uint64_t bit, bool value);
std::uint64_t read_msr(CpuState& state, std::uint32_t index);
void write_msr(CpuState& state, std::uint32_t index, std::uint64_t value);
std::uint64_t read_xcr(CpuState& state, std::uint32_t index);
void write_xcr(CpuState& state, std::uint32_t index, std::uint64_t value);
std::uint64_t truncate(std::uint64_t value, std::size_t width);
bool even_parity(std::uint8_t value);
std::uint64_t sign_extend(std::uint64_t value, std::size_t width);
ExecutionResult memory_fault(ExecutionContext& ctx, std::uint64_t address);
ExecutionResult read_memory_checked(ExecutionContext& ctx, std::uint64_t address, void* value, std::size_t width);
ExecutionResult write_memory_checked(ExecutionContext& ctx, std::uint64_t address, const void* value, std::size_t width);
ExecutionResult read_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, std::uint64_t& value);
ExecutionResult write_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::uint64_t value, std::size_t width);
ExecutionResult dispatch_interrupt(ExecutionContext& ctx, std::uint8_t vector, std::uint64_t return_rip,
                                   std::optional<std::uint32_t> error_code = std::nullopt,
                                   bool push_rf_in_frame = false);

std::uint64_t debug_data_breakpoint_hits(CpuState& state, std::uint64_t address, std::size_t size, bool is_read, bool is_write) noexcept;

[[nodiscard]] inline bool note_debug_break(ExecutionContext& ctx, std::uint64_t hit_bits, bool will_continue) noexcept {
  if (hit_bits == 0) return false;
  ctx.debug_hit_bits |= hit_bits;
  ctx.push_rf_for_debug = ctx.push_rf_for_debug || will_continue;
  if (will_continue) ctx.control_flow_taken = true;
  return true;
}

std::size_t register_width(iced_x86::Register reg);
std::size_t operand_width(const iced_x86::Instruction& instr, std::uint32_t operand_index);
std::uint64_t memory_address(ExecutionContext& ctx);
std::uint64_t read_register(CpuState& state, iced_x86::Register reg);
void write_register(CpuState& state, iced_x86::Register reg, std::uint64_t value, std::size_t width_override = 0);
std::uint64_t immediate_value(const iced_x86::Instruction& instr, std::uint32_t operand_index);
std::uint64_t read_operand(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, bool* ok = nullptr);
bool write_operand(ExecutionContext& ctx, std::uint32_t operand_index, std::uint64_t value, std::size_t width);

template <typename T>
[[nodiscard]] inline ExecutionResult read_memory_checked(ExecutionContext& ctx, std::uint64_t address, T& value) {
  return read_memory_checked(ctx, address, &value, sizeof(T));
}

template <typename T>
[[nodiscard]] inline ExecutionResult write_memory_checked(ExecutionContext& ctx, std::uint64_t address, const T& value) {
  return write_memory_checked(ctx, address, &value, sizeof(T));
}

template <typename T>
[[nodiscard]] inline ExecutionResult read_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, T& value) {
  std::uint64_t raw = 0;
  const auto result = read_operand_checked(ctx, operand_index, width, raw);
  if (result.ok()) {
    value = static_cast<T>(raw);
  }
  return result;
}

template <typename T>
[[nodiscard]] inline ExecutionResult write_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, const T value, std::size_t width) {
  return write_operand_checked(ctx, operand_index, static_cast<std::uint64_t>(value), width);
}

void set_logic_flags(CpuState& state, std::uint64_t value, std::size_t width);
void set_add_flags(CpuState& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result, std::size_t width, bool carry_in = false);
void set_sub_flags(CpuState& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result, std::size_t width, bool borrow_in = false);
void set_multiply_flags(CpuState& state, std::uint64_t value, std::size_t width, bool overflow);
ExecutionResult divide_fault(ExecutionContext& ctx);

[[nodiscard]] inline ExecutionResult read_divisor_checked(ExecutionContext& ctx, std::size_t width, std::uint64_t& divisor) {
  bool ok = false;
  divisor = read_operand(ctx, 0, width, &ok);
  if (!ok) return memory_fault(ctx, memory_address(ctx));
  if (divisor == 0) return divide_fault(ctx);
  return {};
}


}  // namespace detail
}  // namespace seven


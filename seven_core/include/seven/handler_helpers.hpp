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

// Flag-write elimination for the block liveness pass: set_flag() drops a write to any bit in the
// current mask. A bit may only be marked dead when nothing between it and the next write to the same
// bit can observe it. Scoped to the six ALU status bits, and defaults to masking nothing.

// AVX-512 writemasking exists in exactly one place, simd_int.cpp's apply_masked_lanes. Elsewhere an
// EVEX form naming a mask register would write every lane regardless, which is silently wrong rather
// than missing, so handlers that cannot honour a mask ask this first and stop cleanly. K0 and NONE
// both mean no masking, the common case for these shared helpers.
[[nodiscard]] inline bool has_active_opmask(const iced_x86::Instruction& instr) noexcept {
  const auto opmask = instr.op_mask();
  return opmask != iced_x86::Register::NONE && opmask != iced_x86::Register::K0;
}

[[nodiscard]] inline ExecutionResult unsupported_opmask(ExecutionContext& ctx) {
  return {StopReason::unsupported_instruction, 0,
          ExceptionInfo{StopReason::unsupported_instruction, ctx.state.rip, 0}, ctx.instr.code()};
}

void set_dead_flags_mask(std::uint64_t mask) noexcept;
[[nodiscard]] std::uint64_t dead_flags_mask() noexcept;
std::uint64_t read_msr(CpuState& state, std::uint32_t index);
// False when the write would have to create a new entry and the guest has already created more
// than any real machine implements -- the caller raises #GP, as hardware does for an MSR it has no
// storage for.
[[nodiscard]] bool write_msr(CpuState& state, std::uint32_t index, std::uint64_t value);
std::uint64_t read_xcr(CpuState& state, std::uint32_t index);
void write_xcr(CpuState& state, std::uint32_t index, std::uint64_t value);
std::uint64_t truncate(std::uint64_t value, std::size_t width);
bool even_parity(std::uint8_t value);
std::uint64_t sign_extend(std::uint64_t value, std::size_t width);
ExecutionResult memory_fault(ExecutionContext& ctx, std::uint64_t address);
// Legacy (non-VEX) SSE/SSE2/SSE3/SSE4 instructions with a full 128-bit (m128) memory operand
// require it 16-byte aligned, raising #GP(0) if not -- unlike VEX-encoded forms of the same
// operation, which never impose this. Returns the #GP ExecutionResult when operand_index names a
// misaligned memory operand, or std::nullopt when there's nothing to fault on (operand is a
// register, or the memory operand is already aligned) -- callers still do their own read/write
// afterward, this only gates entry to it.
[[nodiscard]] std::optional<ExecutionResult> require_aligned_memory_operand(ExecutionContext& ctx,
                                                                             std::uint32_t operand_index,
                                                                             std::uint64_t alignment_mask);
ExecutionResult read_memory_checked(ExecutionContext& ctx, std::uint64_t address, void* value, std::size_t width);
ExecutionResult write_memory_checked(ExecutionContext& ctx, std::uint64_t address, const void* value, std::size_t width);
ExecutionResult read_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, std::uint64_t& value);
ExecutionResult write_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::uint64_t value, std::size_t width);
// software_interrupt selects the INT n / INT3 / INTO privilege rule: those three check the gate's
// DPL against the current CPL, while a hardware-generated exception ignores it. Defaults to the
// exception behaviour so the executor's own fault paths need no change.
ExecutionResult dispatch_interrupt(ExecutionContext& ctx, std::uint8_t vector, std::uint64_t return_rip,
                                   std::optional<std::uint32_t> error_code = std::nullopt,
                                   bool push_rf_in_frame = false,
                                   bool software_interrupt = false);

std::uint64_t debug_data_breakpoint_hits(CpuState& state, std::uint64_t address, std::size_t size, bool is_read, bool is_write) noexcept;

// The stack slot a push writes and a pop reads is implicit, so it never appears in the
// instruction's operand list and the executor's generic watchpoint sweep cannot see it. Report it
// from the handler instead, or a guest evades a data breakpoint just by pointing rsp at the watched
// address and pushing.
inline void note_stack_access(ExecutionContext& ctx, std::uint64_t slot, std::size_t width, bool is_write) {
  if (ctx.state.dr[7] == 0) {
    return;
  }
  ctx.debug_hit_bits |= debug_data_breakpoint_hits(ctx.state, slot, width, !is_write, is_write);
}

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
// Same, with an extra offset folded in before the address wraps and before the segment base is
// added. The bit-string instructions displace their operand by whole elements and need both.
std::uint64_t memory_address_with_displacement(ExecutionContext& ctx, std::uint64_t extra);
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

// How wide a string instruction's index and count registers are. Only an address-size prefix
// changes it, and iced records that in the operand kind rather than the Code (STOSB is one Code for
// all three sizes), so this is the only place to read it from. Masking on entry and after every step
// is enough, since writing back to a 64-bit register zero-extends anyway.

// A string instruction's DS:rSI source takes a segment override; its ES:rDI destination does not,
// which is why only the source-reading handlers ask. The override applies to the linear address
// only, so rSI still steps and wraps within the address size on its own.
[[nodiscard]] inline std::uint64_t string_source_segment_base(const CpuState& state,
                                                              const iced_x86::Instruction& instr) noexcept {
  if (instr.segment_prefix() == iced_x86::Register::FS) return state.fs_base;
  if (instr.segment_prefix() == iced_x86::Register::GS) return state.gs_base;
  return 0;
}

[[nodiscard]] inline std::uint64_t string_address_mask(const iced_x86::Instruction& instr) noexcept {
  for (std::uint32_t i = 0; i < instr.op_count(); ++i) {
    switch (instr.op_kind(i)) {
      case iced_x86::OpKind::MEMORY_SEG_SI:
      case iced_x86::OpKind::MEMORY_ESDI:
        return 0xFFFFull;
      case iced_x86::OpKind::MEMORY_SEG_ESI:
      case iced_x86::OpKind::MEMORY_ESEDI:
        return 0xFFFFFFFFull;
      default:
        break;
    }
  }
  return ~0ull;
}

}  // namespace detail
}  // namespace seven


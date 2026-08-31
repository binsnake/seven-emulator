#pragma once

#include <cstdint>
#include <span>

#include <iced_x86/instruction.hpp>

namespace seven {

// One instruction in a lifted block. `instr` must already be decoded, and `dead_flags_mask` is
// filled in with the flags this instruction writes that nothing later in the block reads.
struct FlagLivenessInstr {
  const iced_x86::Instruction* instr = nullptr;
  std::uint64_t dead_flags_mask = 0;
};

// Backward liveness over the six ALU status flags as independent variables. `insts` is in program
// order and every dead_flags_mask is overwritten in place. This only skips a provably unobservable
// write, but the caller decides whether skipping is sound at all: instruction, code and access hooks
// can read rflags mid-block, so it must not run while any are registered.
void compute_flag_liveness(std::span<FlagLivenessInstr> insts) noexcept;

// Whether this decoded instruction can fault (memory operand, DIV/IDIV, or implicit stack access
// like CALL/RET/PUSH/POP) -- exposed for external callers with the same "can this actually fault"
// question, not just compute_flag_liveness's own internal use.
[[nodiscard]] bool can_fault(const iced_x86::Instruction& instr) noexcept;

}  // namespace seven

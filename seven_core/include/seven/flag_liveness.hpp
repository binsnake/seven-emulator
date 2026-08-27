#pragma once

#include <cstdint>
#include <span>

#include <iced_x86/instruction.hpp>

namespace seven {

// One instruction within a basic block being lifted for the backward flag-liveness pass. `instr`
// must already be decoded; `dead_flags_mask` is an out-param the pass fills in with the subset of
// kAluStatusFlagsMask that this instruction writes but nothing later in the block (or anything
// past the block, since live-out-of-block is always conservatively "everything live") ever reads.
struct FlagLivenessInstr {
  const iced_x86::Instruction* instr = nullptr;
  std::uint64_t dead_flags_mask = 0;
};

// Standard backward liveness dataflow over the six ALU status flags (CF/PF/AF/ZF/SF/OF), treating
// each bit as an independent "variable". `insts` is in program order (the order the block will
// execute in), and every entry's `dead_flags_mask` is overwritten in place.
//
// This only ever *skips* a write that is provably unobservable -- it never changes what value a
// flag ends up holding when that value is actually read. Callers are responsible for only invoking
// this when it's sound to skip writes at all: instruction/code hooks and memory-access hooks can
// observe rflags at points iced's per-instruction read/write tables don't know about (mid-block,
// via ExecutionContext), so this must not be used while any of those are registered. Trap and
// (address-)execution hooks don't have that problem, and don't need to gate this.
void compute_flag_liveness(std::span<FlagLivenessInstr> insts) noexcept;

// Whether this decoded instruction can fault (memory operand, DIV/IDIV, or implicit stack access
// like CALL/RET/PUSH/POP) -- exposed for external callers with the same "can this actually fault"
// question, not just compute_flag_liveness's own internal use.
[[nodiscard]] bool can_fault(const iced_x86::Instruction& instr) noexcept;

}  // namespace seven

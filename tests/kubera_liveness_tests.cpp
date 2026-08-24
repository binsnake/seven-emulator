#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "seven/handler_helpers.hpp"

// Regression tests for the flag-liveness masking soundness fix: masking a flag write because a
// later instruction in the same lifted block "covers" it is only safe if that later instruction
// is GUARANTEED to actually run before anything external can observe rflags. These tests target
// the two runtime conditions (beyond the caller simply not continuing, which
// KuberaScalar.ImulAndMulFlagSemantics / IncAndShiftEdgeCases already cover for a bare step()
// call) that can interrupt a block mid-way even when driven through Executor::run(): a fault on
// a later instruction, and the single-step trap flag. See Flag Liveness Execution Model
// Problem.md.

namespace {

constexpr std::uint64_t kBase = 0x1000;
constexpr std::uint64_t kIdtBase = 0x8000;
constexpr std::uint64_t kDbHandler = 0x9000;
constexpr std::uint64_t kStackTop = 0x5000;

void write_bytes(seven::Memory& memory, std::uint64_t base, const std::vector<std::uint8_t>& bytes) {
  memory.map(base, bytes.size() + 0x100);
  (void)memory.write(base, bytes.data(), bytes.size());
}

void write_idt_gate64(seven::Memory& memory,
                      std::uint64_t idt_base,
                      std::uint8_t vector,
                      std::uint16_t selector,
                      std::uint64_t offset,
                      std::uint8_t type_attr = 0x8F) {
  const std::uint64_t lo =
      (offset & 0xFFFFull) |
      (static_cast<std::uint64_t>(selector) << 16) |
      (static_cast<std::uint64_t>(type_attr) << 40) |
      (((offset >> 16) & 0xFFFFull) << 48);
  const std::uint64_t hi = (offset >> 32) & 0xFFFFFFFFull;
  const auto entry = idt_base + static_cast<std::uint64_t>(vector) * 16ull;
  (void)memory.write(entry, &lo, sizeof(lo));
  (void)memory.write(entry + 8, &hi, sizeof(hi));
}

}  // namespace

TEST(KuberaLiveness, MaskedWriteSurvivesMidBlockFaultViaRun) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  memory.map(0x4000, 0x2000);
  // add rax, rbx (register-only, unconditional all-flags write) ; add [rdx], rcx (memory
  // operand -- can page-fault). Without the can_fault liveness barrier, the first add's flags
  // would be masked as "covered" by the second -- but the second never completes its write
  // because it faults reading [rdx]. rax/rbx chosen so CF and ZF are unambiguous (unsigned
  // wraparound to zero).
  write_bytes(memory, kBase, {0x48, 0x01, 0xD8, 0x48, 0x01, 0x0A});
  state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;  // rax
  state.gpr[3] = 1ull;                   // rbx
  state.gpr[2] = 0xDEAD0000ull;          // rdx -- deliberately unmapped

  const auto result = executor.run(state, memory, 100);
  ASSERT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[0], 0u);
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
}

TEST(KuberaLiveness, TrapFlagDisablesMaskingViaRun) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  state.rflags = 0x202 | seven::kFlagTF;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  // add rax, rbx ; add rcx, rdx -- both register-only, unconditional all-flags writes. The
  // second would "cover" the first's write under plain liveness (no fault risk here), but TF
  // means the CPU model must stop after exactly the first instruction, same as single-stepping
  // -- the cover never runs before the debug interrupt fires and exposes rflags. The debug
  // handler halts (rather than iretq-ing back) so execution stops cleanly right after the first
  // instruction's interrupt instead of resuming into the second add (and then TF-trapping again,
  // and so on) -- this test only needs to observe the state at that first stop.
  write_bytes(memory, kBase, {0x48, 0x01, 0xD8, 0x48, 0x01, 0xD1});
  const std::uint8_t hlt[] = {0xF4};
  (void)memory.write(kDbHandler, hlt, sizeof(hlt));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;  // rax
  state.gpr[3] = 1ull;                   // rbx

  // max_instructions >= kMaxBlockLiftLength(64) so run()'s budget-headroom gate would otherwise
  // allow masking at the very first dispatch -- the trap flag is the only thing that should stop
  // it here.
  const auto result = executor.run(state, memory, 64);
  ASSERT_EQ(result.reason, seven::StopReason::halted);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
}

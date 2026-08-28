#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "seven/handler_helpers.hpp"

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

void write_idt_gate32(seven::Memory& memory,
                      std::uint64_t idt_base,
                      std::uint8_t vector,
                      std::uint16_t selector,
                      std::uint32_t offset,
                      std::uint8_t type_attr = 0x8F) {
  const std::uint64_t desc =
      (static_cast<std::uint64_t>(offset & 0xFFFFu)) |
      (static_cast<std::uint64_t>(selector) << 16) |
      (static_cast<std::uint64_t>(type_attr) << 40) |
      ((static_cast<std::uint64_t>((offset >> 16) & 0xFFFFu)) << 48);
  (void)memory.write(idt_base + static_cast<std::uint64_t>(vector) * 8ull, &desc, sizeof(desc));
}

}  // namespace

TEST(KuberaDebug, ExecuteBreakpointAndRfSuppression) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x90, 0x90});
  const std::uint8_t iretq[] = {0x48, 0xCF};
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kBase;
  state.dr[7] = 0x1;

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_NE(state.dr[6] & 0x1u, 0u);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase);

  state.rflags |= seven::kFlagRF;
  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 1);
}

TEST(KuberaDebug, MovSsSuppressesExecuteBreakpointAndDelaysTf) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[0] = 0x2Bull;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  state.rflags = 0x202 | seven::kFlagTF;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x8E, 0xD0, 0x90, 0x90});  // mov ss, ax; nop; nop
  const std::uint8_t iretq[] = {0x48, 0xCF};
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kBase + 2;
  state.dr[7] = 0x1;

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.sreg[2], 0x2Bu);
  EXPECT_EQ(state.dr[6] & (1ull << 14), 0u);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_NE(state.dr[6] & (1ull << 14), 0u);

  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 3);

  const auto r4 = executor.step(state, memory);
  ASSERT_EQ(r4.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
}

TEST(KuberaDebug, RepMovsbDataBreakpointResumesWithRf) {
  constexpr std::uint64_t kSrc = 0x3000;
  constexpr std::uint64_t kDst = 0x3800;
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[6] = kSrc;
  state.gpr[7] = kDst;
  state.gpr[1] = 3;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  memory.map(kSrc, 0x1000);
  memory.map(kDst, 0x1000);
  write_bytes(memory, kBase, {0xF3, 0xA4});  // rep movsb
  const std::uint8_t src_bytes[] = {0x11, 0x22, 0x33};
  (void)memory.write(kSrc, src_bytes, sizeof(src_bytes));
  const std::uint8_t iretq[] = {0x48, 0xCF};
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kDst + 1;
  state.dr[7] = 0x1 | (0x1ull << 16);

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_EQ(state.gpr[6], kSrc + 2);
  EXPECT_EQ(state.gpr[7], kDst + 2);
  EXPECT_EQ(state.gpr[1], 1u);
  std::uint8_t out0 = 0, out1 = 0, out2 = 0;
  (void)memory.read(kDst, &out0, sizeof(out0));
  (void)memory.read(kDst + 1, &out1, sizeof(out1));
  (void)memory.read(kDst + 2, &out2, sizeof(out2));
  EXPECT_EQ(out0, 0x11);
  EXPECT_EQ(out1, 0x22);
  EXPECT_EQ(out2, 0x00);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase);
  EXPECT_NE(state.rflags & seven::kFlagRF, 0u);

  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.gpr[1], 0u);
  (void)memory.read(kDst + 2, &out2, sizeof(out2));
  EXPECT_EQ(out2, 0x33);
}

TEST(KuberaDebug, RepMovsbExecuteBreakpointRestarts) {
  constexpr std::uint64_t kSrc = 0x3200;
  constexpr std::uint64_t kDst = 0x3A00;
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[6] = kSrc;
  state.gpr[7] = kDst;
  state.gpr[1] = 3;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  memory.map(kSrc, 0x1000);
  memory.map(kDst, 0x1000);
  write_bytes(memory, kBase, {0xF3, 0xA4});  // rep movsb
  const std::uint8_t src_bytes[] = {0x41, 0x42, 0x43};
  (void)memory.write(kSrc, src_bytes, sizeof(src_bytes));
  const std::uint8_t iretq[] = {0x48, 0xCF};
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kBase;
  state.dr[7] = 0x1;

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  std::uint8_t out0 = 0, out1 = 0, out2 = 0;
  (void)memory.read(kDst, &out0, sizeof(out0));
  (void)memory.read(kDst + 1, &out1, sizeof(out1));
  (void)memory.read(kDst + 2, &out2, sizeof(out2));
  EXPECT_EQ(out0, 0x00);
  EXPECT_EQ(out1, 0x00);
  EXPECT_EQ(out2, 0x00);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase);
  state.rflags |= seven::kFlagRF;

  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.gpr[1], 0u);
  (void)memory.read(kDst, &out0, sizeof(out0));
  (void)memory.read(kDst + 1, &out1, sizeof(out1));
  (void)memory.read(kDst + 2, &out2, sizeof(out2));
  EXPECT_EQ(out0, 0x41);
  EXPECT_EQ(out1, 0x42);
  EXPECT_EQ(out2, 0x43);
}

TEST(KuberaDebug, RepMovsbYieldsAfterIterationCapAndResumesCorrectly) {
  // A guest-controlled RCX far above the per-call iteration cap must not run the whole
  // rep movsb inside one uninterruptible step() call -- see kMaxRepIterationsPerCall's
  // rationale comment in movs.cpp. This proves the yield-and-resume mechanism itself,
  // with no debug registers involved at all.
  constexpr std::uint64_t kSrc = 0x20000;
  constexpr std::uint64_t kDst = 0x30000;
  constexpr std::uint64_t kIterationCap = 4096;
  constexpr std::uint64_t kCount = kIterationCap * 2 + 100;  // needs exactly 3 step() calls
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  memory.map(kSrc, 0x10000);
  memory.map(kDst, 0x10000);
  write_bytes(memory, kBase, {0xF3, 0xA4});  // rep movsb

  std::vector<std::uint8_t> src_bytes(kCount);
  for (std::uint64_t i = 0; i < kCount; ++i) {
    src_bytes[i] = static_cast<std::uint8_t>(i);
  }
  (void)memory.write(kSrc, src_bytes.data(), src_bytes.size());

  state.gpr[6] = kSrc;
  state.gpr[7] = kDst;
  state.gpr[1] = kCount;

  int step_count = 0;
  while (state.rip == kBase) {
    const auto result = executor.step(state, memory);
    ASSERT_EQ(result.reason, seven::StopReason::none);
    ++step_count;
    ASSERT_LT(step_count, 10) << "should not need more than a handful of step() calls";
  }

  EXPECT_EQ(step_count, 3) << "a count of cap*2 + remainder must take exactly 3 step() calls";
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.gpr[1], 0u);
  EXPECT_EQ(state.gpr[6], kSrc + kCount);
  EXPECT_EQ(state.gpr[7], kDst + kCount);

  std::vector<std::uint8_t> dst_bytes(kCount);
  ASSERT_TRUE(memory.read(kDst, dst_bytes.data(), dst_bytes.size()));
  EXPECT_EQ(dst_bytes, src_bytes);
}

TEST(KuberaDebug, SingleSteppingARepTrapsAfterEachIterationNotAfterTheWholeLoop) {
  // A rep is interruptible between iterations, so a guest with TF set gets one #DB per iteration
  // and can watch RCX/RSI/RDI count down. Running the whole loop inside a single step() and
  // trapping once at the end is directly observable, and seven-fuzzer's string family saw it
  // against hardware on its very first run.
  constexpr std::uint64_t kSrc = 0x20000;
  constexpr std::uint64_t kDst = 0x30000;
  constexpr std::uint64_t kCount = 3;
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
  memory.map(kSrc, 0x1000);
  memory.map(kDst, 0x1000);
  memory.map(0x4000, 0x2000);  // the #DB frame's stack
  write_bytes(memory, kBase, {0xF3, 0xA4});  // rep movsb
  const std::uint8_t iretq[] = {0x48, 0xCF};
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);

  state.gpr[6] = kSrc;
  state.gpr[7] = kDst;
  state.gpr[1] = kCount;

  for (std::uint64_t done = 1; done <= kCount; ++done) {
    const auto trapped = executor.step(state, memory);
    ASSERT_EQ(trapped.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler) << "iteration " << done << " should have trapped";
    EXPECT_EQ(state.gpr[1], kCount - done) << "only one iteration may retire per step";
    EXPECT_EQ(state.gpr[6], kSrc + done);
    EXPECT_EQ(state.gpr[7], kDst + done);

    const auto returned = executor.step(state, memory);  // iretq
    ASSERT_EQ(returned.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, done == kCount ? kBase + 2 : kBase);
  }
}

TEST(KuberaDebug, PopSsSuppressesExecuteBreakpointAndDelaysTf) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::compat32;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.sreg[1] = 0x23;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  state.rflags = 0x202 | seven::kFlagTF;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x17, 0x90, 0x90});  // pop ss; nop; nop
  const std::uint8_t iretd[] = {0xCF};
  const std::uint32_t new_ss = 0x2B;
  (void)memory.write(kDbHandler, iretd, sizeof(iretd));
  (void)memory.write(kStackTop, &new_ss, sizeof(new_ss));
  write_idt_gate32(memory, kIdtBase, 1, 0x23, static_cast<std::uint32_t>(kDbHandler));
  state.dr[0] = kBase + 1;
  state.dr[7] = 0x1;

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip & 0xFFFFFFFFu, kBase + 1);
  EXPECT_EQ(state.sreg[2], 0x2Bu);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_NE(state.dr[6] & (1ull << 14), 0u);

  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip & 0xFFFFFFFFu, kBase + 2);

  const auto r4 = executor.step(state, memory);
  ASSERT_EQ(r4.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
}

TEST(KuberaDebug, DelayedDataAndRfSemantics) {
  {
    constexpr std::uint64_t kDataAddr = 0x3400;
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    state.gpr[0] = 0x2Bull;
    state.gpr[3] = kDataAddr;
    state.sreg[1] = 0x33;
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    state.rflags = 0x202 | seven::kFlagTF;
    memory.map(kIdtBase, 0x1000);
    memory.map(kDbHandler, 0x1000);
    memory.map(0x4000, 0x2000);
    memory.map(kDataAddr, 0x1000);
    write_bytes(memory, kBase, {0x8E, 0xD0, 0xC6, 0x03, 0x7F, 0x90});  // mov ss, ax; mov byte ptr [rbx], 0x7f; nop
    const std::uint8_t iretq[] = {0x48, 0xCF};
    (void)memory.write(kDbHandler, iretq, sizeof(iretq));
    write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
    state.dr[0] = kDataAddr;
    state.dr[7] = 0x1 | (0x1ull << 16);

    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
    const auto r2 = executor.step(state, memory);
    ASSERT_EQ(r2.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler);
    EXPECT_NE(state.dr[6] & (1ull << 14), 0u);
    EXPECT_NE(state.dr[6] & 0x1u, 0u);
    std::uint8_t stored = 0;
    (void)memory.read(kDataAddr, &stored, sizeof(stored));
    EXPECT_EQ(stored, 0x7F);
  }

  {
    constexpr std::uint64_t kDataAddr = 0x3700;
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    state.gpr[0] = kDataAddr;
    state.sreg[1] = 0x33;
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    state.rflags = 0x202 | seven::kFlagRF;
    memory.map(kIdtBase, 0x1000);
    memory.map(kDbHandler, 0x1000);
    memory.map(0x4000, 0x2000);
    memory.map(kDataAddr, 0x1000);
    write_bytes(memory, kBase, {0xC6, 0x00, 0x7F});  // mov byte ptr [rax], 0x7f
    const std::uint8_t iretq[] = {0x48, 0xCF};
    (void)memory.write(kDbHandler, iretq, sizeof(iretq));
    write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
    state.dr[0] = kDataAddr;
    state.dr[7] = 0x1 | (0x1ull << 16);

    const auto r1 = executor.step(state, memory);
    ASSERT_EQ(r1.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler);
    EXPECT_NE(state.dr[6] & 0x1u, 0u);
    std::uint8_t stored = 0;
    (void)memory.read(kDataAddr, &stored, sizeof(stored));
    EXPECT_EQ(stored, 0x7F);
  }
}

TEST(KuberaDebug, SsLoadInstructionDataBreakpointsAreDelayed) {
  {
    constexpr std::uint64_t kSelAddr = 0x3800;
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    state.gpr[3] = kSelAddr;
    state.sreg[1] = 0x33;
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    memory.map(kIdtBase, 0x1000);
    memory.map(kDbHandler, 0x1000);
    memory.map(0x4000, 0x2000);
    memory.map(kSelAddr, 0x1000);
    write_bytes(memory, kBase, {0x8E, 0x13, 0x90, 0x90});  // mov ss, word ptr [rbx]; nop; nop
    const std::uint8_t iretq[] = {0x48, 0xCF};
    const std::uint16_t new_ss = 0x2B;
    (void)memory.write(kDbHandler, iretq, sizeof(iretq));
    (void)memory.write(kSelAddr, &new_ss, sizeof(new_ss));
    write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
    state.dr[0] = kSelAddr;
    state.dr[7] = 0x1 | (0x3ull << 16);

    const auto r1 = executor.step(state, memory);
    ASSERT_EQ(r1.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kBase + 2);
    EXPECT_EQ(state.dr[6] & 0x1u, 0u);
    EXPECT_EQ(state.sreg[2], 0x2Bu);

    const auto r2 = executor.step(state, memory);
    ASSERT_EQ(r2.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler);
    EXPECT_NE(state.dr[6] & 0x1u, 0u);
  }

  {
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::compat32;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    state.sreg[1] = 0x23;
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    memory.map(kIdtBase, 0x1000);
    memory.map(kDbHandler, 0x1000);
    memory.map(0x4000, 0x2000);
    write_bytes(memory, kBase, {0x17, 0x90, 0x90});  // pop ss; nop; nop
    const std::uint8_t iretd[] = {0xCF};
    const std::uint32_t new_ss = 0x2B;
    (void)memory.write(kDbHandler, iretd, sizeof(iretd));
    (void)memory.write(kStackTop, &new_ss, sizeof(new_ss));
    write_idt_gate32(memory, kIdtBase, 1, 0x23, static_cast<std::uint32_t>(kDbHandler));
    state.dr[0] = kStackTop;
    state.dr[7] = 0x1 | (0x3ull << 16);

    const auto r1 = executor.step(state, memory);
    ASSERT_EQ(r1.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip & 0xFFFFFFFFu, kBase + 1);
    EXPECT_EQ(state.dr[6] & 0x1u, 0u);
    EXPECT_EQ(state.sreg[2], 0x2Bu);

    const auto r2 = executor.step(state, memory);
    ASSERT_EQ(r2.reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler);
    EXPECT_NE(state.dr[6] & 0x1u, 0u);
  }
}

TEST(KuberaDebug, DelayedSsLoadDataBreakpointIsLostOnFault) {
  constexpr std::uint64_t kSelAddr = 0x3900;
  constexpr std::uint64_t kFaultAddr = 0x6000;
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[3] = kSelAddr;
  state.gpr[1] = kFaultAddr;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  memory.map(kSelAddr, 0x1000);
  write_bytes(memory, kBase, {0x8E, 0x13, 0x8A, 0x01, 0x90});  // mov ss,[rbx]; mov al,[rcx]; nop
  const std::uint8_t iretq[] = {0x48, 0xCF};
  const std::uint16_t new_ss = 0x2B;
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  (void)memory.write(kSelAddr, &new_ss, sizeof(new_ss));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kSelAddr;
  state.dr[7] = 0x1 | (0x3ull << 16);

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::page_fault);
  ASSERT_TRUE(r2.exception.has_value());
  EXPECT_EQ(r2.exception->address, kFaultAddr);

  memory.map(kFaultAddr, 0x1000);
  const std::uint8_t value = 0x7F;
  (void)memory.write(kFaultAddr, &value, sizeof(value));
  state.dr[6] = 0;

  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 4);
  EXPECT_EQ(state.dr[6] & 0x1u, 0u);

  const auto r4 = executor.step(state, memory);
  ASSERT_EQ(r4.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 5);
  EXPECT_EQ(state.dr[6] & 0x1u, 0u);
}


TEST(KuberaDebug, DelayedDebugCombinesMultipleDr6Reasons) {
  constexpr std::uint64_t kSelAddr = 0x3A00;
  constexpr std::uint64_t kDataAddr = 0x3B00;
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[0] = kDataAddr;
  state.gpr[3] = kSelAddr;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  state.rflags = 0x202 | seven::kFlagTF;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  memory.map(kSelAddr, 0x1000);
  memory.map(kDataAddr, 0x1000);
  write_bytes(memory, kBase, {0x8E, 0x13, 0xC6, 0x00, 0x7F});  // mov ss,[rbx]; mov byte ptr [rax],0x7f
  const std::uint8_t iretq[] = {0x48, 0xCF};
  const std::uint16_t new_ss = 0x2B;
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  (void)memory.write(kSelAddr, &new_ss, sizeof(new_ss));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kSelAddr;
  state.dr[1] = kDataAddr;
  state.dr[7] = 0x1 | (0x1ull << 2) | (0x7ull << 16) | (0x1ull << 20);
  
  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.dr[6] & 0x3u, 0u);
  
  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_NE(state.dr[6] & 0x1u, 0u);
  EXPECT_NE(state.dr[6] & 0x2u, 0u);
  EXPECT_NE(state.dr[6] & (1ull << 14), 0u);
  std::uint8_t stored = 0;
  (void)memory.read(kDataAddr, &stored, sizeof(stored));
  EXPECT_EQ(stored, 0x7F);
}

TEST(KuberaDebug, DelayedDebugAfterMovSsEntersSoftwareInterruptHandlerFirst) {
  constexpr std::uint64_t kSelAddr = 0x3C00;
  constexpr std::uint64_t kInt3Handler = 0xA000;
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[3] = kSelAddr;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  state.rflags = 0x202 | seven::kFlagTF;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(kInt3Handler, 0x1000);
  memory.map(0x4000, 0x2000);
  memory.map(kSelAddr, 0x1000);
  write_bytes(memory, kBase, {0x8E, 0x13, 0xCC, 0x90});  // mov ss,[rbx]; int3; nop
  const std::uint8_t iretq[] = {0x48, 0xCF};
  const std::uint16_t new_ss = 0x2B;
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  (void)memory.write(kInt3Handler, iretq, sizeof(iretq));
  (void)memory.write(kSelAddr, &new_ss, sizeof(new_ss));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  // DPL 3, since int3 is issued from CPL 3 here and a DPL 0 gate would (correctly) #GP.
  write_idt_gate64(memory, kIdtBase, 3, 0x33, kInt3Handler, 0xEF);
  state.dr[0] = kSelAddr;
  state.dr[7] = 0x1 | (0x7ull << 16);
  
  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.dr[6] & 0x1u, 0u);
  
  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kDbHandler);
  EXPECT_NE(state.dr[6] & 0x1u, 0u);
  EXPECT_NE(state.dr[6] & (1ull << 14), 0u);
  
  std::uint64_t db_return_rip = 0;
  std::uint64_t int3_return_rip = 0;
  (void)memory.read(state.gpr[4], &db_return_rip, sizeof(db_return_rip));
  (void)memory.read(state.gpr[4] + 18, &int3_return_rip, sizeof(int3_return_rip));
  EXPECT_EQ(db_return_rip, kInt3Handler);
  EXPECT_EQ(int3_return_rip, kBase + 3);
  
  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kInt3Handler);
  
  const auto r4 = executor.step(state, memory);
  ASSERT_EQ(r4.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 3);
}

// Regression test for an integer-overflow bug in debug_data_breakpoint_hits' range-overlap check
// (handler_helpers.cpp): it computed the access's one-past-the-end address as address + size with
// no wraparound guard. address is fully guest-controlled (any register value), and size can be up
// to 64 bytes for a real instruction (a ZMM memory operand) -- picking an address near the top of
// the 64-bit address space wraps that sum back down past zero, which the old non-wrapping interval
// test then read as "no overlap" even when the access's real (wrapping) span does touch a watched
// byte range. That silently let a guest evade a hardware data breakpoint just by choosing where it
// writes, no different in spirit from choosing where NOT to write to dodge a watchpoint -- except
// this let it write exactly where a debugger/anti-tamper tool was watching and still not be seen.
TEST(KuberaDebug, DataBreakpointStillFiresOnWraparoundAccess) {
  seven::CpuState state{};
  // DR0 watches [0x8, 0x10): L0 enabled (bit 0), R/W0 = read-or-write (0b11 at bits 16-17), LEN0 =
  // 8 bytes (0b10 at bits 18-19).
  state.dr[0] = 0x8;
  state.dr[7] = 0x1 | (0x3ull << 16) | (0x2ull << 18);

  // A 64-byte access starting 16 bytes before the top of the address space wraps to cover
  // [0xFFFFFFFFFFFFFFF0, 0xFFFFFFFFFFFFFFFF] then [0x0, 0x2F] -- genuinely touching DR0's [0x8,0x10)
  // watched range through the wrap, which the buggy non-wrapping check would have missed entirely.
  constexpr std::uint64_t kWraparoundAddress = 0xFFFF'FFFF'FFFF'FFF0ull;
  constexpr std::size_t kAccessSize = 64;

  const auto hit_bits = seven::detail::debug_data_breakpoint_hits(state, kWraparoundAddress, kAccessSize,
                                                                    /*is_read=*/false, /*is_write=*/true);
  EXPECT_NE(hit_bits & 0x1ull, 0u) << "DR0's watchpoint must fire for a wrapping access that genuinely touches it";
}

TEST(KuberaDebug, StackPushAndPopTriggerDataBreakpoints) {
  // The stack slot a PUSH writes and a POP reads is an implicit operand, so it never shows up in
  // the instruction's iced operand list. Watchpoints still have to see it, otherwise a guest can
  // park RSP on a watched address and write through it without ever tripping DR0-DR3.
  {
    constexpr std::uint64_t kWatched = 0x3400;
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[0] = 0xDEADBEEFull;
    state.gpr[4] = kWatched + 8;
    state.sreg[1] = 0x33;
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    state.rflags = 0x202;
    memory.map(kIdtBase, 0x1000);
    memory.map(kDbHandler, 0x1000);
    memory.map(kWatched, 0x1000);
    write_bytes(memory, kBase, {0x50});  // push rax
    const std::uint8_t iretq[] = {0x48, 0xCF};
    (void)memory.write(kDbHandler, iretq, sizeof(iretq));
    write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
    // DR0 watches [kWatched, kWatched+8) for writes, LEN0 = 8 bytes.
    state.dr[0] = kWatched;
    state.dr[7] = 0x1 | (0x1ull << 16) | (0x2ull << 18);

    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler) << "push onto a watched stack slot must raise #DB";
    EXPECT_NE(state.dr[6] & 0x1ull, 0u);
    std::uint64_t stored = 0;
    (void)memory.read(kWatched, &stored, sizeof(stored));
    EXPECT_EQ(stored, 0xDEADBEEFull);
  }

  {
    constexpr std::uint64_t kWatched = 0x3400;
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[4] = kWatched;
    state.sreg[1] = 0x33;
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    state.rflags = 0x202;
    memory.map(kIdtBase, 0x1000);
    memory.map(kDbHandler, 0x1000);
    memory.map(kWatched, 0x1000);
    write_bytes(memory, kBase, {0x58});  // pop rax
    const std::uint64_t secret = 0x1122334455667788ull;
    (void)memory.write(kWatched, &secret, sizeof(secret));
    const std::uint8_t iretq[] = {0x48, 0xCF};
    (void)memory.write(kDbHandler, iretq, sizeof(iretq));
    write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
    // DR0 watches [kWatched, kWatched+8) for reads or writes, LEN0 = 8 bytes.
    state.dr[0] = kWatched;
    state.dr[7] = 0x1 | (0x3ull << 16) | (0x2ull << 18);

    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kDbHandler) << "pop off a watched stack slot must raise #DB";
    EXPECT_NE(state.dr[6] & 0x1ull, 0u);
    EXPECT_EQ(state.gpr[0], secret);
  }
}

// DR6 and DR7 have bits that read as 1 on every real x86-64 regardless of what is written, and the
// address registers hold linear addresses, so a non-canonical one faults the way any other
// non-canonical linear address does.
TEST(KuberaDebug, DebugRegisterReservedBitsAndCanonicalChecks) {
  seven::Memory memory{};
  seven::Executor executor{};
  memory.map(0x1000, 0x1000);

  const auto run_one = [&](const std::vector<std::uint8_t>& code, seven::CpuState& state) {
    EXPECT_TRUE(memory.write(0x1000, code.data(), code.size()));
    state.mode = seven::ExecutionMode::long64;
    state.rip = 0x1000;
    return executor.step(state, memory);
  };

  {
    seven::CpuState state{};
    EXPECT_EQ(state.dr[6] & 0xFFFF0FF0ull, 0xFFFF0FF0ull) << "DR6 reset value";
    EXPECT_EQ(state.dr[7] & 0x400ull, 0x400ull) << "DR7 reset value";
  }
  {
    // mov dr7, rax with every architectural bit set -- the reserved zeros must not stick.
    seven::CpuState state{};
    state.gpr[0] = 0xFFFFFFFFull;
    ASSERT_EQ(run_one({0x0F, 0x23, 0xF8}, state).reason, seven::StopReason::none);
    EXPECT_EQ(state.dr[7] & 0x400ull, 0x400ull) << "DR7 bit 10 must stay set";
    EXPECT_EQ(state.dr[7] & 0xFFFFFFFF00000000ull, 0u) << "DR7 is 32 bits of architectural state";
    EXPECT_EQ(state.dr[7] & 0xC000ull, 0u) << "DR7 bits 14-15 are reserved zero";
  }
  {
    // Setting anything above bit 31 is #GP in 64-bit mode.
    seven::CpuState state{};
    state.gpr[0] = 1ull << 32;
    EXPECT_EQ(run_one({0x0F, 0x23, 0xF8}, state).reason, seven::StopReason::general_protection);
  }
  {
    // mov dr6, rax likewise.
    seven::CpuState state{};
    state.gpr[0] = 0;
    ASSERT_EQ(run_one({0x0F, 0x23, 0xF0}, state).reason, seven::StopReason::none);
    EXPECT_EQ(state.dr[6] & 0xFFFF0FF0ull, 0xFFFF0FF0ull) << "DR6 reserved ones must read back set";
  }
  {
    // mov dr0, rax with a non-canonical address.
    seven::CpuState state{};
    state.gpr[0] = 0x0000'8000'0000'0000ull;
    EXPECT_EQ(run_one({0x0F, 0x23, 0xC0}, state).reason, seven::StopReason::general_protection);
  }
}

// The interrupt frame is three separate stores and any of them can fault. Committing rsp store by
// store left the emulator half a frame in when one did: rsp lowered by the slots that landed, the
// rest unwritten, and nothing in the returned fault to say how far it got. An embedder that maps
// the page and restarts then builds a second frame underneath the first.
TEST(KuberaDebug, AnInterruptFrameThatFaultsPartwayLeavesRspWhereItStarted) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  constexpr std::uint64_t kFramePage = 0x40000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(kFramePage, 0x1000);
  write_bytes(memory, kBase, {0xCC});  // int3
  write_idt_gate64(memory, kIdtBase, 3, 0x33, kDbHandler, 0xEF);  // DPL 3, reachable from ring 3

  // rflags (8) lands at 0x40002 and cs (2) at 0x40000; rip (8) would land below the page.
  const std::uint64_t entry_sp = kFramePage + 10;
  state.gpr[4] = entry_sp;

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[4], entry_sp);
  EXPECT_EQ(state.rip, kBase) << "the interrupt never took";
}

// The stack slot a pushfq writes and a popfq reads is implicit: it is not an operand, so the
// executor's own watchpoint sweep over the operand list cannot see it. push/pop report theirs from
// the handler for exactly this reason and these two did not, so a guest evaded a write watchpoint
// just by pointing rsp at the watched address and pushing the flags onto it.
TEST(KuberaDebug, PushfqCannotStepPastADataWatchpointOnItsOwnStackSlot) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x33;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x9C});  // pushfq
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.gpr[4] = kStackTop;
  state.dr[0] = kStackTop - 8;      // the slot pushfq is about to write
  state.dr[7] = 0x1 | (0x1 << 16) | (0x2 << 18);  // DR0 enabled, write, 8 bytes

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_NE(state.dr[6] & 0x1u, 0u) << "the push slipped past the watchpoint";
  EXPECT_EQ(state.rip, kDbHandler);
}

// An instruction breakpoint is a fault, not a trap: the frame carries the breakpointed
// instruction's own rip, so the handler's iret lands right back on it. RF in the saved rflags image
// is the only thing that stops the same breakpoint firing again on that return, and the frame went
// out without it. Any guest that arms DR0 on an instruction and whose #DB handler simply irets
// never retires that instruction -- it re-delivers on every step, forever.
TEST(KuberaDebug, ExecuteBreakpointFrameSetsRfSoTheIretMakesProgress) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.sreg[1] = 0x33;
  state.rflags = 0x202;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x90, 0x90});
  const std::uint8_t iretq[] = {0x48, 0xCF};
  (void)memory.write(kDbHandler, iretq, sizeof(iretq));
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kBase;
  state.dr[7] = 0x1;

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  ASSERT_EQ(state.rip, kDbHandler);

  // rip (8) then cs (2) sit below it, so the pushed rflags image is at rsp + 10.
  std::uint64_t frame_rflags = 0;
  ASSERT_TRUE(memory.read(state.gpr[4] + 10, &frame_rflags, sizeof(frame_rflags)));
  EXPECT_NE(frame_rflags & seven::kFlagRF, 0u) << "the frame's rflags image has no RF";

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  ASSERT_EQ(state.rip, kBase);
  EXPECT_NE(state.rflags & seven::kFlagRF, 0u) << "the iret restored no RF";

  const auto r3 = executor.step(state, memory);
  ASSERT_EQ(r3.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 1) << "the breakpoint re-delivered instead of retiring the nop";
}

// RF and the SS-load shadow are consumed at the top of the instruction, but an instruction that
// faults never ran. A fault hook asking for a retry re-entered with both already spent, so the
// second attempt fired the very execute breakpoint they exist to suppress -- and the guest's #DB
// handler then irets back onto the same faulting instruction.
TEST(KuberaDebug, AFaultRetryKeepsRfSuppressingTheExecuteBreakpoint) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  constexpr std::uint64_t kDataPage = 0x70000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[3] = kDataPage;
  state.sreg[1] = 0x33;
  state.rflags = 0x202 | seven::kFlagRF;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x8B, 0x03});  // mov eax, [rbx]
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kBase;
  state.dr[7] = 0x1;

  int hook_calls = 0;
  const auto id = executor.add_fault_hook([&](const seven::FaultHookEvent& event) {
    if (++hook_calls > 1) {
      return seven::FaultHookAction::stop;
    }
    event.memory.map(kDataPage, 0x1000);
    return seven::FaultHookAction::restart_instruction;
  });
  ASSERT_NE(id, 0u);

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(hook_calls, 1);
  EXPECT_EQ(state.rip, kBase + 2) << "the retry lost RF and took the execute breakpoint";
  EXPECT_EQ(state.dr[6] & 0x1u, 0u);
}

TEST(KuberaDebug, AFaultRetryKeepsTheSsShadowSuppressingTheExecuteBreakpoint) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  constexpr std::uint64_t kDataPage = 0x70000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.gpr[0] = 0x2Bull;
  state.gpr[3] = kDataPage;
  state.sreg[1] = 0x33;
  state.rflags = 0x202;
  state.idtr.base = kIdtBase;
  state.idtr.limit = 0x1000 - 1;
  memory.map(kIdtBase, 0x1000);
  memory.map(kDbHandler, 0x1000);
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x8E, 0xD0, 0x8B, 0x03});  // mov ss, ax; mov eax, [rbx]
  write_idt_gate64(memory, kIdtBase, 1, 0x33, kDbHandler);
  state.dr[0] = kBase + 2;
  state.dr[7] = 0x1;

  int hook_calls = 0;
  const auto id = executor.add_fault_hook([&](const seven::FaultHookEvent& event) {
    if (++hook_calls > 1) {
      return seven::FaultHookAction::stop;
    }
    event.memory.map(kDataPage, 0x1000);
    return seven::FaultHookAction::restart_instruction;
  });
  ASSERT_NE(id, 0u);

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  ASSERT_EQ(state.rip, kBase + 2);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(hook_calls, 1);
  EXPECT_EQ(state.rip, kBase + 4) << "the retry lost the SS shadow and took the execute breakpoint";
  EXPECT_EQ(state.dr[6] & 0x1u, 0u);
}

// stop_reason_counts_ is sized from the last enumerator, and every fault path indexes it with the
// reason it is about to report. Two of those reasons come straight out of an embedder hook's
// ExecutionResult, so a value outside the enum walked off the end of the vector.
TEST(KuberaDebug, AnOutOfRangeStopReasonFromAHookStaysInsideTheCountsVector) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.rflags = 0x202;
  memory.map(0x4000, 0x2000);
  write_bytes(memory, kBase, {0x90});

  const auto id = executor.add_instruction_hook([&](seven::InstructionHookContext&) {
    seven::InstructionHookResult hook_result{};
    hook_result.action = seven::InstructionHookAction::stop;
    hook_result.stop_result = seven::ExecutionResult{static_cast<seven::StopReason>(200), 0, std::nullopt, std::nullopt};
    return hook_result;
  });
  ASSERT_NE(id, 0u);

  const auto before = executor.stop_reason_counts()[0];
  const auto result = executor.step(state, memory);
  EXPECT_EQ(static_cast<unsigned>(result.reason), 200u);
  EXPECT_EQ(executor.stop_reason_counts()[0], before + 1) << "the count landed outside the vector";
}

// BOUND out of range is #BR, vector 5. It was reporting a plain #GP, so a guest with a real #BR
// handler never reached it. Only compat32 and real16 can decode the instruction at all: in long
// mode 62 is the EVEX prefix.
TEST(KuberaDebug, BoundOutOfRangeDispatchesTheBoundRangeVector) {
  constexpr std::uint64_t kData = 0x6000;
  constexpr std::uint32_t kBrHandler = 0x9000;

  const auto set_up = [&](seven::CpuState& state, seven::Memory& memory) {
    state.mode = seven::ExecutionMode::compat32;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    state.gpr[0] = 100;      // eax, well outside the bounds below
    state.gpr[3] = kData;    // ebx
    write_bytes(memory, kBase, {0x62, 0x03});  // bound eax, [ebx]
    memory.map(kData, 0x1000);
    memory.map(kStackTop - 0x1000, 0x1000);
    const std::uint32_t bounds[2] = {10u, 20u};
    (void)memory.write(kData, bounds, sizeof(bounds));
  };

  {
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    set_up(state, memory);
    state.idtr.base = kIdtBase;
    state.idtr.limit = 0x1000 - 1;
    memory.map(kIdtBase, 0x1000);
    memory.map(kBrHandler, 0x1000);
    write_idt_gate32(memory, kIdtBase, 5, 0x08, kBrHandler, 0x8E);

    const auto result = executor.step(state, memory);
    EXPECT_EQ(result.reason, seven::StopReason::none) << "the #BR should have been delivered";
    EXPECT_EQ(state.rip, kBrHandler) << "and it should land in the guest's own handler";
  }

  {
    // With no IDT there is nothing to deliver to, and dispatch_interrupt's own fallback is the #GP
    // this used to report unconditionally. That keeps the no-IDT guest seeing what it always saw.
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    set_up(state, memory);
    EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::general_protection);
  }

  {
    // In range is still unremarkable.
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    set_up(state, memory);
    state.gpr[0] = 15;
    EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
    EXPECT_EQ(state.rip, kBase + 2);
  }
}

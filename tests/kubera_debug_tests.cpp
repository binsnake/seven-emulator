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
  write_idt_gate64(memory, kIdtBase, 3, 0x33, kInt3Handler);
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
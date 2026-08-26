#include <gtest/gtest.h>

#include "kubera_test_support.hpp"

namespace {

using namespace kubera::test;

TEST(KuberaScalar, ImulAndMulFlagSemantics) {
  run_single(seven::parse_hex_bytes("6B C1 05"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFF'FFFF'1234'5678ull;
               state.gpr[1] = 7;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 35u);
               EXPECT_EQ(state.rip, kBase + 3);
             });

  run_single(seven::parse_hex_bytes("F6 E3"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x81;
               state.gpr[3] = 0x02;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFFFu, 0x0102u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_NE(state.rflags & seven::kFlagOF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagAF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagZF, 0u);
               EXPECT_EQ(state.rflags & seven::kFlagSF, 0u);
               EXPECT_EQ((state.rflags & seven::kFlagPF) != 0u, seven::even_parity(0x02));
             });
}

TEST(KuberaScalar, DivIdivAndDivideErrorSemantics) {
  run_single(seven::parse_hex_bytes("F6 F3"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x0014;
               state.gpr[3] = 0x03;
               state.rflags = 0x202 | seven::kFlagCF | seven::kFlagPF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFFFu, 0x0206u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_NE(state.rflags & seven::kFlagPF, 0u);
             });

  run_single(seven::parse_hex_bytes("F6 FB"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFA;
               state.gpr[3] = 0xFD;
               state.rflags = 0x202 | seven::kFlagOF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFFFu, 0x0002u);
               EXPECT_NE(state.rflags & seven::kFlagOF, 0u);
             });

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  const auto bytes = seven::parse_hex_bytes("F6 F3");
  write_bytes(memory, kBase, bytes);
  state.gpr[0] = 0x0014;
  state.gpr[3] = 0x00;
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::divide_error);
}

TEST(KuberaScalar, IncAndShiftEdgeCases) {
  run_single(seven::parse_hex_bytes("FF C0"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFF'FFFF'FFFF'FFFFull;
               state.rflags = 0x202 | seven::kFlagCF;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0], 0u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
               EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
             });

  run_single(seven::parse_hex_bytes("C0 E0 08"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x81;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFu, 0u);
               EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
             });
}

TEST(KuberaScalar, WritablePagesAreReadable) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = 0x4000;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
  const auto code = seven::parse_hex_bytes("8B 03");
  const std::uint32_t value = 0x12345678u;
  write_bytes(memory, kBase, code);
  EXPECT_TRUE(memory.write_unchecked(0x4000, value));
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(static_cast<std::uint32_t>(state.gpr[0]), value);
}


TEST(KuberaScalar, RtlInterlockedFlushSListSequenceMatchesExpected) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kBase + 0x800 - 8;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, seven::kMemoryPermissionAll);
  const auto code = seven::parse_hex_bytes("0F 0D 09 53 4C 8B D1 48 8B 01 48 8B 51 08 33 C9 48 8B D8 66 33 DB F0 49 0F C7 0A 75 F1 48 8B C2 24 F0 5B C3");
  write_bytes(memory, kBase, code);
  const std::uint64_t return_target = kBase + static_cast<std::uint64_t>(code.size());
  EXPECT_TRUE(memory.write_unchecked(state.gpr[4], return_target));
  state.gpr[0] = 0x1111222233334444ull;
  state.gpr[1] = 0x4000ull;
  state.gpr[2] = 0xAAAABBBBCCCCDDD5ull;
  state.gpr[3] = 0xDEADBEEFCAFEBABEu;
  const std::uint64_t old_low = 0x1234567890ABCDEFull;
  const std::uint64_t old_high = 0xAAAABBBBCCCCDDD5ull;
  EXPECT_TRUE(memory.write_unchecked(0x4000, old_low));
  EXPECT_TRUE(memory.write_unchecked(0x4008, old_high));
  bool reached_return = false;
  for (int i = 0; i < 16; ++i) {
    const auto result = executor.step(state, memory);
    ASSERT_EQ(result.reason, seven::StopReason::none);
    if (state.rip == return_target) {
      reached_return = true;
      break;
    }
  }
  ASSERT_TRUE(reached_return);
  const std::uint64_t expected_low = old_low & ~0xFFFFull;
  std::uint64_t actual_low = 0;
  std::uint64_t actual_high = 0;
  EXPECT_TRUE(memory.read_unchecked(0x4000, actual_low));
  EXPECT_TRUE(memory.read_unchecked(0x4008, actual_high));
  EXPECT_EQ(actual_low, expected_low);
  EXPECT_EQ(actual_high, 0u);
  EXPECT_EQ(state.gpr[0], old_high & ~0xFull);
  EXPECT_EQ(state.gpr[3], 0xDEADBEEFCAFEBABEu);
}


TEST(KuberaScalar, RtlInterlockedPushListSListSequenceMatchesExpected) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kBase + 0x800 - 8;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, seven::kMemoryPermissionAll);
  memory.map(0x5000, 0x1000, seven::kMemoryPermissionAll);
  const auto code = seven::parse_hex_bytes("0F 0D 09 53 48 8B 01 4C 8B D9 48 8B CA 49 8B 53 08 4C 8B D2 41 80 E2 F0 4D 89 10 48 8D 98 00 00 02 00 66 42 8D 1C 08 F0 49 0F C7 0B 75 E3 49 8B C2 5B C3");
  write_bytes(memory, kBase, code);
  const std::uint64_t return_target = kBase + static_cast<std::uint64_t>(code.size());
  EXPECT_TRUE(memory.write_unchecked(state.gpr[4], return_target));
  const std::uint64_t old_low = 0x0000000000010002ull;
  const std::uint64_t old_head = 0x1234567890ABCDEFull;
  const std::uint64_t new_head = 0x5000ull;
  const std::uint64_t out_ptr = 0x5010ull;
  EXPECT_TRUE(memory.write_unchecked(0x4000, old_low));
  EXPECT_TRUE(memory.write_unchecked(0x4008, old_head));
  state.gpr[1] = 0x4000ull;
  state.gpr[2] = new_head;
  state.gpr[8] = out_ptr;
  state.gpr[9] = 3;
  bool reached_return = false;
  for (int i = 0; i < 16; ++i) {
    const auto result = executor.step(state, memory);
    ASSERT_EQ(result.reason, seven::StopReason::none);
    if (state.rip == return_target) {
      reached_return = true;
      break;
    }
  }
  ASSERT_TRUE(reached_return);
  std::uint64_t actual_low = 0;
  std::uint64_t actual_head = 0;
  std::uint64_t old_head_out = 0;
  EXPECT_TRUE(memory.read_unchecked(0x4000, actual_low));
  EXPECT_TRUE(memory.read_unchecked(0x4008, actual_head));
  EXPECT_TRUE(memory.read_unchecked(out_ptr, old_head_out));
  EXPECT_EQ(actual_head, new_head);
  EXPECT_EQ(actual_low, ((old_low + 0x20000ull) & ~0xFFFFull) | ((old_low + 3) & 0xFFFFull));
  EXPECT_EQ(old_head_out, old_head & ~0xFull);
  EXPECT_EQ(state.gpr[0], old_head & ~0xFull);
}


TEST(KuberaScalar, OneOperandImul64HandlesNegativeHighHalf) {
  run_single(seven::parse_hex_bytes("48 F7 E9"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = static_cast<std::uint64_t>(-5ll);
               state.gpr[1] = 7;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(static_cast<std::int64_t>(state.gpr[0]), -35ll);
               EXPECT_EQ(static_cast<std::int64_t>(state.gpr[2]), -1ll);
             });
}


TEST(KuberaScalar, JneRel32BranchesWhenZfClear) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 85 04 00 00 00 90 90 90 90"));
  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 10);
}


TEST(KuberaScalar, PrefetchDoesNotFaultUnmappedMemory) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[0] = 0xDEAD'BEEF0000ull;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 18 08"));
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 3);
}


TEST(KuberaScalar, FxrstorReadsWritablePages) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[0] = 0x4000;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F AE 08"));
  std::array<std::byte, 512> fx{};
  EXPECT_TRUE(memory.write_unchecked(0x4000, fx.data(), fx.size()));
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
}


TEST(KuberaScalar, BtsWithHugeBitIndexFaultsGeneralProtectionNotPageFault) {
  // bts [rdi], r8 -- the register-sourced bit index extends the effective address by
  // (index >> 6) * 8 per the SDM, so a huge index (as seven-fuzzer's random register generation
  // produces routinely) pushes the real access into non-canonical territory. Real hardware raises
  // #GP(0) for that, checked before any page walk -- confirmed via a standalone probe. Before this
  // fix, seven_core had no canonical-address check anywhere, so this came back as a generic
  // page_fault instead, a real divergence from hardware seven-fuzzer's BT-family findings caught.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("4C 0F AB 07"));
  state.gpr[7] = 0x4000;                       // rdi: canonical, mapped base
  state.gpr[8] = 0xA94F08C0EA937D27ull;         // r8: bit index, huge/negative-looking

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, BtsWithSmallBitIndexStillWorksAfterCanonicalCheck) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(0x4000, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("4C 0F AB 07"));
  state.gpr[7] = 0x4000;
  state.gpr[8] = 5;  // r8: small in-range bit index

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  std::uint8_t byte0 = 0;
  ASSERT_TRUE(memory.read(0x4000, &byte0, 1));
  EXPECT_EQ(byte0, 0x20);
}

TEST(KuberaScalar, FnstenvToUnmappedMemoryFaultsInsteadOfReportingSuccess) {
  // store_x87_env() returned void and threw away all seven of its ctx.memory.write() results, and
  // both FNSTENV handlers ignored it and returned {} unconditionally -- so an FNSTENV aimed at
  // unmapped memory silently "succeeded" with nothing written. The ~400-site memory_fault() sweep
  // missed this family because the helpers return void, so there was no `if (!...read/write)` shape
  // to match on.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 33"));  // fnstenv [rbx]
  state.gpr[3] = 0x50000;                                        // rbx: canonical but never mapped

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
}

TEST(KuberaScalar, FnsaveToUnmappedMemoryFaultsAndLeavesFpuStateIntact) {
  // Same void-helper gap as above, plus a second-order bug: fsave() ran x87_reset() unconditionally
  // after the (silently failed) stores, so a save that never landed still wiped the guest's FPU
  // stack. Validating the whole 160-byte footprint up front -- the way fxsave already did -- makes
  // the fault happen before any state is touched, which is also what real hardware does.
  // Doubles as the regression test for the orphaned-handler half of this fix: DD /6 decodes to
  // FNSAVE_M108BYTE, which was never registered, so this instruction previously stopped as
  // unsupported_instruction and fsave() was unreachable dead code.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("DD 33"));  // fnsave [rbx]
  state.gpr[3] = 0x50000;                                        // rbx: canonical but never mapped
  ASSERT_TRUE(state.x87_push(seven::X87Scalar(1)));
  const auto top_before = state.get_x87_top();

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.get_x87_top(), top_before);
  EXPECT_FALSE(state.x87_is_empty(0));
}

TEST(KuberaScalar, FnstenvToNonCanonicalAddressFaultsGeneralProtection) {
  // Once the writes are actually checked, the canonical-address check every other handler gets via
  // detail::memory_fault() applies here too -- a non-canonical destination is #GP, not #PF.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 33"));  // fnstenv [rbx]
  state.gpr[3] = 0x8000'1234'5678'9AB0ull;                       // rbx: non-canonical

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, NonCanonicalAddressFaultsGeneralProtectionNotPageFault) {
  // mov rax, [rbx] with rbx pointing at a non-canonical address. Every handler's memory-fault
  // path now funnels through detail::memory_fault(), which checks canonicality before treating
  // the access as an ordinary page_fault -- previously only the ~169 sites that already called
  // memory_fault() got this for free, while ~400 other sites (including this MOV handler)
  // constructed the page_fault ExecutionResult inline and skipped the check entirely.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 8B 03"));
  state.gpr[3] = 0x8000'1234'5678'9ABCull;  // rbx: non-canonical

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, MovCrRejectsReservedControlRegisterIndex) {
  // Real hardware only defines CR0, CR2, CR3, CR4, and (64-bit mode) CR8 as valid MOV CR
  // operands -- CR1, CR5-CR7, CR9-CR15 #UD. state.cr is sized to cover all 16 possible encodings
  // so an unfiltered access would never go out of bounds, but it would let a guest treat a
  // register that doesn't exist in silicon at all as ordinary read/write storage.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 20 C8"));  // mov eax, cr1

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::invalid_opcode);
}

TEST(KuberaScalar, MovCrRequiresCplZero) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CS selector with RPL 3 -- CPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 20 C0"));  // mov eax, cr0 -- valid register, wrong CPL

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, MovCrRoundTripsArchitecturallyValidRegister) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("44 0F 22 C0 44 0F 20 C3"));  // mov cr8,eax ; mov ebx,cr8
  state.gpr[0] = 0xFull;  // eax

  const auto r1 = executor.step(state, memory);
  ASSERT_EQ(r1.reason, seven::StopReason::none);
  EXPECT_EQ(state.cr[8], 0xFull);

  const auto r2 = executor.step(state, memory);
  ASSERT_EQ(r2.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[3], 0xFull);  // ebx, zero-extended
}

TEST(KuberaScalar, CltsRequiresCplZero) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CS selector with RPL 3 -- CPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 06"));  // clts

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, SwapgsRequiresCplZero) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 01 F8"));  // swapgs

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, WrmsrRequiresCplZero) {
  // Ordinary user-mode code being able to rewrite an arbitrary MSR (STAR/LSTAR/FMASK/
  // KERNEL_GS_BASE and friends) is a real privilege violation, not just a fidelity gap.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 30"));  // wrmsr
  state.gpr[1] = 0xC0000082;  // ecx: LSTAR
  state.gpr[0] = 0xDEAD;      // eax
  state.gpr[2] = 0;           // edx

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, RdmsrRequiresCplZero) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 32"));  // rdmsr
  state.gpr[1] = 0xC0000082;  // ecx: LSTAR

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, XsetbvRequiresCplZero) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 01 D1"));  // xsetbv
  state.gpr[1] = 0;  // ecx: XCR0

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, CliRequiresCplLessThanOrEqualIopl) {
  // CLI/STI are only unconditionally allowed at CPL0 -- at CPL>IOPL real hardware #GPs. IOPL 0
  // (the default, unset rflags bits 12-13) means any CPL>0 must fault.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3, IOPL 0 -- CPL > IOPL
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("FA"));  // cli

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, StiRequiresCplLessThanOrEqualIopl) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3, IOPL 0 -- CPL > IOPL
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("FB"));  // sti

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, CliAllowedWhenCplWithinIopl) {
  // CPL 3 with IOPL raised to 3 in rflags -- CPL <= IOPL, so CLI must still succeed. Proves the
  // fix is the real CPL<=IOPL comparison, not a blunt CPL0-only gate.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;               // CPL 3
  state.rflags = 0x202 | (3ull << 12);  // IOPL 3
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("FA"));  // cli

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rflags & seven::kFlagIF, 0u);
}

TEST(KuberaScalar, WrfsbaseRejectsNonCanonicalAddress) {
  // WRFSBASE/WRGSBASE never route through the ordinary memory-operand fault path (they write
  // FS.base/GS.base directly, not memory), so they need their own canonical-address check --
  // real hardware #GP(0)s on a non-canonical operand here just like it does for any other
  // 4-level-paging address.
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("F3 48 0F AE D0"));  // wrfsbase rax (REX.W -- the R64 form is the only one registered)
  state.gpr[0] = 0x8000'1234'5678'9ABCull;  // non-canonical

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
}

TEST(KuberaScalar, WrfsbaseAllowsCanonicalAddress) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("F3 48 0F AE D0"));  // wrfsbase rax (REX.W -- the R64 form is the only one registered)
  state.gpr[0] = 0x0000'7FF6'1234'0000ull;  // canonical

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.fs_base, state.gpr[0]);
}

TEST(KuberaScalar, RdsspReportsNoShadowStack) {
  run_single(seven::parse_hex_bytes("F3 48 0F 1E CA"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[2] = 0xFFFF'FFFF'FFFF'FFFFull;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[2], 0u);
             });
}


}  // namespace

TEST(KuberaScalar, MovSegmentRegisterToAndFromMemoryUsesTheM16Form) {
  constexpr std::uint64_t kData = 0x4000;

  // 8C 03 -- mov word ptr [rbx], es. Default operand size in 64-bit mode makes this
  // MOV_R32M16_SREG, whose handler used to write the register named by op 0 regardless of whether
  // op 0 was a register at all. For a memory operand iced leaves op_register(0) as NONE, which
  // write_register maps to gpr[0], so this stored nothing and destroyed rax instead.
  run_single(seven::parse_hex_bytes("8C 03"),
             [](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[0] = 0xCAFEF00DDEADBEEFull;
               state.gpr[3] = kData;
               state.sreg[0] = 0x2B;
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory& memory) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(state.gpr[0], 0xCAFEF00DDEADBEEFull) << "rax is not an operand of this instruction";
               std::uint16_t stored = 0;
               ASSERT_TRUE(memory.read(kData, &stored, sizeof(stored)));
               EXPECT_EQ(stored, 0x2Bu);
             });

  // 48 8C 03 -- REX.W mov word ptr [rbx], es. Still an m16 store.
  run_single(seven::parse_hex_bytes("48 8C 03"),
             [](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[0] = 0x1111222233334444ull;
               state.gpr[3] = kData;
               state.sreg[0] = 0x33;
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory& memory) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(state.gpr[0], 0x1111222233334444ull);
               std::uint16_t stored = 0;
               ASSERT_TRUE(memory.read(kData, &stored, sizeof(stored)));
               EXPECT_EQ(stored, 0x33u);
             });

  // 8C C8 -- mov eax, cs. The register form still zero-extends the selector to the full width.
  run_single(seven::parse_hex_bytes("8C C8"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;
               state.sreg[1] = 0x33;
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(state.gpr[0], 0x33u);
             });
}

TEST(KuberaScalar, MovSegmentRegisterFromMemoryReadsOnlyTwoBytes) {
  // 8E 03 -- mov es, word ptr [rbx]. The handler read 4 bytes (8 with REX.W) from an m16 operand,
  // so an operand sitting in the last two bytes of a mapped page straddled into the next page and
  // raised a #PF that hardware never raises.
  constexpr std::uint64_t kData = 0x4000;
  constexpr std::uint64_t kLastWord = kData + 0x1000 - 2;

  run_single(seven::parse_hex_bytes("8E 03"),
             [](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               const std::uint16_t selector = 0x2B;
               ASSERT_TRUE(memory.write(kLastWord, &selector, sizeof(selector)));
               state.gpr[3] = kLastWord;
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none) << "an m16 load must not touch the next page";
               EXPECT_EQ(state.sreg[0], 0x2Bu);
             });

  run_single(seven::parse_hex_bytes("48 8E 03"),
             [](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               const std::uint16_t selector = 0x33;
               ASSERT_TRUE(memory.write(kLastWord, &selector, sizeof(selector)));
               state.gpr[3] = kLastWord;
             },
             [](const seven::ExecutionResult& result, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(result.reason, seven::StopReason::none);
               EXPECT_EQ(state.sreg[0], 0x33u);
             });
}

TEST(KuberaScalar, FnsaveWritesOnlyTheArchitecturalImage) {
  // fsave() ignored the size its handler passed and always validated and wrote 160 bytes: a
  // 24-byte private environment plus eight 16-byte ST slots. The architectural FNSAVE image is
  // 108 bytes (a 28-byte environment plus eight 10-byte slots) for the 32-bit form and 94 for the
  // 16-bit one, so this scribbled 52 bytes of unrelated guest memory past the end of a correctly
  // sized buffer, and refused a 108-byte buffer that ended on a page boundary.
  constexpr std::uint64_t kSave = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kSave, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("DD 33"));  // fnsave [rbx]
  state.gpr[3] = kSave;

  const std::vector<std::uint8_t> guard(64, 0xEE);
  ASSERT_TRUE(memory.write(kSave + 108, guard.data(), guard.size()));
  ASSERT_TRUE(state.x87_push(seven::X87Scalar(1)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  std::vector<std::uint8_t> after(guard.size(), 0);
  ASSERT_TRUE(memory.read(kSave + 108, after.data(), after.size()));
  EXPECT_EQ(after, guard) << "fnsave must not write past the 108-byte image";
}

TEST(KuberaScalar, FnsaveFrstorRoundTripKeepsTheStackWhenTopIsNotZero) {
  // The environment's tag word is two bits per PHYSICAL register, but the data slots are
  // top-relative. The old code built the tag word by physical index and then read and wrote the
  // slots by ST index, so after an fld1 (which leaves TOP at 7) the save wrote ST0's data into a
  // slot the restore then read as a different register, and ST(0) came back empty.
  constexpr std::uint64_t kSave = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kSave, 0x1000);
  // fld1; fnsave [rbx]; frstor [rbx]
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 DD 33 DD 23"));
  state.gpr[3] = kSave;

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld1
  ASSERT_EQ(state.get_x87_top(), 7u) << "sanity: fld1 must decrement TOP to 7";
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fnsave
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // frstor

  EXPECT_EQ(state.get_x87_top(), 7u);
  ASSERT_FALSE(state.x87_is_empty(0)) << "ST(0) must survive the round trip";
  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 1.0);
}

TEST(KuberaScalar, FnstenvWritesOnlyTheRequestedEnvironmentSize) {
  // store_x87_env() wrote 26 bytes for both forms, so the 14-byte form clobbered 12 bytes past its
  // image, and the 28-byte form left its last two bytes stale while laying the fields out at the
  // 16-bit offsets.
  constexpr std::uint64_t kEnv = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kEnv, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("66 D9 33"));  // fnstenv [rbx], 14-byte form
  state.gpr[3] = kEnv;

  const std::vector<std::uint8_t> guard(32, 0xEE);
  ASSERT_TRUE(memory.write(kEnv + 14, guard.data(), guard.size()));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  std::vector<std::uint8_t> after(guard.size(), 0);
  ASSERT_TRUE(memory.read(kEnv + 14, after.data(), after.size()));
  EXPECT_EQ(after, guard) << "the 14-byte fnstenv must not write past its image";

  // Control word round-trips at offset 0 in both forms.
  std::uint16_t fcw = 0;
  ASSERT_TRUE(memory.read(kEnv, &fcw, sizeof(fcw)));
  EXPECT_EQ(fcw, state.get_x87_control_word());
}

TEST(KuberaScalar, FstpTbyteWritesTheArchitecturalEightyBitEncoding) {
  // softfloat's extFloat80M swaps its two fields on LITTLEENDIAN, and the build only defined that
  // for softfloat's own sources. seven_core therefore disagreed with the library it links about
  // where signif and signExp live, so every direct field access read the wrong offset: 1.0 came
  // out of encode_ext80 as ff3f followed by eight zero bytes instead of the real encoding. That
  // corrupts every 80-bit memory format the guest can see, fstp tbyte included.
  constexpr std::uint64_t kData = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kData, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 DB 3B"));  // fld1; fstp tbyte ptr [rbx]
  state.gpr[3] = kData;

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  std::vector<std::uint8_t> stored(10, 0);
  ASSERT_TRUE(memory.read(kData, stored.data(), stored.size()));
  // 1.0: significand 0x8000000000000000 little-endian, then the biased exponent 0x3FFF.
  const std::vector<std::uint8_t> expected = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFF, 0x3F};
  EXPECT_EQ(stored, expected);
}

TEST(KuberaScalar, LoopWithoutAnAddressSizePrefixCountsDownTheFullRcx) {
  // E2 FE -- loop $-0. In 64-bit mode the default address size is 64, so this decrements RCX and
  // branches while RCX is still nonzero. The counter register is chosen by the ADDRESS size, not
  // the operand size, and iced's Jb2 decode handler was reading operand_size -- which is 32 here
  // for want of a REX.W that LOOP can never have. That made the plain form decode as the ECX
  // variant, so a 33-bit count collapsed to zero after one iteration instead of counting down.
  run_single(seven::parse_hex_bytes("E2 FE"),
             [](seven::CpuState& state, seven::Memory&) { state.gpr[1] = 0x0000'0001'0000'0001ull; },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[1], 0x0000'0001'0000'0000ull) << "loop decrements the full RCX";
               EXPECT_EQ(state.rip, kBase) << "RCX is still nonzero, so the branch is taken";
             });
}

TEST(KuberaScalar, LoopWithAnAddressSizePrefixCountsDownEcx) {
  // 67 E2 FD -- the 0x67 form really does use ECX, and zero-extends it into RCX.
  run_single(seven::parse_hex_bytes("67 E2 FD"),
             [](seven::CpuState& state, seven::Memory&) { state.gpr[1] = 0x0000'0001'0000'0001ull; },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[1], 0ull) << "ECX drops to zero and zero-extends";
               EXPECT_EQ(state.rip, kBase + 3) << "ECX hit zero, so the branch is not taken";
             });
}

TEST(KuberaScalar, JrcxzTestsTheFullRcxNotJustEcx) {
  // E3 FE -- jrcxz $-0. RCX's low half is zero but RCX itself is not, so this must NOT branch.
  // Decoded as the JECXZ form it would branch, which is a visible control-flow divergence.
  run_single(seven::parse_hex_bytes("E3 FE"),
             [](seven::CpuState& state, seven::Memory&) { state.gpr[1] = 0x0000'0001'0000'0000ull; },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[1], 0x0000'0001'0000'0000ull) << "jrcxz never writes the counter";
               EXPECT_EQ(state.rip, kBase + 2) << "RCX is nonzero, so the branch is not taken";
             });
}

TEST(KuberaScalar, NearIndirectJumpThroughMemoryReadsAFullEightByteTarget) {
  constexpr std::uint64_t kData = 0x4000;
  constexpr std::uint64_t kTarget = 0x0000'0007'1234'5000ull;

  // FF 27 -- jmp qword ptr [rdi]. FF /2 and FF /4 have a forced 64-bit operand size in long mode,
  // but iced's Evj decode handler picked the code straight off operand_size, which is 32 without a
  // REX.W these forms can never carry. That reported an RM32 form with a 4-byte memory size, so a
  // target above 4GB came back truncated to its low half.
  run_single(seven::parse_hex_bytes("FF 27"),
             [kData, kTarget](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[7] = kData;
               ASSERT_TRUE(memory.write(kData, &kTarget, sizeof(kTarget)));
             },
             [kTarget](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.rip, kTarget) << "the whole 64-bit target, not just its low half";
             });
}

TEST(KuberaScalar, NearIndirectCallThroughARegisterUsesTheFullSixtyFourBits) {
  constexpr std::uint64_t kStack = 0x8000;
  constexpr std::uint64_t kTarget = 0x0000'0007'1234'5000ull;

  // FF D1 -- call rcx. Same forced-64 rule as the jump above; decoded as an RM32 form the operand
  // register came back as ECX, so the call landed at the low 32 bits of the target.
  run_single(seven::parse_hex_bytes("FF D1"),
             [kStack, kTarget](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kStack, 0x1000);
               state.gpr[1] = kTarget;
               state.gpr[4] = kStack + 0x800;
             },
             [kTarget](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory& memory) {
               EXPECT_EQ(state.rip, kTarget) << "the whole 64-bit target, not just its low half";
               std::uint64_t pushed = 0;
               ASSERT_TRUE(memory.read(state.gpr[4], &pushed, sizeof(pushed)));
               EXPECT_EQ(pushed, kBase + 2) << "return address is the next instruction";
             });
}

TEST(KuberaScalar, XaddWithTheSameRegisterTwiceDoublesItRatherThanCancelling) {
  // 4D 0F C1 FF -- xadd r15, r15. Intel's order is TEMP := SRC + DEST; SRC := DEST; DEST := TEMP,
  // so the destination write lands last and the register ends up doubled. Writing the source
  // afterwards instead put the original value straight back and the instruction did nothing at
  // all. Hardware and Unicorn both double it; seven was the odd one out.
  run_single(seven::parse_hex_bytes("4D 0F C1 FF"),
             [](seven::CpuState& state, seven::Memory&) { state.gpr[15] = 0xE8C842B14C7FFB7Aull; },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[15], 0xD190856298FFF6F4ull);
             });

  // 0F C0 C0 -- xadd al, al, the 8-bit form of the same aliasing.
  run_single(seven::parse_hex_bytes("0F C0 C0"),
             [](seven::CpuState& state, seven::Memory&) { state.gpr[0] = 0xFFFFFFFFFFFFFF25ull; },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[0] & 0xFFull, 0x4Aull) << "0x25 + 0x25";
               EXPECT_EQ(state.gpr[0] >> 8, 0x00FFFFFFFFFFFFFFull) << "the rest of rax is untouched";
             });

  // Distinct registers still exchange as usual. ModRM C1 makes rcx the r/m destination and rax
  // the reg source.
  run_single(seven::parse_hex_bytes("48 0F C1 C1"),
             [](seven::CpuState& state, seven::Memory&) {
               state.gpr[0] = 0x1111;
               state.gpr[1] = 0x2222;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[1], 0x3333ull) << "destination gets the sum";
               EXPECT_EQ(state.gpr[0], 0x2222ull) << "source gets the old destination";
             });
}

TEST(KuberaScalar, MovToAnAbsoluteMoffsAddressStoresTheAccumulator) {
  constexpr std::uint64_t kData = 0x4000;

  // A2 00 40 00 00 00 00 00 00 -- mov byte ptr [0x4000], al. The A0/A1 load forms read operand 1
  // and write operand 0, and the A2/A3 store forms had simply been given the same shape with the
  // indices swapped, which makes them load as well: the store never happened and the accumulator
  // was overwritten with whatever the address held.
  run_single(seven::parse_hex_bytes("A2 00 40 00 00 00 00 00 00"),
             [](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[0] = 0xCAFEF00DDEADBE5Aull;
               const std::uint8_t existing = 0xC3;
               ASSERT_TRUE(memory.write(kData, &existing, sizeof(existing)));
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory& memory) {
               std::uint8_t stored = 0;
               ASSERT_TRUE(memory.read(kData, &stored, sizeof(stored)));
               EXPECT_EQ(stored, 0x5Au) << "al is stored to the absolute address";
               EXPECT_EQ(state.gpr[0], 0xCAFEF00DDEADBE5Aull) << "rax is the source, not the destination";
             });

  // A3 with REX.W -- mov qword ptr [0x4000], rax, the 64-bit form of the same store.
  run_single(seven::parse_hex_bytes("48 A3 00 40 00 00 00 00 00 00"),
             [](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[0] = 0x1122334455667788ull;
             },
             [](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory& memory) {
               std::uint64_t stored = 0;
               ASSERT_TRUE(memory.read(kData, &stored, sizeof(stored)));
               EXPECT_EQ(stored, 0x1122334455667788ull);
               EXPECT_EQ(state.gpr[0], 0x1122334455667788ull) << "rax is unchanged";
             });
}

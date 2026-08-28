#include <cstring>
#include <tuple>

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

// WRMSRNS encodes no operands. Reading operand slot 0 anyway got back the value-initialised
// (REGISTER, NONE) slot, which resolves to register index zero, so the MSR index came from EAX
// instead of ECX and every write landed somewhere the guest did not name.
TEST(KuberaScalar, WrmsrnsTakesItsIndexFromEcx) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[0] = 0x11112222;  // rax: the index the bug would have used
  state.gpr[1] = 0x00000174;  // rcx: IA32_SYSENTER_CS, the index the instruction names
  state.gpr[2] = 0x33334444;  // rdx: high half of the value
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F 01 C6"));  // wrmsrns

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  ASSERT_TRUE(state.msr.contains(0x174u));
  EXPECT_EQ(state.msr.at(0x174u), 0x3333444411112222ull);
  EXPECT_FALSE(state.msr.contains(0x11112222u)) << "index came from eax";
}

// SYSRET and SYSEXIT are the kernel's way back out to user code and are CPL0-only on hardware.
// None of the four forms checked, and SYSRETQ additionally loaded rflags wholesale out of R11 --
// so a ring 3 guest could pick its own IOPL and hand itself back everything the CLI/STI gate two
// tests up exists to deny, in two instructions.
TEST(KuberaScalar, SysretDoesNotLetRingThreeRaiseItsOwnIopl) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3, IOPL 0
  state.gpr[11] = 0x3202;  // what the guest would like rflags to become: IOPL 3, IF set
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 0F 07"));  // sysretq

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ((state.rflags >> 12) & 0x3u, 0u) << "ring 3 picked its own IOPL";
}

TEST(KuberaScalar, EveryReturnToUserFormIsCplZeroOnly) {
  struct Case { const char* name; const char* bytes; };
  const Case cases[] = {
      {"sysretd", "0F 07"},
      {"sysretq", "48 0F 07"},
      {"sysexitd", "0F 35"},
      {"sysexitq", "48 0F 35"},
  };
  for (const auto& c : cases) {
    seven::Executor executor{};
    seven::CpuState state{};
    seven::Memory memory{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.sreg[1] = 0x2B;  // CPL 3
    memory.map(kBase, 0x1000);
    write_bytes(memory, kBase, seven::parse_hex_bytes(c.bytes));

    const auto result = executor.step(state, memory);
    EXPECT_EQ(result.reason, seven::StopReason::general_protection) << c.name;
    EXPECT_EQ(state.mode, seven::ExecutionMode::long64) << c.name << " changed mode from ring 3";
  }
}

// The same instruction at CPL0 is a legitimate kernel return and must still work.
TEST(KuberaScalar, SysretqStillReturnsAtCplZero) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x08;  // CPL 0
  state.gpr[1] = kBase + 0x100;  // rcx: the address to return to
  state.gpr[11] = 0x202;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 0F 07"));  // sysretq

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, kBase + 0x100);
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

TEST(KuberaScalar, Crc32OverMemorySourcesUsesTheOperandsRealWidth) {
  constexpr std::uint64_t kData = 0x4000;
  constexpr std::uint64_t kValue = 0x0123456789ABCDEFull;
  constexpr std::uint32_t kSeed = 0xB0051228u;

  // F2 0F 38 F1 57 00 -- crc32 edx, dword ptr [rdi]. operand_width fed read_operand a width taken
  // from a double-converted memory_size: 4 bytes came back as 8 and 8 came back as 64. The 64 case
  // is the sharp one -- read_operand copies that many bytes into a uint64_t on its own stack, so
  // `crc32 r64, qword ptr [mem]` overran the frame and took the process down with an access
  // violation. Guest bytes, host stack.
  run_single(seven::parse_hex_bytes("F2 0F 38 F1 57 00"),
             [kData, kSeed, kValue](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[7] = kData;
               state.gpr[2] = kSeed;
               ASSERT_TRUE(memory.write(kData, &kValue, sizeof(kValue)));
             },
             [kSeed, kValue](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[2], crc32c_update(kSeed, kValue, 4))
                   << "only the low four bytes feed the checksum";
             });

  // F2 48 0F 38 F1 57 00 -- crc32 rdx, qword ptr [rdi], the width that used to ask for 64 bytes.
  run_single(seven::parse_hex_bytes("F2 48 0F 38 F1 57 00"),
             [kData, kSeed, kValue](seven::CpuState& state, seven::Memory& memory) {
               memory.map(kData, 0x1000);
               state.gpr[7] = kData;
               state.gpr[2] = kSeed;
               ASSERT_TRUE(memory.write(kData, &kValue, sizeof(kValue)));
             },
             [kSeed, kValue](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[2], crc32c_update(kSeed, kValue, 8));
             });

  // A register source was always fine, and has to stay that way.
  run_single(seven::parse_hex_bytes("F2 48 0F 38 F1 D1"),
             [kSeed, kValue](seven::CpuState& state, seven::Memory&) {
               state.gpr[2] = kSeed;
               state.gpr[1] = kValue;
             },
             [kSeed, kValue](const seven::ExecutionResult&, const seven::CpuState& state, const seven::Memory&) {
               EXPECT_EQ(state.gpr[2], crc32c_update(kSeed, kValue, 8));
             });
}

// sreg[1] is the only thing in this emulator that says what privilege level the guest is at, and
// there are no descriptor tables, so a far branch writing it straight from a guest-supplied
// selector hands the guest ring 0. Every CPL gate in the tree is derived from the same field, so
// one RETF voids all of them at once. Real hardware faults instead: a far return can stay where it
// is or move outward, never inward, and it validates the selector through the GDT/LDT to get there.

TEST(KuberaScalar, AFarReturnCannotLowerTheCurrentPrivilegeLevel) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x33;  // CS selector with RPL 3 -- CPL 3
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  state.gpr[4] = kStack + 0x800;

  write_bytes(memory, kBase, seven::parse_hex_bytes("48 CB"));  // retfq

  // The frame this emulator's retf_width expects: 8-byte target, then a 2-byte selector.
  const std::uint64_t target = kBase + 0x100;
  const std::uint16_t ring0_selector = 0x08;  // RPL 0
  ASSERT_TRUE(memory.write(kStack + 0x800, &target, sizeof(target)));
  ASSERT_TRUE(memory.write(kStack + 0x808, &ring0_selector, sizeof(ring0_selector)));

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.sreg[1] & 0x3u, 3u) << "CPL must not have dropped";
}

namespace {

// Writes one 80-bit extended value in the architectural in-memory layout: significand first,
// little-endian, then the sign-and-exponent halfword.
void write_extf80(seven::Memory& memory, std::uint64_t address, std::uint64_t significand,
                  std::uint16_t sign_exp) {
  ASSERT_TRUE(memory.write(address, &significand, sizeof(significand)));
  ASSERT_TRUE(memory.write(address + 8, &sign_exp, sizeof(sign_exp)));
}

constexpr std::uint64_t kX87Data = 0x4000;

}  // namespace

// seven::ldexp used to add FSCALE's shift count to the biased exponent as an int, and the shift is
// whatever the guest left in ST(1). It also fell back to std::ldexp on a double for the subnormal
// and underflow cases, which are exactly the values a double cannot hold, so a result that should
// have been a representable denormal came back as zero or infinity.
TEST(KuberaScalar, FscaleReachesTheDenormalRangeInsteadOfFlushing) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0x8000000000000000ull, 0xBFFF);       // -1.0
  write_extf80(memory, kX87Data + 16, 0x8000000000000000ull, 0x0001);  // 2^-16382, smallest normal
  // fld tbyte [rbx] ; fld tbyte [rbx+16] ; fscale
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B DB 6B 10 D9 FD"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  const auto result = state.x87_get(0);
  EXPECT_EQ(result.val.signExp, 0x0000u) << "2^-16383 is a denormal, so the biased exponent is zero";
  EXPECT_EQ(result.val.signif, 0x4000000000000000ull) << "and the significand keeps its one bit";
}

// This one is a tripwire rather than a repro: the old int addition wrapped to a large negative
// value, which fell into the underflow branch, which returned infinity anyway -- so the answer was
// right by luck while the arithmetic was undefined. It fails under a sanitizer, not at -O2.
TEST(KuberaScalar, FscaleWithAHugeShiftDoesNotOverflowTheExponentArithmetic) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  const std::uint32_t shift = 0x7FFFFFFFu;  // INT_MAX, straight into the exponent addition
  ASSERT_TRUE(memory.write(kX87Data + 16, &shift, sizeof(shift)));
  // fild dword [rbx+16] ; fld1 ; fscale
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 43 10 D9 E8 D9 FD"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  EXPECT_TRUE(seven::isinf(state.x87_get(0)));
  EXPECT_FALSE(seven::signbit(state.x87_get(0)));
}

TEST(KuberaScalar, FpremByAnInfiniteDivisorReturnsTheDividend) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0x8000000000000000ull, 0x7FFF);       // +inf
  write_extf80(memory, kX87Data + 16, 0x8000000000000000ull, 0x3FFF);  // 1.0
  // fld tbyte [rbx] ; fld tbyte [rbx+16] ; fprem
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B DB 6B 10 D9 F8"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  const auto result = state.x87_get(0);
  EXPECT_FALSE(seven::isnan(result)) << "trunc(a/inf) is zero and zero times infinity is NaN";
  EXPECT_EQ(static_cast<double>(result), 1.0);
}

// FST/FSTP m32 rounds the 80-bit value to single precision once. Going through a double first
// rounds twice, and a value sitting exactly on the double midpoint rounds to even there and then
// lands a full ulp away from where one rounding puts it.
TEST(KuberaScalar, FstpToSinglePrecisionRoundsOnce) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  state.gpr[1] = kX87Data + 32;
  memory.map(kX87Data, 0x1000);
  // 1 + 2^-24 + 2^-53
  write_extf80(memory, kX87Data, 0x8000008000000400ull, 0x3FFF);
  // fld tbyte [rbx] ; fstp dword [rcx]
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B D9 19"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  std::uint32_t stored = 0;
  ASSERT_TRUE(memory.read(kX87Data + 32, &stored, sizeof(stored)));
  EXPECT_EQ(stored, 0x3F800001u) << "rounding through double first collapses this to 1.0f";
}

TEST(KuberaScalar, Fyl2xOnALargeExponentDoesNotSaturateThroughDouble) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0x8000000000000000ull, 0x3FFF);       // 1.0
  write_extf80(memory, kX87Data + 16, 0x8000000000000000ull, 0x444B);  // 2^1100
  // fld tbyte [rbx] ; fld tbyte [rbx+16] ; fyl2x
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B DB 6B 10 D9 F1"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 1100.0)
      << "2^1100 does not fit in a double, so narrowing before taking the log gave infinity";
}

namespace {

// Every x87 test below wants the same shape: data at kX87Data, code at kBase, long mode.
struct X87Fixture {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};

  X87Fixture() {
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[3] = kX87Data;
    memory.map(kX87Data, 0x1000);
  }

  void load_code(std::string_view hex) { write_bytes(memory, kBase, seven::parse_hex_bytes(hex)); }

  void run(int steps) {
    for (int i = 0; i < steps; ++i) {
      ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
    }
  }

  [[nodiscard]] std::uint16_t status() const { return state.get_x87_status_word(); }
};

}  // namespace

TEST(KuberaScalar, DecstpAndIncstpClearC1) {
  for (const auto* code : {"D9 F6", "D9 F7"}) {  // fdecstp, fincstp
    X87Fixture f;
    f.state.set_x87_status_word(static_cast<std::uint16_t>(f.state.get_x87_status_word() | 0x0200u));
    f.load_code(code);
    f.run(1);
    EXPECT_EQ(f.status() & 0x0200u, 0u) << code << " must clear C1";
  }
}

// A masked stack underflow does not skip the instruction on hardware. The register-destination forms
// already knew that; the memory-destination ones returned having written nothing and having left TOP
// where it was, so the guest read a stale destination and an unmoved stack.
TEST(KuberaScalar, StoringFromAnEmptyStackWritesTheIndefiniteAndStillPops) {
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    const std::uint64_t sentinel = 0x0123456789ABCDEFull;
    ASSERT_TRUE(f.memory.write(kX87Data + 64, &sentinel, sizeof(sentinel)));
    f.load_code("DD 19");  // fstp qword [rcx]
    f.run(1);

    std::uint64_t stored = 0;
    ASSERT_TRUE(f.memory.read(kX87Data + 64, &stored, sizeof(stored)));
    EXPECT_EQ(stored, 0xFFF8000000000000ull) << "the double-precision floating-point indefinite";
    EXPECT_EQ(f.state.get_x87_top(), 1u) << "the pop still retires";
    EXPECT_EQ(f.status() & 0x0041u, 0x0041u) << "IE and SF";
  }
  {
    // FIST/FISTP store the INTEGER indefinite, which is a different value entirely.
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    f.load_code("DB 19");  // fistp dword [rcx]
    f.run(1);

    std::uint32_t stored = 0;
    ASSERT_TRUE(f.memory.read(kX87Data + 64, &stored, sizeof(stored)));
    EXPECT_EQ(stored, 0x80000000u);
    EXPECT_EQ(f.state.get_x87_top(), 1u);
  }
  {
    // And packed decimal has a third one, FFFFC000_00000000_0000.
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    f.load_code("DF 31");  // fbstp tbyte [rcx]
    f.run(1);

    std::array<std::uint8_t, 10> stored{};
    ASSERT_TRUE(f.memory.read(kX87Data + 64, stored.data(), stored.size()));
    const std::array<std::uint8_t, 10> expected{0, 0, 0, 0, 0, 0, 0, 0xC0, 0xFF, 0xFF};
    EXPECT_EQ(stored, expected);
    EXPECT_EQ(f.state.get_x87_top(), 1u);
  }
  {
    // FSTP m80 writes the 80-bit indefinite straight through.
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    f.load_code("DB 39");  // fstp tbyte [rcx]
    f.run(1);

    std::uint64_t significand = 0;
    std::uint16_t sign_exp = 0;
    ASSERT_TRUE(f.memory.read(kX87Data + 64, &significand, sizeof(significand)));
    ASSERT_TRUE(f.memory.read(kX87Data + 72, &sign_exp, sizeof(sign_exp)));
    EXPECT_EQ(significand, 0xC000000000000000ull);
    EXPECT_EQ(sign_exp, 0xFFFFu);
    EXPECT_EQ(f.state.get_x87_top(), 1u);
  }
  {
    // FSTP ST(i) with an empty top leaves the indefinite in the destination register and pops.
    X87Fixture f;
    f.load_code("DD DA");  // fstp st(2)
    f.run(1);

    ASSERT_FALSE(f.state.x87_is_empty(1)) << "what was ST(2) before the pop";
    EXPECT_EQ(f.state.x87_get(1).val.signExp, 0xFFFFu);
    EXPECT_EQ(f.state.x87_get(1).val.signif, 0xC000000000000000ull);
    EXPECT_EQ(f.state.get_x87_top(), 1u);
  }
}

// FXCH is both source and destination on both operands, so a masked underflow fills whichever one is
// empty and the exchange still happens. It used to return without swapping anything.
TEST(KuberaScalar, FxchWithAnEmptyRegisterFillsItAndStillSwaps) {
  X87Fixture f;
  f.load_code("D9 E8 D9 C9");  // fld1 ; fxch st(1)
  f.run(2);

  EXPECT_EQ(f.state.x87_get(0).val.signExp, 0xFFFFu) << "ST(1) was empty, so ST(0) gets its indefinite";
  EXPECT_EQ(f.state.x87_get(0).val.signif, 0xC000000000000000ull);
  ASSERT_FALSE(f.state.x87_is_empty(1));
  EXPECT_EQ(static_cast<double>(f.state.x87_get(1)), 1.0) << "and the 1.0 moves down to ST(1)";
}

// An instruction that faults must not have touched its destination. FIST wrote the integer
// indefinite first and only then asked whether #IA was masked.
TEST(KuberaScalar, FistpLeavesTheDestinationAloneWhenInvalidIsUnmasked) {
  X87Fixture f;
  f.state.gpr[1] = kX87Data + 64;
  write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x7FFF);  // +inf, out of range for any int
  const std::uint16_t unmasked_invalid = 0x037E;
  ASSERT_TRUE(f.memory.write(kX87Data + 32, &unmasked_invalid, sizeof(unmasked_invalid)));
  const std::uint32_t sentinel = 0xDEADBEEFu;
  ASSERT_TRUE(f.memory.write(kX87Data + 64, &sentinel, sizeof(sentinel)));
  // fldcw [rbx+0x20] ; fld tbyte [rbx] ; fistp dword [rcx]
  f.load_code("D9 6B 20 DB 2B DB 19");
  f.run(2);

  EXPECT_EQ(f.executor.step(f.state, f.memory).reason, seven::StopReason::floating_point_exception);
  std::uint32_t stored = 0;
  ASSERT_TRUE(f.memory.read(kX87Data + 64, &stored, sizeof(stored)));
  EXPECT_EQ(stored, sentinel) << "the store must not have happened";
}

// softfloat_roundingMode was never assigned anywhere in the tree, so extF80_add and friends rounded
// to nearest-even no matter what the guest put in FCW.RC.
TEST(KuberaScalar, TheGuestsRoundingControlReachesTheSoftFloatArithmetic) {
  X87Fixture f;
  write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x3FFF);       // 1.0
  write_extf80(f.memory, kX87Data + 16, 0x8000000000000000ull, 0x3FBE);  // 2^-65, a quarter of an ulp
  const std::uint16_t round_up = 0x0B7F;  // RC = 10, toward +infinity
  ASSERT_TRUE(f.memory.write(kX87Data + 32, &round_up, sizeof(round_up)));
  // fldcw [rbx+0x20] ; fld tbyte [rbx] ; fld tbyte [rbx+0x10] ; faddp st(1), st(0)
  f.load_code("D9 6B 20 DB 2B DB 6B 10 DE C1");
  f.run(4);

  EXPECT_EQ(f.state.x87_get(0).val.signExp, 0x3FFFu);
  EXPECT_EQ(f.state.x87_get(0).val.signif, 0x8000000000000001ull)
      << "rounding to nearest collapses this back to 1.0";
  EXPECT_EQ(softfloat_roundingMode, softfloat_round_near_even)
      << "the mode is a process-wide global, so the guard has to put it back";
}

// x87_classify_result read the exceptions off the result alone: any infinity was an overflow and any
// NaN was an invalid operand, whatever the operands had been.
TEST(KuberaScalar, AnInfiniteOperandIsNotAnOverflowAndAQuietNanPropagatesSilently) {
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x7FFF);  // +inf
    const double one = 1.0;
    ASSERT_TRUE(f.memory.write(kX87Data + 64, &one, sizeof(one)));
    f.load_code("DB 2B DC 01");  // fld tbyte [rbx] ; fadd qword [rcx]
    f.run(2);

    EXPECT_TRUE(seven::isinf(f.state.x87_get(0)));
    EXPECT_EQ(f.status() & 0x0008u, 0u) << "inf + 1 is inf, not an overflow";
  }
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0xC000000000000000ull, 0x7FFF);  // a quiet NaN
    const double one = 1.0;
    ASSERT_TRUE(f.memory.write(kX87Data + 64, &one, sizeof(one)));
    f.load_code("DB 2B DC 01");
    f.run(2);

    EXPECT_TRUE(seven::isnan(f.state.x87_get(0)));
    EXPECT_EQ(f.status() & 0x0001u, 0u) << "a quiet NaN propagates without raising #IA";
  }
}

// The infinity a division by zero produces is #Z and nothing else. Reading the exception off the
// result made it an overflow as well, so the guest saw two exceptions where hardware raises one.
TEST(KuberaScalar, DividingByZeroIsNotAlsoAnOverflow) {
  X87Fixture f;
  f.state.gpr[1] = kX87Data + 64;
  const double zero = 0.0;
  ASSERT_TRUE(f.memory.write(kX87Data + 64, &zero, sizeof(zero)));
  f.load_code("D9 E8 DC 31");  // fld1 ; fdiv qword [rcx]
  f.run(2);

  EXPECT_TRUE(seven::isinf(f.state.x87_get(0)));
  EXPECT_EQ(f.status() & 0x0004u, 0x0004u) << "#Z";
  EXPECT_EQ(f.status() & 0x0008u, 0u) << "and not #O as well";
}

// #P used to come out of x87_precision_from_binary's algebraic probes instead of the arithmetic.
// 1 + 2^-65 rounds back to 1, and the probe reads (result - rhs == lhs) as proof that nothing was
// lost, so the one addition here that is inexact by construction reported itself exact.
TEST(KuberaScalar, PrecisionComesFromTheRealInexactFlagNotAnAlgebraicProbe) {
  X87Fixture f;
  f.state.gpr[1] = kX87Data + 64;
  const std::uint64_t quarter_ulp = 0x3BE0000000000000ull;  // 2^-65
  ASSERT_TRUE(f.memory.write(kX87Data + 64, &quarter_ulp, sizeof(quarter_ulp)));
  f.load_code("D9 E8 DC 01");  // fld1 ; fadd qword [rcx]
  softfloat_exceptionFlags = 0;
  f.run(2);

  EXPECT_EQ(f.state.x87_get(0).val.signif, 0x8000000000000000ull) << "the sum rounds back to 1.0";
  EXPECT_EQ(f.state.x87_get(0).val.signExp, 0x3FFFu);
  EXPECT_EQ(f.status() & 0x0020u, 0x0020u) << "#P: a quarter of an ulp went missing";
  EXPECT_EQ(softfloat_exceptionFlags, 0u)
      << "the flags are a global too, so reading them means putting them back";
}

// #U was raised for any denormal result at all. The architecture wants tininess AND inexactness,
// and 2^-16382 halved is a denormal that lost nothing on the way there.
TEST(KuberaScalar, AnExactDenormalResultIsNeitherAnUnderflowNorInexact) {
  X87Fixture f;
  f.state.gpr[1] = kX87Data + 64;
  write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x0001);  // 2^-16382, smallest normal
  const std::uint64_t two = 0x4000000000000000ull;
  ASSERT_TRUE(f.memory.write(kX87Data + 64, &two, sizeof(two)));
  f.load_code("DB 2B DC 31");  // fld tbyte [rbx] ; fdiv qword [rcx]
  f.run(2);

  EXPECT_EQ(f.state.x87_get(0).val.signExp, 0x0000u) << "2^-16383, a denormal";
  EXPECT_EQ(f.state.x87_get(0).val.signif, 0x4000000000000000ull);
  EXPECT_EQ(f.status() & 0x0010u, 0u) << "#U also needs the result to be inexact";
  EXPECT_EQ(f.status() & 0x0020u, 0u) << "and nothing was lost, so no #P either";
}

// #D says an OPERAND was denormal. It was being read off the result, so ordinary arithmetic on two
// normal numbers reported a denormal operand it never had.
TEST(KuberaScalar, DenormalIsAnOperandConditionNotAResultCondition) {
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x8000000000000003ull, 0x0001);  // normal, low bits set
    const std::uint64_t four = 0x4010000000000000ull;
    ASSERT_TRUE(f.memory.write(kX87Data + 64, &four, sizeof(four)));
    f.load_code("DB 2B DC 31");  // fld tbyte [rbx] ; fdiv qword [rcx]
    f.run(2);

    EXPECT_EQ(f.state.x87_get(0).val.signExp, 0x0000u) << "a denormal, and an inexact one";
    EXPECT_EQ(f.status() & 0x0030u, 0x0030u) << "#U and #P are both real here";
    EXPECT_EQ(f.status() & 0x0002u, 0u) << "but neither operand was denormal";
  }
  {
    // The operand side still reports, which is the half that was always right.
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x4000000000000000ull, 0x0000);  // a denormal operand
    const double one = 1.0;
    ASSERT_TRUE(f.memory.write(kX87Data + 64, &one, sizeof(one)));
    f.load_code("DB 2B DC 01");  // fld tbyte [rbx] ; fadd qword [rcx]
    f.run(2);

    EXPECT_EQ(f.status() & 0x0002u, 0x0002u) << "#D";
  }
}

// Arithmetic on an encoding that contradicts its own exponent has no answer. seven only knew about
// those in FXAM, so an unnormal walked into extF80_add and came back out as an ordinary number.
TEST(KuberaScalar, AnUnsupportedOperandRaisesInvalidAndYieldsTheIndefinite) {
  X87Fixture f;
  f.state.gpr[1] = kX87Data + 64;
  write_extf80(f.memory, kX87Data, 0x4000000000000000ull, 0x4000);  // unnormal
  const double one = 1.0;
  ASSERT_TRUE(f.memory.write(kX87Data + 64, &one, sizeof(one)));
  f.load_code("DB 2B DC 01");  // fld tbyte [rbx] ; fadd qword [rcx]
  f.run(2);

  EXPECT_EQ(f.status() & 0x0001u, 0x0001u) << "#IA";
  EXPECT_EQ(f.state.x87_get(0).val.signExp, 0xFFFFu) << "and the masked answer is the indefinite";
  EXPECT_EQ(f.state.x87_get(0).val.signif, 0xC000000000000000ull);
}

// FST/FSTP to m32 and m64 narrow an 80-bit value and computed no exceptions at all, so a value that
// did not fit the destination was stored rounded, or as an infinity, with a clean status word.
TEST(KuberaScalar, NarrowingStoresReportWhatTheConversionCost) {
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x8000000000000001ull, 0x3FFF);  // 1 + 2^-63
    f.load_code("DB 2B D9 19");  // fld tbyte [rbx] ; fstp dword [rcx]
    f.run(2);

    std::uint32_t stored = 0;
    ASSERT_TRUE(f.memory.read(kX87Data + 64, &stored, sizeof(stored)));
    EXPECT_EQ(stored, 0x3F800000u) << "1.0f, everything below it rounded away";
    EXPECT_EQ(f.status() & 0x0020u, 0x0020u) << "#P";
    EXPECT_EQ(f.status() & 0x0008u, 0u) << "and it is not an overflow";
  }
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x40C7);  // 2^200
    f.load_code("DB 2B D9 19");  // fld tbyte [rbx] ; fstp dword [rcx]
    f.run(2);

    std::uint32_t stored = 0;
    ASSERT_TRUE(f.memory.read(kX87Data + 64, &stored, sizeof(stored)));
    EXPECT_EQ(stored, 0x7F800000u) << "the masked #O answer is an infinity";
    EXPECT_EQ(f.status() & 0x0028u, 0x0028u) << "#O and #P";
  }
  {
    X87Fixture f;
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x382F);  // 2^-2000
    f.load_code("DB 2B DD 19");  // fld tbyte [rbx] ; fstp qword [rcx]
    f.run(2);

    std::uint64_t stored = 0;
    ASSERT_TRUE(f.memory.read(kX87Data + 64, &stored, sizeof(stored)));
    EXPECT_EQ(stored, 0u) << "too small for a double subnormal";
    EXPECT_EQ(f.status() & 0x0030u, 0x0030u) << "#U and #P";
  }
}

// FSQRT asked whether sqrt(x) differed from x and called that a precision loss, which is true of
// almost every square root there is. Its own negative-operand check also reported #IA and returned
// without leaving the masked answer behind, so ST(0) kept the operand.
TEST(KuberaScalar, FsqrtReportsSoftFloatsExactnessAndTheMaskedInvalidAnswer) {
  {
    X87Fixture f;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x4001);  // 4.0
    f.load_code("DB 2B D9 FA");  // fld tbyte [rbx] ; fsqrt
    f.run(2);

    EXPECT_EQ(f.state.x87_get(0).val.signExp, 0x4000u) << "2.0";
    EXPECT_EQ(f.status() & 0x0020u, 0u) << "an exact root is not a precision loss";
  }
  {
    X87Fixture f;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0x4000);  // 2.0
    f.load_code("DB 2B D9 FA");
    f.run(2);
    EXPECT_EQ(f.status() & 0x0020u, 0x0020u) << "sqrt(2) does not fit, so #P";
  }
  {
    X87Fixture f;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0xBFFF);  // -1.0
    f.load_code("DB 2B D9 FA");
    f.run(2);

    EXPECT_EQ(f.status() & 0x0001u, 0x0001u) << "#IA";
    EXPECT_EQ(f.state.x87_get(0).val.signExp, 0xFFFFu) << "and the masked answer is the indefinite";
    EXPECT_EQ(f.state.x87_get(0).val.signif, 0xC000000000000000ull);
  }
}

// FCHS and FABS edit the sign bit and nothing else. The SDM gives them #IS and no other exception,
// signalling NaNs and unsupported encodings included, but they went through the same result-reading
// classifier as the arithmetic and so reported a denormal operand as an underflow.
TEST(KuberaScalar, FabsAndFchsRaiseNothingAtAll) {
  for (const auto* code : {"DB 2B D9 E1", "DB 2B D9 E0"}) {  // fld tbyte [rbx] ; fabs / fchs
    X87Fixture f;
    write_extf80(f.memory, kX87Data, 0x4000000000000000ull, 0x0000);  // a denormal
    f.load_code(code);
    f.run(2);

    EXPECT_EQ(f.status() & 0x003Fu, 0u) << code << " is a sign-bit edit, not arithmetic";
    EXPECT_EQ(f.state.x87_get(0).val.signif, 0x4000000000000000ull) << "and it keeps the value";
  }
}

// IEEE 754 7.5 and SDM 4.9.1.5 both hand an enabled underflow trap a result whose exponent has been
// biased back into range rather than leaving the destination stale. The fault used to be reported
// with ST(0) still holding the dividend.
TEST(KuberaScalar, AnUnmaskedUnderflowDeliversTheBiasedResult) {
  const auto divide_by_four = [](std::uint16_t control_word, X87Fixture& f) {
    f.state.gpr[1] = kX87Data + 64;
    write_extf80(f.memory, kX87Data, 0x8000000000000003ull, 0x0001);
    const std::uint64_t four = 0x4010000000000000ull;
    ASSERT_TRUE(f.memory.write(kX87Data + 64, &four, sizeof(four)));
    ASSERT_TRUE(f.memory.write(kX87Data + 32, &control_word, sizeof(control_word)));
    // fldcw [rbx+0x20] ; fld tbyte [rbx] ; fdiv qword [rcx]
    f.load_code("D9 6B 20 DB 2B DC 31");
  };

  X87Fixture masked;
  divide_by_four(0x037F, masked);
  masked.run(3);
  const auto gradual = masked.state.x87_get(0);
  ASSERT_EQ(gradual.val.signExp, 0x0000u) << "the masked response is the denormal";

  X87Fixture trapped;
  divide_by_four(0x036F, trapped);  // #U unmasked
  trapped.run(2);
  EXPECT_EQ(trapped.executor.step(trapped.state, trapped.memory).reason,
            seven::StopReason::floating_point_exception);

  const auto expected = seven::ldexp(gradual, 24576);
  EXPECT_EQ(trapped.state.x87_get(0).val.signExp, expected.val.signExp);
  EXPECT_EQ(trapped.state.x87_get(0).val.signif, expected.val.signif);
}

// FUCOM is quiet about quiet NaNs only. The flag used to suppress #IA for signalling ones too.
TEST(KuberaScalar, FucompIsQuietForAQuietNanButNotForASignallingOne) {
  const auto run_with = [](std::uint64_t significand) {
    X87Fixture f;
    write_extf80(f.memory, kX87Data, significand, 0x7FFF);
    f.load_code("DB 2B D9 E8 DA E9");  // fld tbyte [rbx] ; fld1 ; fucompp
    f.run(3);
    return f.status();
  };

  EXPECT_EQ(run_with(0xC000000000000000ull) & 0x0001u, 0u) << "quiet NaN, quiet compare";
  EXPECT_EQ(run_with(0xA000000000000000ull) & 0x0001u, 0x0001u) << "signalling NaN still raises #IA";
}

// FXAM answers about the register, not about a value: C1 is the sign bit even when the register is
// empty, and the encodings that contradict their own exponent get a class of their own.
TEST(KuberaScalar, FxamReportsTheSignOfAnEmptyRegisterAndTheUnsupportedClass) {
  {
    X87Fixture f;
    write_extf80(f.memory, kX87Data, 0x8000000000000000ull, 0xBFFF);  // -1.0
    f.load_code("DB 2B DD C0 D9 E5");  // fld tbyte [rbx] ; ffree st(0) ; fxam
    f.run(3);

    // The class lives in C3, C2 and C0; C1 is the sign and is read separately.
    EXPECT_EQ(f.status() & 0x4500u, 0x4100u) << "still the empty class";
    EXPECT_EQ(f.status() & 0x0200u, 0x0200u) << "C1 is the register's sign bit, empty or not";
  }
  {
    X87Fixture f;
    write_extf80(f.memory, kX87Data, 0x4000000000000000ull, 0x4000);       // unnormal
    write_extf80(f.memory, kX87Data + 16, 0x4000000000000000ull, 0x7FFF);  // pseudo-NaN
    // fld tbyte [rbx] ; fxam ; fld tbyte [rbx+0x10] ; fxam
    f.load_code("DB 2B D9 E5 DB 6B 10 D9 E5");
    f.run(2);
    EXPECT_EQ(f.status() & 0x4500u, 0x0000u) << "an unnormal used to read as an ordinary normal";
    f.run(2);
    EXPECT_EQ(f.status() & 0x4500u, 0x0000u) << "a pseudo-NaN used to read as a NaN";
  }
}

// The RDTSC counter used to be a function-local static, so every guest in the process shared it.
// Two Executors that are meant to be isolated could watch each other's progress through it, and on
// separate threads they raced on the increment. RDTSCP read a hardcoded 0 at the same time, so a
// guest using both saw the clock jump backwards.
TEST(KuberaScalar, TheTimestampCounterIsPerGuestNotPerProcess) {
  const auto run_one = [](seven::CpuState& state, seven::Memory& memory, seven::Executor& executor) {
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    write_bytes(memory, kBase, seven::parse_hex_bytes("0F 31"));  // rdtsc
    return executor.step(state, memory);
  };

  seven::Executor executor_a{};
  seven::CpuState state_a{};
  seven::Memory memory_a{};
  seven::Executor executor_b{};
  seven::CpuState state_b{};
  seven::Memory memory_b{};

  ASSERT_EQ(run_one(state_a, memory_a, executor_a).reason, seven::StopReason::none);
  ASSERT_EQ(run_one(state_a, memory_a, executor_a).reason, seven::StopReason::none);
  state_a.rip = kBase;
  ASSERT_EQ(run_one(state_a, memory_a, executor_a).reason, seven::StopReason::none);
  ASSERT_EQ(run_one(state_b, memory_b, executor_b).reason, seven::StopReason::none);

  EXPECT_EQ(state_b.gpr[0] & 0xFFFFFFFFull, 1u)
      << "the second guest's first rdtsc must not see the first guest's count";
  EXPECT_EQ(state_a.tsc, 3u);
  EXPECT_EQ(state_b.tsc, 1u);

  // rdtscp reads the same counter, and rdmsr on IA32_TIME_STAMP_COUNTER agrees with both.
  seven::CpuState state_c{};
  seven::Memory memory_c{};
  seven::Executor executor_c{};
  state_c.mode = seven::ExecutionMode::long64;
  state_c.rip = kBase;
  write_bytes(memory_c, kBase, seven::parse_hex_bytes("0F 31 0F 01 F9 B9 10 00 00 00 0F 32"));
  ASSERT_EQ(executor_c.step(state_c, memory_c).reason, seven::StopReason::none);  // rdtsc
  ASSERT_EQ(executor_c.step(state_c, memory_c).reason, seven::StopReason::none);  // rdtscp
  EXPECT_EQ(state_c.gpr[0] & 0xFFFFFFFFull, 2u) << "rdtscp must not report a constant zero";
  ASSERT_EQ(executor_c.step(state_c, memory_c).reason, seven::StopReason::none);  // mov ecx, 0x10
  ASSERT_EQ(executor_c.step(state_c, memory_c).reason, seven::StopReason::none);  // rdmsr
  EXPECT_EQ(state_c.gpr[0] & 0xFFFFFFFFull, 2u) << "rdmsr 0x10 must report the same counter";
}

// MOV to a segment register writes sreg[] by index, and index 1 is CS, the only place this
// emulator records the current privilege level. There is no MOV CS encoding on hardware and iced
// declines to decode one, so this pins both halves: the decoder rejects it, and if it ever stopped
// rejecting it the handler would too.
TEST(KuberaScalar, MovToCsCannotSetThePrivilegeLevel) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x33;
  state.gpr[0] = 0x08;
  write_bytes(memory, kBase, seven::parse_hex_bytes("8E C8"));  // mov cs, ax

  const auto result = executor.step(state, memory);
  EXPECT_NE(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.sreg[1] & 0x3u, 3u) << "CPL must not have dropped";
}

// validate_memory_span exists so the x87 image stores can check the whole 108-byte image before
// touching any of it. It computed base + offset as plain uint64, so a destination near the top of
// the address space wrapped and it happily validated page 0, then the store itself faulted partway
// through on Memory's own wrap check, leaving a half-written image behind.
TEST(KuberaScalar, AnX87ImageStoreThatWrapsTheAddressSpaceFaultsBeforeWritingAnything) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kTopPage = 0xFFFFFFFFFFFFF000ull;
  constexpr std::uint64_t kDest = 0xFFFFFFFFFFFFFFC0ull;  // 64 bytes short of wrapping
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kDest;
  memory.map(kTopPage, 0x1000);
  memory.map(0x0, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("DD 33"));  // fnsave [rbx]

  const std::uint8_t sentinel = 0xA5;
  for (std::uint64_t off = 0; off < 0x40; ++off) {
    ASSERT_TRUE(memory.write(kDest + off, &sentinel, 1));
  }

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  for (std::uint64_t off = 0; off < 0x40; ++off) {
    std::uint8_t got = 0;
    ASSERT_TRUE(memory.read(kDest + off, &got, 1));
    EXPECT_EQ(got, sentinel) << "byte at +" << off << " was stored before the fault";
  }
}

// sreg[1] is this emulator's only record of the current privilege level, and dispatch_interrupt
// assigns it straight from the gate's code selector. Hardware only permits that for INT n, INT3 and
// INTO when the gate's DPL is at least the caller's CPL; without that check a guest the embedder
// put at ring 3 walks into any present vector and comes out at whatever ring the gate names, which
// voids every CPL gate in the tree at once (MOV CR, WRMSR, SWAPGS, CLTS, XSETBV, CLI/STI).
TEST(KuberaScalar, ASoftwareInterruptCannotEnterAGateItLacksThePrivilegeFor) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kIdt = 0x8000;
  constexpr std::uint64_t kHandler = 0x9000;
  constexpr std::uint64_t kStack = 0xA000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x33;  // CPL 3
  state.gpr[4] = kStack + 0x800;
  state.idtr.base = kIdt;
  state.idtr.limit = 0x1000 - 1;

  memory.map(kIdt, 0x1000);
  memory.map(kHandler, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("CD 80"));  // int 0x80
  const std::uint8_t hlt[] = {0xF4};
  ASSERT_TRUE(memory.write(kHandler, hlt, sizeof(hlt)));

  // Present 64-bit interrupt gate, DPL 0, pointing at a ring-0 code selector.
  const std::uint64_t selector = 0x08;
  const std::uint64_t type_attr = 0x8E;
  const std::uint64_t desc_lo = (kHandler & 0xFFFFull) | (selector << 16) | (type_attr << 40) |
                                (((kHandler >> 16) & 0xFFFFull) << 48);
  const std::uint64_t desc_hi = (kHandler >> 32) & 0xFFFFFFFFull;
  ASSERT_TRUE(memory.write(kIdt + 0x80 * 16, &desc_lo, sizeof(desc_lo)));
  ASSERT_TRUE(memory.write(kIdt + 0x80 * 16 + 8, &desc_hi, sizeof(desc_hi)));

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.sreg[1] & 0x3u, 3u) << "CPL must not have been raised by the gate";
  EXPECT_NE(state.rip, kHandler) << "the guest must not have entered the handler";
}

TEST(KuberaScalar, AFarReturnThatKeepsOrRaisesItsPrivilegeLevelStillWorks) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x2B;  // CPL 3
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  state.gpr[4] = kStack + 0x800;

  write_bytes(memory, kBase, seven::parse_hex_bytes("48 CB"));  // retfq

  const std::uint64_t target = kBase + 0x100;
  const std::uint16_t same_ring_selector = 0x33;  // RPL 3, same level
  ASSERT_TRUE(memory.write(kStack + 0x800, &target, sizeof(target)));
  ASSERT_TRUE(memory.write(kStack + 0x808, &same_ring_selector, sizeof(same_ring_selector)));

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, target);
  EXPECT_EQ(state.sreg[1], 0x33u);
}

TEST(KuberaScalar, IretDoesNotLetRingThreeRewriteItsOwnIopl) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.sreg[1] = 0x33;   // CPL 3
  state.rflags = 0x202;   // IOPL 0
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  state.gpr[4] = kStack + 0x800;

  write_bytes(memory, kBase, seven::parse_hex_bytes("48 CF"));  // iretq

  const std::uint64_t target = kBase + 0x100;
  const std::uint16_t same_ring_selector = 0x33;
  const std::uint64_t flags = 0x3202;  // IOPL 3
  ASSERT_TRUE(memory.write(kStack + 0x800, &target, sizeof(target)));
  ASSERT_TRUE(memory.write(kStack + 0x808, &same_ring_selector, sizeof(same_ring_selector)));
  ASSERT_TRUE(memory.write(kStack + 0x80A, &flags, sizeof(flags)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ((state.rflags >> 12) & 0x3u, 0u) << "IOPL is writable only from CPL 0";
}

// iced's decoder value-initializes Instruction, and OpKind::REGISTER and Register::NONE are both 0,
// so reading an operand slot the decoder never wrote comes back as (REGISTER, NONE) -- which passes
// an `op_kind(i) != REGISTER` guard. Several x87 handlers read operand 1 (or operands 0 and 1) on
// instructions that carry fewer than that, and x87_st_index(NONE) then underflows to a huge value
// that x87_phys_index masks down to ST(7). The mask is what keeps it memory-safe; it is also what
// hid it.

TEST(KuberaScalar, FpremDoesNotFaultOnAnInstructionWithNoOperands) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // fld1 ; fldpi ; fprem  -- FPREM has op_count 0, so both operand reads return Register::NONE.
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 D9 EB D9 F8"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld1
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fldpi

  const auto result = executor.step(state, memory);  // fprem
  EXPECT_EQ(result.reason, seven::StopReason::none)
      << "fprem has no memory operand and must not raise a page fault";
  EXPECT_NEAR(static_cast<double>(state.x87_get(0)), 0.14159265358979312, 1e-12)
      << "fmod(pi, 1.0)";
}

TEST(KuberaScalar, FstpStiWritesTheRegisterItNames) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // fldz ; fld1 ; fstp st(1)  -- stores ST(0) into ST(1), then pops, leaving 1.0 on top.
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 EE D9 E8 DD D9"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fldz
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld1
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fstp st(1)

  ASSERT_FALSE(state.x87_is_empty(0));
  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 1.0)
      << "the value has to land in ST(1) and survive the pop, not go to ST(7)";
}

TEST(KuberaScalar, FstStiWritesTheRegisterItNames) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // fldz ; fld1 ; fst st(1)  -- copies ST(0) into ST(1) with no pop, so both hold 1.0.
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 EE D9 E8 DD D1"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fldz
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld1
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fst st(1)

  ASSERT_FALSE(state.x87_is_empty(1));
  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 1.0);
  EXPECT_EQ(static_cast<double>(state.x87_get(1)), 1.0)
      << "the source is implicitly ST(0); the only named operand is the destination";
}

TEST(KuberaScalar, FcomppComparesTheTopTwoStackSlots) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // fld1 ; fldz ; fcompp  -- compares ST(0)=0 against ST(1)=1, so C0 (carry-equivalent) is set.
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 D9 EE DE D9"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld1
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fldz
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fcompp

  // C3 is bit 14, C0 is bit 8. 0 < 1 means C3 clear, C0 set. Comparing a slot with itself would
  // report equal: C3 set, C0 clear.
  EXPECT_EQ(state.x87_status_word & (1u << 14), 0u) << "C3: the operands are not equal";
  EXPECT_NE(state.x87_status_word & (1u << 8), 0u) << "C0: ST(0) is less than ST(1)";
}

// fnstcw/fldcw is the most common x87 idiom there is -- every _controlfp, every MSVC __ftol2, every
// rounding-mode save/restore. Real hardware ignores the reserved control-word bits rather than
// validating them, and the architectural default 0x037F has bit 6 set, so a reserved-bit check that
// includes bit 6 faults on the value the FPU resets to.
TEST(KuberaScalar, FldcwAcceptsTheArchitecturalDefaultControlWord) {
  constexpr std::uint64_t kSlot = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kSlot, 0x1000);
  // fnstcw [rbx]; fldcw [rbx]
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 3B D9 2B"));
  state.gpr[3] = kSlot;

  ASSERT_EQ(state.get_x87_control_word(), 0x037Fu) << "sanity: the reset control word";
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fnstcw
  std::uint16_t stored = 0;
  ASSERT_TRUE(memory.read(kSlot, &stored, sizeof(stored)));
  EXPECT_EQ(stored, 0x037Fu);

  EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::none)
      << "fldcw faulted on the control word the FPU had just stored";
  EXPECT_EQ(state.get_x87_control_word(), 0x037Fu);
}

// Same wrong constant on the restore side: the control word has to survive its own save/restore.
TEST(KuberaScalar, FxrstorKeepsTheControlWordFxsaveWrote) {
  constexpr std::uint64_t kSave = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kSave, 0x1000);
  // fxsave [rbx]; fxrstor [rbx]
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F AE 03 0F AE 0B"));
  state.gpr[3] = kSave;

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fxsave
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fxrstor
  EXPECT_EQ(state.get_x87_control_word(), 0x037Fu) << "the restore cleared bits the save wrote";
}

// Hardware puts ST(i) in fxsave slot i while the abridged tag word is indexed by physical register.
// fsave/frstor already pair the two conventions correctly; fxsave/fxrstor used physical for both,
// so any image taken with TOP != 0 had its registers in the wrong slots. Only observable from
// outside the round trip, which is why a save/restore pair alone never caught it.
TEST(KuberaScalar, FxsaveWritesTheStackTopRelative) {
  constexpr std::uint64_t kSave = 0x4000;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kSave, 0x1000);
  // fld1; fxsave [rbx]
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 0F AE 03"));
  state.gpr[3] = kSave;

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld1
  ASSERT_EQ(state.get_x87_top(), 7u) << "sanity: fld1 leaves TOP at 7";
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fxsave

  // 1.0 as an 80-bit extended: significand 0x8000000000000000, exponent 0x3FFF.
  const auto expected = seven::parse_hex_bytes("00 00 00 00 00 00 00 80 FF 3F");
  std::array<std::uint8_t, 10> slot0{};
  ASSERT_TRUE(memory.read(kSave + 32, slot0.data(), slot0.size()));
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), slot0.begin()))
      << "slot 0 must hold ST(0), not physical register 0";
}

// Overflowing the x87 stack sets SF and C1 alongside IE, which is how a guest tells an overflow
// from an underflow. FLD1/FLDZ/FLDPI, the integer loads and the BCD load all go through
// x87_stack_overflow; the plain FLD forms did not -- the memory form raised a bare invalid
// exception, and the register form reported a page fault on an instruction with no memory operand.
TEST(KuberaScalar, OverflowingTheStackWithFldReportsAStackFault) {
  constexpr std::uint64_t kData = 0x4000;
  const auto fill_then = [&](const std::string& tail_hex) {
    seven::Executor executor{};
    seven::CpuState state{};
    seven::Memory memory{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    memory.map(kBase, 0x1000);
    memory.map(kData, 0x1000);
    // fld1 x8 fills the stack, then the instruction under test overflows it.
    write_bytes(memory, kBase,
                seven::parse_hex_bytes("D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 " + tail_hex));
    state.gpr[3] = kData;
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "fill " << i;
    }
    const auto result = executor.step(state, memory);
    return std::pair{result.reason, state.get_x87_status_word()};
  };

  constexpr std::uint16_t kIe = 0x0001;
  constexpr std::uint16_t kSf = 0x0040;
  constexpr std::uint16_t kC1 = 0x0200;

  const auto from_memory = fill_then("D9 03");  // fld dword [rbx]
  EXPECT_EQ(from_memory.first, seven::StopReason::none);
  EXPECT_EQ(from_memory.second & (kIe | kSf | kC1), kIe | kSf | kC1) << "fld m32 overflow";

  const auto from_register = fill_then("D9 C0");  // fld st(0)
  EXPECT_EQ(from_register.first, seven::StopReason::none) << "fld st(0) has no memory operand";
  EXPECT_EQ(from_register.second & (kIe | kSf | kC1), kIe | kSf | kC1) << "fld st(i) overflow";
}

namespace {

struct AdjustResult {
  seven::StopReason reason;
  std::uint64_t rax;
  std::uint64_t rflags;
};

// AAA/AAS/AAD/AAM/DAA/DAS are all #UD in 64-bit mode, so the whole family only decodes in compat32.
AdjustResult run_adjust(std::string_view hex, std::uint64_t rax, std::uint64_t rflags) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::compat32;
  state.rip = kBase;
  state.rflags = rflags;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes(hex));
  state.gpr[0] = rax;
  const auto result = executor.step(state, memory);
  return {result.reason, state.gpr[0], state.rflags};
}

}  // namespace

// Intel spells the AAA adjust as AX := AX + 106H, not as two independent byte adds, so the carry
// out of AL + 6 reaches AH. seven did AL += 6 and AH += 1 separately and dropped that carry.
TEST(KuberaScalar, AaaCarriesOutOfAlIntoAh) {
  const auto r = run_adjust("37", 0x00FF, 0x202);  // aaa with AH:AL = 00:FF
  ASSERT_EQ(r.reason, seven::StopReason::none);
  EXPECT_EQ(r.rax & 0xFFFFu, 0x0205u) << "AL+6 overflows the byte, so AH ends at 2, not 1";
  EXPECT_NE(r.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(r.rflags & seven::kFlagAF, 0u);
}

// Same shape on the subtract side: AX := AX - 6 followed by AH := AH - 1, so a borrow out of
// AL - 6 costs AH two decrements in total.
TEST(KuberaScalar, AasBorrowsOutOfAlIntoAh) {
  const auto r = run_adjust("3F", 0x0200, 0x202 | seven::kFlagAF);  // aas with AH:AL = 02:00
  ASSERT_EQ(r.reason, seven::StopReason::none);
  EXPECT_EQ(r.rax & 0xFFFFu, 0x000Au) << "AL-6 borrows, so AH drops from 2 to 0";
  EXPECT_NE(r.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(r.rflags & seven::kFlagAF, 0u);
}

// DAA's second half ends in an else that forces CF to 0; DAS's does not, so a borrow raised by the
// low-nibble correction survives to the end of the instruction. seven had copied DAA's else into
// DAS and was clearing CF that hardware leaves set.
TEST(KuberaScalar, DasKeepsTheCarryRaisedByTheLowNibbleBorrow) {
  const auto das = run_adjust("2F", 0x0000, 0x202 | seven::kFlagAF);
  ASSERT_EQ(das.reason, seven::StopReason::none);
  EXPECT_EQ(das.rax & 0xFFu, 0xFAu);
  EXPECT_NE(das.rflags & seven::kFlagCF, 0u) << "AL - 6 borrowed out of the byte";
  EXPECT_NE(das.rflags & seven::kFlagAF, 0u);

  const auto daa = run_adjust("27", 0x0000, 0x202 | seven::kFlagAF);
  ASSERT_EQ(daa.reason, seven::StopReason::none);
  EXPECT_EQ(daa.rax & 0xFFu, 0x06u);
  EXPECT_EQ(daa.rflags & seven::kFlagCF, 0u) << "DAA really does clear CF here, DAS does not";
}

// AAD multiplies by its immediate, so every value is legal including zero. seven raised #GP for
// `aad 0`, which is a perfectly ordinary instruction that just clears AH.
TEST(KuberaScalar, AadWithAZeroImmediateIsNotAFault) {
  const auto zero = run_adjust("D5 00", 0x0507, 0x202);
  ASSERT_EQ(zero.reason, seven::StopReason::none);
  EXPECT_EQ(zero.rax & 0xFFFFu, 0x0007u);

  const auto base10 = run_adjust("D5 0A", 0x0203, 0x202);
  ASSERT_EQ(base10.reason, seven::StopReason::none);
  EXPECT_EQ(base10.rax & 0xFFFFu, 0x0017u);
}

// AAM divides by its immediate, and Intel lists #DE (not #GP) for a zero one.
TEST(KuberaScalar, AamWithAZeroImmediateRaisesDivideError) {
  const auto zero = run_adjust("D4 00", 0x0007, 0x202);
  EXPECT_EQ(zero.reason, seven::StopReason::divide_error);

  const auto base10 = run_adjust("D4 0A", 0x001D, 0x202);
  ASSERT_EQ(base10.reason, seven::StopReason::none);
  EXPECT_EQ(base10.rax & 0xFFFFu, 0x0209u);
}

// Only CMPXCHG16B carries the alignment requirement; CMPXCHG8B has none and an unaligned
// destination is a normal locked access. seven was raising #GP for both.
TEST(KuberaScalar, Cmpxchg8bAcceptsAnUnalignedDestinationButCmpxchg16bDoesNot) {
  constexpr std::uint64_t kData = 0x4000;
  constexpr std::uint64_t kUnaligned = kData + 1;

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  memory.map(kData, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F C7 0E"));  // cmpxchg8b [rsi]
  const std::uint64_t seed = 0x1122334455667788ull;
  ASSERT_TRUE(memory.write(kUnaligned, &seed, sizeof(seed)));
  state.gpr[6] = kUnaligned;    // rsi
  state.gpr[2] = 0x11223344;    // edx
  state.gpr[0] = 0x55667788;    // eax
  state.gpr[1] = 0x0A0B0C0D;    // ecx
  state.gpr[3] = 0x0E0F1011;    // ebx

  const auto result = executor.step(state, memory);
  ASSERT_EQ(result.reason, seven::StopReason::none);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
  std::uint64_t stored = 0;
  ASSERT_TRUE(memory.read(kUnaligned, &stored, sizeof(stored)));
  EXPECT_EQ(stored, 0x0A0B0C0D0E0F1011ull);

  seven::Executor wide_executor{};
  seven::CpuState wide_state{};
  seven::Memory wide_memory{};
  wide_state.mode = seven::ExecutionMode::long64;
  wide_state.rip = kBase;
  wide_state.rflags = 0x202;
  wide_memory.map(kBase, 0x1000);
  wide_memory.map(kData, 0x1000);
  write_bytes(wide_memory, kBase, seven::parse_hex_bytes("48 0F C7 0E"));  // cmpxchg16b [rsi]
  wide_state.gpr[6] = kUnaligned;

  const auto wide_result = wide_executor.step(wide_state, wide_memory);
  EXPECT_EQ(wide_result.reason, seven::StopReason::general_protection)
      << "the 16-byte form does require alignment";
}

// XADD computed its flags before attempting the store, so a destination the guest can read but not
// write returned a fault with rflags already rewritten. CMPXCHG and BTS/BTR/BTC in the same
// read-modify-write family already commit flags only after the store lands.
TEST(KuberaScalar, XaddLeavesFlagsAloneWhenTheStoreFaults) {
  constexpr std::uint64_t kData = 0x4000;
  constexpr auto kReadOnly = static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read);

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  memory.map(kData, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("0F C1 06"));  // xadd [rsi], eax
  const std::uint32_t seed = 0xFFFFFFFFu;
  ASSERT_TRUE(memory.write(kData, &seed, sizeof(seed)));
  memory.reprotect(kData, 0x1000, kReadOnly);
  state.gpr[6] = kData;  // rsi
  state.gpr[0] = 1;      // eax -- would carry out and leave a zero result

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.rflags & seven::kAluStatusFlagsMask, 0u)
      << "a store that never landed must not have published CF/ZF/PF/AF";
}

// A push is a fault, not a trap: hardware aborts it and leaves rsp exactly where it was, which is
// the only reason a guard page can grow a stack -- the handler maps the page and the same push runs
// again. Decrementing rsp before the store meant every retry started one slot lower, so a stack that
// faulted twice ended up 16 bytes down with nothing written.
TEST(KuberaScalar, APushThatFaultsDoesNotMoveTheStackPointer) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("50"));  // push rax
  state.gpr[0] = 0xDEADBEEF;
  state.gpr[4] = kStack;  // the slot below is not mapped

  const auto first = executor.step(state, memory);
  EXPECT_EQ(first.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[4], kStack);
  EXPECT_EQ(state.rip, kBase) << "the push never retired";

  const auto second = executor.step(state, memory);
  EXPECT_EQ(second.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[4], kStack) << "rsp walked down on the retry";
}

// pushfq has its own hand-written copy of the push sequence and had the same ordering.
TEST(KuberaScalar, APushfqThatFaultsDoesNotMoveTheStackPointer) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("9C"));  // pushfq
  state.gpr[4] = kStack;

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[4], kStack);
}

// POP m64 addresses its destination through the already-incremented rsp -- that part is
// architectural. Keeping the increment when the store then faults is not.
TEST(KuberaScalar, APopWhoseDestinationFaultsDoesNotMoveTheStackPointer) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("8F 03"));  // pop qword [rbx]
  state.gpr[3] = 0x7000'0000;  // unmapped
  state.gpr[4] = kStack + 0x100;

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[4], kStack + 0x100);
}

// Bits 3, 5, 15 and everything from 22 up do not exist in rflags: they read back as zero whatever
// is written. POPFQ and IRET are the only two instructions that load rflags wholesale out of guest
// memory, so they are the only two that can put a value there at all.
TEST(KuberaScalar, PopfqDropsTheBitsRflagsDoesNotHave) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("9D"));  // popfq
  state.gpr[4] = kStack + 0x100;
  const std::uint64_t all_ones = ~0ull;
  ASSERT_TRUE(memory.write(kStack + 0x100, &all_ones, sizeof(all_ones)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.rflags & ~seven::kRflagsWritableMask, seven::kRflagsReservedOnes)
      << "reserved rflags bits took a guest-supplied value";
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u) << "the bits that do exist still load";
}

TEST(KuberaScalar, IretqDropsTheBitsRflagsDoesNotHave) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.rflags = 0x202;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 CF"));  // iretq

  // The frame this emulator's iret_width expects: 8-byte target, 2-byte selector, 8-byte flags.
  const std::uint64_t target = kBase + 0x100;
  const std::uint16_t selector = 0x08;
  const std::uint64_t all_ones = ~0ull;
  state.gpr[4] = kStack + 0x100;
  ASSERT_TRUE(memory.write(kStack + 0x100, &target, sizeof(target)));
  ASSERT_TRUE(memory.write(kStack + 0x108, &selector, sizeof(selector)));
  ASSERT_TRUE(memory.write(kStack + 0x10A, &all_ones, sizeof(all_ones)));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, target);
  EXPECT_EQ(state.rflags & ~seven::kRflagsWritableMask, seven::kRflagsReservedOnes)
      << "reserved rflags bits took a guest-supplied value";
}

// A branch target out of the non-canonical hole is a #GP against the branch itself. Committing it
// to rip and letting the executor's fetch check catch it on the next step reports the fault against
// the bad address instead of the instruction that produced it, and retires the branch on the way.
TEST(KuberaScalar, AReturnToANonCanonicalAddressFaultsAtTheReturn) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  constexpr std::uint64_t kNonCanonical = 0x0000'8000'0000'0000ull;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C3"));  // ret
  state.gpr[4] = kStack + 0x100;
  ASSERT_TRUE(memory.write(kStack + 0x100, &kNonCanonical, sizeof(kNonCanonical)));

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.rip, kBase) << "rip took the non-canonical value";
  EXPECT_EQ(state.gpr[4], kStack + 0x100) << "a fault commits nothing, the pop included";
  ASSERT_TRUE(result.exception.has_value());
  EXPECT_EQ(result.exception->address, kNonCanonical);
}

TEST(KuberaScalar, AnIndirectJumpToANonCanonicalTargetFaultsAtTheJump) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kNonCanonical = 0x0000'8000'0000'0000ull;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("FF E0"));  // jmp rax
  state.gpr[0] = kNonCanonical;

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.rip, kBase);
}

TEST(KuberaScalar, AnIndirectCallToANonCanonicalTargetPushesNothing) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  constexpr std::uint64_t kNonCanonical = 0x0000'8000'0000'0000ull;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("FF D0"));  // call rax
  state.gpr[0] = kNonCanonical;
  state.gpr[4] = kStack + 0x100;

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.rip, kBase);
  EXPECT_EQ(state.gpr[4], kStack + 0x100);
  std::uint64_t slot = 0xAAAA'AAAA'AAAA'AAAAull;
  ASSERT_TRUE(memory.read(kStack + 0xF8, &slot, sizeof(slot)));
  EXPECT_EQ(slot, 0u) << "the return address was written before the target was checked";
}

// ENTER's nesting level used to be discarded outright, so a level-3 frame came out with the display
// pointers missing and rsp three slots too high -- silently the wrong frame rather than a refused
// one. Level counts the frame pointer itself, so level 3 copies two enclosing pointers and then
// pushes the new frame pointer on top.
TEST(KuberaScalar, EnterBuildsTheDisplayForANonZeroNestingLevel) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  constexpr std::uint64_t kOuterBp = kStack + 0x800;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C8 00 00 03"));  // enter 0, 3

  const std::uint64_t outer_display[2] = {0x1111'1111'1111'1111ull, 0x2222'2222'2222'2222ull};
  ASSERT_TRUE(memory.write(kOuterBp - 8, &outer_display[0], sizeof(outer_display[0])));
  ASSERT_TRUE(memory.write(kOuterBp - 16, &outer_display[1], sizeof(outer_display[1])));

  const std::uint64_t entry_sp = kStack + 0x400;
  state.gpr[4] = entry_sp;
  state.gpr[5] = kOuterBp;

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  const std::uint64_t frame_temp = entry_sp - 8;
  EXPECT_EQ(state.gpr[5], frame_temp) << "rbp is the frame pointer the display ends with";
  EXPECT_EQ(state.gpr[4], frame_temp - 24) << "three more slots below it hold the display";

  const auto slot = [&](std::uint64_t address) {
    std::uint64_t value = 0;
    EXPECT_TRUE(memory.read(address, &value, sizeof(value)));
    return value;
  };
  EXPECT_EQ(slot(entry_sp - 8), kOuterBp);
  EXPECT_EQ(slot(entry_sp - 16), outer_display[0]);
  EXPECT_EQ(slot(entry_sp - 24), outer_display[1]);
  EXPECT_EQ(slot(entry_sp - 32), frame_temp);
}

// Each display copy reads through the enclosing frame chain, and the chain is guest data, so any
// one of them can fault. Nothing may be committed when one does.
TEST(KuberaScalar, AnEnterWhoseDisplayCopyFaultsLeavesTheFrameRegistersAlone) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  constexpr std::uint64_t kStack = 0x8000;
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kStack, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("C8 00 00 02"));  // enter 0, 2

  const std::uint64_t entry_sp = kStack + 0x400;
  state.gpr[4] = entry_sp;
  state.gpr[5] = 0x7000'0000;  // unmapped, so reading the enclosing display pointer faults

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.gpr[4], entry_sp);
  EXPECT_EQ(state.gpr[5], 0x7000'0000ull);
}

namespace {

constexpr std::uint16_t kX87Ie = 0x0001;
constexpr std::uint16_t kX87Ze = 0x0004;
constexpr std::uint16_t kX87Sf = 0x0040;
constexpr std::uint16_t kX87C0 = 0x0100;
constexpr std::uint16_t kX87C1 = 0x0200;
constexpr std::uint16_t kX87C2 = 0x0400;
constexpr std::uint16_t kX87C3 = 0x4000;

bool is_x87_indefinite(const seven::X87Scalar& value) {
  return value.val.signExp == 0xFFFFu && value.val.signif == 0xC000000000000000ull;
}

}  // namespace

// A masked stack overflow is not a no-op on hardware: TOP still moves and the new top gets the QNaN
// indefinite. seven raised the exception and left TOP alone, so from there on every ST(i) the guest
// named resolved to a different physical register than the one hardware would have used -- the
// whole register file was off by one for the rest of the program.
TEST(KuberaScalar, MaskedStackOverflowStillMovesTheStackTop) {
  const auto fill_then_overflow = [](std::uint16_t control_word) {
    seven::Executor executor{};
    seven::CpuState state{};
    seven::Memory memory{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.set_x87_control_word(control_word);
    memory.map(kBase, 0x1000);
    // fld1 x8 fills the stack, the ninth overflows it.
    write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8 D9 E8"));
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "fill " << i;
    }
    EXPECT_EQ(state.get_x87_top(), 0u) << "sanity: eight pushes wrap TOP back to 0";
    const auto reason = executor.step(state, memory).reason;
    return std::tuple{reason, state.get_x87_top(), state.x87_get(0), state.get_x87_status_word()};
  };

  const auto [masked_reason, masked_top, masked_st0, masked_sw] = fill_then_overflow(0x037F);
  EXPECT_EQ(masked_reason, seven::StopReason::none);
  EXPECT_EQ(masked_top, 7u) << "a masked overflow still decrements TOP";
  EXPECT_TRUE(is_x87_indefinite(masked_st0)) << "and writes the indefinite into the new top";
  EXPECT_EQ(masked_sw & (kX87Ie | kX87Sf | kX87C1), kX87Ie | kX87Sf | kX87C1);

  // Unmasking the invalid-operation exception has to keep the old behaviour: fault, complete nothing.
  const auto [raised_reason, raised_top, raised_st0, raised_sw] = fill_then_overflow(0x037E);
  EXPECT_EQ(raised_reason, seven::StopReason::floating_point_exception);
  EXPECT_EQ(raised_top, 0u) << "an unmasked overflow must not move TOP";
  EXPECT_EQ(static_cast<double>(raised_st0), 1.0) << "nor overwrite the register that is still there";
  EXPECT_NE(raised_sw & 0x0080u, 0u) << "ES is set when the exception is not masked";
}

// The other half of the same rule: a masked stack underflow returns the indefinite to the
// destination rather than leaving the instruction unexecuted.
TEST(KuberaScalar, MaskedStackUnderflowLeavesTheIndefiniteInTheDestination) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E0"));  // fchs, on an empty stack

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_FALSE(state.x87_is_empty(0)) << "the destination is written, so it is no longer empty";
  EXPECT_TRUE(is_x87_indefinite(state.x87_get(0)));
  EXPECT_EQ(state.get_x87_status_word() & (kX87Ie | kX87Sf), kX87Ie | kX87Sf);
}

// MM0-MM7 are the low 64 bits of the physical x87 registers, so FXSAVE stores them and FXRSTOR
// brings them back for free. They used to live in an array of their own that no save or restore
// path ever looked at, which lost the whole MMX file across a context switch.
TEST(KuberaScalar, MmxRegistersAliasTheX87RegisterFileAcrossFxsave) {
  constexpr std::uint64_t kSave = 0x4000;
  constexpr std::uint64_t kPattern = 0x0123456789ABCDEFull;
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  memory.map(kSave, 0x1000);
  // movq mm0, rax ; fxsave [rbx] ; movq mm0, rcx ; fxrstor [rbx] ; movq rdx, mm0
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 0F 6E C0 0F AE 03 48 0F 6E C1 0F AE 0B 48 0F 7E C2"));
  state.gpr[0] = kPattern;
  state.gpr[1] = 0xFEDCBA9876543210ull;
  state.gpr[3] = kSave;

  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  std::array<std::uint8_t, 10> slot0{};
  ASSERT_TRUE(memory.read(kSave + 32, slot0.data(), slot0.size()));
  std::uint64_t significand = 0;
  std::uint16_t sign_exp = 0;
  std::memcpy(&significand, slot0.data(), sizeof(significand));
  std::memcpy(&sign_exp, slot0.data() + 8, sizeof(sign_exp));
  EXPECT_EQ(significand, kPattern) << "fxsave stores MM0 as the significand of the aliased register";
  EXPECT_EQ(sign_exp, 0xFFFFu) << "writing an MMX register fills the exponent and sign with ones";
  EXPECT_EQ(state.gpr[2], kPattern) << "fxrstor has to bring MM0 back";
}

// The FXSAVE area must be 16-byte aligned and hardware raises #GP(0) when it is not, the same as
// every other explicitly-aligned SSE operand. seven reported a page fault.
TEST(KuberaScalar, MisalignedFxsaveAreaIsAGeneralProtectionFault) {
  constexpr std::uint64_t kSave = 0x4000;
  const auto run = [](const std::string& hex) {
    seven::Executor executor{};
    seven::CpuState state{};
    seven::Memory memory{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    memory.map(kBase, 0x1000);
    memory.map(kSave, 0x1000);
    write_bytes(memory, kBase, seven::parse_hex_bytes(hex));
    state.gpr[3] = kSave + 1;
    return executor.step(state, memory).reason;
  };

  EXPECT_EQ(run("0F AE 03"), seven::StopReason::general_protection) << "fxsave [rbx]";
  EXPECT_EQ(run("0F AE 0B"), seven::StopReason::general_protection) << "fxrstor [rbx]";
}

// FSINCOS replaces ST(0) with the sine and then pushes the cosine, so the cosine is what ends up on
// top. seven had the two the other way round.
TEST(KuberaScalar, FsincosLeavesTheCosineOnTop) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E8 D9 FB"));  // fld1 ; fsincos

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  EXPECT_NEAR(static_cast<double>(state.x87_get(0)), 0.5403023058681398, 1e-12) << "cos(1) on top";
  EXPECT_NEAR(static_cast<double>(state.x87_get(1)), 0.8414709848078965, 1e-12) << "sin(1) below it";
}

// The trig instructions only reduce arguments below 2^63. Past that hardware sets C2 and leaves both
// the operand and the stack untouched; seven pushed whatever std::sin made of the narrowed value.
TEST(KuberaScalar, TrigOnAnUnreducibleArgumentSetsC2AndLeavesTheStackAlone) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0x8000000000000000ull, 0x403E);  // 2^63, the first rejected value
  // fld tbyte [rbx] ; fsin ; fptan
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B D9 FE D9 F2"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fld
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fsin
  EXPECT_NE(state.get_x87_status_word() & kX87C2, 0u) << "fsin gave up and said so";
  EXPECT_EQ(state.x87_get(0).val.signExp, 0x403Eu) << "and left the operand where it was";
  EXPECT_EQ(state.get_x87_top(), 7u);

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);  // fptan
  EXPECT_NE(state.get_x87_status_word() & kX87C2, 0u);
  EXPECT_EQ(state.get_x87_top(), 7u) << "an out-of-range fptan must not push its 1.0 either";
}

// FPREM reports the low three bits of the quotient, and not in register order: Q2 lands in C0, Q1 in
// C3 and Q0 in C1. seven left all four condition codes exactly as it found them.
TEST(KuberaScalar, FpremReportsTheQuotientBitsAndClearsC2) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0x8000000000000000ull, 0x3FFF);       // 1.0, the divisor
  write_extf80(memory, kX87Data + 16, 0xA000000000000000ull, 0x4001);  // 5.0, the dividend
  // fld tbyte [rbx] ; fld tbyte [rbx+16] ; fprem
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B DB 6B 10 D9 F8"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 0.0);
  const auto sw = state.get_x87_status_word();
  EXPECT_EQ(sw & kX87C2, 0u) << "the reduction finished";
  EXPECT_NE(sw & kX87C0, 0u) << "quotient 5 is 101b, so Q2 is set";
  EXPECT_EQ(sw & kX87C3, 0u) << "Q1 is clear";
  EXPECT_NE(sw & kX87C1, 0u) << "Q0 is set";
}

// One FPREM only makes guaranteed progress while the two exponents are within 64 of each other. Past
// that it reduces part of the way and raises C2 to say so, and the guest is expected to run it again.
// seven always computed the whole remainder, so a guest looping on C2 never saw it clear.
TEST(KuberaScalar, FpremKeepsC2SetUntilTheReductionFinishes) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0xC000000000000000ull, 0x4000);       // 3.0, the divisor
  write_extf80(memory, kX87Data + 16, 0x8000000000000000ull, 0x4045);  // 2^70, the dividend
  // fld tbyte [rbx] ; fld tbyte [rbx+16] ; fprem ; fprem
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B DB 6B 10 D9 F8 D9 F8"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }
  EXPECT_NE(state.get_x87_status_word() & kX87C2, 0u) << "70 bits apart, so one pass cannot finish";
  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 64.0) << "partially reduced, not finished";

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.get_x87_status_word() & kX87C2, 0u) << "the second pass completes it";
  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 1.0) << "2^70 mod 3";
}

// FNCLEX clears the stack fault and busy bits along with the six exception flags and ES. seven's
// mask skipped SF, so a guest that cleared after a stack fault still read one back.
TEST(KuberaScalar, FnclexClearsTheStackFaultBit) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  write_bytes(memory, kBase, seven::parse_hex_bytes("D9 E0 DB E2"));  // fchs on an empty stack ; fnclex

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  ASSERT_NE(state.get_x87_status_word() & kX87Sf, 0u) << "sanity: the underflow raised SF";

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.get_x87_status_word() & (kX87Ie | kX87Sf | 0x0080u), 0u);
}

// log2(0) is minus infinity, which the x87 reports as a divide-by-zero. seven called it an invalid
// operand and then left the stack untouched, so the guest neither got the result nor the pop.
TEST(KuberaScalar, Fyl2xOnZeroIsADivideByZeroNotAnInvalidOperand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0x8000000000000000ull, 0x3FFF);  // 1.0, the multiplier
  write_extf80(memory, kX87Data + 16, 0ull, 0x0000);              // +0.0, the argument
  // fld tbyte [rbx] ; fld tbyte [rbx+16] ; fyl2x
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B DB 6B 10 D9 F1"));

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
  }

  const auto sw = state.get_x87_status_word();
  EXPECT_NE(sw & kX87Ze, 0u) << "ST(0) of zero is a divide-by-zero";
  EXPECT_EQ(sw & kX87Ie, 0u) << "and not an invalid operand";
  EXPECT_TRUE(seven::isinf(state.x87_get(0)));
  EXPECT_TRUE(seven::signbit(state.x87_get(0)));
  EXPECT_EQ(state.get_x87_top(), 7u) << "fyl2x pops";
}

// FSCALE takes its shift count from ST(1), which is guest data, and the count was narrowed straight
// to int. Anything past int64's range came back as softfloat's out-of-range default and truncated to
// zero, so a scale by 2^70 or by infinity quietly returned ST(0) unchanged.
TEST(KuberaScalar, FscaleClampsAShiftCountThatDoesNotFitInAnInt) {
  const auto scale_by = [](std::uint64_t significand, std::uint16_t sign_exp) {
    seven::Executor executor{};
    seven::CpuState state{};
    seven::Memory memory{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[3] = kX87Data;
    memory.map(kX87Data, 0x1000);
    write_extf80(memory, kX87Data, significand, sign_exp);
    // fld tbyte [rbx] ; fld1 ; fscale
    write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B D9 E8 D9 FD"));
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::none) << "step " << i;
    }
    return state.x87_get(0);
  };

  const auto scaled_up = scale_by(0x8000000000000000ull, 0x4045);  // 2^70
  EXPECT_TRUE(seven::isinf(scaled_up)) << "1.0 scaled by 2^70 overflows";
  EXPECT_FALSE(seven::signbit(scaled_up));

  const auto scaled_down = scale_by(0x8000000000000000ull, 0xC045);  // -2^70
  EXPECT_EQ(static_cast<double>(scaled_down), 0.0) << "and scaling down that far underflows to zero";

  const auto scaled_by_infinity = scale_by(0x8000000000000000ull, 0x7FFF);  // +inf
  EXPECT_TRUE(seven::isinf(scaled_by_infinity));
}

// FXTRACT had no handler at all, so it stopped the guest as an unsupported instruction.
TEST(KuberaScalar, FxtractSplitsTheExponentFromTheSignificand) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kX87Data;
  memory.map(kX87Data, 0x1000);
  write_extf80(memory, kX87Data, 0xC000000000000000ull, 0x4002);  // 12.0 = 1.5 * 2^3
  // fld tbyte [rbx] ; fxtract
  write_bytes(memory, kBase, seven::parse_hex_bytes("DB 2B D9 F4"));

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);

  EXPECT_EQ(static_cast<double>(state.x87_get(0)), 1.5) << "the significand is pushed";
  EXPECT_EQ(static_cast<double>(state.x87_get(1)), 3.0) << "the unbiased exponent stays below it";
}

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <span>

#include <iced_x86/decoder.hpp>
#include <iced_x86/instruction_info.hpp>

#include "seven/flag_liveness.hpp"
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

TEST(KuberaLiveness, MaskedWriteSurvivesMovCrUdFaultViaRun) {
  // Same shape as MaskedWriteSurvivesMidBlockFaultViaRun, but the faulting second instruction is
  // MOV r32, CR1 rather than a memory operand -- MOV to/from a control or debug register has two
  // REGISTER-kind operands, so can_fault()'s operand-kind loop can't see it, and it needs its own
  // explicit case (added alongside CALL/RET/PUSH/POP) or this add's flags would be wrongly masked
  // as "covered" by an instruction that never actually completes. CR1 is architecturally reserved
  // (real hardware only defines CR0/CR2/CR3/CR4/CR8), so this UDs before ever touching rax.
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  write_bytes(memory, kBase, {0x48, 0x01, 0xD8, 0x0F, 0x20, 0xC8});  // add rax,rbx ; mov eax,cr1
  state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;  // rax
  state.gpr[3] = 1ull;                   // rbx

  const auto result = executor.run(state, memory, 100);
  ASSERT_EQ(result.reason, seven::StopReason::invalid_opcode);
  EXPECT_EQ(state.gpr[0], 0u);
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
}

TEST(KuberaLiveness, MaskedWriteSurvivesWrmsrGpFaultViaRun) {
  // Same shape as MaskedWriteSurvivesMovCrUdFaultViaRun, but for the CPL0-only system
  // instructions (CLTS/SWAPGS/WRMSR*/RDMSR*/XSETBV) added to can_fault() alongside CR/DR --
  // WRMSR reads its operands from fixed registers (ECX/EAX/EDX), never an OpKind::MEMORY operand,
  // so it needs the same explicit can_fault() case. CPL 3 makes it #GP before ever writing the MSR.
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.sreg[1] = 0x2B;  // CS selector with RPL 3 -- CPL 3
  write_bytes(memory, kBase, {0x48, 0x01, 0xD8, 0x0F, 0x30});  // add rax,rbx ; wrmsr
  state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;  // rax
  state.gpr[3] = 1ull;                   // rbx

  const auto result = executor.run(state, memory, 100);
  ASSERT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.gpr[0], 0u);
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
}

TEST(KuberaLiveness, MaskedWriteSurvivesCliGpFaultViaRun) {
  // Same shape again: CLI reads no memory operand and writes no ALU status flag, but can now #GP
  // at CPL>IOPL, so it needs the same explicit can_fault() case as the other CPL-gated
  // instructions above.
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.sreg[1] = 0x2B;  // CPL 3, IOPL 0 -- CPL > IOPL
  write_bytes(memory, kBase, {0x48, 0x01, 0xD8, 0xFA});  // add rax,rbx ; cli
  state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;  // rax
  state.gpr[3] = 1ull;                   // rbx

  const auto result = executor.run(state, memory, 100);
  ASSERT_EQ(result.reason, seven::StopReason::general_protection);
  EXPECT_EQ(state.gpr[0], 0u);
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u);
}

TEST(KuberaLiveness, JitBypassEligibleReflectsHooksAndTrapState) {
  // jit_bypass_eligible() is a narrow public surface for an external native-codegen consumer (see
  // seven-jit's JitExecutor) to ask "can I run my own code for a span of instructions without
  // going through step()/step_impl() at all" -- it needs to say no for exactly the same reasons
  // flag-liveness masking does: a hook that needs full per-instruction visibility, or a runtime
  // condition (trap flag, active hardware breakpoint) that requires per-instruction stepping
  // regardless of hooks.
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;

  EXPECT_TRUE(executor.jit_bypass_eligible(state, memory));

  const auto hook_id = executor.add_instruction_hook(
      [](seven::InstructionHookContext&) { return seven::InstructionHookResult{}; });
  EXPECT_FALSE(executor.jit_bypass_eligible(state, memory));
  ASSERT_TRUE(executor.remove_hook(hook_id));
  EXPECT_TRUE(executor.jit_bypass_eligible(state, memory));

  state.rflags |= seven::kFlagTF;
  EXPECT_FALSE(executor.jit_bypass_eligible(state, memory));
  state.rflags &= ~seven::kFlagTF;
  EXPECT_TRUE(executor.jit_bypass_eligible(state, memory));

  state.dr[7] = 1;
  EXPECT_FALSE(executor.jit_bypass_eligible(state, memory));
  state.dr[7] = 0;
  EXPECT_TRUE(executor.jit_bypass_eligible(state, memory));
}

// The whole cross-instruction masking argument rests on "a branch is always the LAST instruction in
// a lifted span," which step_impl enforces by stopping the lift at the first instruction whose
// flow_control() isn't NEXT. But this fork's flow_control() is a hand-rolled stub (see
// instruction_info.cpp) that only knows Jcc/JMP/CALL/RET/INT3 -- LOOP, LOOPE/LOOPNE and
// JCXZ/JECXZ/JRCXZ all fall through to NEXT. flag_liveness.cpp separately models plain
// LOOP/JECXZ/JRCXZ as reading and writing nothing, so they're fully transparent to the backward
// pass too. Together that let a span run straight THROUGH a branch: an instruction after the branch
// could "cover" a flag write before it, and when the branch is actually taken that covering
// instruction never executes, leaving the guest looking at a stale flag.
//
// jrcxz with rcx==0 jumps over the cmp, so the cmp can never be the thing that writes ZF here --
// only the add's own (elided) write could have.
TEST(KuberaLiveness, TakenJrcxzIsABlockBoundarySoEarlierFlagWriteIsNotElided) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  //   add eax, ebx     <- writes ZF (0 + 0 == 0, so ZF must end up SET)
  //   jrcxz +2         <- taken (rcx == 0), jumps to the hlt
  //   cmp ecx, edx     <- the "cover"; skipped entirely by the taken branch
  //   hlt              <- stops run() before anything else can touch flags
  write_bytes(memory, kBase, seven::parse_hex_bytes("01 D8 E3 02 39 D1 F4"));
  state.gpr[0] = 0;   // rax
  state.gpr[3] = 0;   // rbx
  state.gpr[1] = 0;   // rcx: zero, so the jrcxz is taken
  state.gpr[2] = 1;   // rdx: makes the skipped cmp a ZF-clearing compare
  state.rflags = 0x202;  // ZF clear going in

  // Budget well above kMaxBlockLiftLength so run() actually enables masking.
  const auto result = executor.run(state, memory, 64);
  ASSERT_EQ(result.reason, seven::StopReason::halted);
  EXPECT_EQ(state.rip, kBase + 6);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u)
      << "add's ZF write was elided by a cmp that the taken jrcxz skipped over";
}

// can_fault() decides two separate things: whether flag liveness must stay conservative across an
// instruction, and (in seven-jit) whether the JIT's callout bridge may inline it. Its operand loop
// tests for OpKind::MEMORY, but iced gives the string instructions their own operand kinds --
// MEMORY_SEG_RSI / MEMORY_ESRDI and the 16/32-bit variants, all distinct enum values from MEMORY --
// so MOVS/CMPS/SCAS/STOS/LODS fell through to the explicit switch, which never listed them, and
// can_fault() reported false for instructions whose whole purpose is touching guest memory. Exactly
// the implicit-memory-access gap the CALL/RET and PUSH/POP entries in that switch already exist to
// close.
TEST(KuberaLiveness, StringInstructionsAreRecognizedAsFaultCapable) {
  struct Case { const char* name; const char* bytes; };
  // rep-prefixed and bare forms both decode to the same underlying string Code. The maskmov pair
  // is here for the same reason: their destination is an implicit ES:[rDI] operand, so they were
  // reported as unable to fault while writing up to 16 bytes of guest memory, which also made them
  // eligible for the JIT's callout bridge and let a self-modifying maskmovdqu leave the rest of a
  // compiled block running the bytes it was compiled from.
  const Case cases[] = {
      {"movsb", "A4"},   {"movsq", "48 A5"}, {"cmpsb", "A6"},   {"cmpsq", "48 A7"},
      {"scasb", "AE"},   {"scasq", "48 AF"}, {"stosb", "AA"},   {"stosq", "48 AB"},
      {"lodsb", "AC"},   {"lodsq", "48 AD"},
      {"maskmovdqu", "66 0F F7 C1"}, {"maskmovq", "0F F7 C1"},
      {"vmaskmovdqu", "C5 F9 F7 C1"},
  };
  for (const auto& c : cases) {
    const auto raw = seven::parse_hex_bytes(c.bytes);
    iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(raw.data(), raw.size()), 0x1000);
    const auto decoded = decoder.decode();
    ASSERT_TRUE(decoded.has_value()) << c.name;
    EXPECT_TRUE(seven::can_fault(decoded.value()))
        << c.name << " reads or writes guest memory, so it must be treated as fault-capable";
  }
}

// InstructionExtensions::encoding() used to return EncodingKind::LEGACY unconditionally, with a
// comment admitting it was a placeholder. Executor::simd_profile_allows() gates the AVX and AVX-512
// build profiles on it, so both of those checks were dead code: a build configured with
// SEVEN_ENABLE_AVX512=0 still accepted EVEX instructions. The surviving vector-width check hides
// this for ZMM/YMM operands, so the case that actually slipped through was an EVEX-encoded
// instruction on XMM registers, which is also where the opmask semantics live.
TEST(KuberaLiveness, InstructionEncodingIsClassifiedNotAssumedLegacy) {
  struct Case { const char* name; const char* bytes; iced_x86::EncodingKind expected; };
  const Case cases[] = {
      {"add eax, ecx", "01 C8", iced_x86::EncodingKind::LEGACY},
      {"movaps xmm0, xmm1", "0F 28 C1", iced_x86::EncodingKind::LEGACY},
      {"vmovaps xmm0, xmm1", "C5 F8 28 C1", iced_x86::EncodingKind::VEX},
      {"vmovaps ymm0, ymm1", "C5 FC 28 C1", iced_x86::EncodingKind::VEX},
      // EVEX on XMM specifically -- the shape the width check cannot catch.
      {"vpaddd xmm0, xmm1, xmm2", "62 F1 75 08 FE C2", iced_x86::EncodingKind::EVEX},
  };
  for (const auto& c : cases) {
    const auto raw = seven::parse_hex_bytes(c.bytes);
    iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(raw.data(), raw.size()), 0x1000);
    const auto decoded = decoder.decode();
    ASSERT_TRUE(decoded.has_value()) << c.name;
    EXPECT_EQ(iced_x86::InstructionExtensions::encoding(decoded.value()), c.expected) << c.name;
  }
}

// The dead-flags mask lives in one thread_local that step_impl assigns just before it dispatches a
// handler. A handler's memory access can land on an MMIO device, and that host callback is free to
// call run() again; the nested frame overwrote the mask and never put it back, so every flag write
// the outer handler still had to do was filtered through the wrong block's liveness result.
TEST(KuberaLiveness, ANestedRunFromAnMmioCallbackDoesNotClobberTheOuterMask) {
  constexpr std::uint64_t kMmioBase = 0x8000;
  constexpr std::uint64_t kNestedCode = 0x2000;

  seven::Executor executor{};
  seven::Memory memory{};
  seven::CpuState state{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;

  memory.map(kBase, 0x1000);
  memory.map(kNestedCode, 0x1000);
  // mov rbx, 0x8000 ; add rax, [rbx] ; cmp rax, rax ; hlt
  // The cmp overwrites every ALU flag the add just wrote without reading one, so the add is
  // exactly the shape the liveness pass hands a non-empty mask to.
  write_bytes(memory, kBase, seven::parse_hex_bytes("48 BB 00 80 00 00 00 00 00 00 48 03 03 48 39 C0 F4"));
  write_bytes(memory, kNestedCode, seven::parse_hex_bytes("48 31 C0 F4"));  // xor rax, rax ; hlt

  bool ran_nested = false;
  std::uint64_t mask_at_entry = 0;
  std::uint64_t mask_after_nested = 0;

  const auto id = memory.map_mmio(
      kMmioBase, 0x1000,
      [&](std::uint64_t, void* dst, std::size_t size) {
        std::memset(dst, 0, size);
        if (!ran_nested) {
          ran_nested = true;
          mask_at_entry = seven::detail::dead_flags_mask();
          seven::CpuState nested_state{};
          nested_state.mode = seven::ExecutionMode::long64;
          nested_state.rip = kNestedCode;
          (void)executor.run(nested_state, memory, 200);
          mask_after_nested = seven::detail::dead_flags_mask();
        }
        return true;
      },
      [](std::uint64_t, const void*, std::size_t) { return true; });
  ASSERT_NE(id, 0u);

  (void)executor.run(state, memory, 200);

  ASSERT_TRUE(ran_nested);
  ASSERT_NE(mask_at_entry, 0u) << "nothing is being proved unless the outer add really carried a mask";
  EXPECT_EQ(mask_after_nested, mask_at_entry)
      << "the nested run left its own mask installed for the rest of the outer handler";
}

// The fault-capable case exists so that nothing after a possible fault can be used to justify
// dropping a flag write before it. It used to apply that only to writes made EARLIER in the block:
// the boost was folded into `live` for the next iteration, while the current instruction's own
// dead mask had already been computed against the unboosted value.
//
// That leaves the faulting instruction's own write unprotected, and it is observable. seven_core's
// handlers commit flags before attempting the write-back, so a store that faults still updates the
// flags, and the instruction that was supposed to overwrite them never runs. A JIT-vs-interpreter
// fuzz lane caught this as the two engines reporting different flags after the same faulting store.
TEST(KuberaLiveness, AFaultCapableInstructionKeepsItsOwnFlagWrite) {
  // xor [rbp+0x26], edx then sar rdi, 1. The sar overwrites every flag the xor writes, which is
  // exactly the reasoning that used to mark the xor's write dead.
  const auto raw = seven::parse_hex_bytes("31 55 26 48 D1 FF");
  iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(raw.data(), raw.size()), 0x1000);
  std::vector<iced_x86::Instruction> instrs;
  for (int i = 0; i < 2; ++i) {
    const auto decoded = decoder.decode();
    ASSERT_TRUE(decoded.has_value());
    instrs.push_back(decoded.value());
  }
  ASSERT_TRUE(seven::can_fault(instrs[0]));

  std::vector<seven::FlagLivenessInstr> liveness;
  for (const auto& instr : instrs) liveness.push_back({&instr, 0});
  seven::compute_flag_liveness(liveness);

  EXPECT_EQ(liveness[0].dead_flags_mask, 0u)
      << "a store that can fault must keep the flag write the fault would expose";
}

// The lift's boundary test ignores handler coverage, so an unhandled opcode does sit mid-span. What
// keeps that sound is FlagsInfo defaulting read to all flags, which pins `live` and blocks any cover
// across it. RDRAND stands in: register only, no trap kind, not control flow.
TEST(KuberaLiveness, MaskedWriteSurvivesAnUnsupportedInstructionMidBlock) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  state.rflags = 0x202;
  memory.map(0x4000, 0x2000);
  // add eax, ebx ; rdrand eax ; cmp eax, ebx
  write_bytes(memory, kBase, {0x01, 0xD8, 0x0F, 0xC7, 0xF0, 0x39, 0xD8});
  state.gpr[0] = 5ull;
  state.gpr[3] = 0xFFFFFFFBull;

  const auto result = executor.run(state, memory, 64);
  ASSERT_EQ(result.reason, seven::StopReason::unsupported_instruction);
  EXPECT_EQ(state.rip, kBase + 2);
  EXPECT_EQ(state.gpr[0], 0u);
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u)
      << "the add's flag write was masked as covered by a cmp the run never reached";
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u);
}


// The invariant the test above leans on. The flags table decides who may cover an earlier write;
// handled_codes.def decides who can actually run. Nothing ties them together, so an entry for a code
// with no handler would drop every flag write in front of it and then never execute. dead_flags_mask
// over [add eax,ebx ; C] is exactly what liveness would drop, so a non-zero mask means C claims a cover.
TEST(KuberaLiveness, KeepsEveryFlagsTableEntryExecutable) {
  const std::vector<std::vector<std::uint8_t>> prefixes = {
      {}, {0x0F}, {0x66}, {0x66, 0x0F}, {0xF2}, {0xF3}, {0xF3, 0x0F},
      {0x48}, {0x48, 0x0F}, {0x4C}, {0x66, 0x48, 0x0F},
      {0x0F, 0x38}, {0x66, 0x0F, 0x38}, {0x0F, 0x3A}, {0x66, 0x0F, 0x3A},
      {0xC4, 0xE2, 0x78}, {0xC4, 0xE2, 0xF8},
  };
  const std::uint8_t probe_bytes[] = {0x01, 0xD8};  // add eax, ebx
  iced_x86::Decoder probe_decoder(64, std::span<const std::uint8_t>(probe_bytes, sizeof(probe_bytes)), 0x1000);
  const auto probe = probe_decoder.decode();
  ASSERT_TRUE(probe.has_value());
  const auto probe_instr = probe.value();

  constexpr std::uint64_t kCode = 0x400000ull;
  constexpr std::uint64_t kData = 0x500000ull;
  seven::Executor executor{};
  seven::Memory memory{};
  memory.map(kCode, 0x1000);
  memory.map(kData, 0x2000);

  std::size_t claiming = 0;
  std::vector<std::string> unrunnable;
  for (const auto& prefix : prefixes) {
    for (unsigned opcode = 0; opcode < 256; ++opcode) {
      for (unsigned modrm = 0; modrm < 256; ++modrm) {
        std::vector<std::uint8_t> bytes = prefix;
        bytes.push_back(static_cast<std::uint8_t>(opcode));
        bytes.push_back(static_cast<std::uint8_t>(modrm));
        for (int i = 0; i < 6; ++i) bytes.push_back(0x11);
        ASSERT_TRUE(memory.write_unchecked(kCode, bytes.data(), bytes.size()));
        iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(bytes.data(), bytes.size()), kCode);
        const auto decoded = decoder.decode();
        if (!decoded.has_value() || decoded.value().code() == iced_x86::Code::INVALID) continue;
        const auto candidate = decoded.value();
        seven::FlagLivenessInstr items[2] = {{&probe_instr, 0}, {&candidate, 0}};
        seven::compute_flag_liveness(std::span<seven::FlagLivenessInstr>(items, 2));
        if (items[0].dead_flags_mask == 0) continue;
        ++claiming;

        seven::CpuState state{};
        state.mode = seven::ExecutionMode::long64;
        state.rip = kCode;
        state.rflags = 0x202;
        state.sreg[1] = 0x08;
        for (std::size_t i = 0; i < 16; ++i) state.gpr[i] = kData + 0x800;
        state.gpr[4] = kData + 0x1000;
        const auto result = executor.step(state, memory);
        if (result.reason != seven::StopReason::unsupported_instruction) continue;
        if (unrunnable.size() < 8) {
          char buf[64] = {};
          std::snprintf(buf, sizeof(buf), "code=%d", static_cast<int>(candidate.code()));
          unrunnable.emplace_back(buf);
        }
      }
    }
  }

  EXPECT_GT(claiming, 5000u) << "nothing is being proved unless the sweep reached the flags table";
  EXPECT_TRUE(unrunnable.empty())
      << "these claim an unconditional flag write but have no handler, so liveness would drop a "
         "preceding write that nothing restores: "
      << (unrunnable.empty() ? std::string{} : unrunnable[0]);
}

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

// Masking a flag write because a later instruction covers it is only safe if that instruction is
// guaranteed to run first. These target the two things that can interrupt a block mid-way even
// under run(): a fault on a later instruction, and the single-step trap flag.

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
  // add rax, rbx ; add [rdx], rcx. Without the can_fault barrier the first add's flags are masked
  // as covered by the second, which never completes because it faults. rax/rbx wrap to zero so CF
  // and ZF are unambiguous.
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
  // add rax, rbx ; add rcx, rdx. The second would cover the first under plain liveness, but TF
  // stops after the first, so the cover never runs before the debug interrupt exposes rflags. The
  // handler halts rather than iretq-ing back, so execution stops at that first trap.
  write_bytes(memory, kBase, {0x48, 0x01, 0xD8, 0x48, 0x01, 0xD1});
  const std::uint8_t hlt[] = {0xF4};
  (void)memory.write(kDbHandler, hlt, sizeof(hlt));
  // Ring-0 gate selector, unlike the 0x33 the other IDT tests use, because the gate's selector
  // becomes the handler's CPL and HLT is CPL0-only. A real exception gate points at ring 0 anyway.
  write_idt_gate64(memory, kIdtBase, 1, 0x08, kDbHandler);
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
  // Same shape, but the faulting instruction is MOV r32, CR1: two REGISTER-kind operands, so
  // can_fault()'s operand loop cannot see it and it needs its own case. CR1 is reserved, so this
  // UDs before touching rax.
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
  // Same shape, for the CPL0-only system instructions. WRMSR reads fixed registers, never a memory
  // operand, so it needs its own can_fault() case; CPL 3 makes it #GP before writing the MSR.
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

// The last three CPL0-only instructions with no privilege check. Their bodies do nothing, so it
// read as a fidelity gap until the hardware oracle ran its lanes at ring 3: 159 divergences in 20k
// iterations, all three. Same can_fault() requirement, since none has a memory operand.
TEST(KuberaLiveness, MaskedWriteSurvivesInvdGpFaultViaRun) {
  const std::vector<std::pair<std::vector<std::uint8_t>, const char*>> cases = {
      {{0x0F, 0x08}, "invd"},
      {{0x0F, 0x09}, "wbinvd"},
      {{0xF4}, "hlt"},
  };
  for (const auto& [tail, name] : cases) {
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    state.sreg[1] = 0x2B;  // CPL 3
    std::vector<std::uint8_t> code = {0x48, 0x01, 0xD8};  // add rax,rbx
    code.insert(code.end(), tail.begin(), tail.end());
    write_bytes(memory, kBase, code);
    state.gpr[0] = 0xFFFFFFFFFFFFFFFFull;
    state.gpr[3] = 1ull;

    const auto result = executor.run(state, memory, 100);
    ASSERT_EQ(result.reason, seven::StopReason::general_protection) << name << " must #GP at CPL 3";
    EXPECT_EQ(state.gpr[0], 0u) << name;
    EXPECT_NE(state.rflags & seven::kFlagCF, 0u) << name << " must not lose the covering flag write";
    EXPECT_NE(state.rflags & seven::kFlagZF, 0u) << name;
  }
}

TEST(KuberaLiveness, TheCpl0OnlyCacheInstructionsStillRunAtRing0) {
  // The other half of the gate: the check is on privilege, not a blanket refusal.
  for (const std::vector<std::uint8_t> code : {std::vector<std::uint8_t>{0x0F, 0x08},
                                               std::vector<std::uint8_t>{0x0F, 0x09}}) {
    seven::CpuState state{};
    seven::Memory memory{};
    seven::Executor executor{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kBase;
    state.gpr[4] = kStackTop;
    write_bytes(memory, kBase, code);
    EXPECT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  }
}

TEST(KuberaLiveness, JitBypassEligibleReflectsHooksAndTrapState) {
  // jit_bypass_eligible() lets an external codegen consumer ask whether it may run a span without
  // going through step(). It has to say no for the same reasons masking does: a hook needing
  // per-instruction visibility, or TF/DR7 requiring per-instruction stepping regardless.
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

// Masking rests on a branch ending its lifted span, which the lifter took off flow_control(), a stub
// reporting NEXT for LOOP and JCXZ. A span could then run through a branch and let a later
// instruction cover a flag write that the taken branch skips.
TEST(KuberaLiveness, TakenJrcxzIsABlockBoundarySoEarlierFlagWriteIsNotElided) {
  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  memory.map(kBase, 0x1000);
  // add sets ZF, the taken jrcxz jumps over the cmp that was supposed to cover it, hlt stops run()
  // before anything else touches flags.
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

// can_fault() gates both liveness and JIT inlining, and its operand loop tested OpKind::MEMORY only,
// which the string instructions do not use, so it reported false for instructions built to touch
// guest memory.
TEST(KuberaLiveness, StringInstructionsAreRecognizedAsFaultCapable) {
  struct Case { const char* name; const char* bytes; };
  // rep-prefixed and bare forms share a Code. The maskmov pair is here for the same reason: an
  // implicit ES:[rDI] destination reported as unable to fault while writing 16 bytes, which also
  // let a self-modifying maskmovdqu leave the rest of a block running stale bytes.
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

// encoding() used to return LEGACY unconditionally, so simd_profile_allows()'s AVX and AVX-512
// gates were dead code and SEVEN_ENABLE_AVX512=0 still accepted EVEX. The width check hides that
// for ZMM/YMM, so what slipped through was EVEX on XMM, where the opmask semantics live.
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

// The dead-flags mask is one thread_local set just before dispatch. An MMIO callback is free to call
// run() again, and the nested frame overwrote the mask without restoring it.
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

// The fault-capable boost covered earlier writes only, not the faulting instruction's own. Handlers
// commit flags before the write-back, so a faulting store updates them and its cover never runs.
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


// The flags table decides who may cover an earlier write and handled_codes.def decides who can run,
// with nothing tying them together, so an entry with no handler drops writes and never executes.
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

// The context-sync callbacks hand out a CpuState at every instruction boundary, the same mid-span
// observation hazard the hook checks cover. The JIT's gate accounted for them, the masking gate did
// not.
TEST(KuberaLiveness, ContextSyncCallbacksSeeFlagsAtEveryBoundary) {
  const auto flags_after_first = [&](const std::vector<std::uint8_t>& code) {
    seven::Memory memory{};
    memory.map(0x1000, 0x1000);
    EXPECT_TRUE(memory.write(0x1000, code.data(), code.size()));
    seven::Executor executor{};
    std::vector<std::uint64_t> seen;
    executor.set_context_write_callback([&](seven::CpuState& state) {
      seen.push_back(state.rflags);
      return true;
    });
    seven::CpuState state{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = 0x1000;
    state.rflags = 0x202;
    state.gpr[0] = 7;   // rax
    state.gpr[3] = 7;   // rbx
    state.gpr[8] = 1;   // r8
    state.gpr[4] = 0x1800;
    (void)executor.run(state, memory, 1000);
    EXPECT_FALSE(seen.empty());
    return seen.empty() ? std::uint64_t{0} : seen.front();
  };

  // cmp rax,rbx ; hlt -- nothing covers the flag write, so nothing can be elided.
  const std::uint64_t alone = flags_after_first({0x48, 0x39, 0xD8, 0xF4});
  // cmp rax,rbx ; test r8,r8 ; hlt -- the test covers every status flag cmp wrote.
  const std::uint64_t covered = flags_after_first({0x48, 0x39, 0xD8, 0x4D, 0x85, 0xC0, 0xF4});

  EXPECT_EQ(covered & seven::kAluStatusFlagsMask, alone & seven::kAluStatusFlagsMask)
      << "the callback was handed flags the covering instruction had not written yet";
}

// A rep-prefixed SCAS reads back the ZF its own compare wrote. The table declares that read, but
// `read` folds into the live set only after the current mask is computed, so it protects the
// instruction before it, not itself -- and SCAS writes AL, so nothing else covered it.
TEST(KuberaLiveness, RepneScasKeepsTheZeroFlagItLoopsOn) {
  constexpr std::uint64_t kProg = 0x1000;
  constexpr std::uint64_t kData = 0x2000;
  seven::Memory memory{};
  seven::Executor executor{};
  memory.map(kProg, 0x1000);
  memory.map(kData, 0x1000);
  // repne scasb ; xor eax,eax ; hlt
  const std::uint8_t code[] = {0xF2, 0xAE, 0x31, 0xC0, 0xF4};
  ASSERT_TRUE(memory.write(kProg, code, sizeof(code)));

  seven::CpuState state{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kProg;
  state.rflags = 0x202 | seven::kFlagZF;  // DF clear, ZF set on entry
  state.gpr[0] = 0x41;                    // AL, never present in the buffer
  state.gpr[1] = 16;                      // RCX
  state.gpr[7] = kData;                   // RDI
  state.gpr[4] = kProg + 0x800;

  const auto result = executor.run(state, memory, 256);
  EXPECT_EQ(result.reason, seven::StopReason::halted);
  EXPECT_EQ(state.gpr[1], 0u) << "the scan stopped early on a zero flag that was masked away";
  EXPECT_EQ(state.gpr[7], kData + 16) << "rdi must have walked the whole buffer";
}

// `aam 0` is the third divide-error source and the only one missing from can_fault(). Its flags entry
// declares an unconditional write, which is what masks the add below, but on the #DE path the handler
// returns before writing any of them. compat32 because AAM does not decode in long mode.
TEST(KuberaLiveness, MaskedWriteSurvivesAamDivideErrorViaRun) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::compat32;
  state.rip = kBase;
  state.gpr[4] = kStackTop;
  memory.map(0x4000, 0x2000);
  // add al, bl ; aam 0
  write_bytes(memory, kBase, {0x00, 0xD8, 0xD4, 0x00});
  state.gpr[0] = 0xFFull;  // AL
  state.gpr[3] = 0x01ull;  // BL
  state.rflags = 0x202;    // CF/ZF/PF/AF all clear on entry, so a stale read is unambiguous

  const auto result = executor.run(state, memory, 100);
  ASSERT_EQ(result.reason, seven::StopReason::divide_error);
  EXPECT_EQ(state.gpr[0] & 0xFFull, 0u) << "the add itself must have retired";
  EXPECT_NE(state.rflags & seven::kFlagCF, 0u) << "add al,bl carried out of bit 7";
  EXPECT_NE(state.rflags & seven::kFlagZF, 0u) << "add al,bl produced zero";
  EXPECT_NE(state.rflags & seven::kFlagPF, 0u) << "add al,bl produced even parity";
  EXPECT_NE(state.rflags & seven::kFlagAF, 0u) << "add al,bl carried out of bit 3";
}

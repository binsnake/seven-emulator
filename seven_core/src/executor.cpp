#include "seven/executor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include <iced_x86/code.hpp>
#include <iced_x86/decoder.hpp>
#include <iced_x86/flow_control.hpp>
#include <iced_x86/instruction_info.hpp>
#include <iced_x86/memory_size_info.hpp>

#include "seven/flag_liveness.hpp"
#include "seven/handler_helpers.hpp"
#include "seven/handlers_fwd.hpp"

namespace seven {

namespace {

[[nodiscard]] bool env_flag_set(const char* name) noexcept {
  if (const char* v = std::getenv(name)) {
    return v[0] != '\0' && v[0] != '0';
  }
  return false;
}

}  // namespace

Executor::Executor()
    : code_execution_counts_(kCodeCount, 0),
      stop_reason_counts_(kStopReasonCount, 0) {
  trace_semantics_ = env_flag_set("SEVEN_TRACE_SEMANTICS");
  collect_code_stats_ = env_flag_set("SEVEN_COLLECT_CODE_STATS");
  decode_cache_disabled_by_env_ = env_flag_set("SEVEN_DISABLE_DECODE_CACHE");
}

namespace {

constexpr bool kEnableAvx = SEVEN_ENABLE_AVX != 0;
constexpr bool kEnableAvx512 = SEVEN_ENABLE_AVX512 != 0;
constexpr std::size_t kMaxFaultRetries = Executor::kMaxFaultRetries;
constexpr std::size_t kMaxStepDepth = Executor::kMaxStepDepth;

constexpr std::size_t kVectorRegisterCount = 32;
constexpr std::size_t kXmmWidth = 16;
constexpr std::size_t kYmmWidth = 32;
constexpr std::size_t kZmmWidth = 64;

[[nodiscard]] std::size_t vector_width_for_register(iced_x86::Register reg) noexcept {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= zmm0 && value < zmm0 + kVectorRegisterCount) return kZmmWidth;
  if (value >= ymm0 && value < ymm0 + kVectorRegisterCount) return kYmmWidth;
  if (value >= xmm0 && value < xmm0 + kVectorRegisterCount) return kXmmWidth;
  return 0;
}

[[nodiscard]] iced_x86::Code normalize_reported_code(iced_x86::Code code) noexcept {
  switch (code) {
    case iced_x86::Code::PUSHD_IMM8:
      return iced_x86::Code::PUSHQ_IMM8;
    case iced_x86::Code::PUSHD_IMM32:
      return iced_x86::Code::PUSHQ_IMM32;
    default:
      return code;
  }
}

[[nodiscard]] std::optional<TrapKind> trap_kind_for_code(iced_x86::Code code) noexcept {
  switch (code) {
    case iced_x86::Code::SYSCALL:
      return TrapKind::syscall;
    case iced_x86::Code::CPUID:
      return TrapKind::cpuid;
    case iced_x86::Code::RDTSC:
      return TrapKind::rdtsc;
    case iced_x86::Code::RDTSCP:
      return TrapKind::rdtscp;
    case iced_x86::Code::INT1:
    case iced_x86::Code::INT3:
    case iced_x86::Code::INT_IMM8:
    case iced_x86::Code::INTO:
      return TrapKind::interrupt;
    default:
      return std::nullopt;
  }
}

// Control flow the flow_control() stub still reports as NEXT, since it only knows
// Jcc/JMP/CALL/RET/INT3. Masking depends on a branch ending the span, so LOOP and JCXZ slipping
// through let an instruction that never runs cover a flag write.
[[nodiscard]] bool ends_lifted_block(const iced_x86::Instruction& instr) noexcept {
  switch (instr.code()) {
    case iced_x86::Code::LOOP_REL8_16_CX: case iced_x86::Code::LOOP_REL8_32_CX:
    case iced_x86::Code::LOOP_REL8_16_ECX: case iced_x86::Code::LOOP_REL8_32_ECX:
    case iced_x86::Code::LOOP_REL8_64_ECX: case iced_x86::Code::LOOP_REL8_16_RCX:
    case iced_x86::Code::LOOP_REL8_64_RCX:
    case iced_x86::Code::LOOPE_REL8_16_CX: case iced_x86::Code::LOOPE_REL8_32_CX:
    case iced_x86::Code::LOOPE_REL8_16_ECX: case iced_x86::Code::LOOPE_REL8_32_ECX:
    case iced_x86::Code::LOOPE_REL8_64_ECX: case iced_x86::Code::LOOPE_REL8_16_RCX:
    case iced_x86::Code::LOOPE_REL8_64_RCX:
    case iced_x86::Code::LOOPNE_REL8_16_CX: case iced_x86::Code::LOOPNE_REL8_32_CX:
    case iced_x86::Code::LOOPNE_REL8_16_ECX: case iced_x86::Code::LOOPNE_REL8_32_ECX:
    case iced_x86::Code::LOOPNE_REL8_64_ECX: case iced_x86::Code::LOOPNE_REL8_16_RCX:
    case iced_x86::Code::LOOPNE_REL8_64_RCX:
    case iced_x86::Code::JECXZ_REL8_16: case iced_x86::Code::JECXZ_REL8_32:
    case iced_x86::Code::JECXZ_REL8_64:
    case iced_x86::Code::JRCXZ_REL8_16: case iced_x86::Code::JRCXZ_REL8_64:
    // Not branches, but they end the run just as hard -- execution never reaches the next
    // sequential instruction, so nothing after them can be trusted to cover anything.
    case iced_x86::Code::SYSENTER: case iced_x86::Code::SYSEXITD: case iced_x86::Code::SYSEXITQ:
    case iced_x86::Code::SYSRETD: case iced_x86::Code::SYSRETQ:
    case iced_x86::Code::IRETW: case iced_x86::Code::IRETD: case iced_x86::Code::IRETQ:
    case iced_x86::Code::HLT:
    case iced_x86::Code::UD0: case iced_x86::Code::UD0_R16_RM16:
    case iced_x86::Code::UD0_R32_RM32: case iced_x86::Code::UD0_R64_RM64:
    case iced_x86::Code::UD1_R16_RM16: case iced_x86::Code::UD1_R32_RM32:
    case iced_x86::Code::UD1_R64_RM64: case iced_x86::Code::UD2:
    case iced_x86::Code::XBEGIN_REL16: case iced_x86::Code::XBEGIN_REL32:
      return true;
    default:
      return iced_x86::InstructionExtensions::flow_control(instr) != iced_x86::FlowControl::NEXT;
  }
}

struct DebugMemoryAccess {
  std::uint64_t address = 0;
  std::size_t size = 0;
  bool is_read = false;
  bool is_write = false;
};

[[nodiscard]] bool op_access_reads(iced_x86::OpAccess access) noexcept {
  switch (access) {
    case iced_x86::OpAccess::READ:
    case iced_x86::OpAccess::COND_READ:
    case iced_x86::OpAccess::READ_WRITE:
    case iced_x86::OpAccess::READ_COND_WRITE:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool op_access_writes(iced_x86::OpAccess access) noexcept {
  switch (access) {
    case iced_x86::OpAccess::WRITE:
    case iced_x86::OpAccess::COND_WRITE:
    case iced_x86::OpAccess::READ_WRITE:
    case iced_x86::OpAccess::READ_COND_WRITE:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] std::size_t dr_len_from_encoding(std::uint64_t len_bits) noexcept {
  switch (len_bits & 0x3u) {
    case 0u: return 1;
    case 1u: return 2;
    case 2u: return 8;
    case 3u:
    default:
      return 4;
  }
}

[[nodiscard]] std::vector<DebugMemoryAccess> collect_debug_memory_accesses(CpuState& state, const iced_x86::Instruction& instr) {
  std::vector<DebugMemoryAccess> accesses;
  iced_x86::InstructionInfoFactory info_factory;
  const auto& info = info_factory.info(instr);
  for (const auto& used_mem : info.used_memory()) {
    const auto access = used_mem.access;
    const bool is_read = op_access_reads(access);
    const bool is_write = op_access_writes(access);
    if (!is_read && !is_write) {
      continue;
    }

    std::uint64_t address = 0;
    if (used_mem.base != iced_x86::Register::NONE) {
      address += detail::read_register(state, used_mem.base);
    }
    if (used_mem.index != iced_x86::Register::NONE) {
      address += detail::read_register(state, used_mem.index) * used_mem.scale;
    }
    address += used_mem.displacement;
    if (used_mem.segment == iced_x86::Register::FS) {
      address += state.fs_base;
    } else if (used_mem.segment == iced_x86::Register::GS) {
      address += state.gs_base;
    }
    address = mask_linear_address(state, address);

    const auto size = std::max<std::size_t>(1u, iced_x86::memory_size_ext::get_size(used_mem.memory_size));
    accesses.push_back(DebugMemoryAccess{address, size, is_read, is_write});
  }
  return accesses;
}

// DR7[0:7] are the four slots' local/global enable pairs; nothing else in the register can arm a
// breakpoint. Testing the whole register instead treats a DR7 of 0x400 -- bit 10 reads as 1 on real
// hardware, so that is what an OS writes to mean "none armed" -- as though a breakpoint were live.
[[nodiscard]] bool has_enabled_breakpoints(const CpuState& state) noexcept {
  return (state.dr[7] & 0xFFu) != 0;
}

[[nodiscard]] bool has_enabled_execute_breakpoints(const CpuState& state) noexcept {
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) {
      continue;
    }
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw == 0u) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool has_enabled_data_breakpoints(const CpuState& state) noexcept {
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) {
      continue;
    }
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw != 0u) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::uint64_t collect_execute_breakpoint_hits(CpuState& state, std::uint64_t instruction_rip) noexcept {
  std::uint64_t hit_bits = 0;
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) {
      continue;
    }
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw != 0u) {
      continue;
    }
    const auto watch = mask_linear_address(state, state.dr[i]);
    if (watch == instruction_rip) {
      hit_bits |= (1ull << i);
    }
  }
  return hit_bits;
}

[[nodiscard]] std::uint64_t collect_data_breakpoint_hits(CpuState& state, const std::vector<DebugMemoryAccess>& accesses) noexcept {
  std::uint64_t hit_bits = 0;
  for (const auto& access : accesses) {
    hit_bits |= detail::debug_data_breakpoint_hits(state, access.address, access.size, access.is_read, access.is_write);
  }
  return hit_bits;
}

}  // namespace

constexpr std::size_t Executor::stop_reason_to_index(StopReason reason) noexcept {
  // Two of these reasons come straight out of an embedder hook, so the value is not guaranteed to
  // be a real enumerator. Bucket anything outside the enum rather than write past the vector.
  const auto index = static_cast<std::size_t>(reason);
  return index < kStopReasonCount ? index : 0;
}

void Executor::reset_stats() {
  total_steps_ = 0;
  total_retired_ = 0;
  std::fill(code_execution_counts_.begin(), code_execution_counts_.end(), 0);
  std::fill(stop_reason_counts_.begin(), stop_reason_counts_.end(), 0);
}

const std::vector<std::uint64_t>& Executor::code_execution_counts() const noexcept {
  return code_execution_counts_;
}

const std::vector<std::uint64_t>& Executor::stop_reason_counts() const noexcept {
  return stop_reason_counts_;
}

std::uint64_t Executor::total_steps() const noexcept {
  return total_steps_;
}

std::uint64_t Executor::total_retired() const noexcept {
  return total_retired_;
}

void Executor::set_context_read_callback(ContextSyncCallback fn) {
  context_read_cb_ = std::move(fn);
}

void Executor::set_context_write_callback(ContextSyncCallback fn) {
  context_write_cb_ = std::move(fn);
}

void Executor::request_stop() noexcept {
  stop_requested_ = true;
}

void Executor::clear_stop_request() noexcept {
  stop_requested_ = false;
}

bool Executor::stop_requested() const noexcept {
  return stop_requested_;
}

bool Executor::has_violation() const noexcept {
  return has_violation_;
}

void Executor::clear_violation() noexcept {
  has_violation_ = false;
  violation_ip_ = 0;
  violation_address_ = 0;
  violation_reason_ = StopReason::none;
}

std::uint64_t Executor::violation_ip() const noexcept {
  return violation_ip_;
}

std::uint64_t Executor::violation_address() const noexcept {
  return violation_address_;
}

StopReason Executor::violation_reason() const noexcept {
  return violation_reason_;
}

// Masking is sound only if the covering instruction is guaranteed to run. Three things break that,
// all handled: a caller that never steps again, a mid-span fault, and TF/DR7 or a hook registered
// after caching. An async request_stop() landing mid-span is the accepted residual.
constexpr bool kFlagLivenessTablesTrustworthy = true;

bool Executor::block_liveness_eligible(const Memory& memory) const noexcept {
  // Instruction, code and access hooks see rflags mid-block at points the per-instruction tables
  // do not know about. Trap and execution hooks cannot: one gets only an address, and the other is
  // always block-terminal.
  return !has_instruction_hooks_ && !has_code_hooks_ &&
         !has_execution_hooks_ && !has_execution_address_hooks_ &&
         !memory.has_access_hooks();
}

ExecutionResult Executor::step(CpuState& state, Memory& memory) {
  return step_impl(state, memory, false);
}

bool Executor::jit_bypass_eligible(const CpuState& state, const Memory& memory) const noexcept {
  // Reuses block_liveness_eligible's hook check, and the context-sync callbacks count as the same
  // per-instruction machinery: a bypassing consumer ran against a CpuState nobody refreshed.
  return (state.rflags & kFlagTF) == 0 && !has_enabled_breakpoints(state) && !context_read_cb_ && !context_write_cb_ &&
         block_liveness_eligible(memory);
}

ExecutionResult Executor::step_impl(CpuState& state, Memory& memory, bool allow_masking) {
  StepDepthScope depth_scope{*this};
  // Each level is an embedder callback re-entering step(). Nothing bounded it, so a hook that
  // re-enters unconditionally exhausted the host stack instead of returning something actionable.
  if (step_depth_ > kMaxStepDepth) {
    return {StopReason::execution_limit, 0, std::nullopt, std::nullopt};
  }
  // The level cap says nothing about bytes per level, so measure the descent instead. Volatile so
  // the probe keeps an address, and compared by magnitude so it assumes no stack direction.
  volatile char stack_probe = 0;
  const auto probe_here = reinterpret_cast<std::uintptr_t>(const_cast<const char*>(&stack_probe));
  if (step_depth_ == 1) {
    step_stack_base_ = probe_here;
  } else if (step_stack_base_ != 0) {
    const auto descended = step_stack_base_ > probe_here ? step_stack_base_ - probe_here
                                                         : probe_here - step_stack_base_;
    if (descended > kMaxStepStackBytes) {
      return {StopReason::execution_limit, 0, std::nullopt, std::nullopt};
    }
  }
  // The decode cache is direct-mapped on (rip >> 1), so a nested step 0x4000 away lands on the outer
  // frame's live slot and runs its handler against the nested operands. Nested frames get their own.
  const bool nested = step_depth_ > 1;
  // Nested frames use neither cache, so tagging on their behalf only mislabels what the outer frame
  // cached: a hook re-entering with a second Memory left the tag naming it while the outer frame
  // kept filling entries from the first, and the next step then ran the wrong guest's bytes.
  if (!nested && cache_memory_instance_ != memory.instance_id()) [[unlikely]] {
    // Both caches validate on (rip, page epoch, mode), and epochs come from a per-Memory counter
    // starting near zero, so an Executor reused across two would run the first one's bytes.
    for (auto& entry : *decode_cache_) entry.valid = false;
    for (auto& entry : *code_page_cache_) entry.valid = false;
    cache_memory_instance_ = memory.instance_id();
  }
  clear_violation();
  if (collect_code_stats_) { ++total_steps_; }
  if (stop_requested_) {
    const ExecutionResult stopped{StopReason::stop_requested, 0, std::nullopt, std::nullopt};
    ++stop_reason_counts_[stop_reason_to_index(stopped.reason)];
    notify_stop_hooks(state, memory, stopped, state.rip);
    return stopped;
  }
  // Copied locally like the MMIO dispatch: a callback that reinstalls itself from inside its own
  // body would destroy the functor whose operator() is still on the stack.
  if (auto read_cb = context_read_cb_) read_cb(state);
  struct WriteSync {
    Executor& self; CpuState& state;
    ~WriteSync() { if (auto write_cb = self.context_write_cb_) write_cb(state); }
  } write_sync{*this, state};
  state.rip = mask_instruction_pointer(state, state.rip);
  state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
  const auto instruction_start_rip = state.rip;
  const auto record_violation = [&](const ExecutionResult& result, std::uint64_t fault_address) {
    if (result.reason == StopReason::none ||
        result.reason == StopReason::halted ||
        result.reason == StopReason::execution_limit ||
        result.reason == StopReason::stop_requested) {
      return;
    }
    has_violation_ = true;
    violation_reason_ = result.reason;
    violation_ip_ = instruction_start_rip;
    violation_address_ = fault_address;
  };
  const auto fault_address_of = [&](const ExecutionResult& result, std::uint64_t fallback) -> std::uint64_t {
    if (result.exception.has_value()) {
      return result.exception->address;
    }
    return fallback;
  };

  // RF and the SS-load shadow are spent per attempt, but a faulting instruction never ran, so a
  // retry has to get them back or it fires the breakpoint they exist to suppress.
  bool rf_consumed = false;
  std::uint8_t debug_suppression_consumed = 0;

  for (std::size_t attempt = 0; attempt < kMaxFaultRetries; ++attempt) {
    const auto try_recover_fault = [&](const ExecutionResult& fault, std::uint64_t fault_address) -> bool {
      const auto action = run_fault_hooks(FaultHookEvent{state, memory, fault, instruction_start_rip, fault_address});
      if (trace_semantics_) {
        std::fprintf(stderr,
            "[seven-fault] rip=0x%llx mode=%u reason=%u addr=0x%llx action=%u\n",
            static_cast<unsigned long long>(instruction_start_rip),
            static_cast<unsigned>(state.mode),
            static_cast<unsigned>(fault.reason),
            static_cast<unsigned long long>(fault_address),
            static_cast<unsigned>(action));
      }
      if (action != FaultHookAction::retry && action != FaultHookAction::restart_instruction) {
        return false;
      }
      if (rf_consumed) {
        state.rflags |= kFlagRF;
        rf_consumed = false;
      }
      if (debug_suppression_consumed != 0) {
        state.debug_suppression = debug_suppression_consumed;
        debug_suppression_consumed = 0;
      }
      if (action == FaultHookAction::restart_instruction) {
        state.rip = instruction_start_rip;
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
      }
      return true;
    };

    // memory_fault() applies this to data references, but the fetch path builds its faults inline
    // and never went through it, so a jump to a non-canonical address came back as a page fault at
    // that address instead of #GP. Real hardware checks this ahead of any page walk.
    if (!is_canonical_address(state.rip)) [[unlikely]] {
      const ExecutionResult fault{StopReason::general_protection, 0,
                                  ExceptionInfo{StopReason::general_protection, state.rip, 0}, std::nullopt};
      if (try_recover_fault(fault, state.rip)) {
        continue;
      }
      record_violation(fault, state.rip);
      ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
      notify_stop_hooks(state, memory, fault, state.rip);
      return fault;
    }

    const auto rip_page = state.rip / Memory::kPageSize;
    const auto rip_page_epoch = memory.page_code_epoch(rip_page);
    // Epoch of the page holding the final byte, which only differs across a boundary. `spans` is
    // false when the instruction runs off the end of the address space, which is never cacheable.
    const auto last_byte_epoch = [&](std::uint32_t length, bool& spans) -> std::uint64_t {
      const auto last_byte = state.rip + (length - 1);
      spans = last_byte >= state.rip;
      if (!spans) {
        return 0;
      }
      const auto last_page = last_byte / Memory::kPageSize;
      return last_page == rip_page ? rip_page_epoch : memory.page_code_epoch(last_page);
    };
    const bool can_use_decode_cache =
        !nested && !memory.has_fetch_access_hooks() && !decode_cache_disabled_by_env_;
    const auto cache_index = static_cast<std::size_t>((state.rip >> 1) & (kDecodeCacheSize - 1));
    const auto scratch_slot = [this]() -> DecodedInstructionCacheEntry& {
      const auto index = step_depth_ - 2;
      while (nested_decode_scratch_.size() <= index) {
        nested_decode_scratch_.push_back(std::make_unique<DecodedInstructionCacheEntry>());
      }
      auto& slot = *nested_decode_scratch_[index];
      slot.valid = false;  // one slot per depth, reused across calls, so never serve a stale hit
      return slot;
    };
    auto& cache_entry = nested ? scratch_slot() : (*decode_cache_)[cache_index];
    bool hit_spans_address_space = false;
    const bool cache_hit = can_use_decode_cache &&
                           cache_entry.valid &&
                           cache_entry.rip == state.rip &&
                           cache_entry.page_epoch == rip_page_epoch &&
                           cache_entry.mode == state.mode &&
                           cache_entry.last_page_epoch ==
                               last_byte_epoch(cache_entry.instruction_length, hit_spans_address_space);

    if (!cache_hit) [[unlikely]] {
      std::array<std::uint8_t, iced_x86::IcedConstants::_MAX_INSTRUCTION_LENGTH> bytes{};
      const std::uint8_t* decode_bytes = bytes.data();
      std::size_t decode_size = bytes.size();
      bool fetched = false;
      if (can_use_decode_cache) {
        const auto page_base = state.rip & ~static_cast<std::uint64_t>(Memory::kPageSize - 1);
        const auto page_offset = static_cast<std::size_t>(state.rip - page_base);
        if (page_offset + bytes.size() <= Memory::kPageSize) {
          const auto page_index = page_base / Memory::kPageSize;
          const auto page_epoch = memory.page_code_epoch(page_index);
          auto& page_cache = (*code_page_cache_)[static_cast<std::size_t>(page_index & (kCodePageCacheSize - 1))];
          if (!page_cache.valid || page_cache.page_base != page_base || page_cache.page_epoch != page_epoch) {
            if (memory.read_code_page(page_base, page_cache.bytes.data())) {
              page_cache.page_base = page_base;
              page_cache.page_epoch = page_epoch;
              page_cache.valid = true;
            } else {
              page_cache.valid = false;
            }
          }
          if (page_cache.valid && page_cache.page_base == page_base && page_cache.page_epoch == page_epoch) {
            decode_bytes = page_cache.bytes.data() + page_offset;
            decode_size = Memory::kPageSize - page_offset;
            fetched = true;
          }
        }
      }
      bool truncated_to_page = false;
      if (!fetched) {
        if (!memory.read(state.rip, bytes.data(), bytes.size(), MemoryAccessKind::instruction_fetch)) {
          // A fetch only faults on the bytes the instruction needs. Retry with what this page holds
          // and let the decoder say whether that was enough.
          const auto in_page =
              static_cast<std::size_t>(Memory::kPageSize - (state.rip % Memory::kPageSize));
          truncated_to_page = in_page < bytes.size() &&
                              memory.read(state.rip, bytes.data(), in_page, MemoryAccessKind::instruction_fetch);
          if (!truncated_to_page) {
            const ExecutionResult fault{StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, state.rip, 0}, std::nullopt};
            if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
              continue;
            }
            record_violation(fault, fault_address_of(fault, state.rip));
            ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
            notify_stop_hooks(state, memory, fault, state.rip);
            return fault;
          }
          decode_size = in_page;
        }
      }

      iced_x86::Decoder decoder(
          decoder_bitness(state.mode),
          std::span<const std::uint8_t>(decode_bytes, decode_size),
          state.rip,
          iced_x86::DecoderOptions::NO_INVALID_CHECK);
      const auto decoded = decoder.decode();
      if (!decoded.has_value() && truncated_to_page &&
          decoded.error().error == iced_x86::DecoderError::NO_MORE_BYTES) {
        // The instruction really does run into the next page, so the suppressed fetch fault was
        // right after all. Wraps to zero on the last page, which is #GP, not a page fault.
        const auto next_page = (state.rip | (Memory::kPageSize - 1)) + 1;
        const auto reason = next_page == 0 ? StopReason::general_protection : StopReason::page_fault;
        const auto fault_rip = next_page == 0 ? state.rip : next_page;
        const ExecutionResult fault{reason, 0, ExceptionInfo{reason, fault_rip, 0}, std::nullopt};
        if (try_recover_fault(fault, fault_address_of(fault, fault_rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, fault_rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }
      if (!decoded.has_value()) {
        if (trace_semantics_) {
          if (decode_bytes != bytes.data()) {
            std::memcpy(bytes.data(), decode_bytes, bytes.size());
          }
          std::fprintf(stderr,
              "[seven-decode-fail] rip=0x%llx mode=%u bytes=%02x %02x %02x %02x %02x %02x %02x %02x\n",
              static_cast<unsigned long long>(state.rip),
              static_cast<unsigned>(state.mode),
              static_cast<unsigned>(bytes[0]),
              static_cast<unsigned>(bytes[1]),
              static_cast<unsigned>(bytes[2]),
              static_cast<unsigned>(bytes[3]),
              static_cast<unsigned>(bytes[4]),
              static_cast<unsigned>(bytes[5]),
              static_cast<unsigned>(bytes[6]),
              static_cast<unsigned>(bytes[7]));
        }
        const ExecutionResult fault{StopReason::decode_error, 0, ExceptionInfo{StopReason::decode_error, state.rip, 0}, std::nullopt};
        if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }

      cache_entry.rip = state.rip;
      cache_entry.page_epoch = rip_page_epoch;
      cache_entry.mode = state.mode;
      cache_entry.instr = decoded.value();
      cache_entry.simd_allowed = simd_profile_allows(cache_entry.instr);
      cache_entry.reported_code = normalize_reported_code(cache_entry.instr.code());
      const auto trap = trap_kind_for_code(cache_entry.instr.code());
      cache_entry.trap_kind = trap.has_value() ? static_cast<std::uint8_t>(*trap) : 0xFFu;
      cache_entry.instruction_length = std::max<std::uint32_t>(1u, cache_entry.instr.length());
      bool fits_in_address_space = false;
      cache_entry.last_page_epoch = last_byte_epoch(cache_entry.instruction_length, fits_in_address_space);
      cache_entry.valid = can_use_decode_cache && fits_in_address_space;
      cache_entry.dead_flags_mask = 0;

      // Lift the rest of the block so the backward liveness pass has more than one instruction.
      // Only pre-populates decode entries step() would fill anyway; nothing about dispatch changes.
      if (fetched && cache_entry.valid && cache_entry.trap_kind == 0xFFu && cache_entry.simd_allowed &&
          !ends_lifted_block(cache_entry.instr)) {
        std::array<std::size_t, kMaxBlockLiftLength> lifted_indices{};
        lifted_indices[0] = cache_index;
        std::size_t lifted_count = 1;
        while (lifted_count < kMaxBlockLiftLength) {
          const auto next_decoded = decoder.decode();
          if (!next_decoded.has_value()) {
            break;
          }
          const auto next_rip = next_decoded.value().ip();
          if (has_execution_address_hooks_ && execution_address_hooks_.contains(next_rip)) {
            break;
          }
          const auto next_index = static_cast<std::size_t>((next_rip >> 1) & (kDecodeCacheSize - 1));
          // The index folds rip and rip+1 onto one slot, so a run of 1-byte instructions can
          // collide with a slot this same lift already claimed -- including cache_entry itself,
          // still pending dispatch below.
          bool collides = false;
          for (std::size_t i = 0; i < lifted_count; ++i) {
            if (lifted_indices[i] == next_index) {
              collides = true;
              break;
            }
          }
          if (collides) {
            break;
          }
          // The lift runs off the page cache, gated on the whole fetch window fitting one page, so
          // every instruction shares rip's epoch. Bail rather than stamp the wrong one.
          const auto next_last_byte = next_rip + (next_decoded.value().length() - 1);
          if (next_rip / Memory::kPageSize != rip_page || next_last_byte < next_rip ||
              next_last_byte / Memory::kPageSize != rip_page) {
            break;
          }
          auto& next_entry = (*decode_cache_)[next_index];
          next_entry.rip = next_rip;
          next_entry.page_epoch = rip_page_epoch;
          next_entry.last_page_epoch = rip_page_epoch;
          next_entry.mode = state.mode;
          next_entry.instr = next_decoded.value();
          next_entry.simd_allowed = simd_profile_allows(next_entry.instr);
          next_entry.reported_code = normalize_reported_code(next_entry.instr.code());
          const auto next_trap = trap_kind_for_code(next_entry.instr.code());
          next_entry.trap_kind = next_trap.has_value() ? static_cast<std::uint8_t>(*next_trap) : 0xFFu;
          next_entry.instruction_length = std::max<std::uint32_t>(1u, next_entry.instr.length());
          next_entry.dead_flags_mask = 0;
          next_entry.valid = true;
          lifted_indices[lifted_count++] = next_index;
          const bool is_boundary = !next_entry.simd_allowed || next_entry.trap_kind != 0xFFu ||
              ends_lifted_block(next_entry.instr);
          if (is_boundary) {
            break;
          }
        }
        if (kFlagLivenessTablesTrustworthy && lifted_count > 1 && block_liveness_eligible(memory)) {
          std::array<FlagLivenessInstr, kMaxBlockLiftLength> liveness{};
          for (std::size_t i = 0; i < lifted_count; ++i) {
            liveness[i].instr = &(*decode_cache_)[lifted_indices[i]].instr;
          }
          compute_flag_liveness(std::span<FlagLivenessInstr>(liveness.data(), lifted_count));
          for (std::size_t i = 0; i < lifted_count; ++i) {
            (*decode_cache_)[lifted_indices[i]].dead_flags_mask = liveness[i].dead_flags_mask;
          }
        }
      }
    }

    const auto& instr = cache_entry.instr;
    const auto next_rip = mask_instruction_pointer(state, state.rip + cache_entry.instruction_length);
    const auto reported_code = cache_entry.reported_code;
    const std::uint64_t dr7 = state.dr[7];
    const bool check_execute_breakpoints = dr7 != 0 && has_enabled_execute_breakpoints(state);
    const bool check_data_breakpoints = dr7 != 0 && has_enabled_data_breakpoints(state);
    const auto debug_memory_accesses = check_data_breakpoints
        ? collect_debug_memory_accesses(state, instr)
        : std::vector<DebugMemoryAccess>{};

    if (instr.code() == iced_x86::Code::INVALID) {
      TrapHookContext trap_ctx{state, memory, instr, next_rip, TrapKind::invalid_opcode};
      const auto trap_result = run_trap_hooks(trap_ctx);
      if (trap_result.action == TrapHookAction::handled) {
        if (collect_code_stats_) { ++total_retired_; }
        state.rip = next_rip;
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        return {StopReason::none, 1, std::nullopt, iced_x86::Code::INVALID};
      }
      if (trap_result.action == TrapHookAction::stop) {
        const auto fault = trap_result.stop_result.value_or(
            ExecutionResult{StopReason::invalid_opcode, 0, ExceptionInfo{StopReason::invalid_opcode, state.rip, 0}, iced_x86::Code::INVALID});
        if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }
      const ExecutionResult fault{StopReason::invalid_opcode, 0, ExceptionInfo{StopReason::invalid_opcode, state.rip, 0}, iced_x86::Code::INVALID};
      if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
        continue;
      }
      record_violation(fault, fault_address_of(fault, state.rip));
      ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
      notify_stop_hooks(state, memory, fault, state.rip);
      return fault;
    }

    if (!cache_entry.simd_allowed) {
      const ExecutionResult fault{StopReason::unsupported_instruction, 0, ExceptionInfo{StopReason::unsupported_instruction, state.rip, 0}, instr.code()};
      if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
        continue;
      }
      record_violation(fault, fault_address_of(fault, state.rip));
      ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
      notify_stop_hooks(state, memory, fault, state.rip);
      return fault;
    }

    const bool rf_suppressed = (state.rflags & kFlagRF) != 0;
    if (rf_suppressed) {
      state.rflags &= ~kFlagRF;
      rf_consumed = true;
    }
    const bool debug_suppressed = state.debug_suppression != 0;
    if (debug_suppressed) {
      debug_suppression_consumed = state.debug_suppression;
      state.debug_suppression = 0;
    }
    const bool tf_active = (state.rflags & kFlagTF) != 0;
    if (!rf_suppressed && !debug_suppressed && check_execute_breakpoints) {
      const auto exec_hit_bits = collect_execute_breakpoint_hits(state, instruction_start_rip);
      if (exec_hit_bits != 0) {
        state.dr[6] |= exec_hit_bits;
        ExecutionContext db_ctx{state, memory, instr, next_rip, false};
        // An instruction breakpoint is a fault, so the frame carries its own rip and the iret lands
        // back on it. RF in the saved image is the only thing stopping it firing forever.
        const auto db_result = detail::dispatch_interrupt(db_ctx, 1u, instruction_start_rip, std::nullopt, true);
        if (db_result.reason != StopReason::none) {
          if (try_recover_fault(db_result, fault_address_of(db_result, state.rip))) {
            continue;
          }
          record_violation(db_result, fault_address_of(db_result, state.rip));
          ++stop_reason_counts_[stop_reason_to_index(db_result.reason)];
          notify_stop_hooks(state, memory, db_result, state.rip);
          return db_result;
        }
        state.rip = mask_instruction_pointer(state, state.rip);
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        return {StopReason::none, 0, std::nullopt, reported_code};
      }
    }

    if (cache_entry.trap_kind != 0xFFu) {
      const auto trap_kind = static_cast<TrapKind>(cache_entry.trap_kind);
      TrapHookContext trap_ctx{state, memory, instr, next_rip, trap_kind};
      const auto trap_result = run_trap_hooks(trap_ctx);
      if (trap_result.action == TrapHookAction::handled) {
        if (state.rip == instruction_start_rip) {
          state.rip = next_rip;
        } else {
          state.rip = mask_instruction_pointer(state, state.rip);
        }
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        if (collect_code_stats_ && static_cast<std::size_t>(instr.code()) < code_execution_counts_.size()) {
          ++code_execution_counts_[static_cast<std::size_t>(instr.code())];
        }
        if (collect_code_stats_) { ++total_retired_; }
        return {StopReason::none, 1, std::nullopt, reported_code};
      }
      if (trap_result.action == TrapHookAction::stop) {
        const auto fault = trap_result.stop_result.value_or(
            ExecutionResult{StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, state.rip, 0}, instr.code()});
        if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }
    }

    ExecutionContext ctx{state, memory, instr, next_rip, false};
    if (has_execution_hooks_ || has_execution_address_hooks_) {
      // One scope across both loops, since a hook in the first can remove one from the second.
      // These were the only unguarded dispatches, so a self-removing one-shot breakpoint erased
      // from the vector its own range-for was walking.
      HookDispatchScope scope{*this};
      if (has_execution_hooks_) {
        for (auto& [id, hook] : execution_hooks_) {
          (void)id;
          hook(state.rip);
        }
      }
      if (has_execution_address_hooks_) {
        const auto exec_addr_it = execution_address_hooks_.find(state.rip);
        if (exec_addr_it != execution_address_hooks_.end()) {
          for (auto& [id, hook] : exec_addr_it->second) {
            (void)id;
            hook(state.rip);
          }
        }
      }
    }
    const bool need_instruction_hooks = has_instruction_hooks_ || has_code_hooks_;
    InstructionHookContext hook_ctx{state, memory, instr, next_rip};
    ExecutionResult hook_stop_result{};
    const auto hook_action = need_instruction_hooks
        ? run_instruction_hooks(hook_ctx, hook_stop_result)
        : InstructionHookAction::continue_to_core;
    if (hook_action == InstructionHookAction::stop) {
      if (!hook_stop_result.code.has_value()) {
        hook_stop_result.code = reported_code;
      }
      if (hook_stop_result.reason != StopReason::none) {
        if (try_recover_fault(hook_stop_result, fault_address_of(hook_stop_result, state.rip))) {
          continue;
        }
        record_violation(hook_stop_result, fault_address_of(hook_stop_result, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(hook_stop_result.reason)];
        notify_stop_hooks(state, memory, hook_stop_result, state.rip);
      }
      return hook_stop_result;
    }
    if (hook_action == InstructionHookAction::skip_core) {
      if (state.rip == instruction_start_rip) {
        state.rip = next_rip;
      } else {
        state.rip = mask_instruction_pointer(state, state.rip);
      }
      state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
      if (collect_code_stats_ && static_cast<std::size_t>(instr.code()) < code_execution_counts_.size()) {
        ++code_execution_counts_[static_cast<std::size_t>(instr.code())];
      }
      if (collect_code_stats_) { ++total_retired_; }
      return {StopReason::none, 1, std::nullopt, reported_code};
    }

    ExecutionResult result{};
    // Dispatch on the normalized code: iced reports PUSH imm as the dword form, but in 64-bit mode
    // it takes a 64-bit stack slot. Dispatching on the raw code ran the 4-byte handler and corrupted
    // the stack for any guest that pushes an immediate.
    const auto code = reported_code;
    // The mask is only trustworthy if the caller keeps draining the block, TF is clear, no debug
    // register is armed, and no hook arrived after the lift -- a cached mask outlives that one
    // check. Context-sync counts too: it hands the host a CpuState at every boundary.
    const bool masking_safe_now = allow_masking && (state.rflags & kFlagTF) == 0 &&
                                   !has_enabled_breakpoints(state) && !context_read_cb_ && !context_write_cb_ &&
                                   block_liveness_eligible(memory);
    // Saved and put back, not assigned: a handler's access can reach a device callback that
    // re-enters run(), and the nested frame's mask would otherwise stay installed and filter
    // everything this handler writes afterwards.
    const struct DeadFlagsMaskScope {
      std::uint64_t saved;
      ~DeadFlagsMaskScope() { detail::set_dead_flags_mask(saved); }
    } dead_flags_scope{detail::dead_flags_mask()};
    detail::set_dead_flags_mask(masking_safe_now ? cache_entry.dead_flags_mask : 0);
    result = dispatch_handler(ctx, code);
    result.code = reported_code;
    if (result.reason == StopReason::none) {
      if (collect_code_stats_ && static_cast<std::size_t>(code) < code_execution_counts_.size()) {
        ++code_execution_counts_[static_cast<std::size_t>(code)];
      }
      if (collect_code_stats_) { ++total_retired_; }
      const bool current_instruction_set_shadow = state.debug_suppression != 0;
      state.rip = ctx.control_flow_taken ? mask_instruction_pointer(state, state.rip) : ctx.next_rip;
      state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
      result.retired = 1;
      if (!current_instruction_set_shadow && state.pending_debug_hit_bits == 0 && !check_data_breakpoints &&
          !state.pending_single_step && !tf_active) {
        return result;
      }
      // A handler reporting its own hits (implicit stack slots, string-op source/destination) does
      // not mean the instruction's explicit operands missed, so take both rather than either.
      const auto current_data_hit_bits =
          ctx.debug_hit_bits | collect_data_breakpoint_hits(state, debug_memory_accesses);
      if (current_instruction_set_shadow) {
        state.pending_debug_hit_bits |= current_data_hit_bits;
        return result;
      }

      const auto data_hit_bits = state.pending_debug_hit_bits | current_data_hit_bits;
      const bool tf_hit = state.pending_single_step || (!debug_suppressed && tf_active);
      if (data_hit_bits != 0 || tf_hit) {
        if (tf_hit) {
          state.dr[6] |= (1ull << 14);
          state.pending_single_step = false;
        }
        state.dr[6] |= data_hit_bits;
        state.pending_debug_hit_bits = 0;
        const auto return_rip = state.rip;
        ExecutionContext db_ctx{state, memory, instr, return_rip, false};
        const auto db_result = detail::dispatch_interrupt(db_ctx, 1u, return_rip, std::nullopt, ctx.push_rf_for_debug);
        if (db_result.reason != StopReason::none) {
          record_violation(db_result, fault_address_of(db_result, state.rip));
          ++stop_reason_counts_[stop_reason_to_index(db_result.reason)];
          notify_stop_hooks(state, memory, db_result, state.rip);
          return db_result;
        }
      }
      return result;
    }
    if (try_recover_fault(result, fault_address_of(result, state.rip))) {
      continue;
    }
    if (debug_suppressed) {
      state.pending_single_step = false;
      state.pending_debug_hit_bits = 0;
    }
    record_violation(result, fault_address_of(result, state.rip));
    ++stop_reason_counts_[stop_reason_to_index(result.reason)];
    notify_stop_hooks(state, memory, result, state.rip);
    return result;
  }

  const ExecutionResult fault{StopReason::execution_limit, 0, std::nullopt, std::nullopt};
  ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
  notify_stop_hooks(state, memory, fault, state.rip);
  return fault;
}

ExecutionResult Executor::run(CpuState& state, Memory& memory, std::size_t max_instructions) {
  ExecutionResult last{};
  for (std::size_t i = 0; i < max_instructions; ++i) {
    if (stop_requested_) {
      const ExecutionResult stopped{StopReason::stop_requested, i, std::nullopt, std::nullopt};
      ++stop_reason_counts_[stop_reason_to_index(stopped.reason)];
      notify_stop_hooks(state, memory, stopped, state.rip);
      return stopped;
    }
    // Mask only with enough budget left that any block started now finishes inside this call, so
    // run() never returns mid-span. Near the tail it falls back to the unmasked dispatch.
    const bool allow_masking = (max_instructions - i) >= kMaxBlockLiftLength;
    last = step_impl(state, memory, allow_masking);
    if (last.reason != StopReason::none) {
      last.retired += i;
      return last;
    }
  }
  ++stop_reason_counts_[stop_reason_to_index(StopReason::execution_limit)];
  const ExecutionResult limit{StopReason::execution_limit, max_instructions, std::nullopt, std::nullopt};
  notify_stop_hooks(state, memory, limit, state.rip);
  return limit;
}

Executor::HookId Executor::add_instruction_hook(InstructionHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      instruction_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    instruction_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_code_hook(iced_x86::Code code, InstructionHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, code, hook = std::move(hook)]() mutable {
      code_hooks_[code].emplace_back(id, std::move(hook));
    });
  } else {
    code_hooks_[code].emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_execution_hook(std::function<void(std::uint64_t)> hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      execution_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    execution_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_execution_hook(std::uint64_t address, std::function<void(std::uint64_t)> hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, address, hook = std::move(hook)]() mutable {
      execution_address_hooks_[address].emplace_back(id, std::move(hook));
    });
  } else {
    execution_address_hooks_[address].emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_stop_hook(StopHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      stop_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    stop_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_fault_hook(FaultHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      fault_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    fault_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_trap_hook(TrapKind kind, TrapHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, kind, hook = std::move(hook)]() mutable {
      trap_hooks_[kind].emplace_back(id, std::move(hook));
    });
  } else {
    trap_hooks_[kind].emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

bool Executor::remove_hook(HookId id) {
  auto remove_now = [this, id]() {
    for (auto it = execution_hooks_.begin(); it != execution_hooks_.end(); ++it) {
      if (it->first == id) {
        execution_hooks_.erase(it);
        return true;
      }
    }
    for (auto map_it = execution_address_hooks_.begin(); map_it != execution_address_hooks_.end(); ++map_it) {
      auto& hooks = map_it->second;
      for (auto it = hooks.begin(); it != hooks.end(); ++it) {
        if (it->first == id) {
          hooks.erase(it);
          if (hooks.empty()) {
            execution_address_hooks_.erase(map_it);
          }
          return true;
        }
      }
    }
    for (auto it = instruction_hooks_.begin(); it != instruction_hooks_.end(); ++it) {
      if (it->first == id) {
        instruction_hooks_.erase(it);
        return true;
      }
    }
    for (auto& [code, hooks] : code_hooks_) {
      (void)code;
      for (auto it = hooks.begin(); it != hooks.end(); ++it) {
        if (it->first == id) {
          hooks.erase(it);
          return true;
        }
      }
    }
    for (auto it = stop_hooks_.begin(); it != stop_hooks_.end(); ++it) {
      if (it->first == id) {
        stop_hooks_.erase(it);
        return true;
      }
    }
    for (auto it = fault_hooks_.begin(); it != fault_hooks_.end(); ++it) {
      if (it->first == id) {
        fault_hooks_.erase(it);
        return true;
      }
    }
    for (auto& [kind, hooks] : trap_hooks_) {
      (void)kind;
      for (auto it = hooks.begin(); it != hooks.end(); ++it) {
        if (it->first == id) {
          hooks.erase(it);
          return true;
        }
      }
    }
    return false;
  };

  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([remove_now]() mutable { (void)remove_now(); });
    return true;
  }
  const bool removed = remove_now();
  refresh_hook_flags();
  return removed;
}

void Executor::clear_hooks() {
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this]() {
      execution_hooks_.clear();
      execution_address_hooks_.clear();
      instruction_hooks_.clear();
      code_hooks_.clear();
      stop_hooks_.clear();
      fault_hooks_.clear();
      trap_hooks_.clear();
      pending_hook_mutations_.clear();
    });
    return;
  }
  execution_hooks_.clear();
  execution_address_hooks_.clear();
  instruction_hooks_.clear();
  code_hooks_.clear();
  stop_hooks_.clear();
  fault_hooks_.clear();
  trap_hooks_.clear();
  pending_hook_mutations_.clear();
  refresh_hook_flags();
}

InstructionHookAction Executor::run_instruction_hooks(InstructionHookContext& ctx, ExecutionResult& stop_result) {
  // Only at depth 0: a queued emplace_back run from inside a callback reallocates the vector an
  // outer dispatch is still walking. The outermost scope flushing on its way out is the safe point.
  if (!dispatching_hooks_ && !pending_hook_mutations_.empty()) {
    apply_pending_hook_mutations();
  }

  const auto it = code_hooks_.find(ctx.instr.code());
  const bool has_code_hooks = it != code_hooks_.end() && !it->second.empty();
  if (instruction_hooks_.empty() && !has_code_hooks) {
    return InstructionHookAction::continue_to_core;
  }

  HookDispatchScope scope{*this};
  for (auto& [id, hook] : instruction_hooks_) {
    (void)id;
    const auto result = hook(ctx);
    if (result.action == InstructionHookAction::stop) {
      stop_result = result.stop_result.value_or(ExecutionResult{StopReason::unsupported_instruction, 0, std::nullopt, ctx.instr.code()});
      return result.action;
    }
    if (result.action == InstructionHookAction::skip_core) {
      return result.action;
    }
  }
  if (has_code_hooks) {
    for (auto& [id, hook] : it->second) {
      (void)id;
      const auto result = hook(ctx);
      if (result.action == InstructionHookAction::stop) {
        stop_result = result.stop_result.value_or(ExecutionResult{StopReason::unsupported_instruction, 0, std::nullopt, ctx.instr.code()});
        return result.action;
      }
      if (result.action == InstructionHookAction::skip_core) {
        return result.action;
      }
    }
  }
  return InstructionHookAction::continue_to_core;
}

TrapHookResult Executor::run_trap_hooks(TrapHookContext& ctx) {
  const auto it = trap_hooks_.find(ctx.kind);
  if (it == trap_hooks_.end()) {
    return {};
  }
  HookDispatchScope scope{*this};
  for (auto& [id, hook] : it->second) {
    (void)id;
    const auto result = hook(ctx);
    if (result.action != TrapHookAction::continue_to_core) {
      return result;
    }
  }
  return {};
}

bool Executor::report_external_fault(CpuState& state, Memory& memory, const ExecutionResult& fault,
                                     std::uint64_t fault_address) {
  const auto action = run_fault_hooks(FaultHookEvent{state, memory, fault, state.rip, fault_address});
  if (action != FaultHookAction::stop) {
    // Both surviving actions mean "attempt this instruction again". step_impl's restart also rewinds
    // rip, which is already where it needs to be here.
    state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
    return true;
  }
  if (fault.reason != StopReason::none && fault.reason != StopReason::halted &&
      fault.reason != StopReason::execution_limit && fault.reason != StopReason::stop_requested) {
    has_violation_ = true;
    violation_reason_ = fault.reason;
    violation_ip_ = state.rip;
    violation_address_ = fault_address;
  }
  ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
  notify_stop_hooks(state, memory, fault, state.rip);
  return false;
}

FaultHookAction Executor::run_fault_hooks(const FaultHookEvent& event) {
  if (fault_hooks_.empty()) {
    return FaultHookAction::stop;
  }
  HookDispatchScope scope{*this};
  for (auto& [id, hook] : fault_hooks_) {
    (void)id;
    const auto action = hook(event);
    if (action != FaultHookAction::stop) {
      return action;
    }
  }
  return FaultHookAction::stop;
}

void Executor::notify_stop_hooks(CpuState& state, Memory& memory, const ExecutionResult& result, std::uint64_t fault_address) const {
  if (stop_hooks_.empty()) {
    return;
  }
  auto& self = const_cast<Executor&>(*this);
  HookDispatchScope scope{self};
  const StopHookEvent event{state, memory, result, fault_address};
  for (const auto& [id, hook] : stop_hooks_) {
    (void)id;
    hook(event);
  }
}

Executor::HookDispatchScope::HookDispatchScope(Executor& executor) noexcept
    : self(executor), was_dispatching(executor.dispatching_hooks_) {
  self.dispatching_hooks_ = true;
}

Executor::HookDispatchScope::~HookDispatchScope() {
  self.dispatching_hooks_ = was_dispatching;
  if (!was_dispatching) {
    self.apply_pending_hook_mutations();
  }
}

Executor::StepDepthScope::StepDepthScope(Executor& executor) noexcept : self(executor) {
  ++self.step_depth_;
}

Executor::StepDepthScope::~StepDepthScope() { --self.step_depth_; }

void Executor::apply_pending_hook_mutations() {
  if (pending_hook_mutations_.empty()) {
    return;
  }
  auto queued = std::move(pending_hook_mutations_);
  pending_hook_mutations_.clear();
  for (auto& mutation : queued) {
    mutation();
  }
  refresh_hook_flags();
}

void Executor::refresh_hook_flags() noexcept {
  has_instruction_hooks_ = !instruction_hooks_.empty();
  has_code_hooks_ = false;
  for (const auto& [code, hooks] : code_hooks_) {
    (void)code;
    if (!hooks.empty()) {
      has_code_hooks_ = true;
      break;
    }
  }
  has_execution_hooks_ = !execution_hooks_.empty();
  has_execution_address_hooks_ = !execution_address_hooks_.empty();
  has_stop_hooks_ = !stop_hooks_.empty();
  has_fault_hooks_ = !fault_hooks_.empty();
  has_trap_hooks_ = false;
  for (const auto& [kind, hooks] : trap_hooks_) {
    (void)kind;
    if (!hooks.empty()) {
      has_trap_hooks_ = true;
      break;
    }
  }
}

std::size_t Executor::supported_code_count() const noexcept {
#define KUBERA_CODE(code) +1
  static constexpr std::size_t kSupportedCodeCount = 0
#include "seven/handled_codes.def"
      ;
#undef KUBERA_CODE
  return kSupportedCodeCount;
}

ExecutionResult Executor::unsupported(ExecutionContext& ctx) {
  return {StopReason::unsupported_instruction, 0, ExceptionInfo{StopReason::unsupported_instruction, ctx.state.rip, 0}, std::nullopt};
}

bool Executor::is_trap_instruction(iced_x86::Code code) noexcept {
  return trap_kind_for_code(code).has_value();
}

bool Executor::simd_profile_allows(const iced_x86::Instruction& instr) noexcept {
  const auto encoding = iced_x86::InstructionExtensions::encoding(instr);
  if (encoding == iced_x86::EncodingKind::EVEX && !kEnableAvx512) {
    return false;
  }
  if (encoding == iced_x86::EncodingKind::VEX && !kEnableAvx) {
    return false;
  }
  for (std::uint32_t i = 0; i < instr.op_count(); ++i) {
    if (instr.op_kind(i) == iced_x86::OpKind::REGISTER &&
        vector_width_for_register(instr.op_register(i)) > kVectorBytes) {
      return false;
    }
  }
  return true;
}

// forceinline so step_impl's dispatch stays as fast as before this got pulled out of it
__forceinline ExecutionResult Executor::dispatch_handler(ExecutionContext& ctx, iced_x86::Code code) {
  switch (code) {
#define KUBERA_CODE(code) \
    case iced_x86::Code::code: return handlers::handle_code_##code(ctx);
#include "seven/handled_codes.def"
#undef KUBERA_CODE
    default:
      return unsupported(ctx);
  }
}

std::vector<std::uint8_t> parse_hex_bytes(std::string_view text) {
  std::vector<std::uint8_t> bytes;
  std::string compact;
  compact.reserve(text.size());
  for (const auto ch : text) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      compact.push_back(static_cast<char>(ch));
    }
  }
  if ((compact.size() % 2) != 0) {
    throw std::runtime_error("hex byte string must contain an even number of digits");
  }
  for (std::size_t i = 0; i < compact.size(); i += 2) {
    unsigned value = 0;
    const auto pair = compact.substr(i, 2);
    const auto result = std::from_chars(pair.data(), pair.data() + pair.size(), value, 16);
    if (result.ec != std::errc{}) {
      throw std::runtime_error("invalid hex byte string");
    }
    bytes.push_back(static_cast<std::uint8_t>(value));
  }
  return bytes;
}

}  // namespace seven

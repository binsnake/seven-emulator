#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <iced_x86/decoder.hpp>
#include <iced_x86/iced_constants.hpp>
#include <iced_x86/instruction.hpp>

#include "seven/hooks.hpp"
#include "seven/memory.hpp"
#include "seven/types.hpp"

namespace seven {

struct ExecutionContext {
  CpuState& state;
  Memory& memory;
  const iced_x86::Instruction& instr;
  std::uint64_t next_rip;
  bool control_flow_taken = false;
  std::uint64_t debug_hit_bits = 0;
  bool push_rf_for_debug = false;
};

class Executor {
 public:
  using HookId = std::uint64_t;
  using ContextSyncCallback = std::function<bool(CpuState&)>;

  Executor();

  [[nodiscard]] ExecutionResult step(CpuState& state, Memory& memory);
  [[nodiscard]] ExecutionResult run(CpuState& state, Memory& memory, std::size_t max_instructions);
  void set_context_read_callback(ContextSyncCallback fn);
  void set_context_write_callback(ContextSyncCallback fn);
  void request_stop() noexcept;
  void clear_stop_request() noexcept;
  [[nodiscard]] bool stop_requested() const noexcept;
  [[nodiscard]] bool has_violation() const noexcept;
  void clear_violation() noexcept;
  [[nodiscard]] std::uint64_t violation_ip() const noexcept;
  [[nodiscard]] std::uint64_t violation_address() const noexcept;
  [[nodiscard]] StopReason violation_reason() const noexcept;
  [[nodiscard]] HookId add_instruction_hook(InstructionHook hook);
  [[nodiscard]] HookId add_code_hook(iced_x86::Code code, InstructionHook hook);
  [[nodiscard]] HookId add_execution_hook(std::function<void(std::uint64_t)> hook);
  [[nodiscard]] HookId add_execution_hook(std::uint64_t address, std::function<void(std::uint64_t)> hook);
  [[nodiscard]] HookId add_stop_hook(StopHook hook);
  [[nodiscard]] HookId add_fault_hook(FaultHook hook);
  [[nodiscard]] HookId add_trap_hook(TrapKind kind, TrapHook hook);
  // How many times step() will let a fault hook ask for the same instruction again before giving up.
  static constexpr std::size_t kMaxFaultRetries = 8;
  // How deep an embedder callback may re-enter step(). Three is legitimate: a device callback
  // inside a fault handler inside a step.
  static constexpr std::size_t kMaxStepDepth = 8;
  // Levels alone do not bound bytes: one re-entry cycle costs ~105 KB, so the level cap permits
  // ~950 KB of a 1 MB stack. Bounded by actual descent as well.
  static constexpr std::size_t kMaxStepStackBytes = 256u * 1024u;
  // step()'s fault path, for an engine that raised the fault in its own generated code. state.rip
  // must already point at the faulting instruction. True means a hook wants it retried.
  [[nodiscard]] bool report_external_fault(CpuState& state, Memory& memory, const ExecutionResult& fault,
                                           std::uint64_t fault_address);
  [[nodiscard]] bool remove_hook(HookId id);
  void clear_hooks();
  [[nodiscard]] std::size_t supported_code_count() const noexcept;
  [[nodiscard]] const std::vector<std::uint64_t>& code_execution_counts() const noexcept;
  [[nodiscard]] const std::vector<std::uint64_t>& stop_reason_counts() const noexcept;
  [[nodiscard]] std::uint64_t total_steps() const noexcept;
  [[nodiscard]] std::uint64_t total_retired() const noexcept;
  void reset_stats();
  // Whether an external engine may run a span from state.rip without going through step() at all.
  // False means every instruction must, so hooks and traps keep firing at the expected granularity.
  [[nodiscard]] bool jit_bypass_eligible(const CpuState& state, const Memory& memory) const noexcept;
  // step_impl()'s own per-opcode dispatch, exposed so an external caller can run the real handler.
  // Runs only the handler: the caller sets the dead-flags mask and commits rip/rsp itself.
  [[nodiscard]] static ExecutionResult dispatch_handler(ExecutionContext& ctx, iced_x86::Code code);
  // Codes step_impl() routes to a trap hook rather than dispatch_handler(), so an external caller
  // knows which ones it cannot hand off the same way.
  [[nodiscard]] static bool is_trap_instruction(iced_x86::Code code) noexcept;
  // Whether this SIMD encoding and width are allowed under the compiled-in profile, the same gate
  // step_impl() applies. dispatch_handler() does NOT re-check it, so an external caller running a
  // vector instruction directly must call this first or a disabled op runs anyway.
  [[nodiscard]] static bool simd_profile_allows(const iced_x86::Instruction& instr) noexcept;

 private:
  using CodeIndex = std::size_t;
  static constexpr std::size_t kCodeCount = static_cast<std::size_t>(iced_x86::IcedConstants::CODE_ENUM_COUNT);
  static constexpr std::size_t kStopReasonCount = static_cast<std::size_t>(StopReason::stop_requested) + 1;

  static constexpr std::size_t stop_reason_to_index(StopReason reason) noexcept;
  [[nodiscard]] InstructionHookAction run_instruction_hooks(InstructionHookContext& ctx, ExecutionResult& stop_result);
  [[nodiscard]] TrapHookResult run_trap_hooks(TrapHookContext& ctx);
  [[nodiscard]] FaultHookAction run_fault_hooks(const FaultHookEvent& event);
  void notify_stop_hooks(CpuState& state, Memory& memory, const ExecutionResult& result, std::uint64_t fault_address) const;
  void apply_pending_hook_mutations();
  void refresh_hook_flags() noexcept;
  // Held across hook dispatch so add/remove/clear queue instead of mutating a container being
  // walked. Saved and restored, not assigned: a callback may re-enter, and a nested dispatch
  // clearing the flag would leave the outer loop walking with deferral off.
  struct HookDispatchScope {
    explicit HookDispatchScope(Executor& executor) noexcept;
    ~HookDispatchScope();
    HookDispatchScope(const HookDispatchScope&) = delete;
    HookDispatchScope& operator=(const HookDispatchScope&) = delete;

    Executor& self;
    bool was_dispatching;
  };
  // Counts step_impl frames on the stack, so a nested one leaves the shared decode cache alone.
  struct StepDepthScope {
    explicit StepDepthScope(Executor& executor) noexcept;
    ~StepDepthScope();
    StepDepthScope(const StepDepthScope&) = delete;
    StepDepthScope& operator=(const StepDepthScope&) = delete;

    Executor& self;
  };

  static ExecutionResult unsupported(ExecutionContext& ctx);
  // Shared dispatch core behind step() and run(). allow_masking gates whether a cached block's
  // dead_flags_mask may be applied: only run()'s loop guarantees it will keep advancing through the
  // block, so public step() always passes false.
  [[nodiscard]] ExecutionResult step_impl(CpuState& state, Memory& memory, bool allow_masking);
  static constexpr std::size_t kDecodeCacheSize = 8192;
  static constexpr std::size_t kCodePageCacheSize = 64;
  struct CachedCodePageEntry {
    std::uint64_t page_base = 0;
    // Per-page, not the process-wide code_epoch(): that moves on a write to any executable page,
    // so one guest store anywhere threw away every cached page and decode.
    std::uint64_t page_epoch = 0;
    bool valid = false;
    std::array<std::uint8_t, Memory::kPageSize> bytes{};
  };
  struct DecodedInstructionCacheEntry {
    std::uint64_t rip = 0;
    // First and last byte's page epochs, which differ only across a page boundary. Both must match.
    std::uint64_t page_epoch = 0;
    std::uint64_t last_page_epoch = 0;
    ExecutionMode mode = ExecutionMode::long64;
    bool valid = false;
    bool simd_allowed = true;
    std::uint8_t trap_kind = 0xFF;  // 0xFF == none
    std::uint32_t instruction_length = 0;
    iced_x86::Code reported_code = iced_x86::Code::INVALID;
    iced_x86::Instruction instr{};
    // Flags this instruction writes that the liveness pass proved dead. 0 outside an eligible
    // block, which is the safe default of computing every flag.
    std::uint64_t dead_flags_mask = 0;
  };
  static constexpr std::size_t kMaxBlockLiftLength = 64;
  [[nodiscard]] bool block_liveness_eligible(const Memory& memory) const noexcept;
  std::uint64_t total_steps_ = 0;
  std::uint64_t total_retired_ = 0;
  std::vector<std::uint64_t> code_execution_counts_;
  std::vector<std::uint64_t> stop_reason_counts_;
  HookId next_hook_id_ = 1;
  std::vector<std::pair<HookId, InstructionHook>> instruction_hooks_;
  std::unordered_map<iced_x86::Code, std::vector<std::pair<HookId, InstructionHook>>> code_hooks_;
  std::vector<std::pair<HookId, StopHook>> stop_hooks_;
  std::vector<std::pair<HookId, FaultHook>> fault_hooks_;
  std::unordered_map<TrapKind, std::vector<std::pair<HookId, TrapHook>>> trap_hooks_;
  // Heap-allocated: ~1 MB combined would blow a default thread stack.
  std::unique_ptr<std::array<CachedCodePageEntry, kCodePageCacheSize>> code_page_cache_ =
      std::make_unique<std::array<CachedCodePageEntry, kCodePageCacheSize>>();
  std::unique_ptr<std::array<DecodedInstructionCacheEntry, kDecodeCacheSize>> decode_cache_ =
      std::make_unique<std::array<DecodedInstructionCacheEntry, kDecodeCacheSize>>();
  std::vector<std::pair<HookId, std::function<void(std::uint64_t)>>> execution_hooks_;
  std::unordered_map<std::uint64_t, std::vector<std::pair<HookId, std::function<void(std::uint64_t)>>>> execution_address_hooks_;
  bool dispatching_hooks_ = false;
  std::vector<std::function<void()>> pending_hook_mutations_;
  // Cached emptiness so the hot path can skip hook dispatch without touching the containers.
  bool has_instruction_hooks_ = false;
  bool has_code_hooks_ = false;
  bool has_execution_hooks_ = false;
  bool has_execution_address_hooks_ = false;
  bool has_stop_hooks_ = false;
  bool has_fault_hooks_ = false;
  bool has_trap_hooks_ = false;
  // Tracing flags resolved once at construction.
  bool trace_semantics_ = false;
  bool collect_code_stats_ = false;
  // Resolved at construction: std::getenv() measures ~2.4us here, and this used to run on every
  // dispatch, dwarfing the instruction itself.
  bool decode_cache_disabled_by_env_ = false;
  // Which Memory the decode caches were filled from. 0 is never a real id, so the first step
  // always refills.
  std::uint64_t cache_memory_instance_ = 0;
  // One private decode slot per nesting depth, by unique_ptr so an outer frame's reference into
  // its slot survives this vector growing.
  std::size_t step_depth_ = 0;
  // Where the outermost step_impl frame sat, so a nested one can measure the descent.
  std::uintptr_t step_stack_base_ = 0;
  std::vector<std::unique_ptr<DecodedInstructionCacheEntry>> nested_decode_scratch_;
  ContextSyncCallback context_read_cb_{};
  ContextSyncCallback context_write_cb_{};
  bool stop_requested_ = false;
  bool has_violation_ = false;
  std::uint64_t violation_ip_ = 0;
  std::uint64_t violation_address_ = 0;
  StopReason violation_reason_ = StopReason::none;
};

std::vector<std::uint8_t> parse_hex_bytes(std::string_view text);

}  // namespace seven


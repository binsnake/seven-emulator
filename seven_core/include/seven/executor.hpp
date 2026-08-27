#pragma once

#include <array>
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
  // The fault path step() runs, for an execution engine (see seven_jit) that raised the fault in its
  // own generated code instead of going through step(). state.rip must already point at the faulting
  // instruction. Returns true if a fault hook wants that instruction attempted again; otherwise it
  // has recorded the violation and notified the stop hooks, and the caller should report the fault.
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
  // True if it's currently safe for an external consumer (e.g. a native-codegen layer sitting on
  // top of this Executor) to run its own code for a span of instructions starting at `state.rip`
  // without going through step()/step_impl() at all -- meaning no hook that needs full
  // per-instruction visibility is registered, and the CPU isn't in a state (trap flag set, active
  // hardware execute breakpoint) that requires per-instruction interpreter dispatch regardless of
  // hooks. If this returns false, every instruction in that span must go through step() so hooks
  // and traps keep firing at the granularity callers already depend on.
  [[nodiscard]] bool jit_bypass_eligible(const CpuState& state, const Memory& memory) const noexcept;
  // Same per-opcode dispatch step_impl() uses, for external callers that want to run one
  // instruction through the real handler instead of reimplementing it. Just runs the handler --
  // caller sets detail::set_dead_flags_mask() beforehand and commits rip/rsp afterward itself.
  [[nodiscard]] static ExecutionResult dispatch_handler(ExecutionContext& ctx, iced_x86::Code code);
  // True for SYSCALL/CPUID/RDTSC/RDTSCP/INT* -- codes step_impl() routes to a trap hook instead of
  // dispatch_handler(). An external caller doing its own dispatch_handler() call needs this to
  // know which codes it can't just hand off the same way.
  [[nodiscard]] static bool is_trap_instruction(iced_x86::Code code) noexcept;
  // True if this instruction's SIMD encoding/register width is allowed under the compiled-in
  // AVX/AVX-512/max-vector-width profile (SEVEN_ENABLE_AVX, SEVEN_ENABLE_AVX512,
  // SEVEN_MAX_VECTOR_BYTES) -- the same gate step_impl() applies before ever calling
  // dispatch_handler() for a SIMD instruction (see kEnableAvx/kEnableAvx512/kVectorBytes in
  // executor.cpp). dispatch_handler() itself does NOT re-check this internally, so an external
  // caller invoking it directly for a vector-register instruction (e.g. a native-codegen callout
  // bridge) MUST call this first, the same way step_impl() does, or a disabled/oversized SIMD op
  // would silently run anyway.
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
  // Held for the duration of any hook dispatch, so add/remove/clear queue themselves instead of
  // mutating a container that is currently being walked. Saved and restored rather than assigned,
  // for the reason Memory::access_allowed spells out for its own dispatch: a hook callback may
  // re-enter the executor, and a nested dispatch that cleared the flag on the way out would leave
  // the outer loop iterating with deferral switched off. Only the outermost scope flushes the queue.
  struct HookDispatchScope {
    explicit HookDispatchScope(Executor& executor) noexcept;
    ~HookDispatchScope();
    HookDispatchScope(const HookDispatchScope&) = delete;
    HookDispatchScope& operator=(const HookDispatchScope&) = delete;

    Executor& self;
    bool was_dispatching;
  };
  // Counts step_impl frames currently on the stack, so a nested one knows to keep its hands off
  // the shared decode cache -- see step_impl's `nested` for what goes wrong otherwise.
  struct StepDepthScope {
    explicit StepDepthScope(Executor& executor) noexcept;
    ~StepDepthScope();
    StepDepthScope(const StepDepthScope&) = delete;
    StepDepthScope& operator=(const StepDepthScope&) = delete;

    Executor& self;
  };

  static ExecutionResult unsupported(ExecutionContext& ctx);
  // The shared dispatch core behind both step() and run()'s internal loop. `allow_masking` gates
  // whether a cached block's precomputed dead_flags_mask may actually be applied -- see its call
  // sites for why this can never just be "trust the
  // cache": a bare step() call never guarantees the caller will keep advancing through the rest of
  // a lifted block, so the public step() always passes false. Only run()'s own internal loop,
  // which does guarantee that (given enough budget headroom), passes true.
  [[nodiscard]] ExecutionResult step_impl(CpuState& state, Memory& memory, bool allow_masking);
  static constexpr std::size_t kDecodeCacheSize = 8192;
  static constexpr std::size_t kCodePageCacheSize = 64;
  struct CachedCodePageEntry {
    std::uint64_t page_base = 0;
    // Memory::page_code_epoch() for this page, not the process-wide Memory::code_epoch(). The
    // global counter moves on a write to ANY executable page, so keying on it meant one guest store
    // anywhere threw away all 64 cached pages and all 8192 decodes below along with them.
    std::uint64_t page_epoch = 0;
    bool valid = false;
    std::array<std::uint8_t, Memory::kPageSize> bytes{};
  };
  struct DecodedInstructionCacheEntry {
    std::uint64_t rip = 0;
    // Page epochs for the first and last byte of this instruction -- the two can differ only when
    // it straddles a page boundary, and both have to still match for the decode to be reusable.
    std::uint64_t page_epoch = 0;
    std::uint64_t last_page_epoch = 0;
    ExecutionMode mode = ExecutionMode::long64;
    bool valid = false;
    bool simd_allowed = true;
    std::uint8_t trap_kind = 0xFF;  // 0xFF == none
    std::uint32_t instruction_length = 0;
    iced_x86::Code reported_code = iced_x86::Code::INVALID;
    iced_x86::Instruction instr{};
    // ALU status flags (subset of kAluStatusFlagsMask) this instruction writes that the block
    // liveness pass proved dead -- see seven/flag_liveness.hpp. 0 for every instruction outside a
    // liveness-eligible block (the correct, always-safe default: compute every flag).
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
  // Heap-allocated to avoid stack pressure — the combined size (~1 MB) would
  // blow Windows' default 1 MB thread stack if Executors are stack-allocated.
  std::unique_ptr<std::array<CachedCodePageEntry, kCodePageCacheSize>> code_page_cache_ =
      std::make_unique<std::array<CachedCodePageEntry, kCodePageCacheSize>>();
  std::unique_ptr<std::array<DecodedInstructionCacheEntry, kDecodeCacheSize>> decode_cache_ =
      std::make_unique<std::array<DecodedInstructionCacheEntry, kDecodeCacheSize>>();
  std::vector<std::pair<HookId, std::function<void(std::uint64_t)>>> execution_hooks_;
  std::unordered_map<std::uint64_t, std::vector<std::pair<HookId, std::function<void(std::uint64_t)>>>> execution_address_hooks_;
  bool dispatching_hooks_ = false;
  std::vector<std::function<void()>> pending_hook_mutations_;
  // Cached emptiness flags so the per-step hot path can short-circuit hook
  // dispatch without touching any of the underlying containers.
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
  // Resolved once at construction, not per-dispatch: std::getenv() is not a cheap call (measured
  // ~2.4us on this machine, backed by a linear scan and possibly a lock in some CRTs) and this used
  // to be evaluated on every single step_impl() dispatch, dwarfing the actual cost of decoding and
  // executing an instruction.
  bool decode_cache_disabled_by_env_ = false;
  // Which Memory the decode and code-page caches were filled from -- see step_impl. 0 is never a
  // real instance id, so the first step always refills.
  std::uint64_t cache_memory_instance_ = 0;
  // Depth of step_impl frames on the stack, and one private decode slot per nested depth. Held by
  // unique_ptr so an outer frame's reference into its slot survives this vector growing.
  std::size_t step_depth_ = 0;
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


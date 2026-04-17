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
  [[nodiscard]] bool remove_hook(HookId id);
  void clear_hooks();
  [[nodiscard]] std::size_t supported_code_count() const noexcept;
  [[nodiscard]] const std::vector<std::uint64_t>& code_execution_counts() const noexcept;
  [[nodiscard]] const std::vector<std::uint64_t>& stop_reason_counts() const noexcept;
  [[nodiscard]] std::uint64_t total_steps() const noexcept;
  [[nodiscard]] std::uint64_t total_retired() const noexcept;
  void reset_stats();

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

  static ExecutionResult unsupported(ExecutionContext& ctx);
  static constexpr std::size_t kDecodeCacheSize = 8192;
  static constexpr std::size_t kCodePageCacheSize = 64;
  struct CachedCodePageEntry {
    std::uint64_t page_base = 0;
    std::uint64_t code_epoch = 0;
    bool valid = false;
    std::array<std::uint8_t, Memory::kPageSize> bytes{};
  };
  struct DecodedInstructionCacheEntry {
    std::uint64_t rip = 0;
    std::uint64_t code_epoch = 0;
    ExecutionMode mode = ExecutionMode::long64;
    bool valid = false;
    bool simd_allowed = true;
    std::uint8_t trap_kind = 0xFF;  // 0xFF == none
    std::uint32_t instruction_length = 0;
    iced_x86::Code reported_code = iced_x86::Code::INVALID;
    iced_x86::Instruction instr{};
  };
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
  bool trace_openkey_probe_ = false;
  bool trace_strrchr_ = false;
  bool collect_code_stats_ = false;
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


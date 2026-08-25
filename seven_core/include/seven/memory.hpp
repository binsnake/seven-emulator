#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace seven {

enum class MemoryAccessKind : std::uint8_t {
  instruction_fetch,
  data_read,
  data_write,
};

enum class MemoryPermission : std::uint8_t {
  read = 1u << 0,
  write = 1u << 1,
  execute = 1u << 2,
};

using MemoryPermissionMask = std::uint8_t;

constexpr MemoryPermissionMask operator|(MemoryPermission lhs, MemoryPermission rhs) noexcept {
  return static_cast<MemoryPermissionMask>(lhs) | static_cast<MemoryPermissionMask>(rhs);
}

constexpr MemoryPermissionMask kMemoryPermissionReadWrite =
    static_cast<MemoryPermissionMask>(MemoryPermission::read) |
    static_cast<MemoryPermissionMask>(MemoryPermission::write);
constexpr MemoryPermissionMask kMemoryPermissionAll =
    kMemoryPermissionReadWrite | static_cast<MemoryPermissionMask>(MemoryPermission::execute);

struct MemoryAccessEvent {
  MemoryAccessKind kind = MemoryAccessKind::data_read;
  std::uint64_t address = 0;
  std::size_t size = 0;
  const void* data = nullptr;
  std::size_t data_size = 0;
};

using MemoryAccessKindMask = std::uint8_t;

constexpr MemoryAccessKindMask bit(MemoryAccessKind kind) noexcept {
  return static_cast<MemoryAccessKindMask>(1u << static_cast<std::uint8_t>(kind));
}

constexpr MemoryAccessKindMask kAllMemoryAccessKinds =
    bit(MemoryAccessKind::instruction_fetch) | bit(MemoryAccessKind::data_read) | bit(MemoryAccessKind::data_write);

struct MemoryHookRange {
  std::uint64_t base = 0;
  std::size_t size = 0;
};

class Memory {
 public:
  static constexpr std::size_t kPageSize = 0x1000;
  using HookId = std::uint64_t;
  using AccessHook = std::function<bool(const MemoryAccessEvent&)>;
  using MmioReadCallback = std::function<bool(std::uint64_t offset, void* dst, std::size_t size)>;
  using MmioWriteCallback = std::function<bool(std::uint64_t offset, const void* src, std::size_t size)>;
  using PassthroughReadFn  = std::function<bool(std::uint64_t addr, void* dst, std::size_t size)>;
  using PassthroughWriteFn = std::function<bool(std::uint64_t addr, const void* src, std::size_t size)>;
  struct MmioRegionSnapshot {
    HookId id = 0;
    std::uint64_t base = 0;
    std::size_t size = 0;
  };
  using MmioResolver =
      std::function<std::optional<std::pair<MmioReadCallback, MmioWriteCallback>>(const MmioRegionSnapshot&)>;

  struct PageSnapshot {
    std::uint64_t page_index = 0;
    std::array<std::byte, kPageSize> data{};
    MemoryPermissionMask permissions = kMemoryPermissionAll;
  };

  // JIT-visible fast-path memory TLB: a small, direct-mapped, PUBLIC cache a JIT consumer's
  // generated code can read and populate directly via offsetof-computed addressing, entirely
  // separate from Memory's own PRIVATE tlb_ (which only ever serves read()/write()/is_mapped()
  // internally and is never touched by anything outside this class). Lets compiled code do a
  // bounds+permission-checked load/store straight against host memory, skipping a trampoline call
  // entirely on a hit. Memory's own methods never read or populate this -- keeping it coherent
  // (invalidating it whenever a cached host_data pointer could go stale, gating it off whenever a
  // hook/MMIO/passthrough could legitimately want to intercept an access) is entirely this class's
  // job internally (see unmap()/reprotect()/restore_pages() and the hook/mmio/passthrough setters),
  // but USING it is entirely the JIT layer's responsibility -- Memory itself never reads jit_tlb.
  struct JitTlbSlot {
    std::uint64_t guest_page = ~0ull;  // ~0 is never a valid page index (address space is 2^64/kPageSize wide, but a page this high is unreachable in long/legacy mode); doubles as "empty"
    std::byte* host_data = nullptr;    // this page's raw byte array iff guest_page matches; nullptr otherwise
    std::uint8_t permissions = 0;
  };
  static constexpr std::size_t kJitTlbSize = 16;  // power of two -- direct-mapped by guest_page & (kJitTlbSize-1)
  std::array<JitTlbSlot, kJitTlbSize> jit_tlb{};
  // True whenever an access hook, an MMIO region, or a passthrough callback is active -- any of
  // which means SOME address might need to be intercepted rather than read/written directly, the
  // same condition read()/write() themselves already gate their own internal fast path on. A JIT
  // fast path MUST check this (kept up to date automatically, see refresh_jit_fast_path_blocked())
  // before ever trusting jit_tlb, and treat "blocked" as "always take the slow path," never cache
  // around it.
  bool jit_fast_path_blocked = false;
  // Diagnostic-only counter a JIT consumer's generated code may increment directly on a jit_tlb
  // hit -- exists for the same reason JitExecutor's compile_count()/hint_hit_count() do: correct
  // output alone can't prove the fast path engaged rather than the slow path happening to agree.
  // Memory never reads or writes this itself.
  std::uint64_t jit_fast_path_hits = 0;

  void map(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions = kMemoryPermissionAll);
  void unmap(std::uint64_t base, std::size_t size);
  void reprotect(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions);
  [[nodiscard]] bool is_mapped(std::uint64_t address, std::size_t size) const;
  [[nodiscard]] bool has_permissions(std::uint64_t address, std::size_t size, MemoryPermissionMask required) const;
  [[nodiscard]] bool read(std::uint64_t address, void* dst, std::size_t size, MemoryAccessKind kind = MemoryAccessKind::data_read) const;
  [[nodiscard]] bool read_unchecked(std::uint64_t address, void* dst, std::size_t size) const;
  [[nodiscard]] bool read_code_page(std::uint64_t page_base, void* dst) const;
  [[nodiscard]] bool write(std::uint64_t address, const void* src, std::size_t size, MemoryAccessKind kind = MemoryAccessKind::data_write);
  [[nodiscard]] bool write_unchecked(std::uint64_t address, const void* src, std::size_t size);
  [[nodiscard]] HookId add_access_hook(AccessHook hook, std::optional<MemoryHookRange> range = std::nullopt,
                                       MemoryAccessKindMask kinds = kAllMemoryAccessKinds);
  [[nodiscard]] bool remove_access_hook(HookId id);
  [[nodiscard]] HookId map_mmio(std::uint64_t base, std::size_t size, MmioReadCallback on_read, MmioWriteCallback on_write);
  [[nodiscard]] bool unmap_mmio(HookId id);
  void clear_mmio_regions();
  [[nodiscard]] std::vector<PageSnapshot> snapshot_pages() const;
  void restore_pages(const std::vector<PageSnapshot>& pages);
  [[nodiscard]] std::vector<MmioRegionSnapshot> snapshot_mmio_regions() const;
  void restore_mmio_regions(const std::vector<MmioRegionSnapshot>& regions, const MmioResolver& resolver);
  void set_passthrough(PassthroughReadFn read_fn, PassthroughWriteFn write_fn);
  void clear_passthrough();
  [[nodiscard]] bool has_passthrough() const noexcept { return static_cast<bool>(passthrough_read_); }

  [[nodiscard]] bool has_access_hooks() const noexcept {
    return has_any_access_hooks_;
  }
  [[nodiscard]] bool has_data_access_hooks() const noexcept {
    return (active_access_hook_kinds_ &
            (bit(MemoryAccessKind::data_read) | bit(MemoryAccessKind::data_write))) != 0;
  }
  [[nodiscard]] bool has_fetch_access_hooks() const noexcept {
    return (active_access_hook_kinds_ & bit(MemoryAccessKind::instruction_fetch)) != 0;
  }
  [[nodiscard]] std::uint64_t code_epoch() const noexcept { return code_epoch_; }
  // Per-page version of code_epoch() -- lets a cache invalidate just the pages it covers instead
  // of everything. Values share code_epoch_'s counter so a remapped page never reuses an old one's
  // epoch; an unmapped page reads back 0.
  [[nodiscard]] std::uint64_t page_code_epoch(std::uint64_t page_index) const noexcept {
    const auto* entry = lookup_page(page_index);
    return entry != nullptr ? entry->code_epoch : 0;
  }
  // Exposes what lookup_page() already resolves internally, for a jit_tlb miss to populate a slot
  // with. host_data is valid for as long as this page stays mapped (see unmap()/restore_pages()) --
  // a caller populating jit_tlb from these is trusting the exact same invariant Memory's own
  // internal tlb_ already depends on for the same pointer. Returns nullptr/0 if unmapped.
  [[nodiscard]] std::byte* page_data(std::uint64_t page_index) const noexcept {
    auto* entry = lookup_page(page_index);
    return entry != nullptr ? entry->data.data() : nullptr;
  }
  [[nodiscard]] MemoryPermissionMask page_permissions(std::uint64_t page_index) const noexcept {
    const auto* entry = lookup_page(page_index);
    return entry != nullptr ? entry->permissions : MemoryPermissionMask{0};
  }
  template <typename T>
  [[nodiscard]] bool read(std::uint64_t address, T& value, MemoryAccessKind kind = MemoryAccessKind::data_read) const {
    return read(address, &value, sizeof(T), kind);
  }

  template <typename T>
  [[nodiscard]] bool read_unchecked(std::uint64_t address, T& value) const {
    return read_unchecked(address, &value, sizeof(T));
  }


  template <typename T>
  [[nodiscard]] bool write(std::uint64_t address, const T& value, MemoryAccessKind kind = MemoryAccessKind::data_write) {
    return write(address, &value, sizeof(T), kind);
  }

  template <typename T>
  [[nodiscard]] bool write_unchecked(std::uint64_t address, const T& value) {
    return write_unchecked(address, &value, sizeof(T));
  }

 private:
  struct AccessHookEntry {
    HookId id = 0;
    AccessHook callback;
    std::optional<MemoryHookRange> range;
    MemoryAccessKindMask kinds = kAllMemoryAccessKinds;
  };
  struct MmioRegion {
    HookId id = 0;
    std::uint64_t base = 0;
    std::size_t size = 0;
    MmioReadCallback on_read;
    MmioWriteCallback on_write;
  };
  struct PageEntry {
    std::array<std::byte, kPageSize> data{};
    MemoryPermissionMask permissions = kMemoryPermissionAll;
    // Stamped from code_epoch_ (never independently incremented) whenever this page is (re)mapped
    // or written -- see page_code_epoch().
    std::uint64_t code_epoch = 0;
  };
  void apply_pending_access_hook_ops();
  void refresh_access_hook_state() noexcept;
  // Recomputes jit_fast_path_blocked from the same three conditions read()/write() already check
  // before taking their own internal fast path. Called from every access-hook/MMIO/passthrough
  // mutator -- see jit_fast_path_blocked's doc comment.
  void refresh_jit_fast_path_blocked() noexcept;
  // Resets every jit_tlb slot to empty. Called anywhere a cached host_data pointer could go stale
  // -- unmap() (erases the underlying PageEntry outright), restore_pages() (rebuilds pages_ from
  // scratch), and reprotect() (permissions, also cached per-slot, could change either direction).
  void clear_jit_tlb() noexcept { jit_tlb.fill(JitTlbSlot{}); }
  [[nodiscard]] const MmioRegion* find_mmio_region(std::uint64_t address, std::size_t size) const;
  [[nodiscard]] bool has_permission(MemoryPermissionMask permissions, MemoryAccessKind kind) const;
  [[nodiscard]] bool access_allowed(const MemoryAccessEvent& event) const;

  // Direct-mapped page lookup cache. The std::unordered_map remains the source
  // of truth; this slot table dramatically reduces per-access hash overhead for
  // hot working sets. Pointers into the underlying map remain stable across
  // insertions and reprotection (only erase invalidates them), so we bump
  // tlb_epoch_ on any operation that may erase / replace entries.
  static constexpr std::size_t kTlbSize = 128;  // power of two
  struct TlbSlot {
    std::uint64_t page_index = ~0ull;
    std::uint64_t epoch = 0;
    PageEntry* entry = nullptr;
  };
  [[nodiscard]] PageEntry* lookup_page(std::uint64_t page_index) const noexcept;
  void invalidate_tlb() noexcept { ++tlb_epoch_; }

  std::unordered_map<std::uint64_t, PageEntry> pages_;
  mutable std::array<TlbSlot, kTlbSize> tlb_{};
  std::uint64_t tlb_epoch_ = 1;
  std::vector<AccessHookEntry> access_hooks_;
  std::vector<HookId> pending_removed_access_hooks_;
  std::vector<AccessHookEntry> pending_added_access_hooks_;
  bool dispatching_access_hooks_ = false;
  bool has_any_access_hooks_ = false;
  MemoryAccessKindMask active_access_hook_kinds_ = 0;
  std::vector<MmioRegion> mmio_regions_;
  PassthroughReadFn  passthrough_read_{};
  PassthroughWriteFn passthrough_write_{};
  HookId next_hook_id_ = 1;
  std::uint64_t page_epoch_ = 1;
  std::uint64_t code_epoch_ = 1;
  std::uint64_t mmio_min_base_ = ~0ull;
  std::uint64_t mmio_max_end_ = 0;
};

}  // namespace seven

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

#include "seven/guarded_allocator.hpp"

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

  // A public, direct-mapped TLB that generated code reads and populates itself via offsetof
  // addressing, so a hit needs no trampoline call. Keeping it coherent is this class's job; using it
  // is the JIT layer's, and Memory itself never reads it.
  struct JitTlbSlot {
    std::uint64_t guest_page = ~0ull;  // ~0 is never a valid page index (address space is 2^64/kPageSize wide, but a page this high is unreachable in long/legacy mode); doubles as "empty"
    std::byte* host_data = nullptr;    // this page's raw byte array iff guest_page matches; nullptr otherwise
    std::uint8_t permissions = 0;
  };
  static constexpr std::size_t kJitTlbSize = 16;  // power of two -- direct-mapped by guest_page & (kJitTlbSize-1)
  // Slots hold raw pointers into this object's pages_, so none survives a copy or move.
  // lookup_page() catches a mismatch, but generated code reads this table without calling anything.
  struct JitTlb {
    std::array<JitTlbSlot, kJitTlbSize> slots{};

    JitTlb() noexcept = default;
    JitTlb(const JitTlb&) noexcept {}
    JitTlb& operator=(const JitTlb&) noexcept { slots = {}; return *this; }
    JitTlb(JitTlb&& other) noexcept : slots(other.slots) { other.slots = {}; }
    JitTlb& operator=(JitTlb&& other) noexcept {
      slots = other.slots;
      other.slots = {};
      return *this;
    }

    [[nodiscard]] JitTlbSlot& operator[](std::size_t index) noexcept { return slots[index]; }
    [[nodiscard]] const JitTlbSlot& operator[](std::size_t index) const noexcept { return slots[index]; }
    [[nodiscard]] std::size_t size() const noexcept { return slots.size(); }
    [[nodiscard]] JitTlbSlot* begin() noexcept { return slots.data(); }
    [[nodiscard]] JitTlbSlot* end() noexcept { return slots.data() + slots.size(); }
    [[nodiscard]] const JitTlbSlot* begin() const noexcept { return slots.data(); }
    [[nodiscard]] const JitTlbSlot* end() const noexcept { return slots.data() + slots.size(); }
    void clear() noexcept { slots = {}; }
  };
  JitTlb jit_tlb{};
  // Set whenever a hook, MMIO region or passthrough could need to intercept an access, the same
  // condition read()/write() gate their own fast path on. A JIT fast path MUST check this before
  // trusting jit_tlb and must never cache around it.
  bool jit_fast_path_blocked = false;
  // Diagnostic counter generated code increments on a jit_tlb hit, since correct output alone
  // cannot prove the fast path engaged. Memory never touches it.
  std::uint64_t jit_fast_path_hits = 0;

  // Identifies this exact Memory for the life of the process and never carries over to a copy.
  // Caches need it because code_epoch counters are per-Memory and all start near zero, so two
  // Memories agree almost immediately. An address will not do: they get reused.
  [[nodiscard]] std::uint64_t instance_id() const noexcept { return instance_id_.value(); }

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
  // Whether an mmio region serves this address rather than a real page. A block compiler needs it:
  // prefetching 256 bytes out of a device to compile three instructions is a read the guest never
  // asked for, and a register that clears on read does not care that it was speculative.
  [[nodiscard]] bool is_mmio_address(std::uint64_t address) const;
  // Whether any device region touches [base, base+size). find_mmio_region asks whether ONE region
  // covers the whole range, and says no to a range that merely runs into a device.
  [[nodiscard]] bool mmio_overlaps(std::uint64_t base, std::size_t size) const noexcept;
  [[nodiscard]] std::vector<PageSnapshot> snapshot_pages() const;
  void restore_pages(const std::vector<PageSnapshot>& pages);
  [[nodiscard]] std::vector<MmioRegionSnapshot> snapshot_mmio_regions() const;
  void restore_mmio_regions(const std::vector<MmioRegionSnapshot>& regions, const MmioResolver& resolver);
  void set_passthrough(PassthroughReadFn read_fn, PassthroughWriteFn write_fn);
  void clear_passthrough();
  // Either half counts: the caller wants to know whether page-backed storage is still
  // authoritative, and a write-only passthrough answers that too.
  [[nodiscard]] bool has_passthrough() const noexcept {
    return passthrough_read_ != nullptr || passthrough_write_ != nullptr;
  }

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
  // Bumped whenever an embedder callback runs under an access, so a JIT consumer can tell host code
  // ran mid-block. Counts dispatches, not devices.
  [[nodiscard]] std::uint64_t device_dispatch_count() const noexcept { return device_dispatch_count_; }
  // Per-page code_epoch(), so a cache invalidates only the pages it covers. Values share the one
  // counter, so a remapped page never reuses an old epoch; unmapped reads back 0.
  [[nodiscard]] std::uint64_t page_code_epoch(std::uint64_t page_index) const noexcept {
    // A passthrough write has no page to stamp, so it bumps only the global counter and a per-entry
    // epoch goes stale. Gating on has_passthrough() is what covers a write-only passthrough.
    if (has_passthrough()) {
      return code_epoch_;
    }
    const auto* entry = lookup_page(page_index);
    return entry != nullptr ? entry->code_epoch : 0;
  }
  // What lookup_page() already resolves, for a jit_tlb miss to populate a slot with. Valid for as
  // long as the page stays mapped, the same invariant Memory's own tlb_ depends on. nullptr if not.
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
  // data goes last on purpose. It is what a guest address indexes into, so an overrunning store
  // leaves the allocation where the allocator sees it rather than reaching the permission byte.
  struct PageEntry {
    MemoryPermissionMask permissions = kMemoryPermissionAll;
    // Stamped from code_epoch_, never incremented on its own. See page_code_epoch().
    std::uint64_t code_epoch = 0;
    std::array<std::byte, kPageSize> data{};
  };
  static_assert(offsetof(PageEntry, data) > offsetof(PageEntry, permissions),
                "the page bytes must sit after the metadata they could otherwise overwrite");
  void apply_pending_access_hook_ops();
  void refresh_access_hook_state() noexcept;
  // Recomputes jit_fast_path_blocked from the conditions read()/write() already check. Called from
  // every hook, MMIO and passthrough mutator.
  void refresh_jit_fast_path_blocked() noexcept;
  // Resets every jit_tlb slot to empty. Called anywhere a cached host_data pointer could go stale
  // -- unmap() (erases the underlying PageEntry outright), restore_pages() (rebuilds pages_ from
  // scratch), and reprotect() (permissions, also cached per-slot, could change either direction).
  void clear_jit_tlb() noexcept { jit_tlb.clear(); }
  // Mapping a device changes what a fetch returns without any page being written.
  void invalidate_code_epochs(std::uint64_t base, std::size_t size) noexcept;
  void invalidate_all_code_epochs() noexcept;
  [[nodiscard]] const MmioRegion* find_mmio_region(std::uint64_t address, std::size_t size) const;
  [[nodiscard]] bool has_permission(MemoryPermissionMask permissions, MemoryAccessKind kind) const;
  // Whether every page under [address, size) permits `kind`, by the same test the copy loops use. A
  // multi-page store that commits one page then faults on the next is not what hardware does.
  [[nodiscard]] bool span_permits(std::uint64_t address, std::size_t size, MemoryAccessKind kind) const;
  [[nodiscard]] bool access_allowed(const MemoryAccessEvent& event) const;

  // Direct-mapped lookup cache over pages_, which stays the source of truth. Map nodes are stable
  // across insertion and reprotection, so only an erase needs to bump tlb_epoch_.
  static constexpr std::size_t kTlbSize = 128;  // power of two
  struct TlbSlot {
    std::uint64_t page_index = ~0ull;
    std::uint64_t epoch = 0;
    PageEntry* entry = nullptr;
  };
  [[nodiscard]] PageEntry* lookup_page(std::uint64_t page_index) const noexcept;
  void invalidate_tlb() noexcept { ++tlb_epoch_; }

  // A copy takes a fresh number; a move hands its number to the destination, since the map relocates
  // but its nodes do not. The source of a move takes a fresh one rather than keeping a duplicate,
  // which would let a cache filled for one object be judged current for the other.
  class InstanceIdentity {
   public:
    InstanceIdentity() noexcept : value_(allocate()) {}
    InstanceIdentity(const InstanceIdentity&) noexcept : value_(allocate()) {}
    InstanceIdentity& operator=(const InstanceIdentity&) noexcept { value_ = allocate(); return *this; }
    InstanceIdentity(InstanceIdentity&& other) noexcept : value_(other.value_) { other.value_ = allocate(); }
    InstanceIdentity& operator=(InstanceIdentity&& other) noexcept {
      value_ = other.value_;
      other.value_ = allocate();
      return *this;
    }
    [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

   private:
    static std::uint64_t allocate() noexcept;
    std::uint64_t value_;
  };
  InstanceIdentity instance_id_{};
  // Mutable because read() and read_code_page() are const and still dispatch to a device.
  mutable std::uint64_t device_dispatch_count_ = 0;

  // Under SEVEN_GUARDED_PAGES each node sits with its last byte against an unmapped page, and
  // PageEntry's byte array is its last member, so an overrun off a guest page hits that guard even
  // when it comes from JIT-emitted code no sanitizer can instrument.
  std::unordered_map<std::uint64_t, PageEntry, std::hash<std::uint64_t>, std::equal_to<std::uint64_t>,
                     PageMapAllocator<std::pair<const std::uint64_t, PageEntry>>>
      pages_;
  // Which Memory the caches below were filled for, tagged by instance_id() rather than address since
  // addresses come back around: self-assignment through a copy handed the destination its own address
  // back and kept a table naming storage that had just been freed.
  mutable std::uint64_t cache_owner_id_ = 0;  // 0 is never a real instance number, so a fresh Memory refills once
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
  std::uint64_t code_epoch_ = 1;
  std::uint64_t mmio_min_base_ = ~0ull;
  std::uint64_t mmio_max_end_ = 0;
};

}  // namespace seven

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

  // A public, direct-mapped TLB that a JIT consumer's generated code reads and populates itself via
  // offsetof addressing, separate from Memory's private tlb_. It lets compiled code do a checked
  // load/store straight against host memory with no trampoline call on a hit.
  //
  // Keeping it coherent is this class's job (see unmap/reprotect/restore_pages and the hook, mmio
  // and passthrough setters); using it is the JIT layer's. Memory itself never reads it.
  struct JitTlbSlot {
    std::uint64_t guest_page = ~0ull;  // ~0 is never a valid page index (address space is 2^64/kPageSize wide, but a page this high is unreachable in long/legacy mode); doubles as "empty"
    std::byte* host_data = nullptr;    // this page's raw byte array iff guest_page matches; nullptr otherwise
    std::uint8_t permissions = 0;
  };
  static constexpr std::size_t kJitTlbSize = 16;  // power of two -- direct-mapped by guest_page & (kJitTlbSize-1)
  // Every slot holds a raw pointer into THIS object's pages_, so no slot survives a copy (which
  // deep-copies those pages) or a move (which hands them to somebody else). lookup_page() already
  // notices a table filled for a different instance, but it only gets the chance when something
  // calls a Memory method, and the whole point of this table is that generated code reads it without
  // calling anything -- a compiled block run straight against a moved-from Memory found its slots
  // still warm and stored into the object it had been moved into. Resetting here instead makes it
  // structural: the slots are gone before either object can be used again, whichever path gets there
  // first. Standard-layout with the array first, so offsetof-computed codegen is unaffected.
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

  // Identifies this exact Memory for as long as the process runs, and never carries over to a copy.
  // Anything caching something derived from this object's contents needs it: the code_epoch counters
  // those caches compare against are per-Memory and every Memory starts one near zero, so two of
  // them hold equal values almost immediately and one's cached decode would happily validate
  // against the other's bytes. An address is not enough on its own, since a destroyed Memory's
  // address gets handed straight back out for the next one.
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
  // True when this address is served by an mmio region rather than by a real page. A block compiler
  // needs this: it fetches far more bytes than the instruction it is about to run, and a device read
  // is not something to perform speculatively. Reading 256 bytes of "instruction stream" out of a
  // device to compile three instructions is a read the guest never asked for, and registers that
  // change when read do not care that the fetch was speculative.
  [[nodiscard]] bool is_mmio_address(std::uint64_t address) const;
  // True if any device region touches [base, base+size). find_mmio_region answers a different
  // question -- whether one region covers the whole range -- and says no to a range that merely runs
  // into a device, which is the wrong answer for a caller asking "are these bytes really mine".
  [[nodiscard]] bool mmio_overlaps(std::uint64_t base, std::size_t size) const noexcept;
  [[nodiscard]] std::vector<PageSnapshot> snapshot_pages() const;
  void restore_pages(const std::vector<PageSnapshot>& pages);
  [[nodiscard]] std::vector<MmioRegionSnapshot> snapshot_mmio_regions() const;
  void restore_mmio_regions(const std::vector<MmioRegionSnapshot>& regions, const MmioResolver& resolver);
  void set_passthrough(PassthroughReadFn read_fn, PassthroughWriteFn write_fn);
  void clear_passthrough();
  // Either half counts. set_passthrough takes the two independently, and a caller asking this
  // question wants to know whether page-backed storage is still authoritative -- a write-only
  // passthrough answers that just as much as a read one does.
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
  // Bumped whenever an embedder callback runs underneath an access (an MMIO on_read/on_write, or
  // either half of a passthrough). A JIT consumer compares it across one access to learn that host
  // code ran mid-block. Counts dispatches, not devices: having one mapped is not interesting.
  [[nodiscard]] std::uint64_t device_dispatch_count() const noexcept { return device_dispatch_count_; }
  // Per-page version of code_epoch() -- lets a cache invalidate just the pages it covers instead
  // of everything. Values share code_epoch_'s counter so a remapped page never reuses an old one's
  // epoch; an unmapped page reads back 0.
  [[nodiscard]] std::uint64_t page_code_epoch(std::uint64_t page_index) const noexcept {
    // A passthrough write bumps only the global code_epoch_, never a PageEntry: Memory::write's
    // passthrough branch has no page to stamp. So once either half of a passthrough is installed the
    // per-entry epoch stops being a reliable staleness signal, and self-modifying code written
    // through the passthrough would replay the bytes its block was first compiled from. Report the
    // global counter for every page in that configuration, so any passthrough write invalidates
    // every cached decode and compiled block. Gating on has_passthrough() rather than
    // passthrough_read_ is what covers a write-only passthrough (set_passthrough(nullptr, write_fn)),
    // the same split refresh_jit_fast_path_blocked() already accounts for.
    if (has_passthrough()) {
      return code_epoch_;
    }
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
  // data goes last on purpose. It is the one field a guest address indexes into, and the JIT's
  // fast path writes into it through a raw pointer that no sanitizer instruments. With the
  // metadata ahead of it, a store that runs off the end of the page leaves the allocation, where
  // the allocator and a sanitizer can both see it. Behind it sat this page's own permission byte,
  // so the same store would have quietly handed the guest whatever permissions it wrote -- which
  // is exactly what a mismatch between the fast path's bounds check and its access width would
  // have produced.
  struct PageEntry {
    MemoryPermissionMask permissions = kMemoryPermissionAll;
    // Stamped from code_epoch_ (never independently incremented) whenever this page is (re)mapped
    // or written -- see page_code_epoch().
    std::uint64_t code_epoch = 0;
    std::array<std::byte, kPageSize> data{};
  };
  static_assert(offsetof(PageEntry, data) > offsetof(PageEntry, permissions),
                "the page bytes must sit after the metadata they could otherwise overwrite");
  void apply_pending_access_hook_ops();
  void refresh_access_hook_state() noexcept;
  // Recomputes jit_fast_path_blocked from the same three conditions read()/write() already check
  // before taking their own internal fast path. Called from every access-hook/MMIO/passthrough
  // mutator -- see jit_fast_path_blocked's doc comment.
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
  // Whether every page under [address, size) permits `kind`, by exactly the test the copy loops use.
  // A multi-page access has to answer this before it moves any bytes: a store that commits the first
  // page and then faults on the second is not what hardware does, and a guard page beside a writable
  // one is the ordinary layout rather than a corner case.
  [[nodiscard]] bool span_permits(std::uint64_t address, std::size_t size, MemoryAccessKind kind) const;
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

  // A copy is a different Memory, so it takes a fresh number rather than inheriting one that a
  // consumer's cache is already tagged with. A move hands the number to the destination: the map
  // relocates but its nodes do not, so a cache filled before the move is still describing the right
  // object. The SOURCE of a move takes a fresh one instead of keeping a duplicate -- its pages have
  // just become the destination's while its own caches still name them, and leaving both objects
  // answering to the same number let a cache filled for one be judged current for the other.
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

  // The allocator is std::allocator unless SEVEN_GUARDED_PAGES is defined, in which case each node
  // is placed with its last byte against an unmapped page -- see guarded_allocator.hpp. PageEntry's
  // byte array is its last member (the static_assert above keeps it there), so that page is what an
  // access running off the end of a guest page hits, including one issued by JIT-emitted code that
  // no sanitizer can instrument.
  std::unordered_map<std::uint64_t, PageEntry, std::hash<std::uint64_t>, std::equal_to<std::uint64_t>,
                     PageMapAllocator<std::pair<const std::uint64_t, PageEntry>>>
      pages_;
  // Which Memory the caches below were filled for, by instance_id() rather than address, since a
  // copy brings tlb_/jit_tlb across still pointing into the SOURCE object's pages. A mismatch just
  // drops both and they refill.
  //
  // An address cannot be the tag because addresses come back around: assigning a Memory from a copy
  // of itself handed the destination its own address back, so the check matched and kept a table
  // naming storage the assignment had just freed. Instance numbers only go up, and a move keeps its
  // number, which is correct since a map move relocates the map and not its nodes.
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

#include "seven/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>

namespace seven {

namespace {

// Exclusive end page for [base, base+size), saturating instead of wrapping. Written as
// `(base + size + kPageSize - 1) / kPageSize` this collapses to 0 whenever the range touches the
// very top of the address space, which silently turned unmap() and reprotect() into no-ops there --
// a revoke the caller believes happened but didn't. Both additions can overflow independently, so
// neither is performed: a wrapped end means "through the last page", and the ceiling is taken with
// a remainder test rather than a bias term.
std::uint64_t page_range_end(std::uint64_t base, std::size_t size) noexcept {
  constexpr std::uint64_t kEndOfAddressSpace = (~std::uint64_t{0} / Memory::kPageSize) + 1;
  const auto end = base + size;
  if (end < base) {
    return kEndOfAddressSpace;
  }
  return (end / Memory::kPageSize) + ((end % Memory::kPageSize) != 0 ? 1 : 0);
}

// True when [base, base+size) runs off the top of the address space. Hardware faults there rather
// than wrapping to zero, and the page-walking loops below all advance a plain uint64 cursor, so
// without this they would happily carry on splicing in whatever lives at address 0.
bool access_wraps(std::uint64_t base, std::size_t size) noexcept {
  if (size == 0) {
    return false;
  }
  return base + (static_cast<std::uint64_t>(size) - 1) < base;
}

// The exclusive end of a registered range, held at the top of the address space rather than
// wrapping past it. Every range test below is a plain non-wrapping interval comparison, so an end
// that folded back down to a small number would quietly stop matching: an mmio region would lose
// its own addresses to the raw page path, and an access hook registered over that range would be
// skipped entirely. For a write hook that is a bypass of whatever the hook enforces, since
// access_allowed() runs before the page write. The guest-controlled side of these comparisons is
// already guarded; this is the host-registered side.
std::uint64_t range_end_saturating(std::uint64_t base, std::size_t size) noexcept {
  const auto end = base + size;
  return end < base ? ~std::uint64_t{0} : end;
}

}  // namespace

std::uint64_t Memory::InstanceIdentity::allocate() noexcept {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

Memory::PageEntry* Memory::lookup_page(std::uint64_t page_index) const noexcept {
  if (cache_owner_ != this) [[unlikely]] {
    // Filled for a different Memory -- see cache_owner_. Every path that could act on a cached
    // pointer comes through here first (read/write/is_mapped directly, and the JIT via
    // page_code_epoch/page_data before it trusts a jit_tlb slot), so this is where they get dropped.
    const_cast<Memory&>(*this).invalidate_tlb();
    const_cast<Memory&>(*this).clear_jit_tlb();
    cache_owner_ = this;
  }
  auto& slot = tlb_[page_index & (kTlbSize - 1)];
  if (slot.entry != nullptr && slot.page_index == page_index && slot.epoch == tlb_epoch_) {
    return slot.entry;
  }
  const auto it = pages_.find(page_index);
  if (it == pages_.end()) {
    slot.entry = nullptr;
    slot.page_index = page_index;
    slot.epoch = tlb_epoch_;
    return nullptr;
  }
  slot.entry = const_cast<PageEntry*>(&it->second);
  slot.page_index = page_index;
  slot.epoch = tlb_epoch_;
  return slot.entry;
}

void Memory::set_passthrough(PassthroughReadFn read_fn, PassthroughWriteFn write_fn) {
  passthrough_read_  = std::move(read_fn);
  passthrough_write_ = std::move(write_fn);
  refresh_jit_fast_path_blocked();
}

void Memory::clear_passthrough() {
  passthrough_read_  = nullptr;
  passthrough_write_ = nullptr;
  refresh_jit_fast_path_blocked();
}

void Memory::map(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions) {
  // An empty range must not touch anything. page_range_end rounds its end up, so with an unaligned
  // base a size of 0 otherwise resolves to one page and this quietly maps, erases or reprotects a
  // page the caller never named.
  if (size == 0) {
    return;
  }
  ++page_epoch_;
  // map() may insert new entries; std::unordered_map insertion can rehash and
  // invalidate iterators, but does not invalidate references / pointers to
  // existing elements. However, we still need to invalidate the TLB so that
  // negative cache entries (slots holding nullptr for previously-unmapped
  // pages) are refreshed.
  invalidate_tlb();
  // jit_tlb also caches permissions (not just a pointer), which map() can change on an
  // already-mapped page (try_emplace below is a no-op then, but permissions still get
  // overwritten) -- clear rather than risk a stale cached permission bit surviving this call.
  clear_jit_tlb();
  const auto first_page = base / kPageSize;
  const auto last_page = page_range_end(base, size);
  for (auto page = first_page; page < last_page; ++page) {
    auto [it, inserted] = pages_.try_emplace(page);
    (void)inserted;
    it->second.permissions = permissions;
    // Every (re)mapped page gets a fresh, never-before-used epoch, even one that lands back on a
    // page_index an earlier unmap() vacated -- a stale cache entry compiled against the OLD
    // occupant of this address can never alias the new one, since code_epoch_ only ever goes up.
    it->second.code_epoch = ++code_epoch_;
  }
}

void Memory::unmap(std::uint64_t base, std::size_t size) {
  // An empty range must not touch anything. page_range_end rounds its end up, so with an unaligned
  // base a size of 0 otherwise resolves to one page and this quietly maps, erases or reprotects a
  // page the caller never named.
  if (size == 0) {
    return;
  }
  ++page_epoch_;
  ++code_epoch_;
  invalidate_tlb();  // erase invalidates references; flush TLB
  // jit_tlb may hold host_data pointers straight into PageEntry objects this unmap is about to
  // erase -- those become dangling the instant pages_.erase() runs, not just logically stale.
  clear_jit_tlb();
  const auto first_page = base / kPageSize;
  const auto last_page = page_range_end(base, size);
  for (auto page = first_page; page < last_page; ++page) {
    pages_.erase(page);
  }
}

void Memory::reprotect(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions) {
  // An empty range must not touch anything. page_range_end rounds its end up, so with an unaligned
  // base a size of 0 otherwise resolves to one page and this quietly maps, erases or reprotects a
  // page the caller never named.
  if (size == 0) {
    return;
  }
  ++page_epoch_;
  // Reprotect does not erase entries, so cached PageEntry* pointers stay
  // valid. Permissions are read through the pointer, so we don't have to
  // invalidate the TLB.
  // jit_tlb caches permissions BY VALUE though (not read through host_data the way Memory's own
  // tlb_ re-reads permissions through PageEntry* every time) -- a reprotect can flip the exact bit
  // a cached slot's fast path would trust, so this one does need to clear it.
  clear_jit_tlb();
  const auto first_page = base / kPageSize;
  const auto last_page = page_range_end(base, size);
  for (auto page = first_page; page < last_page; ++page) {
    auto it = pages_.find(page);
    if (it != pages_.end()) {
      it->second.permissions = permissions;
      // A permission change can flip a page executable either direction; stamp it regardless so a
      // block compiled while it was executable doesn't survive a later revoke.
      it->second.code_epoch = ++code_epoch_;
    }
  }
}

bool Memory::is_mapped(std::uint64_t address, std::size_t size) const {
  if (access_wraps(address, size)) {
    return false;
  }
  std::size_t remaining = size;
  std::uint64_t current = address;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    if (lookup_page(page_index) == nullptr) {
      return false;
    }
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    current += chunk;
    remaining -= chunk;
  }
  return true;
}

bool Memory::has_permissions(std::uint64_t address, std::size_t size, MemoryPermissionMask required) const {
  if (access_wraps(address, size)) {
    return false;
  }
  std::size_t remaining = size;
  std::uint64_t current = address;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    const auto* entry = lookup_page(page_index);
    if (entry == nullptr) {
      return false;
    }
    auto effective = entry->permissions;
    if ((effective & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0) {
      effective |= static_cast<MemoryPermissionMask>(MemoryPermission::read);
    }
    if ((effective & required) != required) {
      return false;
    }
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    current += chunk;
    remaining -= chunk;
  }
  return true;
}

bool Memory::read(std::uint64_t address, void* dst, std::size_t size, MemoryAccessKind kind) const {
  if (access_wraps(address, size)) {
    return false;
  }
  // Copied out before the call, matching the MMIO dispatch below: a passthrough is the embedder's
  // whole memory backend, and one that reopens its handle by calling set_passthrough from inside a
  // read would otherwise free the functor it is still running out of.
  if (auto fn = passthrough_read_) {
    // Hooks run against the passthrough too. They are a policy layer over every access, not a
    // feature of the page-backed storage, and returning here without dispatching them meant an
    // embedder that installed both got no hook calls at all and no indication of it.
    if (!access_allowed(MemoryAccessEvent{kind, address, size, nullptr, 0})) {
      return false;
    }
    return fn(address, dst, size);
  }
  // Fast path: most reads in real workloads are entirely within a single page
  // and target a non-MMIO address with no access hooks installed. Inline that
  // case to skip every dynamic check besides the TLB lookup itself.
  if (!has_any_access_hooks_ && mmio_regions_.empty()) [[likely]] {
    const auto first_page = address / kPageSize;
    const auto first_offset = address % kPageSize;
    // Not `first_offset + size <= kPageSize`: size is a full-width size_t, so that sum wraps and a
    // huge size sails through the single-page guard straight into the memcpy below.
    if (size <= kPageSize - first_offset) [[likely]] {
      auto* entry = lookup_page(first_page);
      if (entry == nullptr || !has_permission(entry->permissions, kind)) {
        return false;
      }
      std::memcpy(dst, entry->data.data() + first_offset, size);
      return true;
    }
  }

  if (const auto* mmio = find_mmio_region(address, size)) {
    // Take a copy of the callback and base before anything else runs. mmio points into
    // mmio_regions_, and nothing defers MMIO mutation the way add/remove_access_hook is deferred --
    // so a hook dispatched below calling map_mmio (push_back, may reallocate), unmap_mmio (erase,
    // shifts every later element down) or clear_mmio_regions leaves that pointer naming a different
    // device or freed storage, and the std::function read back out of it is then called. Copying
    // also keeps the callable alive if the device reconfigures its own region from inside the call.
    auto on_read = mmio->on_read;
    const auto region_base = mmio->base;
    if (!access_allowed(MemoryAccessEvent{kind, address, size, nullptr, 0})) {
      return false;
    }
    return on_read != nullptr ? on_read(address - region_base, dst, size) : false;
  }

  const auto copy_from_pages = [&](std::byte* out) {
    std::size_t remaining = size;
    std::uint64_t current = address;
    while (remaining != 0) {
      const auto page_index = current / kPageSize;
      const auto page_offset = current % kPageSize;
      const auto* entry = lookup_page(page_index);
      if (entry == nullptr || !has_permission(entry->permissions, kind)) {
        return false;
      }
      const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
      std::memcpy(out, entry->data.data() + page_offset, chunk);
      out += chunk;
      current += chunk;
      remaining -= chunk;
    }
    return true;
  };

  auto* out = static_cast<std::byte*>(dst);
  if (!has_any_access_hooks_) {
    return copy_from_pages(out);
  }

  constexpr std::size_t kInlineReadBufferSize = 64;
  std::array<std::byte, kInlineReadBufferSize> inline_buffer{};
  std::vector<std::byte> heap_buffer{};
  auto* temp = inline_buffer.data();
  if (size > inline_buffer.size()) {
    heap_buffer.resize(size);
    temp = heap_buffer.data();
  }

  if (!copy_from_pages(temp)) {
    return false;
  }

  if (!access_allowed(MemoryAccessEvent{kind, address, size, temp, size})) {
    return false;
  }

  std::memcpy(dst, temp, size);
  return true;
}

bool Memory::read_unchecked(std::uint64_t address, void* dst, std::size_t size) const {
  if (access_wraps(address, size)) {
    return false;
  }
  if (const auto* mmio = find_mmio_region(address, size)) {
    // Copied out of mmio_regions_ before the call for the reason read()'s hooked path
    // spells out: a device callback is free to reconfigure its own region, which would
    // otherwise destroy or relocate the std::function while its own frame is still live.
    auto on_read = mmio->on_read;
    const auto region_base = mmio->base;
    return on_read != nullptr ? on_read(address - region_base, dst, size) : false;
  }

  auto* out = static_cast<std::byte*>(dst);
  std::size_t remaining = size;
  std::uint64_t current = address;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    const auto* entry = lookup_page(page_index);
    if (entry == nullptr) {
      return false;
    }
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    std::memcpy(out, entry->data.data() + page_offset, chunk);
    out += chunk;
    current += chunk;
    remaining -= chunk;
  }

  return true;
}

bool Memory::read_code_page(std::uint64_t page_base, void* dst) const {
  if (auto fn = passthrough_read_) return fn(page_base, dst, kPageSize);
  if ((page_base % kPageSize) != 0) {
    return false;
  }
  const auto* entry = lookup_page(page_base / kPageSize);
  if (entry == nullptr || !has_permission(entry->permissions, MemoryAccessKind::instruction_fetch)) {
    return false;
  }
  std::memcpy(dst, entry->data.data(), kPageSize);
  return true;
}

bool Memory::write(std::uint64_t address, const void* src, std::size_t size, MemoryAccessKind kind) {
  if (passthrough_write_) {
    // read() rejects a wrapping range before it ever reaches passthrough_read_, and a passthrough
    // is an embedder's whole memory implementation rather than a hook with something to veto -- it
    // should never be handed an (address, size) pair the read side is guaranteed never to see. The
    // documented bridge in examples/live_memory_windows.hpp forwards both straight to
    // Read/WriteProcessMemory.
    if (access_wraps(address, size)) {
      return false;
    }
    auto fn = passthrough_write_;
    // Same as the read side, and it matters more here: a write hook's veto is what blocks the
    // write, and on the page-backed path access_allowed() runs before the page is touched. Skipping
    // it for a passthrough handed the write straight to the backend with the veto never consulted.
    if (!access_allowed(MemoryAccessEvent{kind, address, size, src, size})) {
      return false;
    }
    if (!fn(address, src, size)) {
      return false;
    }
    // A passthrough can't tell us whether what it just wrote was executable, so every write has to
    // count as one that might have rewritten code. page_code_epoch() reports this counter for every
    // page while a passthrough is installed, which is what makes cached decodes and compiled blocks
    // go stale at all in that configuration.
    ++code_epoch_;
    return true;
  }
  // Fast path: no hooks, no MMIO, single-page access. Bypasses access_allowed
  // and all multi-page bookkeeping. We still maintain code_epoch for write
  // through executable pages so the decode cache stays correct.
  if (!has_any_access_hooks_ && mmio_regions_.empty()) [[likely]] {
    const auto first_page = address / kPageSize;
    const auto first_offset = address % kPageSize;
    // Not `first_offset + size <= kPageSize`: size is a full-width size_t, so that sum wraps and a
    // huge size sails through the single-page guard straight into the memcpy below.
    if (size <= kPageSize - first_offset) [[likely]] {
      auto* entry = lookup_page(first_page);
      if (entry == nullptr || !has_permission(entry->permissions, kind)) {
        return false;
      }
      std::memcpy(entry->data.data() + first_offset, src, size);
      if ((entry->permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0) {
        entry->code_epoch = ++code_epoch_;
      }
      return true;
    }
  }

  if (!access_allowed(MemoryAccessEvent{kind, address, size, src, size})) {
    return false;
  }
  // Deliberately after access_allowed rather than at the top of the function: a range-scoped write
  // hook has to keep getting the chance to see and veto a wrapping access, which is the property
  // the overlap-check fix relies on. The single-page fast path above cannot be reached by a
  // wrapping access anyway, since one always straddles the last page boundary.
  if (access_wraps(address, size)) {
    return false;
  }
  if (const auto* mmio = find_mmio_region(address, size)) {
    // Copied out of mmio_regions_ before the call for the reason read()'s hooked path
    // spells out: a device callback is free to reconfigure its own region, which would
    // otherwise destroy or relocate the std::function while its own frame is still live.
    auto on_write = mmio->on_write;
    const auto region_base = mmio->base;
    return on_write != nullptr ? on_write(address - region_base, src, size) : false;
  }

  const auto* in = static_cast<const std::byte*>(src);
  std::size_t remaining = size;
  std::uint64_t current = address;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    auto* entry = lookup_page(page_index);
    if (entry == nullptr || !has_permission(entry->permissions, kind)) {
      return false;
    }
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    std::memcpy(entry->data.data() + page_offset, in, chunk);
    if ((entry->permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0) {
      entry->code_epoch = ++code_epoch_;
    }
    in += chunk;
    current += chunk;
    remaining -= chunk;
  }
  return true;
}

bool Memory::write_unchecked(std::uint64_t address, const void* src, std::size_t size) {
  if (access_wraps(address, size)) {
    return false;
  }
  if (const auto* mmio = find_mmio_region(address, size)) {
    // Copied out of mmio_regions_ before the call for the reason read()'s hooked path
    // spells out: a device callback is free to reconfigure its own region, which would
    // otherwise destroy or relocate the std::function while its own frame is still live.
    auto on_write = mmio->on_write;
    const auto region_base = mmio->base;
    return on_write != nullptr ? on_write(address - region_base, src, size) : false;
  }

  const auto* in = static_cast<const std::byte*>(src);
  std::size_t remaining = size;
  std::uint64_t current = address;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    auto* entry = lookup_page(page_index);
    if (entry == nullptr) {
      return false;
    }
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    std::memcpy(entry->data.data() + page_offset, in, chunk);
    if ((entry->permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0) {
      entry->code_epoch = ++code_epoch_;
    }
    in += chunk;
    current += chunk;
    remaining -= chunk;
  }
  return true;
}

Memory::HookId Memory::add_access_hook(AccessHook hook, std::optional<MemoryHookRange> range, MemoryAccessKindMask kinds) {
  const auto id = next_hook_id_++;
  AccessHookEntry entry{id, std::move(hook), range, kinds};
  if (dispatching_access_hooks_) {
    pending_added_access_hooks_.push_back(std::move(entry));
  } else {
    access_hooks_.push_back(std::move(entry));
  }
  refresh_access_hook_state();
  return id;
}

bool Memory::remove_access_hook(HookId id) {
  auto exists = [&](const auto& hooks) {
    return std::find_if(hooks.begin(), hooks.end(), [&](const auto& e) { return e.id == id; }) != hooks.end();
  };
  if (dispatching_access_hooks_) {
    if (exists(access_hooks_) || exists(pending_added_access_hooks_)) {
      pending_removed_access_hooks_.push_back(id);
      refresh_access_hook_state();
      return true;
    }
    return false;
  }
  for (auto it = access_hooks_.begin(); it != access_hooks_.end(); ++it) {
    if (it->id == id) {
      access_hooks_.erase(it);
      refresh_access_hook_state();
      return true;
    }
  }
  return false;
}

void Memory::refresh_access_hook_state() noexcept {
  MemoryAccessKindMask mask = 0;
  for (const auto& entry : access_hooks_) {
    mask |= entry.kinds;
  }
  for (const auto& entry : pending_added_access_hooks_) {
    mask |= entry.kinds;
  }
  active_access_hook_kinds_ = mask;
  has_any_access_hooks_ = dispatching_access_hooks_ ||
                          !access_hooks_.empty() ||
                          !pending_added_access_hooks_.empty() ||
                          !pending_removed_access_hooks_.empty();
  refresh_jit_fast_path_blocked();
}

void Memory::refresh_jit_fast_path_blocked() noexcept {
  // Both halves of the passthrough matter. set_passthrough takes them independently, so a
  // write-only passthrough used to leave this false: reads went through the page path and
  // warmed jit_tlb slots, then compiled writes stored straight into PageEntry::data and never
  // called passthrough_write_ at all, diverging from what Memory::write itself would do.
  jit_fast_path_blocked = has_any_access_hooks_ || !mmio_regions_.empty() ||
                          passthrough_read_ != nullptr || passthrough_write_ != nullptr;
}

Memory::HookId Memory::map_mmio(std::uint64_t base, std::size_t size, MmioReadCallback on_read, MmioWriteCallback on_write) {
  const auto id = next_hook_id_++;
  mmio_regions_.push_back(MmioRegion{id, base, size, std::move(on_read), std::move(on_write)});
  if (mmio_regions_.size() == 1) {
    mmio_min_base_ = base;
    mmio_max_end_ = range_end_saturating(base, size);
  } else {
    mmio_min_base_ = std::min(mmio_min_base_, base);
    mmio_max_end_ = std::max(mmio_max_end_, range_end_saturating(base, size));
  }
  refresh_jit_fast_path_blocked();
  return id;
}

bool Memory::unmap_mmio(HookId id) {
  for (auto it = mmio_regions_.begin(); it != mmio_regions_.end(); ++it) {
    if (it->id == id) {
      mmio_regions_.erase(it);
      if (mmio_regions_.empty()) {
        mmio_min_base_ = ~0ull;
        mmio_max_end_ = 0;
      } else {
        mmio_min_base_ = mmio_regions_.front().base;
        mmio_max_end_ = range_end_saturating(mmio_regions_.front().base, mmio_regions_.front().size);
        for (const auto& region : mmio_regions_) {
          mmio_min_base_ = std::min(mmio_min_base_, region.base);
          mmio_max_end_ = std::max(mmio_max_end_, range_end_saturating(region.base, region.size));
        }
      }
      refresh_jit_fast_path_blocked();
      return true;
    }
  }
  return false;
}

void Memory::clear_mmio_regions() {
  mmio_regions_.clear();
  mmio_min_base_ = ~0ull;
  mmio_max_end_ = 0;
  refresh_jit_fast_path_blocked();
}

std::vector<Memory::PageSnapshot> Memory::snapshot_pages() const {
  std::vector<PageSnapshot> pages;
  pages.reserve(pages_.size());
  for (const auto& [page_index, entry] : pages_) {
    pages.push_back(PageSnapshot{page_index, entry.data, entry.permissions});
  }
  std::sort(pages.begin(), pages.end(), [](const auto& lhs, const auto& rhs) { return lhs.page_index < rhs.page_index; });
  return pages;
}

void Memory::restore_pages(const std::vector<PageSnapshot>& pages) {
  ++page_epoch_;
  invalidate_tlb();
  clear_jit_tlb();  // pages_.clear() below frees every PageEntry jit_tlb could be pointing at
  pages_.clear();
  for (const auto& snapshot : pages) {
    pages_.emplace(snapshot.page_index, PageEntry{snapshot.data, snapshot.permissions, ++code_epoch_});
  }
}

std::vector<Memory::MmioRegionSnapshot> Memory::snapshot_mmio_regions() const {
  std::vector<MmioRegionSnapshot> regions;
  regions.reserve(mmio_regions_.size());
  for (const auto& region : mmio_regions_) {
    regions.push_back(MmioRegionSnapshot{region.id, region.base, region.size});
  }
  std::sort(regions.begin(), regions.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.base != rhs.base) {
      return lhs.base < rhs.base;
    }
    return lhs.id < rhs.id;
  });
  return regions;
}

void Memory::restore_mmio_regions(const std::vector<MmioRegionSnapshot>& regions, const MmioResolver& resolver) {
  mmio_regions_.clear();
  mmio_min_base_ = ~0ull;
  mmio_max_end_ = 0;
  for (const auto& region : regions) {
    if (!resolver) {
      continue;
    }
    const auto callbacks = resolver(region);
    if (!callbacks.has_value()) {
      continue;
    }
    mmio_regions_.push_back(MmioRegion{
        region.id,
        region.base,
        region.size,
        callbacks->first,
        callbacks->second,
    });
    next_hook_id_ = std::max(next_hook_id_, region.id + 1);
    if (mmio_regions_.size() == 1) {
      mmio_min_base_ = region.base;
      mmio_max_end_ = range_end_saturating(region.base, region.size);
    } else {
      mmio_min_base_ = std::min(mmio_min_base_, region.base);
      mmio_max_end_ = std::max(mmio_max_end_, range_end_saturating(region.base, region.size));
    }
  }
  // Every other mmio mutator refreshes this. Without it a restore that brings regions back into an
  // empty Memory leaves compiled code believing it can still take the raw-page fast path, so the
  // mmio callbacks never fire for anything that is also backed by a real page.
  refresh_jit_fast_path_blocked();
}

void Memory::apply_pending_access_hook_ops() {
  if (!pending_removed_access_hooks_.empty()) {
    for (const auto id : pending_removed_access_hooks_) {
      access_hooks_.erase(
          std::remove_if(access_hooks_.begin(), access_hooks_.end(), [&](const auto& entry) { return entry.id == id; }),
          access_hooks_.end());
      pending_added_access_hooks_.erase(
          std::remove_if(pending_added_access_hooks_.begin(), pending_added_access_hooks_.end(),
                         [&](const auto& entry) { return entry.id == id; }),
          pending_added_access_hooks_.end());
    }
    pending_removed_access_hooks_.clear();
  }
  if (!pending_added_access_hooks_.empty()) {
    for (auto& entry : pending_added_access_hooks_) {
      access_hooks_.push_back(std::move(entry));
    }
    pending_added_access_hooks_.clear();
  }
  refresh_access_hook_state();
}

const Memory::MmioRegion* Memory::find_mmio_region(std::uint64_t address, std::size_t size) const {
  if (mmio_regions_.empty()) {
    return nullptr;
  }
  const auto end = address + static_cast<std::uint64_t>(size);
  // A guest-chosen address near the top of the address space plus size can wrap end back down
  // past zero. Left unchecked, that wrapped (small) end could pass the mmio_max_end_ pre-check
  // and then the per-region range check below despite address itself being nowhere near any
  // registered region -- a false match whose "address - region.base" offset (see read()/write())
  // would be a huge, effectively guest-controlled index into whatever the region's callback
  // trusts that offset to bound. Reject the wrap outright rather than let it reach that check.
  if (end < address) {
    return nullptr;
  }
  if (address < mmio_min_base_ || end > mmio_max_end_) {
    return nullptr;
  }
  for (const auto& region : mmio_regions_) {
    if (address >= region.base && end <= range_end_saturating(region.base, region.size)) {
      return &region;
    }
  }
  return nullptr;
}

bool Memory::has_permission(MemoryPermissionMask permissions, MemoryAccessKind kind) const {
  switch (kind) {
    case MemoryAccessKind::instruction_fetch:
      return (permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0;
    case MemoryAccessKind::data_read:
      return (permissions & static_cast<MemoryPermissionMask>(MemoryPermission::read)) != 0 ||
             (permissions & static_cast<MemoryPermissionMask>(MemoryPermission::write)) != 0 ||
             (permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0;
    case MemoryAccessKind::data_write:
      return (permissions & static_cast<MemoryPermissionMask>(MemoryPermission::write)) != 0;
    default:
      return false;
  }
}

bool Memory::access_allowed(const MemoryAccessEvent& event) const {
  if (!has_any_access_hooks_) {
    return true;
  }
  auto& self = const_cast<Memory&>(*this);

  // Save and restore rather than assign. A hook callback is free to touch guest memory, which
  // re-enters here; if the nested call cleared the flag on its way out, the outer loop below would
  // carry on iterating access_hooks_ with deferral switched off, so a later hook calling
  // remove_access_hook would erase from the vector mid-iteration and leave this range-for walking
  // freed std::function storage. Same reason the pending queue is only flushed at depth 0.
  const bool was_dispatching = self.dispatching_access_hooks_;
  const auto finish_dispatch = [&self, was_dispatching] {
    self.dispatching_access_hooks_ = was_dispatching;
    if (!was_dispatching) {
      self.apply_pending_access_hook_ops();
    }
  };

  self.dispatching_access_hooks_ = true;
  for (const auto& hook : access_hooks_) {
    if ((hook.kinds & bit(event.kind)) == 0) {
      continue;
    }
    if (hook.range.has_value()) {
      const auto range_base = hook.range->base;
      const auto range_end = range_end_saturating(range_base, hook.range->size);
      const auto access_end = event.address + event.size;
      // event.address is guest-controlled (unlike range_base/size, which the host set up when
      // registering the hook) -- a guest picking an address near ~0ull can make access_end wrap
      // back down past zero. A plain non-wrapping interval test against that wrapped value can
      // then wrongly decide "no overlap" for an access whose real (wrapping) span does touch
      // [range_base, range_end), letting it skip a hook meant to see every access to that range
      // -- for a write hook specifically, access_allowed() runs before the underlying page write,
      // so a skipped hook here is a real bypass of whatever the hook enforces, not just a missed
      // notification. Treat the wrap itself as "can't rule out overlap" and fall through to call
      // the hook rather than risk a false negative.
      const bool access_wraps = access_end < event.address;
      if (!access_wraps && (event.address >= range_end || access_end <= range_base)) {
        continue;
      }
    }
    if (!hook.callback(event)) {
      finish_dispatch();
      return false;
    }
  }
  finish_dispatch();
  return true;
}

}  // namespace seven

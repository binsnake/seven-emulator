#include "seven/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>

namespace seven {

namespace {

// Exclusive end page for [base, base+size), saturating instead of wrapping. The obvious
// `(base + size + kPageSize - 1) / kPageSize` collapses to 0 at the top of the address space, which
// silently turned unmap() and reprotect() into no-ops there.
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

// Exclusive end of a registered range, held at the top rather than wrapping. Every range test below
// is a non-wrapping comparison, so a folded-back end loses its own addresses and skips its hooks.
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
  if (cache_owner_id_ != instance_id_.value()) [[unlikely]] {
    // Filled for a different Memory -- see cache_owner_id_. Every path that could act on a cached
    // pointer comes through here first (read/write/is_mapped directly, and the JIT via
    // page_code_epoch/page_data before it trusts a jit_tlb slot), so this is where they get dropped.
    const_cast<Memory&>(*this).invalidate_tlb();
    const_cast<Memory&>(*this).clear_jit_tlb();
    cache_owner_id_ = instance_id_.value();
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

bool Memory::mmio_overlaps(std::uint64_t base, std::size_t size) const noexcept {
  if (mmio_regions_.empty() || size == 0) {
    return false;
  }
  const auto end = range_end_saturating(base, size);
  if (base >= mmio_max_end_ || mmio_min_base_ >= end) {
    return false;
  }
  for (const auto& region : mmio_regions_) {
    if (base < range_end_saturating(region.base, region.size) && region.base < end) {
      return true;
    }
  }
  return false;
}

void Memory::invalidate_code_epochs(std::uint64_t base, std::size_t size) noexcept {
  ++code_epoch_;
  if (size == 0) {
    return;
  }
  const auto first_page = base / kPageSize;
  const auto last_page = page_range_end(base, size);
  // Walks pages_, not the range: a device region can dwarf what is mapped under it.
  for (auto& [page_index, entry] : pages_) {
    if (page_index >= first_page && page_index < last_page) {
      entry.code_epoch = code_epoch_;
    }
  }
}

void Memory::invalidate_all_code_epochs() noexcept {
  ++code_epoch_;
  for (auto& [page_index, entry] : pages_) {
    entry.code_epoch = code_epoch_;
  }
}

void Memory::set_passthrough(PassthroughReadFn read_fn, PassthroughWriteFn write_fn) {
  passthrough_read_  = std::move(read_fn);
  passthrough_write_ = std::move(write_fn);
  invalidate_all_code_epochs();
  refresh_jit_fast_path_blocked();
}

void Memory::clear_passthrough() {
  passthrough_read_  = nullptr;
  passthrough_write_ = nullptr;
  invalidate_all_code_epochs();
  refresh_jit_fast_path_blocked();
}

void Memory::map(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions) {
  // An empty range must not touch anything. page_range_end rounds its end up, so with an unaligned
  // base a size of 0 otherwise resolves to one page and this quietly maps, erases or reprotects a
  // page the caller never named.
  if (size == 0) {
    return;
  }
  // Insertion never invalidates pointers to existing elements, but the TLB still has to go: its
  // negative entries (nullptr for previously-unmapped pages) are now wrong.
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
  // tlb_ survives: it reads permissions back through PageEntry* every time. jit_tlb caches them by
  // value, so a reprotect can flip the exact bit its fast path trusts.
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

bool Memory::span_permits(std::uint64_t address, std::size_t size, MemoryAccessKind kind) const {
  std::size_t remaining = size;
  std::uint64_t current = address;
  while (remaining != 0) {
    const auto* entry = lookup_page(current / kPageSize);
    if (entry == nullptr || !has_permission(entry->permissions, kind)) {
      return false;
    }
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - (current % kPageSize));
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
    ++device_dispatch_count_;
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
    // Copy the callback and base first: mmio points into mmio_regions_ and MMIO mutation is not
    // deferred, so a hook calling map_mmio or unmap_mmio below leaves it naming another device or
    // freed storage. The copy also keeps the callable alive across a self-reconfiguring device.
    auto on_read = mmio->on_read;
    const auto region_base = mmio->base;
    if (!access_allowed(MemoryAccessEvent{kind, address, size, nullptr, 0})) {
      return false;
    }
    if (on_read == nullptr) return false;
    ++device_dispatch_count_;
    return on_read(address - region_base, dst, size);
  }
  // Touching a region without being contained in one would serve the device's bytes out of the page
  // underneath. A fetch is exempt since it always asks for 15 bytes regardless of length.
  if (kind != MemoryAccessKind::instruction_fetch && mmio_overlaps(address, size)) {
    return false;
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
    // Checked up front so a refusal partway leaves dst untouched, which is what the hooked path
    // below already gets for free by staging through temp.
    return span_permits(address, size, kind) && copy_from_pages(out);
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
    if (on_read == nullptr) return false;
    ++device_dispatch_count_;
    return on_read(address - region_base, dst, size);
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
  if (auto fn = passthrough_read_) {
    ++device_dispatch_count_;
    return fn(page_base, dst, kPageSize);
  }
  if ((page_base % kPageSize) != 0) {
    return false;
  }
  // A device anywhere in this page means the page bytes are not what a fetch here returns. Decline
  // and let the caller fetch per instruction through read(), which consults the region.
  if (mmio_overlaps(page_base, kPageSize)) {
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
    // read() rejects a wrapping range before passthrough_read_ ever sees it, and a passthrough is
    // the embedder's whole memory implementation, so the write side must not hand it a pair the read
    // side is guaranteed never to see.
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
    ++device_dispatch_count_;
    if (!fn(address, src, size)) {
      return false;
    }
    // A passthrough cannot say whether what it wrote was executable, so every write counts as one
    // that might have rewritten code. This counter is what page_code_epoch() reports under one.
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
  // After access_allowed, not at the top: a range-scoped write hook still has to get the chance to
  // veto a wrapping access. The fast path above cannot be reached by one anyway.
  if (access_wraps(address, size)) {
    return false;
  }
  if (const auto* mmio = find_mmio_region(address, size)) {
    // Copied out of mmio_regions_ before the call for the reason read()'s hooked path
    // spells out: a device callback is free to reconfigure its own region, which would
    // otherwise destroy or relocate the std::function while its own frame is still live.
    auto on_write = mmio->on_write;
    const auto region_base = mmio->base;
    if (on_write == nullptr) return false;
    ++device_dispatch_count_;
    return on_write(address - region_base, src, size);
  }
  // See read(): an access only partly landing on a device must not be served by the page under it.
  if (mmio_overlaps(address, size)) {
    return false;
  }

  // Every page first, then any bytes: see span_permits.
  if (!span_permits(address, size, kind)) {
    return false;
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
    if (on_write == nullptr) return false;
    ++device_dispatch_count_;
    return on_write(address - region_base, src, size);
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
  // Both halves matter: a write-only passthrough left this false, so reads warmed jit_tlb slots and
  // compiled writes then stored into PageEntry::data without ever calling passthrough_write_.
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
  invalidate_code_epochs(base, size);
  refresh_jit_fast_path_blocked();
  return id;
}

bool Memory::unmap_mmio(HookId id) {
  for (auto it = mmio_regions_.begin(); it != mmio_regions_.end(); ++it) {
    if (it->id == id) {
      const auto gone_base = it->base;
      const auto gone_size = it->size;
      mmio_regions_.erase(it);
      invalidate_code_epochs(gone_base, gone_size);
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
  invalidate_all_code_epochs();
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
  invalidate_tlb();
  clear_jit_tlb();  // pages_.clear() below frees every PageEntry jit_tlb could be pointing at
  pages_.clear();
  for (const auto& snapshot : pages) {
    pages_.emplace(snapshot.page_index,
                   PageEntry{.permissions = snapshot.permissions,
                             .code_epoch = ++code_epoch_,
                             .data = snapshot.data});
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
  invalidate_all_code_epochs();
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

bool Memory::is_mmio_address(std::uint64_t address) const {
  return find_mmio_region(address, 1) != nullptr;
}

const Memory::MmioRegion* Memory::find_mmio_region(std::uint64_t address, std::size_t size) const {
  // Zero bytes has no last byte, so containment below is satisfied by an address one past the end of
  // a region and the callback would be handed an offset of exactly region.size -- outside the
  // [0, size) range the framework contracts. mmio_overlaps already declines this shape.
  if (mmio_regions_.empty() || size == 0) {
    return nullptr;
  }
  const auto end = address + static_cast<std::uint64_t>(size);
  // A guest address near the top of the space wraps end back past zero, and the wrapped value can
  // pass both range checks while address is nowhere near any region. The resulting offset is a huge
  // guest-controlled index handed to a callback that trusts it to be bounded.
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

  // Save and restore, not assign: a callback touching guest memory re-enters here, and a nested
  // call clearing the flag would leave the outer loop walking access_hooks_ with deferral off, so a
  // later remove_access_hook would erase mid-iteration.
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
    // Only the erase is deferred, not the removal: remove_access_hook already returned true, so the
    // caller may have destroyed whatever the hook captured.
    if (!pending_removed_access_hooks_.empty() &&
        std::find(pending_removed_access_hooks_.begin(), pending_removed_access_hooks_.end(), hook.id) !=
            pending_removed_access_hooks_.end()) {
      continue;
    }
    if (hook.range.has_value()) {
      const auto range_base = hook.range->base;
      const auto range_end = range_end_saturating(range_base, hook.range->size);
      const auto access_end = event.address + event.size;
      // A guest-controlled address can wrap access_end past zero, which a non-wrapping test reads as
      // no overlap. That skips a write hook before the page write, so treat a wrap as overlapping.
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

#include "seven/memory.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace seven {

Memory::PageEntry* Memory::lookup_page(std::uint64_t page_index) const noexcept {
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
}

void Memory::clear_passthrough() {
  passthrough_read_  = nullptr;
  passthrough_write_ = nullptr;
}

void Memory::map(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions) {
  ++page_epoch_;
  ++code_epoch_;
  // map() may insert new entries; std::unordered_map insertion can rehash and
  // invalidate iterators, but does not invalidate references / pointers to
  // existing elements. However, we still need to invalidate the TLB so that
  // negative cache entries (slots holding nullptr for previously-unmapped
  // pages) are refreshed.
  invalidate_tlb();
  const auto first_page = base / kPageSize;
  const auto last_page = (base + size + kPageSize - 1) / kPageSize;
  for (auto page = first_page; page < last_page; ++page) {
    auto [it, inserted] = pages_.try_emplace(page);
    (void)inserted;
    it->second.permissions = permissions;
  }
}

void Memory::unmap(std::uint64_t base, std::size_t size) {
  ++page_epoch_;
  ++code_epoch_;
  invalidate_tlb();  // erase invalidates references; flush TLB
  const auto first_page = base / kPageSize;
  const auto last_page = (base + size + kPageSize - 1) / kPageSize;
  for (auto page = first_page; page < last_page; ++page) {
    pages_.erase(page);
  }
}

void Memory::reprotect(std::uint64_t base, std::size_t size, MemoryPermissionMask permissions) {
  ++page_epoch_;
  ++code_epoch_;
  // Reprotect does not erase entries, so cached PageEntry* pointers stay
  // valid. Permissions are read through the pointer, so we don't have to
  // invalidate the TLB.
  const auto first_page = base / kPageSize;
  const auto last_page = (base + size + kPageSize - 1) / kPageSize;
  for (auto page = first_page; page < last_page; ++page) {
    auto it = pages_.find(page);
    if (it != pages_.end()) {
      it->second.permissions = permissions;
    }
  }
}

bool Memory::is_mapped(std::uint64_t address, std::size_t size) const {
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
  if (passthrough_read_) return passthrough_read_(address, dst, size);
  // Fast path: most reads in real workloads are entirely within a single page
  // and target a non-MMIO address with no access hooks installed. Inline that
  // case to skip every dynamic check besides the TLB lookup itself.
  if (!has_any_access_hooks_ && mmio_regions_.empty()) [[likely]] {
    const auto first_page = address / kPageSize;
    const auto first_offset = address % kPageSize;
    if (first_offset + size <= kPageSize) [[likely]] {
      auto* entry = lookup_page(first_page);
      if (entry == nullptr || !has_permission(entry->permissions, kind)) {
        return false;
      }
      std::memcpy(dst, entry->data.data() + first_offset, size);
      return true;
    }
  }

  if (const auto* mmio = find_mmio_region(address, size)) {
    if (!access_allowed(MemoryAccessEvent{kind, address, size, nullptr, 0})) {
      return false;
    }
    return mmio->on_read != nullptr ? mmio->on_read(address - mmio->base, dst, size) : false;
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
  if (const auto* mmio = find_mmio_region(address, size)) {
    return mmio->on_read != nullptr ? mmio->on_read(address - mmio->base, dst, size) : false;
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
  if (passthrough_read_) return passthrough_read_(page_base, dst, kPageSize);
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
  if (passthrough_write_) return passthrough_write_(address, src, size);
  // Fast path: no hooks, no MMIO, single-page access. Bypasses access_allowed
  // and all multi-page bookkeeping. We still maintain code_epoch for write
  // through executable pages so the decode cache stays correct.
  if (!has_any_access_hooks_ && mmio_regions_.empty()) [[likely]] {
    const auto first_page = address / kPageSize;
    const auto first_offset = address % kPageSize;
    if (first_offset + size <= kPageSize) [[likely]] {
      auto* entry = lookup_page(first_page);
      if (entry == nullptr || !has_permission(entry->permissions, kind)) {
        return false;
      }
      std::memcpy(entry->data.data() + first_offset, src, size);
      if ((entry->permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0) {
        ++code_epoch_;
      }
      return true;
    }
  }

  if (!access_allowed(MemoryAccessEvent{kind, address, size, src, size})) {
    return false;
  }
  if (const auto* mmio = find_mmio_region(address, size)) {
    return mmio->on_write != nullptr ? mmio->on_write(address - mmio->base, src, size) : false;
  }

  const auto* in = static_cast<const std::byte*>(src);
  std::size_t remaining = size;
  std::uint64_t current = address;
  bool code_changed = false;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    auto* entry = lookup_page(page_index);
    if (entry == nullptr || !has_permission(entry->permissions, kind)) {
      return false;
    }
    code_changed |= (entry->permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0;
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    std::memcpy(entry->data.data() + page_offset, in, chunk);
    in += chunk;
    current += chunk;
    remaining -= chunk;
  }
  if (code_changed) {
    ++code_epoch_;
  }
  return true;
}

bool Memory::write_unchecked(std::uint64_t address, const void* src, std::size_t size) {
  if (const auto* mmio = find_mmio_region(address, size)) {
    return mmio->on_write != nullptr ? mmio->on_write(address - mmio->base, src, size) : false;
  }

  const auto* in = static_cast<const std::byte*>(src);
  std::size_t remaining = size;
  std::uint64_t current = address;
  bool code_changed = false;
  while (remaining != 0) {
    const auto page_index = current / kPageSize;
    const auto page_offset = current % kPageSize;
    auto* entry = lookup_page(page_index);
    if (entry == nullptr) {
      return false;
    }
    code_changed |= (entry->permissions & static_cast<MemoryPermissionMask>(MemoryPermission::execute)) != 0;
    const auto chunk = std::min<std::size_t>(remaining, kPageSize - page_offset);
    std::memcpy(entry->data.data() + page_offset, in, chunk);
    in += chunk;
    current += chunk;
    remaining -= chunk;
  }
  if (code_changed) {
    ++code_epoch_;
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
}

Memory::HookId Memory::map_mmio(std::uint64_t base, std::size_t size, MmioReadCallback on_read, MmioWriteCallback on_write) {
  const auto id = next_hook_id_++;
  mmio_regions_.push_back(MmioRegion{id, base, size, std::move(on_read), std::move(on_write)});
  if (mmio_regions_.size() == 1) {
    mmio_min_base_ = base;
    mmio_max_end_ = base + size;
  } else {
    mmio_min_base_ = std::min(mmio_min_base_, base);
    mmio_max_end_ = std::max(mmio_max_end_, base + size);
  }
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
        mmio_max_end_ = mmio_regions_.front().base + mmio_regions_.front().size;
        for (const auto& region : mmio_regions_) {
          mmio_min_base_ = std::min(mmio_min_base_, region.base);
          mmio_max_end_ = std::max(mmio_max_end_, region.base + region.size);
        }
      }
      return true;
    }
  }
  return false;
}

void Memory::clear_mmio_regions() {
  mmio_regions_.clear();
  mmio_min_base_ = ~0ull;
  mmio_max_end_ = 0;
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
  ++code_epoch_;
  invalidate_tlb();
  pages_.clear();
  for (const auto& snapshot : pages) {
    pages_.emplace(snapshot.page_index, PageEntry{snapshot.data, snapshot.permissions});
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
      mmio_max_end_ = region.base + region.size;
    } else {
      mmio_min_base_ = std::min(mmio_min_base_, region.base);
      mmio_max_end_ = std::max(mmio_max_end_, region.base + region.size);
    }
  }
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
  const auto end = address + size;
  if (address < mmio_min_base_ || end > mmio_max_end_) {
    return nullptr;
  }
  for (const auto& region : mmio_regions_) {
    if (address >= region.base && end <= (region.base + region.size)) {
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

  self.dispatching_access_hooks_ = true;
  for (const auto& hook : access_hooks_) {
    if ((hook.kinds & bit(event.kind)) == 0) {
      continue;
    }
    if (hook.range.has_value()) {
      const auto range_base = hook.range->base;
      const auto range_end = range_base + hook.range->size;
      const auto access_end = event.address + event.size;
      if (event.address >= range_end || access_end <= range_base) {
        continue;
      }
    }
    if (!hook.callback(event)) {
      self.dispatching_access_hooks_ = false;
      self.apply_pending_access_hook_ops();
      return false;
    }
  }
  self.dispatching_access_hooks_ = false;
  self.apply_pending_access_hook_ops();
  return true;
}

}  // namespace seven

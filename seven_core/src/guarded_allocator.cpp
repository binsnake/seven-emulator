#include "seven/guarded_allocator.hpp"

#if defined(SEVEN_GUARDED_PAGES)

#include <mutex>
#include <new>
#include <unordered_set>

#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h>

namespace seven::detail {

std::size_t guarded_page_size() noexcept {
  static const std::size_t cached = [] {
    SYSTEM_INFO info{};
    ::GetSystemInfo(&info);
    return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize) : std::size_t{0x1000};
  }();
  return cached;
}

namespace {

// Which pages are guards, kept exact rather than approximated by a range since address space gets
// reused. Both are leaked on purpose: a fault during static destruction reaching a destroyed mutex
// would turn a useful finding into a second crash.
std::mutex& registry_lock() {
  static auto* lock = new std::mutex();
  return *lock;
}

std::unordered_set<std::uintptr_t>& guard_pages() {
  static auto* pages = new std::unordered_set<std::uintptr_t>();
  return *pages;
}

}  // namespace

void* guarded_reserve(std::size_t committed_bytes) {
  const std::size_t page = guarded_page_size();
  auto* base = static_cast<std::byte*>(
      ::VirtualAlloc(nullptr, committed_bytes + page, MEM_RESERVE, PAGE_NOACCESS));
  if (base == nullptr) throw std::bad_alloc();
  // The trailing page is left reserved rather than committed, which faults on any access without
  // needing a separate PAGE_NOACCESS commit.
  if (::VirtualAlloc(base, committed_bytes, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
    ::VirtualFree(base, 0, MEM_RELEASE);
    throw std::bad_alloc();
  }
  {
    const std::lock_guard<std::mutex> held(registry_lock());
    guard_pages().insert(reinterpret_cast<std::uintptr_t>(base + committed_bytes));
  }
  return base;
}

void guarded_release(void* base, std::size_t committed_bytes) noexcept {
  {
    const std::lock_guard<std::mutex> held(registry_lock());
    guard_pages().erase(reinterpret_cast<std::uintptr_t>(base) + committed_bytes);
  }
  ::VirtualFree(base, 0, MEM_RELEASE);
}

}  // namespace seven::detail

namespace seven {

GuardHit classify_faulting_address(const void* address) noexcept {
  const auto addr = reinterpret_cast<std::uintptr_t>(address);
  const auto base = addr & ~static_cast<std::uintptr_t>(detail::guarded_page_size() - 1);
  const std::lock_guard<std::mutex> held(detail::registry_lock());
  if (detail::guard_pages().count(base) == 0) return {};
  return {true, base, static_cast<std::size_t>(addr - base)};
}

}  // namespace seven

#endif

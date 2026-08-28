#include "seven/guarded_allocator.hpp"

#if defined(SEVEN_GUARDED_PAGES)

#include <new>

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
  return base;
}

void guarded_release(void* base) noexcept { ::VirtualFree(base, 0, MEM_RELEASE); }

}  // namespace seven::detail

#endif

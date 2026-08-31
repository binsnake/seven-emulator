#pragma once

// A hardware oracle for out-of-bounds accesses to guest pages, which the JIT reaches from code it
// emitted itself and no sanitizer can instrument. Each map node is reserved a page larger than it
// needs and positioned so its final byte sits against that uncommitted page; PageEntry keeps its
// bytes last, so one past the end of a guest page is the guard. Off by default, where this is plain
// std::allocator; the guarded build costs a page of address space per guest page.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace seven {

#if defined(SEVEN_GUARDED_PAGES)

// What a faulting address turned out to be. A fault handler needs to tell "the guest ran off the
// end of a page", which is a finding, apart from an ordinary crash, which is a different one.
struct GuardHit {
  bool is_guard_page = false;
  std::uintptr_t guard_base = 0;
  // The payload ends exactly where the guard starts, so this doubles as how far past the end of
  // the guest page the access reached.
  std::size_t bytes_past_payload = 0;
};

namespace detail {
// Defined in guarded_allocator.cpp. The reservation calls live out of line so that windows.h stays
// out of memory.hpp, which nearly every translation unit here includes -- pulling it in defines
// min/max as macros and float80.hpp stops compiling.
[[nodiscard]] std::size_t guarded_page_size() noexcept;
[[nodiscard]] void* guarded_reserve(std::size_t committed_bytes);
void guarded_release(void* base, std::size_t committed_bytes) noexcept;
}  // namespace detail

// Safe to call from an exception handler: it takes a lock and reads a hash set, and never
// allocates. Only ever contends with map/unmap, never with the faulting access itself.
[[nodiscard]] GuardHit classify_faulting_address(const void* address) noexcept;

template <class T>
struct GuardedPageAllocator {
  using value_type = T;

  GuardedPageAllocator() noexcept = default;
  template <class U>
  GuardedPageAllocator(const GuardedPageAllocator<U>&) noexcept {}

  // Rounded up to T's alignment so the returned pointer stays aligned once it is measured back from
  // a page boundary. The slack that rounding leaves sits between the object and the guard, so it is
  // worth keeping at zero: for the page map's node type it already is.
  [[nodiscard]] static std::size_t footprint(std::size_t n) noexcept {
    const std::size_t bytes = n * sizeof(T);
    const std::size_t align = alignof(T);
    return (bytes + align - 1) / align * align;
  }

  [[nodiscard]] static std::size_t committed_for(std::size_t n) noexcept {
    const std::size_t page = detail::guarded_page_size();
    return (footprint(n) + page - 1) / page * page;
  }

  [[nodiscard]] T* allocate(std::size_t n) {
    const std::size_t committed = committed_for(n);
    auto* base = static_cast<std::byte*>(detail::guarded_reserve(committed));
    return reinterpret_cast<T*>(base + committed - footprint(n));
  }

  void deallocate(T* p, std::size_t n) noexcept {
    if (p == nullptr) return;
    // The allocation ends on a page boundary by construction, so the base is recoverable from the
    // pointer and the size alone and nothing has to be stored alongside it.
    const std::size_t committed = committed_for(n);
    auto* end = reinterpret_cast<std::byte*>(p) + footprint(n);
    detail::guarded_release(end - committed, committed);
  }

  template <class U>
  bool operator==(const GuardedPageAllocator<U>&) const noexcept {
    return true;
  }
  template <class U>
  bool operator!=(const GuardedPageAllocator<U>&) const noexcept {
    return false;
  }
};

template <class T>
using PageMapAllocator = GuardedPageAllocator<T>;

inline constexpr bool kGuardedPagesEnabled = true;

#else

template <class T>
using PageMapAllocator = std::allocator<T>;

inline constexpr bool kGuardedPagesEnabled = false;

#endif

}  // namespace seven

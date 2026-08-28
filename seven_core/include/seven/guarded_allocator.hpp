#pragma once

// A hardware oracle for out-of-bounds accesses to guest page bytes.
//
// AddressSanitizer cannot see the accesses that matter most here: it instruments compiled C++, and
// the JIT's fast path reaches guest memory from code it emitted itself, which no sanitizer touches.
// A page's bytes live inside a PageEntry inside an unordered_map node, so an emitted access that
// runs off the end of a page lands in whatever the allocator put next and reads or writes it
// silently.
//
// This allocator puts an unmapped page there instead. It reserves the node's pages plus one more,
// commits everything but that last page, and hands back a pointer positioned so the object's final
// byte sits directly against it. PageEntry keeps its byte array as its last member (there is a
// static_assert in memory.hpp holding that), so "one past the end of the page bytes" and "first
// byte of the guard page" are the same address, and the access faults in hardware no matter who
// issued it.
//
// Off by default: with SEVEN_GUARDED_PAGES undefined this is std::allocator and the page map is
// byte-for-byte the container it was before. The guarded build costs a page of address space per
// mapped guest page and is meant for tests and fuzzing, not production.

#include <cstddef>
#include <memory>

namespace seven {

#if defined(SEVEN_GUARDED_PAGES)

namespace detail {
// Defined in guarded_allocator.cpp. The reservation calls live out of line so that windows.h stays
// out of memory.hpp, which nearly every translation unit here includes -- pulling it in defines
// min/max as macros and float80.hpp stops compiling.
[[nodiscard]] std::size_t guarded_page_size() noexcept;
[[nodiscard]] void* guarded_reserve(std::size_t committed_bytes);
void guarded_release(void* base) noexcept;
}  // namespace detail

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
    auto* end = reinterpret_cast<std::byte*>(p) + footprint(n);
    detail::guarded_release(end - committed_for(n));
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

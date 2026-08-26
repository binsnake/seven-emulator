#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "seven/executor.hpp"
#include "seven/memory.hpp"

// Regression test for a real integer-overflow bug in Memory::find_mmio_region: address + size can
// wrap past the top of the 64-bit address space for a guest-chosen address near ~0ull, and the
// wrapped (small) end used to be able to pass both the mmio_min_base_/mmio_max_end_ pre-check and
// the per-region range check despite the real address being nowhere near any registered region.
// A false match there would have handed the MMIO callback an "offset" of address - region.base --
// itself still a huge, effectively guest-controlled value -- which any callback that trusts the
// framework's contract that offset always lies within [0, region.size) would use as an unchecked
// index, a genuine guest-to-host out-of-bounds read/write.

TEST(KuberaMemory, WraparoundAccessNeverFalseMatchesMmioRegion) {
  seven::Memory memory{};

  bool callback_invoked = false;
  std::uint64_t seen_offset = 0;
  const auto id = memory.map_mmio(
      0x1000, 0x1000,
      [&](std::uint64_t offset, void* dst, std::size_t size) {
        callback_invoked = true;
        seen_offset = offset;
        std::memset(dst, 0, size);
        return true;
      },
      [&](std::uint64_t offset, const void*, std::size_t) {
        callback_invoked = true;
        seen_offset = offset;
        return true;
      });
  ASSERT_NE(id, 0u);

  // address is 8 bytes from wrapping past ~0ull; size=16 makes address+size wrap to 0x8, which
  // used to fall inside the registered region's [0, 0x1000) range and produce a false match.
  constexpr std::uint64_t kWraparoundAddress = 0xFFFF'FFFF'FFFF'FFF8ull;
  std::uint8_t buffer[16] = {};

  const bool read_ok = memory.read(kWraparoundAddress, buffer, sizeof(buffer));
  EXPECT_FALSE(read_ok) << "a wrapping access must never be treated as landing inside a real MMIO region";
  EXPECT_FALSE(callback_invoked);

  const bool write_ok = memory.write(kWraparoundAddress, buffer, sizeof(buffer));
  EXPECT_FALSE(write_ok);
  EXPECT_FALSE(callback_invoked);
  EXPECT_EQ(seen_offset, 0u);
}

TEST(KuberaMemory, NonWrappingAccessStillReachesMmioCallback) {
  // Sanity check alongside the wraparound test above: the overflow guard must not have made
  // ordinary in-range MMIO access unreachable.
  seven::Memory memory{};

  bool callback_invoked = false;
  std::uint64_t seen_offset = ~0ull;
  (void)memory.map_mmio(
      0x2000, 0x1000,
      [&](std::uint64_t offset, void* dst, std::size_t size) {
        callback_invoked = true;
        seen_offset = offset;
        std::memset(dst, 0x42, size);
        return true;
      },
      [](std::uint64_t, const void*, std::size_t) { return true; });

  std::uint8_t buffer[4] = {};
  ASSERT_TRUE(memory.read(0x2010, buffer, sizeof(buffer)));
  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(seen_offset, 0x10u);
  EXPECT_EQ(buffer[0], 0x42u);
}

// Same overflow shape as find_mmio_region, in Memory::access_allowed()'s range-scoped hook
// overlap check: it computed access_end = event.address + event.size with no wraparound guard.
// A guest picking an address near ~0ull can wrap access_end back down past zero, which a plain
// non-wrapping interval test then wrongly reads as "no overlap" against a hook's registered
// range -- even though the access's real (wrapping) span may well touch it. For a write hook
// specifically this matters more than a missed notification: access_allowed() runs BEFORE the
// underlying page write in Memory::write(), so a hook skipped this way never gets the chance to
// veto (its callback returning false is what blocks the write) -- a real bypass of whatever the
// hook enforces, reachable purely by a guest choosing a wraparound address.
TEST(KuberaMemory, WraparoundWriteStillInvokesRangeScopedAccessHook) {
  seven::Memory memory{};

  bool hook_invoked = false;
  const auto id = memory.add_access_hook(
      [&](const seven::MemoryAccessEvent&) {
        hook_invoked = true;
        return true;
      },
      seven::MemoryHookRange{.base = 0x1000, .size = 0x1000},
      seven::bit(seven::MemoryAccessKind::data_write));
  ASSERT_NE(id, 0u);

  // Same wraparound shape as the MMIO test above: address+size wraps to 0x8, well inside the
  // hooked range numerically, even though the pre-fix skip logic would have missed it because
  // event.address itself is nowhere near range_end under a naive (non-wrapping) comparison.
  constexpr std::uint64_t kWraparoundAddress = 0xFFFF'FFFF'FFFF'FFF8ull;
  std::uint8_t buffer[16] = {};

  // The underlying page write still fails -- nothing is mapped at the top of the address space --
  // but the hook must be given the chance to see (and veto) this access before that ever runs.
  (void)memory.write(kWraparoundAddress, buffer, sizeof(buffer));
  EXPECT_TRUE(hook_invoked);
}

// map()/unmap()/reprotect() all derived their exclusive end page as
// `(base + size + kPageSize - 1) / kPageSize`. For any range reaching the very top of the address
// space that addition wraps, the end page comes out as 0, and the `page < last_page` loop body
// never executes at all -- so the call silently did nothing. unmap() leaving the page mapped is a
// leak, but reprotect() is the sharper one: a host revoking write access on that page is told
// nothing failed while the guest keeps writing through it.
TEST(KuberaMemory, TopOfAddressSpacePageIsActuallyMappedUnmappedAndReprotected) {
  constexpr std::uint64_t kTopPage = 0xFFFF'FFFF'FFFF'F000ull;
  constexpr auto kWrite = static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write);
  constexpr auto kRead = static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read);
  seven::Memory memory{};

  memory.map(kTopPage, 0x1000, seven::kMemoryPermissionAll);
  ASSERT_TRUE(memory.is_mapped(kTopPage, 0x1000));
  ASSERT_TRUE(memory.has_permissions(kTopPage, 0x1000, kWrite));

  // The revoke must actually land: read-only afterwards, not still writable.
  memory.reprotect(kTopPage, 0x1000, kRead);
  EXPECT_FALSE(memory.has_permissions(kTopPage, 0x1000, kWrite));

  memory.unmap(kTopPage, 0x1000);
  EXPECT_FALSE(memory.is_mapped(kTopPage, 0x1000));
}

// Memory::read/write took a single-page fast path guarded by `first_offset + size <= kPageSize`.
// size is a full-width size_t, so that sum wraps: an access based in page 0 whose size covers the
// rest of the address space produced a sum of exactly 2^64, which reads as 0 and sails through the
// guard straight into a memcpy of ~2^64 bytes out of (or into) a 4096-byte page buffer.
TEST(KuberaMemory, HugeSizeCannotWrapPastTheSinglePageFastPath) {
  seven::Memory memory{};
  memory.map(0, seven::Memory::kPageSize);

  // Last byte of page 0, sized so that kBase + kSize lands exactly on 2^64. The access itself does
  // not wrap, so a top-of-address-space check alone would let it past, but first_offset + kSize
  // does. Basing it on the final byte keeps the correct slow path down to a single byte before it
  // runs out of mapped pages, so the buffer below is all the destination this can legitimately use.
  constexpr std::uint64_t kBase = seven::Memory::kPageSize - 1;
  constexpr std::size_t kSize = static_cast<std::size_t>(~std::uint64_t{0} - kBase + 1);

  std::uint8_t scratch[64]{};
  EXPECT_FALSE(memory.read(kBase, scratch, kSize));
  EXPECT_FALSE(memory.write(kBase, scratch, kSize));
}

// The page-walking loops advance a plain uint64 cursor, so an access starting near the top of the
// address space used to wrap to zero and carry on reading (or writing) page 0. Hardware faults
// there instead of wrapping, and quietly splicing in unrelated memory is worse than either.
TEST(KuberaMemory, AccessRunningOffTheTopOfMemoryFailsInsteadOfWrappingToPageZero) {
  seven::Memory memory{};
  constexpr std::uint64_t kTopPage = ~std::uint64_t{0} - seven::Memory::kPageSize + 1;
  memory.map(kTopPage, seven::Memory::kPageSize);
  memory.map(0, seven::Memory::kPageSize);

  const std::vector<std::uint8_t> marker(seven::Memory::kPageSize, 0xAB);
  ASSERT_TRUE(memory.write(0, marker.data(), marker.size()));

  constexpr std::uint64_t kStart = ~std::uint64_t{0} - 0xFF;  // last 256 bytes
  std::array<std::uint8_t, 0x200> buffer{};
  EXPECT_FALSE(memory.read(kStart, buffer.data(), buffer.size()));
  EXPECT_FALSE(memory.write(kStart, buffer.data(), buffer.size()));
  EXPECT_FALSE(memory.is_mapped(kStart, buffer.size()));

  // Page 0 must be untouched by the rejected write.
  std::array<std::uint8_t, 8> readback{};
  ASSERT_TRUE(memory.read(0, readback.data(), readback.size()));
  EXPECT_EQ(readback[0], 0xAB);

  // An access ending exactly on the last byte is legal and must still work.
  EXPECT_TRUE(memory.read(kStart, buffer.data(), 0x100));
}

// page_range_end rounds its end up, so with an unaligned base a size of 0 used to resolve to one
// page. reprotect is the sharp edge: an empty request could revoke (or grant) permissions on a
// page the caller never named.
TEST(KuberaMemory, ZeroSizedRangeNeverTouchesAPage) {
  seven::Memory memory{};
  memory.map(0x1000, seven::Memory::kPageSize, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read) |
                                                   static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));

  const std::uint32_t value = 0x11223344;
  ASSERT_TRUE(memory.write(0x1000, &value, sizeof(value)));

  memory.reprotect(0x1800, 0, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read));
  std::uint32_t probe = 0;
  EXPECT_TRUE(memory.write(0x1000, &value, sizeof(value))) << "a zero-sized reprotect must not revoke write access";
  EXPECT_TRUE(memory.read(0x1000, &probe, sizeof(probe)));
  EXPECT_EQ(probe, value);

  memory.unmap(0x1800, 0);
  EXPECT_TRUE(memory.is_mapped(0x1000, 1)) << "a zero-sized unmap must not erase a page";
}

// Every other mmio mutator refreshes jit_fast_path_blocked. restore_mmio_regions did not, so a
// snapshot restore left compiled code believing the raw-page fast path was still safe while mmio
// regions were live again, and the callbacks never fired for anything also backed by a real page.
TEST(KuberaMemory, RestoringMmioRegionsBlocksTheJitFastPathAgain) {
  seven::Memory memory{};
  ASSERT_FALSE(memory.jit_fast_path_blocked);

  const auto id = memory.map_mmio(
      0x4000, seven::Memory::kPageSize,
      [](std::uint64_t, void*, std::size_t) { return true; },
      [](std::uint64_t, const void*, std::size_t) { return true; });
  ASSERT_NE(id, 0u);
  ASSERT_TRUE(memory.jit_fast_path_blocked);

  const auto regions = memory.snapshot_mmio_regions();
  ASSERT_EQ(regions.size(), 1u);

  seven::Memory restored{};
  ASSERT_FALSE(restored.jit_fast_path_blocked);
  restored.restore_mmio_regions(regions, [](const seven::Memory::MmioRegionSnapshot&) {
    return std::make_optional(std::make_pair(
        seven::Memory::MmioReadCallback{[](std::uint64_t, void*, std::size_t) { return true; }},
        seven::Memory::MmioWriteCallback{[](std::uint64_t, const void*, std::size_t) { return true; }}));
  });
  EXPECT_TRUE(restored.jit_fast_path_blocked);
}

// access_allowed assigned dispatching_access_hooks_ = true/false rather than saving and restoring
// it, and flushed the pending add/remove queue at every nesting level. A hook callback that reads
// guest memory re-enters access_allowed, and the nested call cleared the flag on its way out --
// so a later hook in the OUTER loop calling remove_access_hook took the immediate branch and
// erased from access_hooks_ while the outer range-for was still walking it, leaving the loop
// calling through freed std::function storage.
TEST(KuberaMemory, NestedAccessHookDispatchDoesNotInvalidateTheOuterLoop) {
  seven::Memory memory{};
  memory.map(0x1000, seven::Memory::kPageSize);

  seven::Memory::HookId to_remove = 0;
  int reentrant_calls = 0;
  int remover_calls = 0;
  int tail_calls = 0;

  // First hook re-enters Memory, which re-enters access_allowed and used to clear the guard.
  const auto reentrant = memory.add_access_hook([&](const seven::MemoryAccessEvent& event) {
    if (event.kind == seven::MemoryAccessKind::data_read) {
      return true;  // this is our own nested read, let it through without recursing further
    }
    ++reentrant_calls;
    std::uint32_t sink = 0;
    (void)memory.read(0x1000, &sink, sizeof(sink));
    return true;
  });
  ASSERT_NE(reentrant, 0u);

  // Second hook removes a hook mid-dispatch. With the guard wrongly cleared this erased in place.
  const auto remover = memory.add_access_hook([&](const seven::MemoryAccessEvent& event) {
    if (event.kind == seven::MemoryAccessKind::data_read) {
      return true;
    }
    ++remover_calls;
    EXPECT_TRUE(memory.remove_access_hook(to_remove));
    return true;
  });
  ASSERT_NE(remover, 0u);

  // Third hook is what the outer loop walks into after the erase. It has to still be callable.
  to_remove = memory.add_access_hook([&](const seven::MemoryAccessEvent& event) {
    if (event.kind == seven::MemoryAccessKind::data_read) {
      return true;
    }
    ++tail_calls;
    return true;
  });
  ASSERT_NE(to_remove, 0u);

  const std::uint32_t value = 0xA5A5A5A5;
  EXPECT_TRUE(memory.write(0x1000, &value, sizeof(value)));

  EXPECT_EQ(reentrant_calls, 1);
  EXPECT_EQ(remover_calls, 1);
  EXPECT_EQ(tail_calls, 1) << "the removal must be deferred until dispatch unwinds";

  // The deferred removal must actually have landed once dispatch finished.
  EXPECT_FALSE(memory.remove_access_hook(to_remove)) << "hook should already be gone";
  EXPECT_TRUE(memory.remove_access_hook(reentrant));
  EXPECT_TRUE(memory.remove_access_hook(remover));
}

// An instruction fetch is only allowed to fault on bytes the instruction actually needs. seven used
// to ask for a full 15 bytes every time, so a one-byte instruction in the last stretch of a page
// faulted whenever the following page was unmapped or non-executable -- the ordinary layout of the
// last instruction before a guard page. It also gave the guest a way to map out what the host had
// placed around it without ever issuing an access to those addresses.

TEST(KuberaMemory, ShortInstructionAtAPageBoundaryDoesNotFetchIntoTheNextPage) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  const std::uint8_t nop = 0x90;
  ASSERT_TRUE(memory.write(0x1FFF, &nop, 1));

  state.rip = 0x1FFF;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, 0x2000u);
}

TEST(KuberaMemory, ShortInstructionAtAPageBoundaryIgnoresTheNextPagesPermissions) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  memory.map(0x2000, 0x1000, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read) |
                                 static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
  const std::uint8_t nop = 0x90;
  ASSERT_TRUE(memory.write(0x1FFF, &nop, 1));

  state.rip = 0x1FFF;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.rip, 0x2000u);
}

TEST(KuberaMemory, InstructionThatGenuinelyStraddlesAnUnmappedPageStillFaults) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  // mov eax, imm32 is five bytes; only three of them fit before the boundary.
  const std::uint8_t head[] = {0xB8, 0x00, 0x00};
  ASSERT_TRUE(memory.write(0x1FFD, head, sizeof(head)));

  state.rip = 0x1FFD;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.rip, 0x1FFDu);
  ASSERT_TRUE(result.exception.has_value());
  EXPECT_EQ(result.exception->address, 0x2000u) << "the fault belongs to the page it ran into";
}

// CpuState::msr is an unordered_map keyed on the 32-bit MSR index, and WRMSR used to insert
// unconditionally. A CPL0 guest walking ECX through a wrmsr loop therefore made the host allocate
// one node per index -- around 200 GB before the counter wraps, with no cooperative-cancellation
// path able to interrupt it. Real hardware implements a few hundred MSRs and faults on the rest.

TEST(KuberaMemory, WrmsrCannotGrowTheMsrMapWithoutBound) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  const std::uint8_t wrmsr[] = {0x0F, 0x30};
  ASSERT_TRUE(memory.write(0x1000, wrmsr, sizeof(wrmsr)));

  const auto initial = state.msr.size();
  bool faulted = false;
  for (std::uint32_t index = 0; index < 100000u && !faulted; ++index) {
    state.rip = 0x1000;
    state.gpr[1] = index;  // ECX
    state.gpr[0] = index;  // EAX
    state.gpr[2] = 0;      // EDX
    const auto result = executor.step(state, memory);
    faulted = result.reason == seven::StopReason::general_protection;
  }

  EXPECT_TRUE(faulted) << "a new index past the ceiling has to fault, not allocate";
  EXPECT_LT(state.msr.size(), initial + 100000u);

  // An index that is already present must keep working after the ceiling is reached.
  state.rip = 0x1000;
  state.gpr[1] = 0xC0000080u;  // EFER, mapped from the start
  state.gpr[0] = 0x0D01;
  state.gpr[2] = 0;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.msr.at(0xC0000080u), 0x0D01u);
}

// Memory is copyable, and a copy deep-copies pages_ into fresh PageEntry objects. Its two page
// caches came across holding raw pointers into the source object's pages, so the copy silently read
// and wrote through to the original -- and would have kept doing so after the original was gone.

TEST(KuberaMemory, CopyingDoesNotInheritPointersIntoTheOriginalsPages) {
  seven::Memory original{};
  original.map(0x1000, 0x1000);
  const std::uint32_t before = 0xAAAAAAAA;
  ASSERT_TRUE(original.write(0x1000, &before, sizeof(before)));

  seven::Memory copy = original;
  for (const auto& slot : copy.jit_tlb) {
    EXPECT_EQ(slot.host_data, nullptr) << "a copy must not start out pointing into the original";
  }

  const std::uint32_t after = 0xBBBBBBBB;
  ASSERT_TRUE(copy.write(0x1000, &after, sizeof(after)));

  std::uint32_t original_value = 0;
  ASSERT_TRUE(original.read(0x1000, &original_value, sizeof(original_value)));
  EXPECT_EQ(original_value, before) << "the copy wrote into the original's page";

  std::uint32_t copy_value = 0;
  ASSERT_TRUE(copy.read(0x1000, &copy_value, sizeof(copy_value)));
  EXPECT_EQ(copy_value, after);
}

TEST(KuberaMemory, CopyAssignmentAlsoDropsTheInheritedCaches) {
  seven::Memory original{};
  original.map(0x1000, 0x1000);
  const std::uint32_t before = 0x11111111;
  ASSERT_TRUE(original.write(0x1000, &before, sizeof(before)));

  seven::Memory target{};
  target.map(0x9000, 0x1000);
  std::uint32_t warm = 0;
  ASSERT_TRUE(target.read(0x9000, &warm, sizeof(warm)));

  target = original;
  const std::uint32_t after = 0x22222222;
  ASSERT_TRUE(target.write(0x1000, &after, sizeof(after)));

  std::uint32_t original_value = 0;
  ASSERT_TRUE(original.read(0x1000, &original_value, sizeof(original_value)));
  EXPECT_EQ(original_value, before);
}

// Executor's decode and code-page caches are validated on (rip, code_epoch, mode), and code_epoch
// is a per-Memory counter that every Memory starts near zero. Reusing one Executor across two of
// them -- separate guests, or one guest torn down and rebuilt -- meant the second could hit a
// cached decode belonging to the first and execute its bytes at that address.

TEST(KuberaMemory, ReusingAnExecutorAcrossTwoMemoriesDoesNotReuseTheirDecodes) {
  seven::Executor executor{};
  seven::CpuState state{};
  state.mode = seven::ExecutionMode::long64;

  seven::Memory first{};
  first.map(0x1000, 0x1000);
  const std::uint8_t inc_rax[] = {0x48, 0xFF, 0xC0};
  ASSERT_TRUE(first.write(0x1000, inc_rax, sizeof(inc_rax)));
  state.rip = 0x1000;
  state.gpr[0] = 0;
  ASSERT_EQ(executor.step(state, first).reason, seven::StopReason::none);
  ASSERT_EQ(state.gpr[0], 1u);

  seven::Memory second{};
  second.map(0x1000, 0x1000);
  const std::uint8_t dec_rax[] = {0x48, 0xFF, 0xC8};
  ASSERT_TRUE(second.write(0x1000, dec_rax, sizeof(dec_rax)));
  state.rip = 0x1000;
  state.gpr[0] = 10;
  ASSERT_EQ(executor.step(state, second).reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 9u) << "the second memory ran the first memory's instruction";
}

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "seven/executor.hpp"
#include "seven/guarded_allocator.hpp"
#include "seven/memory.hpp"

#if defined(_MSC_VER)
#include <excpt.h>

namespace {

constexpr unsigned kAccessViolation = 0xC0000005u;

// __try/__except cannot share a frame with anything that unwinds, so the probes are their own
// functions. The volatile pointer keeps the access from being optimized away.
bool reading_faults(const volatile std::byte* p) {
  __try {
    (void)*p;
    return false;
  } __except (GetExceptionCode() == kAccessViolation ? EXCEPTION_EXECUTE_HANDLER
                                                     : EXCEPTION_CONTINUE_SEARCH) {
    return true;
  }
}

bool writing_faults(volatile std::byte* p) {
  __try {
    *p = std::byte{0xCC};
    return false;
  } __except (GetExceptionCode() == kAccessViolation ? EXCEPTION_EXECUTE_HANDLER
                                                     : EXCEPTION_CONTINUE_SEARCH) {
    return true;
  }
}

}  // namespace
#endif

// Integer overflow in find_mmio_region: an address near ~0ull wraps address + size and the wrapped
// end passes both range checks, handing the callback a huge guest-controlled offset where its
// contract promises one inside [0, region.size).

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

// Same overflow in access_allowed()'s hook overlap check. It runs before the page write, so a
// skipped write hook never gets to veto: a real bypass from choosing a wraparound address.
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

// A passthrough is the storage backend, an access hook the policy layer over it. read()/write()
// returned through the passthrough before dispatching hooks, so an embedder with both got no hook
// calls at all, and a write hook's false return is what blocks a write.
TEST(KuberaMemory, AccessHooksStillRunWhenAPassthroughBacksMemory) {
  seven::Memory memory{};

  std::vector<std::uint8_t> backing(0x1000, 0u);
  memory.set_passthrough(
      [&](std::uint64_t address, void* dst, std::size_t size) {
        if (address + size > backing.size()) return false;
        std::memcpy(dst, backing.data() + address, size);
        return true;
      },
      [&](std::uint64_t address, const void* src, std::size_t size) {
        if (address + size > backing.size()) return false;
        std::memcpy(backing.data() + address, src, size);
        return true;
      });

  int reads_seen = 0;
  int writes_seen = 0;
  const auto id = memory.add_access_hook([&](const seven::MemoryAccessEvent& event) {
    if (event.kind == seven::MemoryAccessKind::data_write) {
      ++writes_seen;
      return false;  // veto
    }
    ++reads_seen;
    return true;
  });
  ASSERT_NE(id, 0u);

  const std::uint32_t value = 0xDEADBEEFu;
  EXPECT_FALSE(memory.write(0x100, &value, sizeof(value)));
  EXPECT_EQ(writes_seen, 1);

  std::uint32_t read_back = 0xFFFFFFFFu;
  EXPECT_TRUE(memory.read(0x100, &read_back, sizeof(read_back)));
  EXPECT_EQ(reads_seen, 1);
  EXPECT_EQ(read_back, 0u) << "the vetoed write reached the backing store anyway";
}

// set_passthrough takes its two halves independently, so a write-only passthrough is a legitimate
// configuration and has_passthrough() has to report it.
TEST(KuberaMemory, AWriteOnlyPassthroughStillCountsAsOne) {
  seven::Memory memory{};
  EXPECT_FALSE(memory.has_passthrough());
  memory.set_passthrough(nullptr, [](std::uint64_t, const void*, std::size_t) { return true; });
  EXPECT_TRUE(memory.has_passthrough());
  memory.clear_passthrough();
  EXPECT_FALSE(memory.has_passthrough());
}

// The other half: a sane access against a range whose own base + size runs off the top. The folded
// end made `event.address >= range_end` true for every address the host asked to watch.
TEST(KuberaMemory, HookRangeRunningOffTheTopStillCoversItsOwnAddresses) {
  seven::Memory memory{};
  memory.map(0xFFFF'FFFF'FFFF'F000ull, 0x1000, seven::kMemoryPermissionAll);

  int hook_calls = 0;
  const auto id = memory.add_access_hook(
      [&](const seven::MemoryAccessEvent&) {
        ++hook_calls;
        return false;  // veto, so a skipped hook shows up as a write that went through
      },
      // Ends exactly one byte past the top of the address space.
      seven::MemoryHookRange{.base = 0xFFFF'FFFF'FFFF'F000ull, .size = 0x1001},
      seven::bit(seven::MemoryAccessKind::data_write));
  ASSERT_NE(id, 0u);

  const std::uint32_t value = 0xDEADBEEFu;
  EXPECT_FALSE(memory.write(0xFFFF'FFFF'FFFF'F100ull, &value, sizeof(value)));
  EXPECT_EQ(hook_calls, 1);

  std::uint32_t read_back = 0xFFFFFFFFu;
  ASSERT_TRUE(memory.read(0xFFFF'FFFF'FFFF'F100ull, &read_back, sizeof(read_back)));
  EXPECT_EQ(read_back, 0u) << "the vetoed write reached the page anyway";
}

// The exclusive end page wraps to 0 for any range reaching the top of the address space, so the
// loop body never ran and the call silently did nothing. reprotect() is the sharp one: a host
// revoking write access is told it succeeded while the guest keeps writing.
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

// The single-page fast path guarded on `first_offset + size <= kPageSize`, and size is a full-width
// size_t, so a sum of exactly 2^64 reads as 0 and sails through into a memcpy of ~2^64 bytes.
TEST(KuberaMemory, HugeSizeCannotWrapPastTheSinglePageFastPath) {
  seven::Memory memory{};
  memory.map(0, seven::Memory::kPageSize);

  // Last byte of page 0, sized so kBase + kSize lands on exactly 2^64. The access itself does not
  // wrap, so a top-of-address-space check would let it past, but first_offset + kSize does.
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

// access_allowed assigned the dispatching flag rather than restoring it, so a hook reading guest
// memory re-entered and cleared it, letting a later remove erase from the vector being walked.
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

  // Third hook is the one being removed. It must stop firing immediately (see
  // AHookRemovedMidDispatchStopsFiringForTheRestOfThatAccess), but its slot must stay in the vector
  // until dispatch unwinds.
  to_remove = memory.add_access_hook([&](const seven::MemoryAccessEvent& event) {
    if (event.kind == seven::MemoryAccessKind::data_read) {
      return true;
    }
    ++tail_calls;
    return true;
  });
  ASSERT_NE(to_remove, 0u);

  // The fourth hook is what distinguishes a deferred erase from an in-place one: erasing the third
  // shifts this one into a slot the loop has already passed, so it never runs.
  int after_removed_calls = 0;
  const auto after_removed = memory.add_access_hook([&](const seven::MemoryAccessEvent& event) {
    if (event.kind == seven::MemoryAccessKind::data_read) {
      return true;
    }
    ++after_removed_calls;
    return true;
  });
  ASSERT_NE(after_removed, 0u);

  const std::uint32_t value = 0xA5A5A5A5;
  EXPECT_TRUE(memory.write(0x1000, &value, sizeof(value)));

  EXPECT_EQ(reentrant_calls, 1);
  EXPECT_EQ(remover_calls, 1);
  EXPECT_EQ(tail_calls, 0) << "a hook remove_access_hook already reported gone must not fire again";
  EXPECT_EQ(after_removed_calls, 1) << "the erase must be deferred until dispatch unwinds";

  // The deferred removal must actually have landed once dispatch finished.
  EXPECT_FALSE(memory.remove_access_hook(to_remove)) << "hook should already be gone";
  EXPECT_TRUE(memory.remove_access_hook(reentrant));
  EXPECT_TRUE(memory.remove_access_hook(remover));
  EXPECT_TRUE(memory.remove_access_hook(after_removed));
}

// remove_access_hook returns true from inside a dispatch, so its caller is free to destroy whatever
// the hook captured. The erase has to wait, but the dispatch kept calling the removed hook for the
// rest of that access: a call into an object its owner was just told it could drop.
TEST(KuberaMemory, AHookRemovedMidDispatchStopsFiringForTheRestOfThatAccess) {
  seven::Memory memory{};
  memory.map(0x1000, seven::Memory::kPageSize);

  seven::Memory::HookId victim = 0;
  int victim_calls = 0;
  bool victim_state_alive = true;

  const auto remover = memory.add_access_hook([&](const seven::MemoryAccessEvent&) {
    EXPECT_TRUE(memory.remove_access_hook(victim));
    victim_state_alive = false;  // stands in for the embedder tearing down what the hook captured
    return true;
  });
  ASSERT_NE(remover, 0u);

  victim = memory.add_access_hook([&](const seven::MemoryAccessEvent&) {
    ++victim_calls;
    EXPECT_TRUE(victim_state_alive) << "the removed hook ran against state its owner already dropped";
    return true;
  });
  ASSERT_NE(victim, 0u);

  const std::uint32_t value = 0x5A5A5A5A;
  EXPECT_TRUE(memory.write(0x1000, &value, sizeof(value)));
  EXPECT_EQ(victim_calls, 0) << "remove_access_hook said it was gone, so it must not be called again";
}

// A fetch may only fault on bytes the instruction needs, and seven asked for a full 15 every time,
// so a one-byte instruction before a guard page faulted. It also let a guest map out its
// surroundings without ever issuing an access to them.

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

// The test above uses B8, whose immediate is read through read_u32. An instruction whose missing
// byte is the modrm instead goes down a different exhaustion path in the decoder.
TEST(KuberaMemory, InstructionWhoseModrmStraddlesAnUnmappedPageStillFaults) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  // mov [rax], rbx is three bytes; the modrm is the one that lands past the boundary.
  const std::uint8_t head[] = {0x48, 0x89};
  ASSERT_TRUE(memory.write(0x1FFE, head, sizeof(head)));

  state.rip = 0x1FFE;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::page_fault);
  EXPECT_EQ(state.rip, 0x1FFEu);
  ASSERT_TRUE(result.exception.has_value());
  EXPECT_EQ(result.exception->address, 0x2000u) << "the fault belongs to the page it ran into";
}

// CpuState::msr is an unordered_map and WRMSR inserted unconditionally, so a guest walking ECX
// through a wrmsr loop allocated one node per index, about 200 GB, uninterruptibly.

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

// A move relocates the map but not its nodes, so the destination's inherited jit_tlb is correct.
// The moved-from source is the hazard: its jit_tlb stays warm naming the destination's storage, and
// the JIT fast path reads it directly with none of lookup_page's reconciliation.
TEST(KuberaMemory, AMovedFromMemoryKeepsNoFastPathIntoItsDestination) {
  seven::Memory source{};
  source.map(0x1000, 0x1000);
  const std::uint32_t seed = 0x33333333;
  ASSERT_TRUE(source.write(0x1000, &seed, sizeof(seed)));

  // Warm a fast-path slot by hand: the JIT owns this cache in production, but the property under
  // test is purely that a move clears whatever the source had, and this needs no JIT to state.
  auto& slot = source.jit_tlb[0];
  slot.guest_page = 0x1000 / seven::Memory::kPageSize;
  slot.host_data = source.page_data(slot.guest_page);
  slot.permissions = 0xFF;
  ASSERT_NE(slot.host_data, nullptr);

  seven::Memory destination = std::move(source);

  for (const auto& s : source.jit_tlb) {
    EXPECT_EQ(s.host_data, nullptr) << "a moved-from Memory kept a warm host pointer into the destination";
    EXPECT_EQ(s.guest_page, ~0ull) << "a moved-from Memory kept a warm page tag";
  }
  // The destination still works, and reads its own seed.
  std::uint32_t value = 0;
  ASSERT_TRUE(destination.read(0x1000, &value, sizeof(value)));
  EXPECT_EQ(value, seed);
}

// The decode and code-page caches validate on (rip, code_epoch, mode), and code_epoch is per-Memory
// starting near zero, so an Executor reused across two could run the first one's bytes.

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

TEST(KuberaMemory, PassthroughNeverSeesAWrappingRangeInEitherDirection) {
  seven::Memory memory{};
  bool saw_read = false;
  bool saw_write = false;
  memory.set_passthrough(
      [&](std::uint64_t, void*, std::size_t) { saw_read = true; return true; },
      [&](std::uint64_t, const void*, std::size_t) { saw_write = true; return true; });

  const std::uint64_t near_top = ~std::uint64_t{0} - 3;
  std::uint64_t value = 0;
  EXPECT_FALSE(memory.read(near_top, &value, sizeof(value)));
  EXPECT_FALSE(saw_read);
  EXPECT_FALSE(memory.write(near_top, &value, sizeof(value)));
  EXPECT_FALSE(saw_write) << "the write side used to forward a wrapping range the read side rejects";

  // A range that stops exactly at the top is not a wrap and must still get through.
  EXPECT_TRUE(memory.read(~std::uint64_t{0} - 7, &value, sizeof(value)));
  EXPECT_TRUE(saw_read);
  EXPECT_TRUE(memory.write(~std::uint64_t{0} - 7, &value, sizeof(value)));
  EXPECT_TRUE(saw_write);
}

// Every hook dispatch guards against mutation from inside a callback except the two execution-hook
// loops, so a self-removing hook (a one-shot breakpoint) erased from the vector being walked.

TEST(KuberaMemory, AnExecutionHookMayRemoveHooksFromInsideItsOwnCallback) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  const std::uint8_t nops[] = {0x90, 0x90, 0x90};
  ASSERT_TRUE(memory.write(0x1000, nops, sizeof(nops)));

  int first_calls = 0;
  int second_calls = 0;
  int third_calls = 0;
  seven::Executor::HookId second = 0;
  seven::Executor::HookId third = 0;

  const auto first = executor.add_execution_hook([&](std::uint64_t) {
    ++first_calls;
    // Drop the two hooks queued behind this one, from inside the dispatch that is walking them.
    (void)executor.remove_hook(second);
    (void)executor.remove_hook(third);
  });
  ASSERT_NE(first, 0u);
  second = executor.add_execution_hook([&](std::uint64_t) { ++second_calls; });
  third = executor.add_execution_hook([&](std::uint64_t) { ++third_calls; });
  ASSERT_NE(second, 0u);
  ASSERT_NE(third, 0u);

  state.rip = 0x1000;
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(first_calls, 1);
  EXPECT_EQ(second_calls, 1) << "a removal must not take effect until the walk finishes";
  EXPECT_EQ(third_calls, 1) << "a removal must not take effect until the walk finishes";

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(first_calls, 2);
  EXPECT_EQ(second_calls, 1) << "the deferred removal has to actually land";
  EXPECT_EQ(third_calls, 1) << "the deferred removal has to actually land";
}

TEST(KuberaMemory, AnExecutionHookMayAddAHookFromInsideItsOwnCallback) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  const std::uint8_t nops[] = {0x90, 0x90, 0x90};
  ASSERT_TRUE(memory.write(0x1000, nops, sizeof(nops)));

  int added_calls = 0;
  bool has_added = false;
  const auto id = executor.add_execution_hook([&](std::uint64_t) {
    if (has_added) {
      return;
    }
    has_added = true;
    // An immediate emplace_back here would reallocate the vector the dispatch loop is walking.
    (void)executor.add_execution_hook([&](std::uint64_t) { ++added_calls; });
  });
  ASSERT_NE(id, 0u);

  state.rip = 0x1000;
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(added_calls, 0) << "a hook added mid-walk must not run until the next instruction";
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(added_calls, 1);
}

// step_impl holds a reference to a direct-mapped decode slot across every hook dispatch and the
// handler call. A nested step 0x4000 away lands on the same slot, so the outer frame runs the
// handler it picked from the original opcode against the nested instruction's operands.

TEST(KuberaMemory, AReentrantHookCannotSwapTheInstructionUnderTheOuterHandler) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  memory.map(0x5000, 0x1000);  // (0x1000 >> 1) & 8191 == (0x5000 >> 1) & 8191

  const std::uint8_t outer[] = {0x48, 0x01, 0xD8};  // add rax, rbx
  const std::uint8_t inner[] = {0x48, 0x01, 0xD1};  // add rcx, rdx
  ASSERT_TRUE(memory.write(0x1000, outer, sizeof(outer)));
  ASSERT_TRUE(memory.write(0x5000, inner, sizeof(inner)));

  seven::CpuState nested{};
  nested.mode = seven::ExecutionMode::long64;
  nested.rip = 0x5000;

  bool reentered = false;
  const auto id = executor.add_execution_hook([&](std::uint64_t) {
    if (reentered) {
      return;
    }
    reentered = true;
    (void)executor.step(nested, memory);
  });
  ASSERT_NE(id, 0u);

  state.rip = 0x1000;
  state.gpr[0] = 1;    // rax
  state.gpr[3] = 2;    // rbx
  state.gpr[1] = 100;  // rcx
  state.gpr[2] = 200;  // rdx

  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  ASSERT_TRUE(reentered);
  EXPECT_EQ(state.gpr[0], 3u) << "the outer add rax, rbx has to be the one that ran";
  EXPECT_EQ(state.gpr[1], 100u) << "the nested instruction's operands must not leak into it";
}

// A passthrough replaces the page table wholesale, so nothing stamps a PageEntry and every page
// reported epoch 0 forever. Self-modifying code kept running the bytes it was first decoded from.

TEST(KuberaMemory, SelfModifyingCodeThroughAPassthroughInvalidatesTheDecodeCache) {
  std::array<std::uint8_t, 0x1000> backing{};
  backing[0] = 0x48;  // add rax, rcx
  backing[1] = 0x01;
  backing[2] = 0xC8;

  seven::Memory memory{};
  memory.set_passthrough(
      [&](std::uint64_t address, void* dst, std::size_t size) {
        if (address < 0x1000 || address + size > 0x1000 + backing.size()) return false;
        std::memcpy(dst, backing.data() + (address - 0x1000), size);
        return true;
      },
      [&](std::uint64_t address, const void* src, std::size_t size) {
        if (address < 0x1000 || address + size > 0x1000 + backing.size()) return false;
        std::memcpy(backing.data() + (address - 0x1000), src, size);
        return true;
      });

  seven::CpuState state{};
  state.mode = seven::ExecutionMode::long64;
  state.gpr[0] = 10;  // rax
  state.gpr[1] = 3;   // rcx

  seven::Executor executor{};
  state.rip = 0x1000;
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 13u) << "sanity: add rax, rcx";

  const std::uint8_t sub_opcode = 0x29;  // add -> sub
  ASSERT_TRUE(memory.write(0x1001, &sub_opcode, 1));

  state.rip = 0x1000;
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 10u) << "the rewritten sub rax, rcx has to be what ran";
}

// page_code_epoch() gated on passthrough_read_ alone, but a write-only passthrough is legitimate and
// its writes bump only the global counter, so a page kept reporting its stale per-entry epoch.
TEST(KuberaMemory, AWriteOnlyPassthroughWriteMovesThePageEpoch) {
  seven::Memory memory{};
  memory.map(0x1000, 0x1000);
  memory.set_passthrough(nullptr, [](std::uint64_t, const void*, std::size_t) { return true; });
  // Captured after the install: set_passthrough re-stamps every page epoch, so the only thing left
  // to move page_code_epoch is the write itself -- which is exactly what the bug missed.
  const auto before = memory.page_code_epoch(1);
  std::uint8_t byte = 0;
  ASSERT_TRUE(memory.write(0x1000, &byte, 1));
  EXPECT_NE(memory.page_code_epoch(1), before)
      << "a passthrough write bumps only the global epoch; page_code_epoch has to report it";
}

// The same hole one layer up: a decode cache keyed on page_code_epoch() replayed pre-modification
// bytes when the store went through a write-only passthrough. Mirrors the read+write case above,
// with reads left on the page-backed path so PageEntries actually exist to go stale.
TEST(KuberaMemory, SelfModifyingCodeThroughAWriteOnlyPassthroughInvalidatesTheDecodeCache) {
  seven::Memory memory{};
  memory.map(0x1000, 0x1000);
  memory.set_passthrough(nullptr, [&](std::uint64_t address, const void* src, std::size_t size) {
    auto* page = memory.page_data(address / 0x1000);
    if (page == nullptr) return false;
    std::memcpy(page + (address % 0x1000), src, size);
    return true;
  });

  const std::uint8_t add_code[3] = {0x48, 0x01, 0xC8};  // add rax, rcx
  ASSERT_TRUE(memory.write(0x1000, add_code, sizeof(add_code)));

  seven::CpuState state{};
  state.mode = seven::ExecutionMode::long64;
  state.gpr[0] = 10;  // rax
  state.gpr[1] = 3;   // rcx

  seven::Executor executor{};
  state.rip = 0x1000;
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 13u) << "sanity: add rax, rcx";

  const std::uint8_t sub_opcode = 0x29;  // add -> sub
  ASSERT_TRUE(memory.write(0x1001, &sub_opcode, 1));

  state.rip = 0x1000;
  ASSERT_EQ(executor.step(state, memory).reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], 10u) << "the rewritten sub rax, rcx has to be what ran";
}

// jit_bypass_eligible listed hooks, the trap flag and DR7 but not the context-sync callbacks, which
// are per-instruction machinery too, so a bypassing consumer silently stopped mirroring.

TEST(KuberaMemory, ContextSyncCallbacksBlockTheCodegenBypass) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = 0x1000;
  memory.map(0x1000, 0x1000);

  ASSERT_TRUE(executor.jit_bypass_eligible(state, memory)) << "baseline: nothing registered";

  executor.set_context_read_callback([](seven::CpuState&) { return true; });
  EXPECT_FALSE(executor.jit_bypass_eligible(state, memory)) << "a read callback has to run per instruction";

  executor.set_context_read_callback(nullptr);
  ASSERT_TRUE(executor.jit_bypass_eligible(state, memory));

  executor.set_context_write_callback([](seven::CpuState&) { return true; });
  EXPECT_FALSE(executor.jit_bypass_eligible(state, memory)) << "so does a write callback";
}

// Asked by the block compiler before it fetches. It reads far more than the instruction it is
// about to run needs, and doing that against a device is a read the guest never asked for.
TEST(KuberaMemory, AnMmioAddressIsDistinguishableFromOrdinaryMemory) {
  seven::Memory memory{};
  constexpr std::uint64_t kBase = 0x50000;
  memory.map(0x10000, 0x1000);
  EXPECT_FALSE(memory.is_mmio_address(0x10000)) << "an ordinary mapped page is not mmio";
  EXPECT_FALSE(memory.is_mmio_address(kBase)) << "nothing is registered here yet";

  const auto id = memory.map_mmio(
      kBase, 0x1000, [](std::uint64_t, void*, std::size_t) { return true; },
      [](std::uint64_t, const void*, std::size_t) { return true; });
  ASSERT_NE(id, 0u);

  EXPECT_TRUE(memory.is_mmio_address(kBase));
  EXPECT_TRUE(memory.is_mmio_address(kBase + 0xFFF)) << "the last byte is still inside the region";
  EXPECT_FALSE(memory.is_mmio_address(kBase + 0x1000)) << "one past the end is outside it";
  EXPECT_FALSE(memory.is_mmio_address(kBase - 1));
  EXPECT_FALSE(memory.is_mmio_address(0x10000));

  EXPECT_TRUE(memory.unmap_mmio(id));
  EXPECT_FALSE(memory.is_mmio_address(kBase)) << "the region is gone";
}

// The counter has to move for calls that actually reached a device and stay put for ones that did
// not: a counter that moved on ordinary RAM would stall a compiled block after every memory operand
// for as long as any device stayed mapped.
TEST(KuberaMemory, DeviceDispatchCountTracksCallbacksNotConfiguration) {
  seven::Memory memory{};
  constexpr std::uint64_t kRam = 0x10000;
  constexpr std::uint64_t kDev = 0x20000;
  memory.map(kRam, 0x1000);

  std::uint64_t value = 0;
  ASSERT_TRUE(memory.write(kRam, &value, sizeof(value)));
  ASSERT_TRUE(memory.read(kRam, &value, sizeof(value)));
  EXPECT_EQ(memory.device_dispatch_count(), 0u) << "plain ram must not look like a device";

  int reads = 0;
  int writes = 0;
  const auto id = memory.map_mmio(
      kDev, 0x1000,
      [&](std::uint64_t, void* dst, std::size_t size) {
        ++reads;
        std::memset(dst, 0, size);
        return true;
      },
      [&](std::uint64_t, const void*, std::size_t) {
        ++writes;
        return true;
      });
  ASSERT_NE(id, 0u);
  EXPECT_EQ(memory.device_dispatch_count(), 0u)
      << "mapping a device is not a dispatch, only calling one is";

  ASSERT_TRUE(memory.read(kDev, &value, sizeof(value)));
  EXPECT_EQ(memory.device_dispatch_count(), 1u);
  ASSERT_TRUE(memory.write(kDev, &value, sizeof(value)));
  EXPECT_EQ(memory.device_dispatch_count(), 2u);
  EXPECT_EQ(reads, 1);
  EXPECT_EQ(writes, 1);

  // A ram access while a device is mapped still must not move it -- that is the whole point.
  ASSERT_TRUE(memory.read(kRam, &value, sizeof(value)));
  ASSERT_TRUE(memory.write(kRam, &value, sizeof(value)));
  EXPECT_EQ(memory.device_dispatch_count(), 2u)
      << "an access that never reached a device moved the counter anyway";
}

// A device mapped over an address changes what a fetch there returns, so a cached decode is stale.
TEST(KuberaMemory, MappingADeviceOverCodeInvalidatesCachedDecodes) {
  constexpr std::uint64_t kProg = 0x1000;
  seven::Memory memory{};
  seven::Executor executor{};
  memory.map(kProg, 0x1000);
  const std::vector<std::uint8_t> code = {0x83, 0xC0, 0x01, 0x83, 0xC0, 0x01, 0xF4};  // add eax,1 x2 ; hlt
  ASSERT_TRUE(memory.write(kProg, code.data(), code.size()));

  const auto go = [&] {
    seven::CpuState state{};
    state.mode = seven::ExecutionMode::long64;
    state.rip = kProg;
    state.rflags = 0x202;
    state.gpr[4] = kProg + 0x800;
    const auto result = executor.run(state, memory, 64);
    return std::pair{result.reason, state.gpr[0]};
  };

  const auto before = go();
  ASSERT_EQ(before.first, seven::StopReason::halted);
  ASSERT_EQ(before.second, 2u) << "the warm-up did not run the adds, so nothing got cached";

  // The device answers every fetch with hlt, so the adds are simply not there any more.
  const auto id = memory.map_mmio(
      kProg, 0x1000,
      [](std::uint64_t, void* dst, std::size_t size) {
        std::memset(dst, 0xF4, size);
        return true;
      },
      [](std::uint64_t, const void*, std::size_t) { return true; });
  ASSERT_NE(id, 0u);

  const auto after = go();
  EXPECT_EQ(after.first, seven::StopReason::halted);
  EXPECT_EQ(after.second, 0u) << "the cached decode kept running instructions the device replaced";

  ASSERT_TRUE(memory.unmap_mmio(id));
  const auto restored = go();
  EXPECT_EQ(restored.first, seven::StopReason::halted);
  EXPECT_EQ(restored.second, 2u) << "removing the device left the decode it served cached";
}

// One Executor across two Memory objects is what the decode cache's instance tag exists for. The
// tag was claimed once per step() rather than per fault-retry attempt, so a hook that re-entered
// with the other Memory left the outer frame filling entries under the wrong owner's name.
TEST(KuberaMemory, AReenteringFaultHookCannotMisclaimTheDecodeCache) {
  seven::Executor executor{};
  seven::Memory a{};
  seven::Memory b{};
  b.map(0x1000, 0x1000);

  const std::uint8_t a_code[] = {0x83, 0xC0, 0x01};  // add eax,1
  const std::uint8_t b_code[] = {0x83, 0xE8, 0x01};  // sub eax,1
  ASSERT_TRUE(b.write(0x1000, b_code, sizeof(b_code)));

  bool reentered = false;
  const auto id = executor.add_fault_hook([&](const seven::FaultHookEvent& event) {
    if (reentered) {
      return seven::FaultHookAction::stop;
    }
    reentered = true;
    seven::CpuState nested{};
    nested.mode = seven::ExecutionMode::long64;
    nested.rip = 0x1000;
    nested.rflags = 0x202;
    nested.gpr[4] = 0x1800;
    (void)executor.step(nested, b);
    event.memory.map(0x1000, 0x1000);
    EXPECT_TRUE(event.memory.write(0x1000, a_code, sizeof(a_code)));
    return seven::FaultHookAction::restart_instruction;
  });
  ASSERT_NE(id, 0u);

  seven::CpuState state_a{};
  state_a.mode = seven::ExecutionMode::long64;
  state_a.rip = 0x1000;
  state_a.rflags = 0x202;
  state_a.gpr[4] = 0x1800;
  state_a.gpr[0] = 10;
  // One step, not run(): every step_impl call reclaims the tag, so a second one would paper over
  // the entry this one cached under the wrong owner before anything could read it back.
  const auto ran_a = executor.step(state_a, a);
  ASSERT_EQ(ran_a.reason, seven::StopReason::none);
  ASSERT_EQ(state_a.gpr[0], 11u) << "a should have added";

  seven::CpuState state_b{};
  state_b.mode = seven::ExecutionMode::long64;
  state_b.rip = 0x1000;
  state_b.rflags = 0x202;
  state_b.gpr[4] = 0x1800;
  state_b.gpr[0] = 10;
  const auto ran_b = executor.step(state_b, b);
  EXPECT_EQ(ran_b.reason, seven::StopReason::none);
  EXPECT_EQ(state_b.gpr[0], 9u) << "b ran the decode cached for a";
}

// Bits 63:47 of a linear address must all match bit 47. Data references get this check through
// memory_fault(); the fetch path builds its faults inline and never did.
TEST(KuberaMemory, FetchingFromANonCanonicalAddressIsAGeneralProtectionFault) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = 0x0000'8000'0000'0000ull;  // one past the top of the low half
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  ASSERT_TRUE(result.exception.has_value());
  EXPECT_EQ(result.exception->address, state.rip);
}

// The truncated-fetch retry reports the fault against the page the instruction ran into, which for
// the last page in the address space is not a page at all.
TEST(KuberaMemory, AnInstructionRunningOffTheTopOfTheAddressSpaceFaultsThere) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0xFFFF'FFFF'FFFF'F000ull, 0x1000);
  const std::uint8_t head[] = {0xB8};  // mov eax, imm32, with none of the immediate present
  ASSERT_TRUE(memory.write(0xFFFF'FFFF'FFFF'FFFFull, head, sizeof(head)));

  state.rip = 0xFFFF'FFFF'FFFF'FFFFull;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::general_protection);
  ASSERT_TRUE(result.exception.has_value());
  EXPECT_EQ(result.exception->address, state.rip) << "the fault wrapped around to page zero";
}

// A store that faults partway must leave nothing behind. This is the property a guard page depends
// on: a writable page abutting an unmapped one is the ordinary layout, and hardware checks the whole
// footprint before it commits any of it.
TEST(KuberaMemory, AWriteStraddlingOntoAnUnwritablePageCommitsNothing) {
  const std::array<std::uint8_t, 8> value = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  const auto first_page_after = [&](bool map_second_read_only) {
    seven::Memory memory{};
    memory.map(0x1000, 0x1000, seven::kMemoryPermissionReadWrite);
    if (map_second_read_only) {
      memory.map(0x2000, 0x1000, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read));
    }
    EXPECT_FALSE(memory.write(0x1FFC, value.data(), value.size()));
    std::array<std::uint8_t, 4> tail{};
    EXPECT_TRUE(memory.read(0x1FFC, tail.data(), tail.size()));
    return tail;
  };

  const std::array<std::uint8_t, 4> untouched = {0, 0, 0, 0};
  EXPECT_EQ(first_page_after(true), untouched) << "bytes landed on the first page before the fault";
  EXPECT_EQ(first_page_after(false), untouched) << "same, with the second page not mapped at all";
}

// find_mmio_region matches only when one region covers the whole access, so an access that merely
// runs into a device matched nothing and fell through to the page underneath it -- writing straight
// past the device with its callback never invoked.
TEST(KuberaMemory, AnAccessStraddlingADeviceEdgeDoesNotSlipPastIt) {
  seven::Memory memory{};
  memory.map(0x1000, 0x2000, seven::kMemoryPermissionReadWrite);
  int writes_seen = 0;
  const auto id = memory.map_mmio(
      0x2000, 0x1000,
      [](std::uint64_t, void* dst, std::size_t size) {
        std::memset(dst, 0xCD, size);
        return true;
      },
      [&](std::uint64_t, const void*, std::size_t) {
        ++writes_seen;
        return true;
      });
  ASSERT_NE(id, 0u);

  const std::array<std::uint8_t, 8> value = {1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_FALSE(memory.write(0x1FFC, value.data(), value.size()))
      << "half of this lands on the device, so it cannot quietly go to the page underneath";
  EXPECT_EQ(writes_seen, 0);

  std::array<std::uint8_t, 4> tail{};
  ASSERT_TRUE(memory.read(0x1FFC, tail.data(), tail.size()));
  const std::array<std::uint8_t, 4> untouched = {0, 0, 0, 0};
  EXPECT_EQ(tail, untouched);

  std::array<std::uint8_t, 8> readback{};
  EXPECT_FALSE(memory.read(0x1FFC, readback.data(), readback.size()));
}

// The framework contracts an MMIO offset into [0, region.size). A zero-size access at the region's
// end passed the containment test and handed the callback an offset of exactly region.size.
TEST(KuberaMemory, AZeroSizeAccessNeverReachesADeviceCallback) {
  seven::Memory memory{};
  int hits = 0;
  const auto id = memory.map_mmio(
      0x8000, 0x100,
      [&](std::uint64_t, void*, std::size_t) {
        ++hits;
        return true;
      },
      [&](std::uint64_t, const void*, std::size_t) {
        ++hits;
        return true;
      });
  ASSERT_NE(id, 0u);

  std::uint8_t scratch = 0;
  EXPECT_TRUE(memory.read(0x8100, &scratch, 0));
  EXPECT_TRUE(memory.write(0x8100, &scratch, 0));
  EXPECT_EQ(hits, 0);
}

// A 0x67 prefix makes a string instruction use ECX/ESI/EDI. iced records that only in the operand
// kind, never in the Code, so a handler reading the 64-bit registers executes an entirely different
// instruction: an RCX whose low half is zero looks like four billion iterations rather than none.
TEST(KuberaMemory, StringOperationsHonourTheAddressSizePrefix) {
  seven::Memory memory{};
  seven::Executor executor{};
  memory.map(0x1000, 0x1000);
  memory.map(0x4000, 0x1000, seven::kMemoryPermissionReadWrite);

  const auto run_one = [&](const std::vector<std::uint8_t>& code, seven::CpuState& state) {
    EXPECT_TRUE(memory.write(0x1000, code.data(), code.size()));
    state.mode = seven::ExecutionMode::long64;
    state.rip = 0x1000;
    state.rflags = 0x202;
    return executor.step(state, memory);
  };

  {
    // 67 F3 AA -- rep stosb with ECX == 0. Hardware stores nothing.
    seven::CpuState state{};
    state.gpr[0] = 0xFF;                  // AL
    state.gpr[1] = 0x0000'0001'0000'0000; // RCX: ECX is zero
    state.gpr[7] = 0x4000;                // RDI
    ASSERT_EQ(run_one({0x67, 0xF3, 0xAA}, state).reason, seven::StopReason::none);
    std::uint8_t first = 0xAB;
    ASSERT_TRUE(memory.read(0x4000, &first, 1));
    EXPECT_EQ(first, 0u) << "ECX was zero, so nothing should have been stored";
    EXPECT_EQ(state.gpr[1], 0x0000'0001'0000'0000ull) << "no iterations, so rcx is untouched";
  }
  {
    // The count itself must come from ECX, not RCX.
    seven::CpuState state{};
    state.gpr[0] = 0x5A;
    state.gpr[1] = 0xFFFF'FFFF'0000'0004;  // ECX == 4
    state.gpr[7] = 0x4100;
    ASSERT_EQ(run_one({0x67, 0xF3, 0xAA}, state).reason, seven::StopReason::none);
    std::array<std::uint8_t, 5> seen{};
    ASSERT_TRUE(memory.read(0x4100, seen.data(), seen.size()));
    const std::array<std::uint8_t, 5> expected = {0x5A, 0x5A, 0x5A, 0x5A, 0x00};
    EXPECT_EQ(seen, expected);
    EXPECT_EQ(state.gpr[1], 0u) << "writing ecx zero-extends";
    EXPECT_EQ(state.gpr[7], 0x4104u);
  }
}

// The DS:rSI source of movs/lods/cmps takes a segment override; the ES:rDI destination does not.
// The handlers read straight through rSI, so an fs-prefixed lods read the unbiased address.
TEST(KuberaMemory, StringSourceOperandHonoursASegmentOverride) {
  seven::CpuState state{};
  seven::Memory memory{};
  seven::Executor executor{};
  state.mode = seven::ExecutionMode::long64;
  memory.map(0x1000, 0x1000);
  memory.map(0x51000, 0x1000);
  state.fs_base = 0x50000;

  const std::uint8_t code[] = {0x64, 0x48, 0xAD};  // fs lodsq
  ASSERT_TRUE(memory.write(0x1000, code, sizeof(code)));
  const std::uint64_t through_fs = 0xFEEDFACECAFEBEEFull;
  ASSERT_TRUE(memory.write(0x51230, &through_fs, sizeof(through_fs)));
  const std::uint64_t unbiased = 0x1111222233334444ull;
  ASSERT_TRUE(memory.write(0x1230, &unbiased, sizeof(unbiased)));

  state.rip = 0x1000;
  state.gpr[6] = 0x1230;
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_EQ(state.gpr[0], through_fs);
  EXPECT_EQ(state.gpr[6], 0x1238u) << "rsi steps by the operand size, not the linear address";
}

// Each level of a hook re-entering step() costs a host stack frame and a decode-scratch slot that is
// never released, so one that re-enters unconditionally exhausted the host stack. Only an embedder
// can reach it, but a silly hook should be a stop, not a crash.
TEST(KuberaMemory, AnAccessHookThatReentersForeverStopsInsteadOfExhaustingTheStack) {
  constexpr std::uint64_t kBase = 0x1000;
  constexpr std::uint64_t kData = 0x4000;

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kData;  // rbx

  memory.map(kBase, 0x1000);
  memory.map(kData, 0x1000);
  const std::uint8_t code[] = {0x48, 0x8B, 0x03};  // mov rax, [rbx]
  ASSERT_TRUE(memory.write(kBase, code, sizeof(code)));

  // Scoped to the data page so the hook does not also fire on the instruction fetch. An unscoped
  // hook makes each frame spawn two nested steps and the recursion fans out exponentially, which
  // is a different (and much noisier) failure than the straight-line descent being bounded here.
  std::size_t depth_now = 0;
  std::size_t depth_max = 0;
  const auto hook = memory.add_access_hook(
      [&](const seven::MemoryAccessEvent&) {
        ++depth_now;
        depth_max = std::max(depth_max, depth_now);
        seven::CpuState nested = state;
        nested.rip = kBase;
        (void)executor.step(nested, memory);
        --depth_now;
        return true;
      },
      seven::MemoryHookRange{kData, 0x1000});
  (void)hook;

  // The outermost step still retires normally. Only the frames past the cap give up, and they give
  // up by returning rather than by taking the process with them, which is the whole point.
  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  EXPECT_GT(depth_max, 0u) << "the hook never ran, so this test proved nothing";
  EXPECT_LE(depth_max, seven::Executor::kMaxStepDepth + 2)
      << "the re-entry ran deeper than the step-depth cap should ever allow";
}

// The cap counts levels, but what must fit is bytes: one re-entry cycle times the cap, a product
// nothing checks. An ASan build, which pads every local, blows a 1 MiB stack partway through it.
// This pins the per-level cost so a frame that grows quietly gets caught here.
TEST(KuberaMemory, ReentrantStepCostsABoundedNumberOfStackBytesPerLevel) {
  constexpr std::uint64_t kBase = 0x1000;
  constexpr std::uint64_t kData = 0x4000;

  seven::Executor executor{};
  seven::CpuState state{};
  seven::Memory memory{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kBase;
  state.gpr[3] = kData;  // rbx

  memory.map(kBase, 0x1000);
  memory.map(kData, 0x1000);
  const std::uint8_t code[] = {0x48, 0x8B, 0x03};  // mov rax, [rbx]
  ASSERT_TRUE(memory.write(kBase, code, sizeof(code)));

  // One probe per nesting level, in entry order, so consecutive differences are the cost of exactly
  // one cycle. Scoped to the data page for the same reason as the test above.
  std::vector<const char*> marks;
  const auto hook = memory.add_access_hook(
      [&](const seven::MemoryAccessEvent&) {
        volatile char probe = 0;
        marks.push_back(const_cast<const char*>(&probe));
        seven::CpuState nested = state;
        nested.rip = kBase;
        (void)executor.step(nested, memory);
        return true;
      },
      seven::MemoryHookRange{kData, 0x1000});
  (void)hook;

  const auto result = executor.step(state, memory);
  EXPECT_EQ(result.reason, seven::StopReason::none);
  ASSERT_GE(marks.size(), 2u) << "need two nesting levels before a per-level cost means anything";

  // Stack grows down, so each deeper probe sits at a lower address than the one before it.
  std::ptrdiff_t worst_level = 0;
  for (std::size_t i = 1; i < marks.size(); ++i) {
    worst_level = std::max(worst_level, marks[i - 1] - marks[i]);
  }
  ASSERT_GT(worst_level, 0) << "probes did not descend, so this measured nothing";

  // The whole descent stays inside the byte budget plus the frame already entered when the check
  // ran. The level cap alone permitted about 950 KB, which made a 1 MiB stack a coin toss.
  const auto total = static_cast<std::size_t>(marks.front() - marks.back());
  const auto allowed = seven::Executor::kMaxStepStackBytes + static_cast<std::size_t>(worst_level);
  EXPECT_LE(total, allowed)
      << "the re-entry descended " << total << " bytes of host stack with a budget of "
      << seven::Executor::kMaxStepStackBytes << " and a per-level cost of " << worst_level
      << " bytes, so the descent is not being bounded by the byte budget any more.";
}

#if defined(_MSC_VER)
// Takes the raw page_data() pointer the JIT's fast path stores through -- which no bounds check or
// sanitizer sees -- and walks off the end of it. The two halves assert opposite things on purpose:
// this has to fail in the default build for the guarded build's pass to mean anything.
TEST(KuberaMemory, GuardedBuildFaultsOnAnAccessOneBytePastAGuestPage) {
  if (!seven::kGuardedPagesEnabled) {
    GTEST_SKIP() << "built without SEVEN_GUARDED_PAGES, so there is no guard page to hit";
  }

  seven::Memory memory;
  memory.map(0x4000, seven::Memory::kPageSize);

  auto* page = memory.page_data(0x4000 / seven::Memory::kPageSize);
  ASSERT_NE(page, nullptr);

  // The last byte the guest owns has to stay readable and writable, or the guard is simply sitting
  // in the wrong place and the test would pass for the wrong reason.
  EXPECT_FALSE(reading_faults(page + seven::Memory::kPageSize - 1));
  EXPECT_FALSE(writing_faults(page + seven::Memory::kPageSize - 1));

  EXPECT_TRUE(reading_faults(page + seven::Memory::kPageSize))
      << "a read one byte past the page did not fault, so the page bytes are not ending against "
         "the guard page";
  EXPECT_TRUE(writing_faults(page + seven::Memory::kPageSize))
      << "a write one byte past the page did not fault, so an emitted store that overruns a page "
         "would still be landing in live heap memory";
}

// A one-byte overrun is the friendly case. The failure this is really aimed at is an access whose
// width disagrees with the bounds check that let it through, which lands several bytes over.
TEST(KuberaMemory, GuardedBuildFaultsOnAWideAccessStraddlingTheEndOfAGuestPage) {
  if (!seven::kGuardedPagesEnabled) {
    GTEST_SKIP() << "built without SEVEN_GUARDED_PAGES, so there is no guard page to hit";
  }

  seven::Memory memory;
  memory.map(0x8000, seven::Memory::kPageSize);

  auto* page = memory.page_data(0x8000 / seven::Memory::kPageSize);
  ASSERT_NE(page, nullptr);

  // Eight bytes starting four from the end: the first four are the guest's, the rest are not.
  for (std::size_t i = 0; i < 8; ++i) {
    const bool past_the_end = (seven::Memory::kPageSize - 4 + i) >= seven::Memory::kPageSize;
    EXPECT_EQ(writing_faults(page + seven::Memory::kPageSize - 4 + i), past_the_end)
        << "byte " << i << " of a straddling 8-byte store behaved unexpectedly";
  }
}
#endif

// classify_faulting_address only exists in the guarded build, so unlike the two tests above this
// one cannot compile at all without the flag and has to be excluded rather than skipped.
#if defined(_MSC_VER) && defined(SEVEN_GUARDED_PAGES)
// A fault handler has to be able to tell a guard hit apart from an ordinary crash, otherwise the
// fuzzer's escape lane can only report "something died" and every finding needs a debugger.
TEST(KuberaMemory, AGuardPageAddressIsIdentifiableFromAFaultHandler) {
  seven::Memory memory;
  memory.map(0xC000, seven::Memory::kPageSize);

  auto* page = memory.page_data(0xC000 / seven::Memory::kPageSize);
  ASSERT_NE(page, nullptr);

  const auto inside = seven::classify_faulting_address(page + seven::Memory::kPageSize - 1);
  EXPECT_FALSE(inside.is_guard_page) << "the last byte the guest owns was reported as a guard page";

  const auto overrun = seven::classify_faulting_address(page + seven::Memory::kPageSize);
  EXPECT_TRUE(overrun.is_guard_page);
  EXPECT_EQ(overrun.bytes_past_payload, 0u);

  const auto further = seven::classify_faulting_address(page + seven::Memory::kPageSize + 24);
  EXPECT_TRUE(further.is_guard_page);
  EXPECT_EQ(further.bytes_past_payload, 24u);
  EXPECT_EQ(further.guard_base, overrun.guard_base);

  // Address space gets handed back and reused, so an unmapped page's guard has to stop being
  // reported as one or the classifier would manufacture findings out of unrelated crashes.
  const auto guard_base = overrun.guard_base;
  memory.unmap(0xC000, seven::Memory::kPageSize);
  EXPECT_FALSE(seven::classify_faulting_address(reinterpret_cast<const void*>(guard_base))
                   .is_guard_page);

  EXPECT_FALSE(seven::classify_faulting_address(&memory).is_guard_page)
      << "a plain stack address was reported as a guard page";
}
#endif

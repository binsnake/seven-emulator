#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

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

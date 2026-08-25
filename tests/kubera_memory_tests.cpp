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
  memory.map_mmio(
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

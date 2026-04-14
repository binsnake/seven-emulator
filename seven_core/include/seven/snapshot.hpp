#pragma once

#include <vector>

#include "seven/memory.hpp"
#include "seven/types.hpp"

namespace seven {

struct MachineSnapshot {
  CpuState state{};
  std::vector<Memory::PageSnapshot> pages;
  std::vector<Memory::MmioRegionSnapshot> mmio_regions;
};

[[nodiscard]] MachineSnapshot capture_snapshot(const CpuState& state, const Memory& memory);
void restore_snapshot(CpuState& state, Memory& memory, const MachineSnapshot& snapshot);
void restore_snapshot(CpuState& state, Memory& memory, const MachineSnapshot& snapshot, const Memory::MmioResolver& mmio_resolver);

}  // namespace seven

#include "seven/snapshot.hpp"

namespace seven {

MachineSnapshot capture_snapshot(const CpuState& state, const Memory& memory) {
  MachineSnapshot snapshot{};
  snapshot.state = state;
  snapshot.pages = memory.snapshot_pages();
  snapshot.mmio_regions = memory.snapshot_mmio_regions();
  return snapshot;
}

void restore_snapshot(CpuState& state, Memory& memory, const MachineSnapshot& snapshot) {
  state = snapshot.state;
  memory.restore_pages(snapshot.pages);
  memory.clear_mmio_regions();
}

void restore_snapshot(CpuState& state, Memory& memory, const MachineSnapshot& snapshot, const Memory::MmioResolver& mmio_resolver) {
  state = snapshot.state;
  memory.restore_pages(snapshot.pages);
  memory.restore_mmio_regions(snapshot.mmio_regions, mmio_resolver);
}

}  // namespace seven

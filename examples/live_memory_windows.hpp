#pragma once

// Windows process-memory passthrough — example only, not compiled into seven_core.
//
// Forwards all Memory reads/writes to a target process via ReadProcessMemory /
// WriteProcessMemory, bypassing the emulator's internal page table entirely.
//
// Usage:
//   WindowsProcessMemory pm{process_handle};
//   memory.set_passthrough(pm.make_read(), pm.make_write());
//
// Notes:
//   - The target process must have PROCESS_VM_READ / PROCESS_VM_WRITE access.
//   - Partial reads (when the remote page is not fully committed) return false,
//     which the emulator treats as a page fault.
//   - Instruction fetch also goes through passthrough, so no pages need to be
//     mapped in the Memory object when using this bridge.

#include <windows.h>
#include "seven/memory.hpp"

struct WindowsProcessMemory {
  HANDLE process_handle = INVALID_HANDLE_VALUE;

  seven::Memory::PassthroughReadFn make_read() {
    return [this](std::uint64_t addr, void* dst, std::size_t size) -> bool {
      SIZE_T bytes_read = 0;
      if (!ReadProcessMemory(process_handle,
                             reinterpret_cast<LPCVOID>(static_cast<ULONG_PTR>(addr)),
                             dst, size, &bytes_read)) {
        return false;
      }
      return bytes_read == size;
    };
  }

  seven::Memory::PassthroughWriteFn make_write() {
    return [this](std::uint64_t addr, const void* src, std::size_t size) -> bool {
      SIZE_T bytes_written = 0;
      if (!WriteProcessMemory(process_handle,
                              reinterpret_cast<LPVOID>(static_cast<ULONG_PTR>(addr)),
                              src, size, &bytes_written)) {
        return false;
      }
      return bytes_written == size;
    };
  }
};

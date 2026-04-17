#pragma once

// Windows live-context bridge — example only, not compiled into seven_core.
//
// Syncs a seven::CpuState to/from a suspended Windows thread's CONTEXT so
// the emulator operates on real register state.
//
// Usage:
//   WindowsContextBridge bridge{thread_handle};
//   executor.set_context_read_callback(bridge.make_read());
//   executor.set_context_write_callback(bridge.make_write());
//
// Notes:
//   - The thread must be suspended before calling step()/run().
//   - CR* and MSR values are NOT available in CONTEXT; the emulator keeps its
//     own CpuState defaults for those.
//   - x87/MMX/XMM/YMM/ZMM state requires CONTEXT_ALL and an extended context
//     area; this bridge handles XMM (CONTEXT_FLOATING_POINT) only.

#include <windows.h>
#include "seven/executor.hpp"
#include "seven/types.hpp"

struct WindowsContextBridge {
  HANDLE thread_handle = INVALID_HANDLE_VALUE;

  seven::Executor::ContextSyncCallback make_read() {
    return [this](seven::CpuState& state) -> bool {
      CONTEXT ctx{};
      ctx.ContextFlags = CONTEXT_FULL | CONTEXT_FLOATING_POINT;
      if (!GetThreadContext(thread_handle, &ctx)) return false;

      state.gpr[0]  = ctx.Rax;
      state.gpr[1]  = ctx.Rcx;
      state.gpr[2]  = ctx.Rdx;
      state.gpr[3]  = ctx.Rbx;
      state.gpr[4]  = ctx.Rsp;
      state.gpr[5]  = ctx.Rbp;
      state.gpr[6]  = ctx.Rsi;
      state.gpr[7]  = ctx.Rdi;
      state.gpr[8]  = ctx.R8;
      state.gpr[9]  = ctx.R9;
      state.gpr[10] = ctx.R10;
      state.gpr[11] = ctx.R11;
      state.gpr[12] = ctx.R12;
      state.gpr[13] = ctx.R13;
      state.gpr[14] = ctx.R14;
      state.gpr[15] = ctx.R15;
      state.rip     = ctx.Rip;
      state.rflags  = ctx.EFlags;

      state.sreg[0] = ctx.SegEs;
      state.sreg[1] = ctx.SegCs;
      state.sreg[2] = ctx.SegSs;
      state.sreg[3] = ctx.SegDs;
      state.sreg[4] = ctx.SegFs;
      state.sreg[5] = ctx.SegGs;

      state.mxcsr = ctx.MxCsr;

      // XMM0–XMM15 → vectors[0]–vectors[15]
      static_assert(sizeof(ctx.Xmm0) == 16);
      const auto* xmm = &ctx.Xmm0;
      for (int i = 0; i < 16; ++i) {
        std::uint64_t lo, hi;
        std::memcpy(&lo, reinterpret_cast<const std::uint8_t*>(xmm + i),     8);
        std::memcpy(&hi, reinterpret_cast<const std::uint8_t*>(xmm + i) + 8, 8);
        state.vectors[i].value = (seven::SimdUint(hi) << 64) | seven::SimdUint(lo);
      }

      return true;
    };
  }

  seven::Executor::ContextSyncCallback make_write() {
    return [this](seven::CpuState& state) -> bool {
      CONTEXT ctx{};
      ctx.ContextFlags = CONTEXT_FULL | CONTEXT_FLOATING_POINT;
      if (!GetThreadContext(thread_handle, &ctx)) return false;

      ctx.Rax    = state.gpr[0];
      ctx.Rcx    = state.gpr[1];
      ctx.Rdx    = state.gpr[2];
      ctx.Rbx    = state.gpr[3];
      ctx.Rsp    = state.gpr[4];
      ctx.Rbp    = state.gpr[5];
      ctx.Rsi    = state.gpr[6];
      ctx.Rdi    = state.gpr[7];
      ctx.R8     = state.gpr[8];
      ctx.R9     = state.gpr[9];
      ctx.R10    = state.gpr[10];
      ctx.R11    = state.gpr[11];
      ctx.R12    = state.gpr[12];
      ctx.R13    = state.gpr[13];
      ctx.R14    = state.gpr[14];
      ctx.R15    = state.gpr[15];
      ctx.Rip    = state.rip;
      ctx.EFlags = static_cast<DWORD>(state.rflags);

      ctx.SegEs  = state.sreg[0];
      ctx.SegCs  = state.sreg[1];
      ctx.SegSs  = state.sreg[2];
      ctx.SegDs  = state.sreg[3];
      ctx.SegFs  = state.sreg[4];
      ctx.SegGs  = state.sreg[5];

      ctx.MxCsr  = state.mxcsr;

      const auto* xmm = &ctx.Xmm0;
      for (int i = 0; i < 16; ++i) {
        const std::uint64_t lo = static_cast<std::uint64_t>(state.vectors[i].value);
        const std::uint64_t hi = static_cast<std::uint64_t>(state.vectors[i].value >> 64);
        std::memcpy(reinterpret_cast<std::uint8_t*>(const_cast<M128A*>(xmm) + i),     &lo, 8);
        std::memcpy(reinterpret_cast<std::uint8_t*>(const_cast<M128A*>(xmm) + i) + 8, &hi, 8);
      }

      return SetThreadContext(thread_handle, &ctx) != FALSE;
    };
  }
};

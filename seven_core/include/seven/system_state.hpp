#pragma once

#include <cstdint>

#include <iced_x86/register.hpp>

#include "seven/types.hpp"

namespace seven {

[[nodiscard]] inline std::uint64_t get_segment_base(const CpuState& state, iced_x86::Register segment) noexcept {
  switch (segment) {
    case iced_x86::Register::FS:
      return state.fs_base;
    case iced_x86::Register::GS:
      return state.gs_base;
    default:
      return 0;
  }
}

inline void set_segment_base(CpuState& state, iced_x86::Register segment, std::uint64_t base) noexcept {
  switch (segment) {
    case iced_x86::Register::FS:
      state.fs_base = base;
      break;
    case iced_x86::Register::GS:
      state.gs_base = base;
      break;
    default:
      break;
  }
}

inline void load_gdt(CpuState& state, std::uint64_t base, std::uint16_t limit) noexcept {
  state.gdtr.base = base;
  state.gdtr.limit = limit;
}

inline void load_idt(CpuState& state, std::uint64_t base, std::uint16_t limit) noexcept {
  state.idtr.base = base;
  state.idtr.limit = limit;
}

[[nodiscard]] inline DescriptorTableRegister gdt(const CpuState& state) noexcept {
  return state.gdtr;
}

[[nodiscard]] inline DescriptorTableRegister idt(const CpuState& state) noexcept {
  return state.idtr;
}

}  // namespace seven


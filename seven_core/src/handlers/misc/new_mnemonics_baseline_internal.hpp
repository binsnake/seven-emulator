#pragma once

#include "seven/handler_helpers.hpp"
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <iced_x86/register.hpp>

namespace seven::handlers {

namespace {
using boost::multiprecision::int128_t;
using boost::multiprecision::uint128_t;

constexpr std::uint64_t width_mask(std::size_t width) {
  return width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
}

iced_x86::Register low_acc_reg(std::size_t width) {
  switch (width) {
    case 1: return iced_x86::Register::AL;
    case 2: return iced_x86::Register::AX;
    case 4: return iced_x86::Register::EAX;
    default: return iced_x86::Register::RAX;
  }
}

iced_x86::Register high_acc_reg(std::size_t width) {
  switch (width) {
    case 1: return iced_x86::Register::AH;
    case 2: return iced_x86::Register::DX;
    case 4: return iced_x86::Register::EDX;
    default: return iced_x86::Register::RDX;
  }
}

int128_t sign_extend_to_i128(std::uint64_t value, std::size_t width) {
  return static_cast<int64_t>(sign_extend(value, width));
}

void write_width_reg(std::uint64_t& full, std::size_t width, std::uint64_t value) {
  const auto mask = width_mask(width);
  full = (full & ~mask) | (value & mask);
}

void update_szp_flags(CpuState& state, std::uint8_t value) {
  state.rflags &= ~(kFlagSF | kFlagZF | kFlagPF);
  if (value == 0) state.rflags |= kFlagZF;
  if ((value & 0x80u) != 0) state.rflags |= kFlagSF;
  if (even_parity(value)) state.rflags |= kFlagPF;
}

}  // namespace

}  // namespace seven::handlers


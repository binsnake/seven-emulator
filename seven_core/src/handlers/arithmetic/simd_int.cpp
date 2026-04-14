#include <bit>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <boost/multiprecision/cpp_int.hpp>
#include <iced_x86/memory_size_info.hpp>

#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

using big_uint = seven::SimdUint;

constexpr std::size_t kXmmBytes = 16;
constexpr std::size_t kYmmBytes = 32;
constexpr std::size_t kZmmBytes = 64;

bool is_vector_register(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  return (value >= xmm0 && value < xmm0 + 32) || (value >= ymm0 && value < ymm0 + 32) || (value >= zmm0 && value < zmm0 + 32);
}

std::size_t vector_index(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= xmm0 && value < xmm0 + 32) return static_cast<std::size_t>(value - xmm0);
  if (value >= ymm0 && value < ymm0 + 32) return static_cast<std::size_t>(value - ymm0);
  if (value >= zmm0 && value < zmm0 + 32) return static_cast<std::size_t>(value - zmm0);
  return 0;
}

std::size_t vector_width(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= zmm0 && value < zmm0 + 32) return kZmmBytes;
  if (value >= ymm0 && value < ymm0 + 32) return kYmmBytes;
  if (value >= xmm0 && value < xmm0 + 32) return kXmmBytes;
  return kXmmBytes;
}

big_uint mask(std::size_t bytes) {
  if (bytes >= sizeof(big_uint)) return ~big_uint(0);
  return (big_uint(1) << (bytes * 8)) - 1;
}

big_uint read_vec(const CpuState& state, iced_x86::Register reg) {
  return state.vectors[vector_index(reg)].value & mask(vector_width(reg));
}

void write_vec(CpuState& state, iced_x86::Register reg, big_uint value, bool zero_upper = false) {
  auto& slot = state.vectors[vector_index(reg)].value;
  const auto m = mask(vector_width(reg));
  if (zero_upper) {
    slot = value & m;
  } else {
    slot = (slot & ~m) | (value & m);
  }
}

big_uint read_mem(ExecutionContext& ctx, std::uint64_t address, std::size_t width, bool* ok) {
  std::array<std::uint8_t, kZmmBytes> bytes{};
  if (!ctx.memory.read(address, bytes.data(), width)) {
    if (ok) *ok = false;
    return 0;
  }
  if (ok) *ok = true;
  big_uint value = 0;
  for (std::size_t i = 0; i < width; ++i) {
    value |= (big_uint(bytes[i]) << (8 * i));
  }
  return value;
}

big_uint read_operand(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, bool* ok) {
  const auto kind = ctx.instr.op_kind(operand_index);
  if (kind == iced_x86::OpKind::REGISTER) {
    const auto reg = ctx.instr.op_register(operand_index);
    if (!is_vector_register(reg)) {
      if (ok) *ok = false;
      return 0;
    }
    if (ok) *ok = true;
    return read_vec(ctx.state, reg) & mask(width);
  }
  if (kind == iced_x86::OpKind::MEMORY) {
    if (ctx.instr.is_broadcast()) {
      const auto element_width = iced_x86::memory_size_ext::get_size(static_cast<iced_x86::MemorySize>(ctx.instr.memory_size()));
      const auto element = read_mem(ctx, detail::memory_address(ctx), element_width, ok);
      if (ok && !*ok) return 0;
      big_uint out = 0;
      const auto lane_value = element & mask(element_width);
      for (std::size_t lane = 0; lane < width; lane += element_width) {
        out |= lane_value << (lane * 8);
      }
      if (ok) *ok = true;
      return out;
    }
    return read_mem(ctx, detail::memory_address(ctx), width, ok);
  }
  if (ok) *ok = false;
  return 0;
}

std::uint64_t read_opmask(const CpuState& state, iced_x86::Register reg) {
  if (reg == iced_x86::Register::NONE || reg == iced_x86::Register::K0) return ~std::uint64_t{0};
  if (reg >= iced_x86::Register::K1 && reg <= iced_x86::Register::K7) {
    return state.opmask[static_cast<std::size_t>(reg) - static_cast<std::size_t>(iced_x86::Register::K0)];
  }
  return ~std::uint64_t{0};
}


template <typename T>
T lane_load(const big_uint& value, std::size_t lane_offset_bytes) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> ((lane_offset_bytes + i) * 8)) & 0xFFu);
  }
  T out{};
  std::memcpy(&out, bytes.data(), sizeof(T));
  return out;
}

template <typename T>
void lane_store(big_uint& value, std::size_t lane_offset_bytes, T lane_value) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &lane_value, sizeof(T));
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    const auto bit_offset = (lane_offset_bytes + i) * 8;
    value &= ~(big_uint(0xFFu) << bit_offset);
    value |= (big_uint(bytes[i]) << bit_offset);
  }
}

template <typename T>
big_uint apply_masked_lanes(ExecutionContext& ctx, iced_x86::Register dst_reg, big_uint original, big_uint computed) {
  const auto opmask = ctx.instr.op_mask();
  if (opmask == iced_x86::Register::NONE || opmask == iced_x86::Register::K0) return computed;

  big_uint out = original & mask(vector_width(dst_reg));
  const auto lane_mask = read_opmask(ctx.state, opmask);
  const auto zeroing = ctx.instr.zeroing_masking();
  std::size_t lane_index = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T), ++lane_index) {
    const auto active = ((lane_mask >> lane_index) & 1ull) != 0ull;
    if (active) {
      lane_store(out, lane, lane_load<T>(computed, lane));
    } else if (zeroing) {
      lane_store(out, lane, T{});
    }
  }
  return out;
}

template <typename T>
T all_ones() {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  bytes.fill(0xFFu);
  T out{};
  std::memcpy(&out, bytes.data(), sizeof(T));
  return out;
}

template <typename T>
T sat_add_unsigned(T lhs, T rhs) {
  const auto max = std::numeric_limits<T>::max();
  return (max - lhs < rhs) ? max : static_cast<T>(lhs + rhs);
}

template <typename T>
T sat_add_signed(T lhs, T rhs) {
  const auto max = std::numeric_limits<T>::max();
  const auto min = std::numeric_limits<T>::min();
  if (rhs > 0 && lhs > static_cast<T>(max - rhs)) return max;
  if (rhs < 0 && lhs < static_cast<T>(min - rhs)) return min;
  return static_cast<T>(lhs + rhs);
}

template <typename T>
T sat_sub_unsigned(T lhs, T rhs) {
  return lhs < rhs ? T{0} : static_cast<T>(lhs - rhs);
}

template <typename T>
T sat_sub_signed(T lhs, T rhs) {
  const auto max = std::numeric_limits<T>::max();
  const auto min = std::numeric_limits<T>::min();
  if (rhs > 0 && lhs < static_cast<T>(min + rhs)) return min;
  if (rhs < 0 && lhs > static_cast<T>(max + rhs)) return max;
  return static_cast<T>(lhs - rhs);
}

template <typename T>
T avg_unsigned(T lhs, T rhs) {
  return static_cast<T>((static_cast<std::make_unsigned_t<T>>(lhs) + static_cast<std::make_unsigned_t<T>>(rhs) + 1u) >> 1);
}

template <typename Fn>
ExecutionResult legacy_custom_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  write_vec(ctx.state, dst_reg, fn(lhs_bits, rhs_bits, vector_width(dst_reg)), zero_upper);
  return {};
}


template <typename T, typename Fn>
ExecutionResult legacy_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = lhs_bits;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    const auto lhs = lane_load<T>(lhs_bits, lane);
    const auto rhs = lane_load<T>(rhs_bits, lane);
    lane_store(out, lane, fn(lhs, rhs));
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto original = read_vec(ctx.state, dst_reg);
  const auto lhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto rhs_bits = read_operand(ctx, 2, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    const auto lhs = lane_load<T>(lhs_bits, lane);
    const auto rhs = lane_load<T>(rhs_bits, lane);
    lane_store(out, lane, fn(lhs, rhs));
  }
  out = apply_masked_lanes<T>(ctx, dst_reg, original, out);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult legacy_compare(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    const auto lhs = lane_load<T>(lhs_bits, lane);
    const auto rhs = lane_load<T>(rhs_bits, lane);
    lane_store(out, lane, fn(lhs, rhs));
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_compare(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto original = read_vec(ctx.state, dst_reg);
  const auto lhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto rhs_bits = read_operand(ctx, 2, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    const auto lhs = lane_load<T>(lhs_bits, lane);
    const auto rhs = lane_load<T>(rhs_bits, lane);
    lane_store(out, lane, fn(lhs, rhs));
  }
  out = apply_masked_lanes<T>(ctx, dst_reg, original, out);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult legacy_bitwise(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs = read_vec(ctx.state, dst_reg);
  const auto rhs = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  write_vec(ctx.state, dst_reg, fn(lhs, rhs), zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_bitwise(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto original = read_vec(ctx.state, dst_reg);
  const auto lhs = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto rhs = read_operand(ctx, 2, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto computed = fn(lhs, rhs);
  if constexpr (std::is_void_v<T>) {
    write_vec(ctx.state, dst_reg, computed, zero_upper);
  } else {
    write_vec(ctx.state, dst_reg, apply_masked_lanes<T>(ctx, dst_reg, original, computed), zero_upper);
  }
  return {};
}

template <typename T>
T shift_left_lane(T value, unsigned count) {
  if (count >= sizeof(T) * 8) return T{0};
  using u = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<u>(value) << count);
}

template <typename T>
T shift_right_logical_lane(T value, unsigned count) {
  if (count >= sizeof(T) * 8) return T{0};
  using u = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<u>(value) >> count);
}

template <typename T>
T shift_right_arithmetic_lane(T value, unsigned count) {
  if (count >= sizeof(T) * 8) return value < 0 ? static_cast<T>(-1) : T{0};
  return static_cast<T>(value >> count);
}

template <typename T, typename Fn>
ExecutionResult legacy_shift_reg(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (ctx.instr.op_kind(1) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(1))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto count_bits = read_vec(ctx.state, ctx.instr.op_register(1));
  const auto count = static_cast<unsigned>(count_bits & 0xFFu);
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_vec(ctx.state, dst_reg);
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    lane_store(out, lane, fn(lane_load<T>(src_bits, lane), count));
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult legacy_shift_imm(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto count = static_cast<unsigned>(ctx.instr.immediate8() & 0xFFu);
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_vec(ctx.state, dst_reg);
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    lane_store(out, lane, fn(lane_load<T>(src_bits, lane), count));
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_shift_imm(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto original = read_vec(ctx.state, dst_reg);
  const auto src_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto count = static_cast<unsigned>(ctx.instr.immediate8() & 0xFFu);
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    lane_store(out, lane, fn(lane_load<T>(src_bits, lane), count));
  }
  out = apply_masked_lanes<T>(ctx, dst_reg, original, out);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_shift_reg(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (ctx.instr.op_kind(1) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(1))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto count_bits = read_operand(ctx, 2, sizeof(std::uint64_t), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto count = static_cast<unsigned>(count_bits & 0xFFu);
  const auto dst_reg = ctx.instr.op_register(0);
  const auto original = read_vec(ctx.state, dst_reg);
  const auto src_bits = read_vec(ctx.state, ctx.instr.op_register(1));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += sizeof(T)) {
    lane_store(out, lane, fn(lane_load<T>(src_bits, lane), count));
  }
  out = apply_masked_lanes<T>(ctx, dst_reg, original, out);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

ExecutionResult legacy_shift_dq_imm(ExecutionContext& ctx, bool shift_right, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_vec(ctx.state, dst_reg);
  const auto width = vector_width(dst_reg);
  const auto count = static_cast<unsigned>(ctx.instr.immediate8()) & 0xFFu;
  const big_uint lane_mask = (big_uint(1) << 128) - 1;
  big_uint out = 0;
  for (std::size_t lane = 0; lane < width; lane += 16) {
    const auto bit_offset = lane * 8;
    const auto lane_bits = (src_bits >> bit_offset) & lane_mask;
    big_uint shifted = 0;
    if (count < 16u) {
      const auto shift_bits = count * 8u;
      shifted = shift_right ? (lane_bits >> shift_bits) : ((lane_bits << shift_bits) & lane_mask);
    }
    out |= (shifted & lane_mask) << bit_offset;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

ExecutionResult vex_shift_dq_imm(ExecutionContext& ctx, bool shift_right, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto width = vector_width(dst_reg);
  const auto src_bits = read_operand(ctx, 1, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto count = static_cast<unsigned>(ctx.instr.immediate8()) & 0xFFu;
  const big_uint lane_mask = (big_uint(1) << 128) - 1;
  big_uint out = 0;
  for (std::size_t lane = 0; lane < width; lane += 16) {
    const auto bit_offset = lane * 8;
    const auto lane_bits = (src_bits >> bit_offset) & lane_mask;
    big_uint shifted = 0;
    if (count < 16u) {
      const auto shift_bits = count * 8u;
      shifted = shift_right ? (lane_bits >> shift_bits) : ((lane_bits << shift_bits) & lane_mask);
    }
    out |= (shifted & lane_mask) << bit_offset;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

struct PcmpSource {
  std::array<std::int32_t, 16> values{};
  std::array<std::uint32_t, 16> raw{};
};

std::size_t pcmp_element_width(unsigned imm) {
  return (imm & 0x1u) != 0u ? 2u : 1u;
}

std::size_t pcmp_element_count(unsigned imm) {
  return pcmp_element_width(imm) == 2u ? 8u : 16u;
}

PcmpSource pcmp_extract_source(const big_uint& bits, unsigned imm) {
  PcmpSource out{};
  const auto elem_width = pcmp_element_width(imm);
  const auto elem_count = pcmp_element_count(imm);
  const auto signed_mode = (imm & 0x2u) != 0u;
  for (std::size_t i = 0; i < elem_count; ++i) {
    const auto offset = i * elem_width;
    if (elem_width == 1u) {
      const auto raw = lane_load<std::uint8_t>(bits, offset);
      out.raw[i] = raw;
      out.values[i] = signed_mode ? static_cast<std::int8_t>(raw) : static_cast<std::int32_t>(raw);
    } else {
      const auto raw = lane_load<std::uint16_t>(bits, offset);
      out.raw[i] = raw;
      out.values[i] = signed_mode ? static_cast<std::int16_t>(raw) : static_cast<std::int32_t>(raw);
    }
  }
  return out;
}

std::size_t pcmp_explicit_length(std::uint64_t raw, std::size_t elem_count) {
  const auto signed_length = static_cast<std::int32_t>(raw & 0xFFFFFFFFu);
  std::uint64_t magnitude = 0;
  if (signed_length == std::numeric_limits<std::int32_t>::min()) {
    magnitude = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) + 1ull;
  } else {
    magnitude = static_cast<std::uint64_t>(signed_length < 0 ? -signed_length : signed_length);
  }
  return static_cast<std::size_t>(std::min<std::uint64_t>(magnitude, elem_count));
}

std::size_t pcmp_implicit_length(const PcmpSource& src, std::size_t elem_count) {
  std::size_t length = 0;
  while (length < elem_count && src.raw[length] != 0u) ++length;
  return length;
}

std::uint32_t pcmp_equal_any(const PcmpSource& lhs, const PcmpSource& rhs, std::size_t len1, std::size_t len2) {
  std::uint32_t res = 0;
  for (std::size_t j = 0; j < len2; ++j) {
    bool match = false;
    for (std::size_t i = 0; i < len1; ++i) {
      if (lhs.values[i] == rhs.values[j]) {
        match = true;
        break;
      }
    }
    if (match) res |= (1u << j);
  }
  return res;
}

std::uint32_t pcmp_ranges(const PcmpSource& lhs, const PcmpSource& rhs, std::size_t len1, std::size_t len2) {
  std::uint32_t res = 0;
  for (std::size_t j = 0; j < len2; ++j) {
    bool match = false;
    for (std::size_t i = 0; i + 1 < len1; i += 2) {
      if (lhs.values[i] <= rhs.values[j] && rhs.values[j] <= lhs.values[i + 1]) {
        match = true;
        break;
      }
    }
    if (match) res |= (1u << j);
  }
  return res;
}

std::uint32_t pcmp_equal_each(const PcmpSource& lhs, const PcmpSource& rhs, std::size_t len1, std::size_t len2, std::size_t elem_count) {
  std::uint32_t res = 0;
  for (std::size_t i = 0; i < elem_count; ++i) {
    bool match = false;
    if (i < len1 && i < len2) match = lhs.values[i] == rhs.values[i];
    else if (i >= len1 && i >= len2) match = true;
    if (match) res |= (1u << i);
  }
  return res;
}

std::uint32_t pcmp_equal_ordered(const PcmpSource& lhs, const PcmpSource& rhs, std::size_t len1, std::size_t len2, std::size_t elem_count) {
  std::uint32_t res = 0;
  for (std::size_t j = 0; j < elem_count; ++j) {
    bool match = true;
    for (std::size_t i = 0; i < elem_count; ++i) {
      if (i >= len1) break;
      const auto rhs_index = j + i;
      if (rhs_index >= len2 || rhs_index >= elem_count || lhs.values[i] != rhs.values[rhs_index]) {
        match = false;
        break;
      }
    }
    if (match) res |= (1u << j);
  }
  return res;
}

std::uint32_t pcmp_apply_polarity(unsigned imm, std::uint32_t intres1, std::size_t len2, std::size_t elem_count) {
  const auto full_mask = (1u << elem_count) - 1u;
  switch ((imm >> 4) & 0x3u) {
    case 0:
    case 2:
      return intres1 & full_mask;
    case 1:
      return (~intres1) & full_mask;
    default: {
      const auto valid_mask = len2 >= elem_count ? full_mask : ((1u << len2) - 1u);
      return (((~intres1) & valid_mask) | (intres1 & ~valid_mask)) & full_mask;
    }
  }
}

void set_pcmpstr_flags(CpuState& state, std::uint32_t intres2, std::size_t len1, std::size_t len2, std::size_t elem_count) {
  std::uint64_t rf = state.rflags;
  rf &= ~(kFlagCF | kFlagPF | kFlagAF | kFlagZF | kFlagSF | kFlagOF);
  if (intres2 != 0u) rf |= kFlagCF;
  if (len2 < elem_count) rf |= kFlagZF;
  if (len1 < elem_count) rf |= kFlagSF;
  if ((intres2 & 1u) != 0u) rf |= kFlagOF;
  state.rflags = rf;
}

ExecutionResult pcmpstr(ExecutionContext& ctx, bool explicit_lengths, bool return_mask) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto imm = static_cast<unsigned>(ctx.instr.immediate8());
  const auto elem_width = pcmp_element_width(imm);
  const auto elem_count = pcmp_element_count(imm);
  bool ok = false;
  const auto lhs_bits = read_vec(ctx.state, ctx.instr.op_register(0));
  const auto rhs_bits = read_operand(ctx, 1, 16, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto lhs = pcmp_extract_source(lhs_bits, imm);
  const auto rhs = pcmp_extract_source(rhs_bits, imm);
  const auto len1 = explicit_lengths ? pcmp_explicit_length(ctx.state.gpr[0], elem_count) : pcmp_implicit_length(lhs, elem_count);
  const auto len2 = explicit_lengths ? pcmp_explicit_length(ctx.state.gpr[2], elem_count) : pcmp_implicit_length(rhs, elem_count);

  std::uint32_t intres1 = 0;
  switch ((imm >> 2) & 0x3u) {
    case 0: intres1 = pcmp_equal_any(lhs, rhs, len1, len2); break;
    case 1: intres1 = pcmp_ranges(lhs, rhs, len1, len2); break;
    case 2: intres1 = pcmp_equal_each(lhs, rhs, len1, len2, elem_count); break;
    default: intres1 = pcmp_equal_ordered(lhs, rhs, len1, len2, elem_count); break;
  }
  const auto intres2 = pcmp_apply_polarity(imm, intres1, len2, elem_count);
  set_pcmpstr_flags(ctx.state, intres2, len1, len2, elem_count);

  if (return_mask) {
    big_uint out = 0;
    if ((imm & 0x40u) == 0u) {
      out = intres2;
    } else {
      for (std::size_t i = 0; i < elem_count; ++i) {
        if (((intres2 >> i) & 1u) == 0u) continue;
        if (elem_width == 1u) lane_store(out, i, all_ones<std::uint8_t>());
        else lane_store(out, i * elem_width, all_ones<std::uint16_t>());
      }
    }
    write_vec(ctx.state, iced_x86::Register::XMM0, out, false);
    return {};
  }

  std::uint64_t index = elem_count;
  if (intres2 != 0u) {
    if ((imm & 0x40u) == 0u) {
      index = static_cast<std::uint64_t>(std::countr_zero(intres2));
    } else {
      for (std::size_t i = elem_count; i-- > 0;) {
        if (((intres2 >> i) & 1u) != 0u) {
          index = i;
          break;
        }
      }
    }
  }
  detail::write_register(ctx.state, iced_x86::Register::ECX, index, 4);
  return {};
}

}  // namespace

#define KUBERA_LEGACY_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return legacy_binary<type>(ctx, fn); }
#define KUBERA_LEGACY_CMP(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return legacy_compare<type>(ctx, fn); }
#define KUBERA_LEGACY_BIT(name, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return legacy_bitwise<void>(ctx, fn); }
#define KUBERA_VEX_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_binary<type>(ctx, fn, true); }
#define KUBERA_VEX_CMP(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_compare<type>(ctx, fn, true); }
#define KUBERA_VEX_BIT(name, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_bitwise<void>(ctx, fn, true); }

KUBERA_LEGACY_BIN(PADDB_XMM_XMMM128, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a + b); })
KUBERA_LEGACY_BIN(PADDW_XMM_XMMM128, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a + b); })
KUBERA_LEGACY_BIN(PADDD_XMM_XMMM128, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_LEGACY_BIN(PADDQ_XMM_XMMM128, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_LEGACY_BIN(PADDUSB_XMM_XMMM128, std::uint8_t, sat_add_unsigned<std::uint8_t>)
KUBERA_LEGACY_BIN(PADDUSW_XMM_XMMM128, std::uint16_t, sat_add_unsigned<std::uint16_t>)
KUBERA_LEGACY_BIN(PADDSB_XMM_XMMM128, std::int8_t, sat_add_signed<std::int8_t>)
KUBERA_LEGACY_BIN(PADDSW_XMM_XMMM128, std::int16_t, sat_add_signed<std::int16_t>)
KUBERA_LEGACY_BIN(PSUBB_XMM_XMMM128, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a - b); })
KUBERA_LEGACY_BIN(PSUBW_XMM_XMMM128, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a - b); })
KUBERA_LEGACY_BIN(PSUBD_XMM_XMMM128, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_LEGACY_BIN(PSUBQ_XMM_XMMM128, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_LEGACY_BIN(PSUBUSB_XMM_XMMM128, std::uint8_t, sat_sub_unsigned<std::uint8_t>)
KUBERA_LEGACY_BIN(PSUBUSW_XMM_XMMM128, std::uint16_t, sat_sub_unsigned<std::uint16_t>)
KUBERA_LEGACY_BIN(PSUBSB_XMM_XMMM128, std::int8_t, sat_sub_signed<std::int8_t>)
KUBERA_LEGACY_BIN(PSUBSW_XMM_XMMM128, std::int16_t, sat_sub_signed<std::int16_t>)
KUBERA_LEGACY_CMP(PCMPEQB_XMM_XMMM128, std::uint8_t, [](auto a, auto b) { return a == b ? all_ones<std::uint8_t>() : std::uint8_t{0}; })
KUBERA_LEGACY_CMP(PCMPEQW_XMM_XMMM128, std::uint16_t, [](auto a, auto b) { return a == b ? all_ones<std::uint16_t>() : std::uint16_t{0}; })
KUBERA_LEGACY_CMP(PCMPEQD_XMM_XMMM128, std::uint32_t, [](auto a, auto b) { return a == b ? all_ones<std::uint32_t>() : std::uint32_t{0}; })
KUBERA_LEGACY_CMP(PCMPEQQ_XMM_XMMM128, std::uint64_t, [](auto a, auto b) { return a == b ? all_ones<std::uint64_t>() : std::uint64_t{0}; })
KUBERA_LEGACY_CMP(PCMPGTB_XMM_XMMM128, std::int8_t, [](auto a, auto b) { return a > b ? all_ones<std::int8_t>() : std::int8_t{0}; })
KUBERA_LEGACY_CMP(PCMPGTW_XMM_XMMM128, std::int16_t, [](auto a, auto b) { return a > b ? all_ones<std::int16_t>() : std::int16_t{0}; })
KUBERA_LEGACY_CMP(PCMPGTD_XMM_XMMM128, std::int32_t, [](auto a, auto b) { return a > b ? all_ones<std::int32_t>() : std::int32_t{0}; })
KUBERA_LEGACY_CMP(PCMPGTQ_XMM_XMMM128, std::int64_t, [](auto a, auto b) { return a > b ? all_ones<std::int64_t>() : std::int64_t{0}; })
KUBERA_LEGACY_BIN(PAVGB_XMM_XMMM128, std::uint8_t, avg_unsigned<std::uint8_t>)
KUBERA_LEGACY_BIN(PAVGW_XMM_XMMM128, std::uint16_t, avg_unsigned<std::uint16_t>)
KUBERA_LEGACY_BIN(PMULLW_XMM_XMMM128, std::uint16_t, [](auto a, auto b) {
  return static_cast<std::uint16_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(a)) * static_cast<std::int32_t>(static_cast<std::int16_t>(b)));
})
ExecutionResult handle_code_PMULHW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary<std::int16_t>(ctx, [](auto a, auto b) {
    return static_cast<std::int16_t>((static_cast<std::int32_t>(a) * static_cast<std::int32_t>(b)) >> 16);
  });
}
ExecutionResult handle_code_PMULHUW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary<std::uint16_t>(ctx, [](auto a, auto b) {
    return static_cast<std::uint16_t>((static_cast<std::uint32_t>(a) * static_cast<std::uint32_t>(b)) >> 16);
  });
}
ExecutionResult handle_code_PMULUDQ_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_custom_binary(ctx, [](const big_uint& lhs_bits, const big_uint& rhs_bits, std::size_t width) {
    big_uint out = 0;
    for (std::size_t lane = 0; lane < width; lane += sizeof(std::uint64_t)) {
      const auto lhs = lane_load<std::uint32_t>(lhs_bits, lane);
      const auto rhs = lane_load<std::uint32_t>(rhs_bits, lane);
      lane_store(out, lane, static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs));
    }
    return out;
  });
}
ExecutionResult handle_code_PMADDWD_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_custom_binary(ctx, [](const big_uint& lhs_bits, const big_uint& rhs_bits, std::size_t width) {
    big_uint out = 0;
    for (std::size_t lane = 0; lane < width; lane += sizeof(std::uint32_t)) {
      const auto lhs_lo = static_cast<std::int32_t>(lane_load<std::int16_t>(lhs_bits, lane));
      const auto lhs_hi = static_cast<std::int32_t>(lane_load<std::int16_t>(lhs_bits, lane + sizeof(std::int16_t)));
      const auto rhs_lo = static_cast<std::int32_t>(lane_load<std::int16_t>(rhs_bits, lane));
      const auto rhs_hi = static_cast<std::int32_t>(lane_load<std::int16_t>(rhs_bits, lane + sizeof(std::int16_t)));
      const auto sum = static_cast<std::int64_t>(lhs_lo) * static_cast<std::int64_t>(rhs_lo) +
                       static_cast<std::int64_t>(lhs_hi) * static_cast<std::int64_t>(rhs_hi);
      lane_store(out, lane, static_cast<std::uint32_t>(static_cast<std::uint64_t>(sum) & 0xFFFFFFFFull));
    }
    return out;
  });
}
ExecutionResult handle_code_PSADBW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_custom_binary(ctx, [](const big_uint& lhs_bits, const big_uint& rhs_bits, std::size_t width) {
    big_uint out = 0;
    for (std::size_t lane = 0; lane < width; lane += sizeof(std::uint64_t)) {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
        const auto lhs = lane_load<std::uint8_t>(lhs_bits, lane + i);
        const auto rhs = lane_load<std::uint8_t>(rhs_bits, lane + i);
        sum += lhs > rhs ? static_cast<std::uint64_t>(lhs - rhs) : static_cast<std::uint64_t>(rhs - lhs);
      }
      lane_store(out, lane, sum);
    }
    return out;
  });
}
ExecutionResult handle_code_PMAXUB_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary<std::uint8_t>(ctx, [](auto a, auto b) { return std::max(a, b); });
}
ExecutionResult handle_code_PMINSW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary<std::int16_t>(ctx, [](auto a, auto b) { return std::min(a, b); });
}

ExecutionResult handle_code_PAND_XMM_XMMM128(ExecutionContext& ctx) { return legacy_bitwise<void>(ctx, [](auto a, auto b) { return a & b; }); }
ExecutionResult handle_code_PANDN_XMM_XMMM128(ExecutionContext& ctx) { return legacy_bitwise<void>(ctx, [](auto a, auto b) { return (~a) & b; }); }
ExecutionResult handle_code_POR_XMM_XMMM128(ExecutionContext& ctx) { return legacy_bitwise<void>(ctx, [](auto a, auto b) { return a | b; }); }
ExecutionResult handle_code_PXOR_XMM_XMMM128(ExecutionContext& ctx) { return legacy_bitwise<void>(ctx, [](auto a, auto b) { return a ^ b; }); }
ExecutionResult handle_code_PSLLW_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::uint16_t>(ctx, shift_left_lane<std::uint16_t>); }
ExecutionResult handle_code_PSLLD_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::uint32_t>(ctx, shift_left_lane<std::uint32_t>); }
ExecutionResult handle_code_PSLLQ_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::uint64_t>(ctx, shift_left_lane<std::uint64_t>); }
ExecutionResult handle_code_PSRLW_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::uint16_t>(ctx, shift_right_logical_lane<std::uint16_t>); }
ExecutionResult handle_code_PSRLD_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::uint32_t>(ctx, shift_right_logical_lane<std::uint32_t>); }
ExecutionResult handle_code_PSRLQ_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::uint64_t>(ctx, shift_right_logical_lane<std::uint64_t>); }
ExecutionResult handle_code_PSRAW_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::int16_t>(ctx, shift_right_arithmetic_lane<std::int16_t>); }
ExecutionResult handle_code_PSRAD_XMM_XMMM128(ExecutionContext& ctx) { return legacy_shift_reg<std::int32_t>(ctx, shift_right_arithmetic_lane<std::int32_t>); }
ExecutionResult handle_code_PSLLW_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::uint16_t>(ctx, shift_left_lane<std::uint16_t>); }
ExecutionResult handle_code_PSLLD_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::uint32_t>(ctx, shift_left_lane<std::uint32_t>); }
ExecutionResult handle_code_PSLLQ_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::uint64_t>(ctx, shift_left_lane<std::uint64_t>); }
ExecutionResult handle_code_PSRLW_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::uint16_t>(ctx, shift_right_logical_lane<std::uint16_t>); }
ExecutionResult handle_code_PSRLD_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::uint32_t>(ctx, shift_right_logical_lane<std::uint32_t>); }
ExecutionResult handle_code_PSRLQ_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::uint64_t>(ctx, shift_right_logical_lane<std::uint64_t>); }
ExecutionResult handle_code_PSRAW_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::int16_t>(ctx, shift_right_arithmetic_lane<std::int16_t>); }
ExecutionResult handle_code_PSRAD_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_imm<std::int32_t>(ctx, shift_right_arithmetic_lane<std::int32_t>); }
ExecutionResult handle_code_PSRLDQ_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_dq_imm(ctx, true); }
ExecutionResult handle_code_PSLLDQ_XMM_IMM8(ExecutionContext& ctx) { return legacy_shift_dq_imm(ctx, false); }

#define KUBERA_VEX_PACKED_INT_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_binary<type>(ctx, fn, true); }
#define KUBERA_VEX_PACKED_INT_CMP(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_compare<type>(ctx, fn, true); }
#define KUBERA_VEX_PACKED_INT_BIT(name, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_bitwise<void>(ctx, fn, true); }

// XMM VEX
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDB_XMM_XMM_XMMM128, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDW_XMM_XMM_XMMM128, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDD_XMM_XMM_XMMM128, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDQ_XMM_XMM_XMMM128, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDUSB_XMM_XMM_XMMM128, std::uint8_t, sat_add_unsigned<std::uint8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDUSW_XMM_XMM_XMMM128, std::uint16_t, sat_add_unsigned<std::uint16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDSB_XMM_XMM_XMMM128, std::int8_t, sat_add_signed<std::int8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDSW_XMM_XMM_XMMM128, std::int16_t, sat_add_signed<std::int16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBB_XMM_XMM_XMMM128, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBW_XMM_XMM_XMMM128, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBD_XMM_XMM_XMMM128, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBQ_XMM_XMM_XMMM128, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBUSB_XMM_XMM_XMMM128, std::uint8_t, sat_sub_unsigned<std::uint8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBUSW_XMM_XMM_XMMM128, std::uint16_t, sat_sub_unsigned<std::uint16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBSB_XMM_XMM_XMMM128, std::int8_t, sat_sub_signed<std::int8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBSW_XMM_XMM_XMMM128, std::int16_t, sat_sub_signed<std::int16_t>)
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQB_XMM_XMM_XMMM128, std::uint8_t, [](auto a, auto b) { return a == b ? all_ones<std::uint8_t>() : std::uint8_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQW_XMM_XMM_XMMM128, std::uint16_t, [](auto a, auto b) { return a == b ? all_ones<std::uint16_t>() : std::uint16_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQD_XMM_XMM_XMMM128, std::uint32_t, [](auto a, auto b) { return a == b ? all_ones<std::uint32_t>() : std::uint32_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQQ_XMM_XMM_XMMM128, std::uint64_t, [](auto a, auto b) { return a == b ? all_ones<std::uint64_t>() : std::uint64_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTB_XMM_XMM_XMMM128, std::int8_t, [](auto a, auto b) { return a > b ? all_ones<std::int8_t>() : std::int8_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTW_XMM_XMM_XMMM128, std::int16_t, [](auto a, auto b) { return a > b ? all_ones<std::int16_t>() : std::int16_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTD_XMM_XMM_XMMM128, std::int32_t, [](auto a, auto b) { return a > b ? all_ones<std::int32_t>() : std::int32_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTQ_XMM_XMM_XMMM128, std::int64_t, [](auto a, auto b) { return a > b ? all_ones<std::int64_t>() : std::int64_t{0}; })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPAVGB_XMM_XMM_XMMM128, std::uint8_t, avg_unsigned<std::uint8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPAVGW_XMM_XMM_XMMM128, std::uint16_t, avg_unsigned<std::uint16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPMULLW_XMM_XMM_XMMM128, std::uint16_t, [](auto a, auto b) {
  return static_cast<std::uint16_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(a)) * static_cast<std::int32_t>(static_cast<std::int16_t>(b)));
})
KUBERA_VEX_PACKED_INT_BIT(VEX_VPAND_XMM_XMM_XMMM128, [](auto a, auto b) { return a & b; })
KUBERA_VEX_PACKED_INT_BIT(VEX_VPANDN_XMM_XMM_XMMM128, [](auto a, auto b) { return (~a) & b; })
KUBERA_VEX_PACKED_INT_BIT(VEX_VPOR_XMM_XMM_XMMM128, [](auto a, auto b) { return a | b; })
KUBERA_VEX_PACKED_INT_BIT(VEX_VPXOR_XMM_XMM_XMMM128, [](auto a, auto b) { return a ^ b; })
ExecutionResult handle_code_VEX_VPSLLW_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::uint16_t>(ctx, shift_left_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSLLD_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::uint32_t>(ctx, shift_left_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSLLQ_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::uint64_t>(ctx, shift_left_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRLW_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::uint16_t>(ctx, shift_right_logical_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSRLD_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::uint32_t>(ctx, shift_right_logical_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSRLQ_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::uint64_t>(ctx, shift_right_logical_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRAW_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::int16_t>(ctx, shift_right_arithmetic_lane<std::int16_t>, true); }
ExecutionResult handle_code_VEX_VPSRAD_XMM_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<std::int32_t>(ctx, shift_right_arithmetic_lane<std::int32_t>, true); }
ExecutionResult handle_code_VEX_VPSLLW_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint16_t>(ctx, shift_left_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSLLD_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint32_t>(ctx, shift_left_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSLLQ_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint64_t>(ctx, shift_left_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRLW_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint16_t>(ctx, shift_right_logical_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSRLD_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint32_t>(ctx, shift_right_logical_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSRLQ_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint64_t>(ctx, shift_right_logical_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRAW_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::int16_t>(ctx, shift_right_arithmetic_lane<std::int16_t>, true); }
ExecutionResult handle_code_VEX_VPSRAD_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::int32_t>(ctx, shift_right_arithmetic_lane<std::int32_t>, true); }
ExecutionResult handle_code_VEX_VPSRLDQ_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_dq_imm(ctx, true, true); }
ExecutionResult handle_code_VEX_VPSRLDQ_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_dq_imm(ctx, true, true); }
ExecutionResult handle_code_VEX_VPSLLDQ_XMM_XMM_IMM8(ExecutionContext& ctx) { return vex_shift_dq_imm(ctx, false, true); }
ExecutionResult handle_code_VEX_VPSLLDQ_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_dq_imm(ctx, false, true); }

// YMM VEX
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDB_YMM_YMM_YMMM256, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDW_YMM_YMM_YMMM256, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDD_YMM_YMM_YMMM256, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDQ_YMM_YMM_YMMM256, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDUSB_YMM_YMM_YMMM256, std::uint8_t, sat_add_unsigned<std::uint8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDUSW_YMM_YMM_YMMM256, std::uint16_t, sat_add_unsigned<std::uint16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDSB_YMM_YMM_YMMM256, std::int8_t, sat_add_signed<std::int8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPADDSW_YMM_YMM_YMMM256, std::int16_t, sat_add_signed<std::int16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBB_YMM_YMM_YMMM256, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBW_YMM_YMM_YMMM256, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBD_YMM_YMM_YMMM256, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBQ_YMM_YMM_YMMM256, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBUSB_YMM_YMM_YMMM256, std::uint8_t, sat_sub_unsigned<std::uint8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBUSW_YMM_YMM_YMMM256, std::uint16_t, sat_sub_unsigned<std::uint16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBSB_YMM_YMM_YMMM256, std::int8_t, sat_sub_signed<std::int8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPSUBSW_YMM_YMM_YMMM256, std::int16_t, sat_sub_signed<std::int16_t>)
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQB_YMM_YMM_YMMM256, std::uint8_t, [](auto a, auto b) { return a == b ? all_ones<std::uint8_t>() : std::uint8_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQW_YMM_YMM_YMMM256, std::uint16_t, [](auto a, auto b) { return a == b ? all_ones<std::uint16_t>() : std::uint16_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQD_YMM_YMM_YMMM256, std::uint32_t, [](auto a, auto b) { return a == b ? all_ones<std::uint32_t>() : std::uint32_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPEQQ_YMM_YMM_YMMM256, std::uint64_t, [](auto a, auto b) { return a == b ? all_ones<std::uint64_t>() : std::uint64_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTB_YMM_YMM_YMMM256, std::int8_t, [](auto a, auto b) { return a > b ? all_ones<std::int8_t>() : std::int8_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTW_YMM_YMM_YMMM256, std::int16_t, [](auto a, auto b) { return a > b ? all_ones<std::int16_t>() : std::int16_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTD_YMM_YMM_YMMM256, std::int32_t, [](auto a, auto b) { return a > b ? all_ones<std::int32_t>() : std::int32_t{0}; })
KUBERA_VEX_PACKED_INT_CMP(VEX_VPCMPGTQ_YMM_YMM_YMMM256, std::int64_t, [](auto a, auto b) { return a > b ? all_ones<std::int64_t>() : std::int64_t{0}; })
KUBERA_VEX_PACKED_INT_BIN(VEX_VPAVGB_YMM_YMM_YMMM256, std::uint8_t, avg_unsigned<std::uint8_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPAVGW_YMM_YMM_YMMM256, std::uint16_t, avg_unsigned<std::uint16_t>)
KUBERA_VEX_PACKED_INT_BIN(VEX_VPMULLW_YMM_YMM_YMMM256, std::uint16_t, [](auto a, auto b) {
  return static_cast<std::uint16_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(a)) * static_cast<std::int32_t>(static_cast<std::int16_t>(b)));
})
KUBERA_VEX_PACKED_INT_BIT(VEX_VPAND_YMM_YMM_YMMM256, [](auto a, auto b) { return a & b; })
KUBERA_VEX_PACKED_INT_BIT(VEX_VPANDN_YMM_YMM_YMMM256, [](auto a, auto b) { return (~a) & b; })
KUBERA_VEX_PACKED_INT_BIT(VEX_VPOR_YMM_YMM_YMMM256, [](auto a, auto b) { return a | b; })
KUBERA_VEX_PACKED_INT_BIT(VEX_VPXOR_YMM_YMM_YMMM256, [](auto a, auto b) { return a ^ b; })
ExecutionResult handle_code_VEX_VPSLLW_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::uint16_t>(ctx, shift_left_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSLLD_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::uint32_t>(ctx, shift_left_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSLLQ_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::uint64_t>(ctx, shift_left_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRLW_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::uint16_t>(ctx, shift_right_logical_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSRLD_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::uint32_t>(ctx, shift_right_logical_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSRLQ_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::uint64_t>(ctx, shift_right_logical_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRAW_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::int16_t>(ctx, shift_right_arithmetic_lane<std::int16_t>, true); }
ExecutionResult handle_code_VEX_VPSRAD_YMM_YMM_YMMM256(ExecutionContext& ctx) { return vex_shift_reg<std::int32_t>(ctx, shift_right_arithmetic_lane<std::int32_t>, true); }
ExecutionResult handle_code_VEX_VPSLLW_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint16_t>(ctx, shift_left_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSLLD_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint32_t>(ctx, shift_left_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSLLQ_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint64_t>(ctx, shift_left_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRLW_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint16_t>(ctx, shift_right_logical_lane<std::uint16_t>, true); }
ExecutionResult handle_code_VEX_VPSRLD_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint32_t>(ctx, shift_right_logical_lane<std::uint32_t>, true); }
ExecutionResult handle_code_VEX_VPSRLQ_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::uint64_t>(ctx, shift_right_logical_lane<std::uint64_t>, true); }
ExecutionResult handle_code_VEX_VPSRAW_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::int16_t>(ctx, shift_right_arithmetic_lane<std::int16_t>, true); }
ExecutionResult handle_code_VEX_VPSRAD_YMM_YMM_IMM8(ExecutionContext& ctx) { return vex_shift_imm<std::int32_t>(ctx, shift_right_arithmetic_lane<std::int32_t>, true); }


// EVEX XMM/YMM/ZMM
#define KUBERA_EVEX_PACKED_INT_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_binary<type>(ctx, fn, true); }
#define KUBERA_EVEX_PACKED_INT_CMP(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_compare<type>(ctx, fn, true); }
#define KUBERA_EVEX_PACKED_INT_BIT(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_bitwise<type>(ctx, fn, true); }

#define KUBERA_EVEX_PACKED_INT_XYZ(base, type, fn) \
  KUBERA_EVEX_PACKED_INT_BIN(EVEX_##base##_XMM_K1Z_XMM_XMMM128, type, fn) \
  KUBERA_EVEX_PACKED_INT_BIN(EVEX_##base##_YMM_K1Z_YMM_YMMM256, type, fn) \
  KUBERA_EVEX_PACKED_INT_BIN(EVEX_##base##_ZMM_K1Z_ZMM_ZMMM512, type, fn)

#define KUBERA_EVEX_PACKED_INT_CMP_XYZ(base, type, fn) \
  KUBERA_EVEX_PACKED_INT_CMP(EVEX_##base##_XMM_K1Z_XMM_XMMM128, type, fn) \
  KUBERA_EVEX_PACKED_INT_CMP(EVEX_##base##_YMM_K1Z_YMM_YMMM256, type, fn) \
  KUBERA_EVEX_PACKED_INT_CMP(EVEX_##base##_ZMM_K1Z_ZMM_ZMMM512, type, fn)

#define KUBERA_EVEX_PACKED_INT_BIT_DQ(base, fn) \
  KUBERA_EVEX_PACKED_INT_BIT(EVEX_##base##D_XMM_K1Z_XMM_XMMM128B32, std::uint32_t, fn) \
  KUBERA_EVEX_PACKED_INT_BIT(EVEX_##base##D_YMM_K1Z_YMM_YMMM256B32, std::uint32_t, fn) \
  KUBERA_EVEX_PACKED_INT_BIT(EVEX_##base##D_ZMM_K1Z_ZMM_ZMMM512B32, std::uint32_t, fn) \
  KUBERA_EVEX_PACKED_INT_BIT(EVEX_##base##Q_XMM_K1Z_XMM_XMMM128B64, std::uint64_t, fn) \
  KUBERA_EVEX_PACKED_INT_BIT(EVEX_##base##Q_YMM_K1Z_YMM_YMMM256B64, std::uint64_t, fn) \
  KUBERA_EVEX_PACKED_INT_BIT(EVEX_##base##Q_ZMM_K1Z_ZMM_ZMMM512B64, std::uint64_t, fn)

#define KUBERA_EVEX_SHIFT_XYZ(base, type, fn) \
  ExecutionResult handle_code_EVEX_##base##_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<type>(ctx, fn, true); } \
  ExecutionResult handle_code_EVEX_##base##_YMM_K1Z_YMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<type>(ctx, fn, true); } \
  ExecutionResult handle_code_EVEX_##base##_ZMM_K1Z_ZMM_XMMM128(ExecutionContext& ctx) { return vex_shift_reg<type>(ctx, fn, true); }

#define KUBERA_EVEX_SHIFT_IMM_XYZ(xmm_name, ymm_name, zmm_name, type, fn) \
  ExecutionResult handle_code_##xmm_name(ExecutionContext& ctx) { return vex_shift_imm<type>(ctx, fn, true); } \
  ExecutionResult handle_code_##ymm_name(ExecutionContext& ctx) { return vex_shift_imm<type>(ctx, fn, true); } \
  ExecutionResult handle_code_##zmm_name(ExecutionContext& ctx) { return vex_shift_imm<type>(ctx, fn, true); }

KUBERA_EVEX_PACKED_INT_XYZ(VPADDB, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a + b); })
KUBERA_EVEX_PACKED_INT_XYZ(VPADDW, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a + b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPADDD_YMM_K1Z_YMM_YMMM256B32, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPADDD_ZMM_K1Z_ZMM_ZMMM512B32, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a + b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPADDQ_XMM_K1Z_XMM_XMMM128B64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPADDQ_YMM_K1Z_YMM_YMMM256B64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPADDQ_ZMM_K1Z_ZMM_ZMMM512B64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a + b); })
KUBERA_EVEX_PACKED_INT_XYZ(VPADDUSB, std::uint8_t, sat_add_unsigned<std::uint8_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPADDUSW, std::uint16_t, sat_add_unsigned<std::uint16_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPADDSB, std::int8_t, sat_add_signed<std::int8_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPADDSW, std::int16_t, sat_add_signed<std::int16_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPSUBB, std::uint8_t, [](auto a, auto b) { return static_cast<std::uint8_t>(a - b); })
KUBERA_EVEX_PACKED_INT_XYZ(VPSUBW, std::uint16_t, [](auto a, auto b) { return static_cast<std::uint16_t>(a - b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPSUBD_XMM_K1Z_XMM_XMMM128B32, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPSUBD_YMM_K1Z_YMM_YMMM256B32, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPSUBD_ZMM_K1Z_ZMM_ZMMM512B32, std::uint32_t, [](auto a, auto b) { return static_cast<std::uint32_t>(a - b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPSUBQ_XMM_K1Z_XMM_XMMM128B64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPSUBQ_YMM_K1Z_YMM_YMMM256B64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_EVEX_PACKED_INT_BIN(EVEX_VPSUBQ_ZMM_K1Z_ZMM_ZMMM512B64, std::uint64_t, [](auto a, auto b) { return static_cast<std::uint64_t>(a - b); })
KUBERA_EVEX_PACKED_INT_XYZ(VPSUBUSB, std::uint8_t, sat_sub_unsigned<std::uint8_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPSUBUSW, std::uint16_t, sat_sub_unsigned<std::uint16_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPSUBSB, std::int8_t, sat_sub_signed<std::int8_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPSUBSW, std::int16_t, sat_sub_signed<std::int16_t>)
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPEQB, std::uint8_t, [](auto a, auto b) { return a == b ? all_ones<std::uint8_t>() : std::uint8_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPEQW, std::uint16_t, [](auto a, auto b) { return a == b ? all_ones<std::uint16_t>() : std::uint16_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPEQD, std::uint32_t, [](auto a, auto b) { return a == b ? all_ones<std::uint32_t>() : std::uint32_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPEQQ, std::uint64_t, [](auto a, auto b) { return a == b ? all_ones<std::uint64_t>() : std::uint64_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPGTB, std::int8_t, [](auto a, auto b) { return a > b ? all_ones<std::int8_t>() : std::int8_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPGTW, std::int16_t, [](auto a, auto b) { return a > b ? all_ones<std::int16_t>() : std::int16_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPGTD, std::int32_t, [](auto a, auto b) { return a > b ? all_ones<std::int32_t>() : std::int32_t{0}; })
KUBERA_EVEX_PACKED_INT_CMP_XYZ(VPCMPGTQ, std::int64_t, [](auto a, auto b) { return a > b ? all_ones<std::int64_t>() : std::int64_t{0}; })
KUBERA_EVEX_PACKED_INT_XYZ(VPAVGB, std::uint8_t, avg_unsigned<std::uint8_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPAVGW, std::uint16_t, avg_unsigned<std::uint16_t>)
KUBERA_EVEX_PACKED_INT_XYZ(VPMULLW, std::uint16_t, [](auto a, auto b) {
  return static_cast<std::uint16_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(a)) * static_cast<std::int32_t>(static_cast<std::int16_t>(b)));
})
KUBERA_EVEX_PACKED_INT_BIT_DQ(VPAND, [](auto a, auto b) { return a & b; })
KUBERA_EVEX_PACKED_INT_BIT_DQ(VPANDN, [](auto a, auto b) { return (~a) & b; })
KUBERA_EVEX_PACKED_INT_BIT_DQ(VPOR, [](auto a, auto b) { return a | b; })
KUBERA_EVEX_PACKED_INT_BIT_DQ(VPXOR, [](auto a, auto b) { return a ^ b; })
KUBERA_EVEX_SHIFT_XYZ(VPSLLW, std::uint16_t, shift_left_lane<std::uint16_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSLLD, std::uint32_t, shift_left_lane<std::uint32_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSLLQ, std::uint64_t, shift_left_lane<std::uint64_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSRLW, std::uint16_t, shift_right_logical_lane<std::uint16_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSRLD, std::uint32_t, shift_right_logical_lane<std::uint32_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSRLQ, std::uint64_t, shift_right_logical_lane<std::uint64_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSRAW, std::int16_t, shift_right_arithmetic_lane<std::int16_t>)
KUBERA_EVEX_SHIFT_XYZ(VPSRAD, std::int32_t, shift_right_arithmetic_lane<std::int32_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSLLW_XMM_K1Z_XMMM128_IMM8, EVEX_VPSLLW_YMM_K1Z_YMMM256_IMM8, EVEX_VPSLLW_ZMM_K1Z_ZMMM512_IMM8, std::uint16_t, shift_left_lane<std::uint16_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSLLD_XMM_K1Z_XMMM128B32_IMM8, EVEX_VPSLLD_YMM_K1Z_YMMM256B32_IMM8, EVEX_VPSLLD_ZMM_K1Z_ZMMM512B32_IMM8, std::uint32_t, shift_left_lane<std::uint32_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSLLQ_XMM_K1Z_XMMM128B64_IMM8, EVEX_VPSLLQ_YMM_K1Z_YMMM256B64_IMM8, EVEX_VPSLLQ_ZMM_K1Z_ZMMM512B64_IMM8, std::uint64_t, shift_left_lane<std::uint64_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSRLW_XMM_K1Z_XMMM128_IMM8, EVEX_VPSRLW_YMM_K1Z_YMMM256_IMM8, EVEX_VPSRLW_ZMM_K1Z_ZMMM512_IMM8, std::uint16_t, shift_right_logical_lane<std::uint16_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSRLD_XMM_K1Z_XMMM128B32_IMM8, EVEX_VPSRLD_YMM_K1Z_YMMM256B32_IMM8, EVEX_VPSRLD_ZMM_K1Z_ZMMM512B32_IMM8, std::uint32_t, shift_right_logical_lane<std::uint32_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSRLQ_XMM_K1Z_XMMM128B64_IMM8, EVEX_VPSRLQ_YMM_K1Z_YMMM256B64_IMM8, EVEX_VPSRLQ_ZMM_K1Z_ZMMM512B64_IMM8, std::uint64_t, shift_right_logical_lane<std::uint64_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSRAW_XMM_K1Z_XMMM128_IMM8, EVEX_VPSRAW_YMM_K1Z_YMMM256_IMM8, EVEX_VPSRAW_ZMM_K1Z_ZMMM512_IMM8, std::int16_t, shift_right_arithmetic_lane<std::int16_t>)
KUBERA_EVEX_SHIFT_IMM_XYZ(EVEX_VPSRAD_XMM_K1Z_XMMM128B32_IMM8, EVEX_VPSRAD_YMM_K1Z_YMMM256B32_IMM8, EVEX_VPSRAD_ZMM_K1Z_ZMMM512B32_IMM8, std::int32_t, shift_right_arithmetic_lane<std::int32_t>)

ExecutionResult handle_code_PCMPISTRI_XMM_XMMM128_IMM8(ExecutionContext& ctx) { return pcmpstr(ctx, false, false); }
ExecutionResult handle_code_PCMPISTRM_XMM_XMMM128_IMM8(ExecutionContext& ctx) { return pcmpstr(ctx, false, true); }
ExecutionResult handle_code_PCMPESTRI_XMM_XMMM128_IMM8(ExecutionContext& ctx) { return pcmpstr(ctx, true, false); }
ExecutionResult handle_code_PCMPESTRM_XMM_XMMM128_IMM8(ExecutionContext& ctx) { return pcmpstr(ctx, true, true); }

#undef KUBERA_LEGACY_BIN
#undef KUBERA_LEGACY_CMP
#undef KUBERA_LEGACY_BIT
#undef KUBERA_VEX_PACKED_INT_BIN
#undef KUBERA_VEX_PACKED_INT_CMP
#undef KUBERA_VEX_PACKED_INT_BIT
#undef KUBERA_EVEX_PACKED_INT_BIN
#undef KUBERA_EVEX_PACKED_INT_CMP
#undef KUBERA_EVEX_PACKED_INT_BIT
#undef KUBERA_EVEX_PACKED_INT_XYZ
#undef KUBERA_EVEX_PACKED_INT_CMP_XYZ
#undef KUBERA_EVEX_PACKED_INT_BIT_DQ
#undef KUBERA_EVEX_SHIFT_XYZ
#undef KUBERA_EVEX_SHIFT_IMM_XYZ

}  // namespace seven::handlers



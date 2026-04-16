#include <bit>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

using big_uint = seven::SimdUint;

constexpr std::size_t kXmmBytes = 16;
constexpr std::size_t kYmmBytes = 32;
constexpr std::size_t kZmmBytes = 64;
constexpr std::uint32_t kSseExceptionInvalid = 0x01u;
constexpr std::uint32_t kSseExceptionDenormal = 0x02u;
constexpr std::uint32_t kSseExceptionZeroDiv = 0x04u;
constexpr std::uint32_t kSseExceptionOverflow = 0x08u;
constexpr std::uint32_t kSseExceptionUnderflow = 0x10u;
constexpr std::uint32_t kSseExceptionPrecision = 0x20u;
constexpr std::uint32_t kSseExceptionMask = 0x3Fu;

bool is_vector_register(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  return (value >= xmm0 && value < xmm0 + 32) || (value >= ymm0 && value < ymm0 + 32) || (value >= zmm0 && value < zmm0 + 32);
}

std::size_t xmm_index(iced_x86::Register reg) {
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
  return state.vectors[xmm_index(reg)].value & mask(vector_width(reg));
}

void write_vec(CpuState& state, iced_x86::Register reg, big_uint value, bool zero_upper = false) {
  auto& slot = state.vectors[xmm_index(reg)].value;
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
    return read_mem(ctx, detail::memory_address(ctx), width, ok);
  }
  if (ok) *ok = false;
  return 0;
}

template <typename T>
T unpack_lane(const big_uint& value, std::size_t lane_offset_bytes) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> ((lane_offset_bytes + i) * 8)) & 0xFFu);
  }
  T out{};
  std::memcpy(&out, bytes.data(), sizeof(T));
  return out;
}

template <typename T>
void pack_lane(big_uint& target, std::size_t lane_offset_bytes, T value) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  const auto lane_mask = (big_uint(1) << (sizeof(T) * 8)) - 1;
  target &= ~(lane_mask << (lane_offset_bytes * 8));
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    target |= (big_uint(bytes[i]) << ((lane_offset_bytes + i) * 8));
  }
}

template <typename T>
std::uint32_t classify_result(T result, T lhs, T rhs) {
  std::uint32_t exceptions = 0;
  if (std::isnan(result)) exceptions |= kSseExceptionInvalid;
  if (std::isinf(result)) exceptions |= kSseExceptionOverflow;
  const T min_normal = std::numeric_limits<T>::min();
  const auto abs_result = std::fabs(result);
  if (result != 0 && abs_result < min_normal) {
    exceptions |= static_cast<std::uint32_t>(kSseExceptionUnderflow | kSseExceptionPrecision);
  }
  if ((lhs != 0 && std::fabs(lhs) < min_normal) || (rhs != 0 && std::fabs(rhs) < min_normal) || (result != 0 && abs_result < min_normal)) {
    exceptions |= kSseExceptionDenormal;
  }
  return exceptions;
}

template <typename T>
std::uint32_t precision_from_binary(T lhs, T rhs, T result) {
  if (std::isnan(result) || std::isinf(result)) return 0;
  if (result == 0 && lhs != 0 && rhs != 0) {
    return static_cast<std::uint32_t>(kSseExceptionUnderflow | kSseExceptionPrecision);
  }
  if ((result - lhs == rhs) || (result - rhs == lhs) || (lhs - result == rhs) || (result + rhs == lhs)) return 0;
  if ((lhs == 0 || rhs == 0) || (result / lhs == rhs) || (result / rhs == lhs) || (rhs != 0 && result * rhs == lhs) || (lhs != 0 && result * lhs == rhs)) return 0;
  return kSseExceptionPrecision;
}

bool sse_exceptions_masked(const CpuState& state, std::uint32_t exceptions) {
  const auto masked = static_cast<std::uint32_t>((state.mxcsr >> 7) & kSseExceptionMask);
  return (exceptions & ~masked) == 0;
}

ExecutionResult sse_exception(ExecutionContext& ctx, std::uint32_t exceptions) {
  ctx.state.mxcsr |= (exceptions & kSseExceptionMask);
  if (sse_exceptions_masked(ctx.state, exceptions)) {
    return {};
  }
  return {StopReason::floating_point_exception, 0, ExceptionInfo{StopReason::floating_point_exception, ctx.state.rip, exceptions}, ctx.instr.code()};
}

void set_sse_cmp_flags(CpuState& state, int relation) {
  std::uint64_t rf = state.rflags;
  rf &= ~(kFlagCF | kFlagPF | kFlagZF | kFlagOF | kFlagSF | kFlagAF);
  if (relation == -2) {
    rf |= (kFlagCF | kFlagPF | kFlagZF);
  } else if (relation < 0) {
    rf |= kFlagCF;
  } else if (relation == 0) {
    rf |= kFlagZF;
  }
  state.rflags = rf;
}

template <typename T, typename Fn>
ExecutionResult packed_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = lhs_bits;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg) / sizeof(T); ++lane) {
    const auto offset = lane * sizeof(T);
    const T lhs = unpack_lane<T>(lhs_bits, offset);
    const T rhs = unpack_lane<T>(rhs_bits, offset);
    const T result = fn(lhs, rhs);
    exceptions |= classify_result(result, lhs, rhs);
    exceptions |= precision_from_binary(lhs, rhs, result);
    pack_lane(out, offset, result);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_packed_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto rhs_bits = read_operand(ctx, 2, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg) / sizeof(T); ++lane) {
    const auto offset = lane * sizeof(T);
    const T lhs = unpack_lane<T>(lhs_bits, offset);
    const T rhs = unpack_lane<T>(rhs_bits, offset);
    const T result = fn(lhs, rhs);
    exceptions |= classify_result(result, lhs, rhs);
    exceptions |= precision_from_binary(lhs, rhs, result);
    pack_lane(out, offset, result);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult scalar_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, sizeof(T), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const T lhs = unpack_lane<T>(lhs_bits, 0);
  const T rhs = unpack_lane<T>(rhs_bits, 0);
  const T result_value = fn(lhs, rhs);
  std::uint32_t exceptions = classify_result(result_value, lhs, rhs);
  exceptions |= precision_from_binary(lhs, rhs, result_value);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = lhs_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_scalar_binary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto rhs_bits = read_operand(ctx, 2, sizeof(T), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const T lhs = unpack_lane<T>(lhs_bits, 0);
  const T rhs = unpack_lane<T>(rhs_bits, 0);
  const T result_value = fn(lhs, rhs);
  std::uint32_t exceptions = classify_result(result_value, lhs, rhs);
  exceptions |= precision_from_binary(lhs, rhs, result_value);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = lhs_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult packed_unary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst_reg = ctx.instr.op_register(0);
  const auto input_bits = read_vec(ctx.state, dst_reg);
  big_uint out = input_bits;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg) / sizeof(T); ++lane) {
    const auto offset = lane * sizeof(T);
    const T input = unpack_lane<T>(input_bits, offset);
    const T result = fn(input);
    exceptions |= classify_result(result, input, T{});
    pack_lane(out, offset, result);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_packed_unary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto input_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = input_bits;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg) / sizeof(T); ++lane) {
    const auto offset = lane * sizeof(T);
    const T input = unpack_lane<T>(input_bits, offset);
    const T result = fn(input);
    exceptions |= classify_result(result, input, T{});
    pack_lane(out, offset, result);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult scalar_unary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  const auto dst_reg = ctx.instr.op_register(0);
  const auto input_bits = read_vec(ctx.state, dst_reg);
  const T input = unpack_lane<T>(input_bits, 0);
  const T result_value = fn(input);
  const std::uint32_t exceptions = classify_result(result_value, input, T{});
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = input_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, typename Fn>
ExecutionResult vex_scalar_unary(ExecutionContext& ctx, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto input_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto scalar_bits = read_operand(ctx, 2, sizeof(T), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const T input = unpack_lane<T>(scalar_bits, 0);
  const T result_value = fn(input);
  const std::uint32_t exceptions = classify_result(result_value, input, T{});
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = input_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

ExecutionResult packed_logic(ExecutionContext& ctx, std::uint32_t lhs_index, std::uint32_t rhs_index, std::uint8_t op, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs = read_operand(ctx, lhs_index, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto rhs = read_operand(ctx, rhs_index, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  switch (op) {
    case 0: out = lhs & rhs; break;
    case 1: out = lhs | rhs; break;
    case 2: out = lhs ^ rhs; break;
    default: out = (~lhs) & rhs; break;
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T>
ExecutionResult scalar_compare(ExecutionContext& ctx, bool quiet) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto lhs_bits = read_vec(ctx.state, ctx.instr.op_register(0));
  const auto rhs_bits = read_operand(ctx, 1, sizeof(T), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const T lhs = unpack_lane<T>(lhs_bits, 0);
  const T rhs = unpack_lane<T>(rhs_bits, 0);
  int relation = 0;
  std::uint32_t exceptions = 0;
  if (std::isnan(lhs) || std::isnan(rhs)) {
    relation = -2;
    if (!quiet) exceptions |= kSseExceptionInvalid;
  } else if (lhs < rhs) {
    relation = -1;
  } else if (lhs > rhs) {
    relation = 1;
  }
  set_sse_cmp_flags(ctx.state, relation);
  if (exceptions != 0) {
    return sse_exception(ctx, exceptions);
  }
  return {};
}

bool is_mmx_register(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto mm0 = static_cast<std::uint32_t>(iced_x86::Register::MM0);
  return value >= mm0 && value < mm0 + 8;
}

std::size_t mmx_index(iced_x86::Register reg) {
  return static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::MM0));
}

unsigned mxcsr_rounding_mode(const CpuState& state) {
  return static_cast<unsigned>((state.mxcsr >> 13) & 0x3u);
}

template <typename T>
bool fp_subnormal(T value) {
  return std::fpclassify(value) == FP_SUBNORMAL;
}

long double round_half_even(long double value) {
  const auto floor_value = std::floor(value);
  const auto fraction = value - floor_value;
  if (fraction < 0.5L) return floor_value;
  if (fraction > 0.5L) return floor_value + 1.0L;
  return std::fmod(std::fabs(floor_value), 2.0L) == 0.0L ? floor_value : floor_value + 1.0L;
}

long double round_by_mxcsr(const CpuState& state, long double value, bool truncate_only = false) {
  if (truncate_only) return std::trunc(value);
  switch (mxcsr_rounding_mode(state)) {
    case 0: return round_half_even(value);
    case 1: return std::floor(value);
    case 2: return std::ceil(value);
    default: return std::trunc(value);
  }
}

template <typename Float>
Float apply_rounding_mode(const CpuState& state, Float nearest, long double exact) {
  if (static_cast<long double>(nearest) == exact) return nearest;
  if (!std::isfinite(nearest)) {
    switch (mxcsr_rounding_mode(state)) {
      case 1: return exact > 0.0L ? std::numeric_limits<Float>::max() : nearest;
      case 2: return exact < 0.0L ? -std::numeric_limits<Float>::max() : nearest;
      case 3: return exact > 0.0L ? std::numeric_limits<Float>::max() : -std::numeric_limits<Float>::max();
      default: return nearest;
    }
  }
  const auto rounded = static_cast<long double>(nearest);
  switch (mxcsr_rounding_mode(state)) {
    case 1:
      if (rounded > exact) return std::nextafter(nearest, -std::numeric_limits<Float>::infinity());
      return nearest;
    case 2:
      if (rounded < exact) return std::nextafter(nearest, std::numeric_limits<Float>::infinity());
      return nearest;
    case 3:
      if ((exact > 0.0L && rounded > exact) || (exact < 0.0L && rounded < exact)) {
        return std::nextafter(nearest, Float{0});
      }
      return nearest;
    default:
      return nearest;
  }
}

template <typename Int>
constexpr std::uint64_t indefinite_integer_bits() {
  using U = std::make_unsigned_t<Int>;
  return static_cast<std::uint64_t>(U{1} << ((sizeof(Int) * 8) - 1));
}

template <typename Int, typename Float>
std::pair<std::uint64_t, std::uint32_t> convert_fp_to_int_bits(const CpuState& state, Float input, bool truncate_only) {
  std::uint32_t exceptions = 0;
  if (fp_subnormal(input)) exceptions |= kSseExceptionDenormal;
  if (std::isnan(input) || std::isinf(input)) {
    exceptions |= kSseExceptionInvalid;
    return {indefinite_integer_bits<Int>(), exceptions};
  }
  const auto exact = static_cast<long double>(input);
  const auto rounded = round_by_mxcsr(state, exact, truncate_only);
  if (rounded < static_cast<long double>(std::numeric_limits<Int>::min()) ||
      rounded > static_cast<long double>(std::numeric_limits<Int>::max())) {
    exceptions |= kSseExceptionInvalid;
    return {indefinite_integer_bits<Int>(), exceptions};
  }
  if (rounded != exact) exceptions |= kSseExceptionPrecision;
  using U = std::make_unsigned_t<Int>;
  return {static_cast<std::uint64_t>(static_cast<U>(static_cast<Int>(rounded))), exceptions};
}

template <typename Float, typename Int>
std::pair<Float, std::uint32_t> convert_int_to_fp(const CpuState& state, Int input) {
  std::uint32_t exceptions = 0;
  if constexpr (std::is_same_v<Float, double> && std::is_same_v<Int, std::int64_t>) {
    using UInt = std::make_unsigned_t<Int>;
    const bool negative = input < 0;
    const auto magnitude = negative ? (UInt{0} - static_cast<UInt>(input)) : static_cast<UInt>(input);
    if (magnitude < (UInt{1} << 53)) {
      return {static_cast<double>(input), 0};
    }

    const auto exponent = std::bit_width(static_cast<std::uint64_t>(magnitude)) - 1u;
    const auto shift = exponent - 52u;
    const auto base = magnitude >> shift;
    const auto remainder_mask = (UInt{1} << shift) - 1u;
    const auto remainder = magnitude & remainder_mask;
    auto rounded_base = base;
    switch (mxcsr_rounding_mode(state)) {
      case 0: {
        const auto half = UInt{1} << (shift - 1u);
        if (remainder > half || (remainder == half && (rounded_base & 1u) != 0u)) ++rounded_base;
        break;
      }
      case 1:
        if (negative && remainder != 0) ++rounded_base;
        break;
      case 2:
        if (!negative && remainder != 0) ++rounded_base;
        break;
      default:
        break;
    }
    const auto rounded_magnitude = rounded_base << shift;
    const auto rounded_exact = negative ? -static_cast<long double>(rounded_magnitude) : static_cast<long double>(rounded_magnitude);
    if (remainder != 0) exceptions |= kSseExceptionPrecision;
    return {static_cast<double>(rounded_exact), exceptions};
  }

  const auto exact = static_cast<long double>(input);
  auto result = static_cast<Float>(input);
  result = apply_rounding_mode(state, result, exact);
  if (static_cast<long double>(result) != exact) exceptions |= kSseExceptionPrecision;
  return {result, exceptions};
}

template <typename FloatOut, typename FloatIn>
std::pair<FloatOut, std::uint32_t> convert_fp_to_fp(const CpuState& state, FloatIn input) {
  std::uint32_t exceptions = 0;
  if (fp_subnormal(input)) exceptions |= kSseExceptionDenormal;
  auto result = static_cast<FloatOut>(input);
  if (std::isnan(input) || std::isinf(input)) return {result, exceptions};
  const auto exact = static_cast<long double>(input);
  result = apply_rounding_mode(state, result, exact);
  const auto rounded = static_cast<long double>(result);
  if (rounded != exact) {
    exceptions |= kSseExceptionPrecision;
    if (std::isinf(result)) {
      exceptions |= kSseExceptionOverflow;
    } else if ((result == FloatOut{0} || fp_subnormal(result)) && input != FloatIn{0}) {
      exceptions |= kSseExceptionUnderflow;
    }
  }
  return {result, exceptions};
}

template <typename T>
bool compare_predicate(unsigned predicate, T lhs, T rhs) {
  const bool unordered = std::isnan(lhs) || std::isnan(rhs);
  switch (predicate & 0x7u) {
    case 0: return !unordered && lhs == rhs;
    case 1: return !unordered && lhs < rhs;
    case 2: return !unordered && lhs <= rhs;
    case 3: return unordered;
    case 4: return unordered || lhs != rhs;
    case 5: return unordered || !(lhs < rhs);
    case 6: return unordered || !(lhs <= rhs);
    default: return !unordered;
  }
}

template <typename T>
std::uint32_t compare_exceptions(unsigned predicate, T lhs, T rhs) {
  if (!std::isnan(lhs) && !std::isnan(rhs)) return 0;
  switch (predicate & 0x7u) {
    case 1:
    case 2:
    case 5:
    case 6:
      return kSseExceptionInvalid;
    default:
      return 0;
  }
}

template <typename T>
ExecutionResult packed_compare_mask(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto predicate = static_cast<unsigned>(ctx.instr.immediate8()) & 0x7u;
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg) / sizeof(T); ++lane) {
    const auto offset = lane * sizeof(T);
    const T lhs = unpack_lane<T>(lhs_bits, offset);
    const T rhs = unpack_lane<T>(rhs_bits, offset);
    const auto mask_value = compare_predicate(predicate, lhs, rhs) ? ~std::uint64_t{0} : std::uint64_t{0};
    pack_lane(out, offset, mask_value);
    exceptions |= compare_exceptions(predicate, lhs, rhs);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

template <typename T>
ExecutionResult scalar_compare_mask(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, sizeof(T), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto predicate = static_cast<unsigned>(ctx.instr.immediate8()) & 0x7u;
  const T lhs = unpack_lane<T>(lhs_bits, 0);
  const T rhs = unpack_lane<T>(rhs_bits, 0);
  const auto mask_value = compare_predicate(predicate, lhs, rhs) ? ~std::uint64_t{0} : std::uint64_t{0};
  std::uint32_t exceptions = compare_exceptions(predicate, lhs, rhs);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = lhs_bits;
  pack_lane(out, 0, mask_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

template <typename T>
ExecutionResult packed_addsub(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = lhs_bits;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg) / sizeof(T); ++lane) {
    const auto offset = lane * sizeof(T);
    const T lhs = unpack_lane<T>(lhs_bits, offset);
    const T rhs = unpack_lane<T>(rhs_bits, offset);
    const T op_rhs = (lane & 1u) == 0u ? rhs : static_cast<T>(-rhs);
    const T result = (lane & 1u) == 0u ? static_cast<T>(lhs - rhs) : static_cast<T>(lhs + rhs);
    exceptions |= classify_result(result, lhs, op_rhs);
    exceptions |= precision_from_binary(lhs, op_rhs, result);
    pack_lane(out, offset, result);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

template <typename T>
ExecutionResult packed_horizontal(ExecutionContext& ctx, bool subtract) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  const std::size_t half_lanes = (vector_width(dst_reg) / sizeof(T)) / 2;
  for (std::size_t pair = 0; pair < half_lanes; ++pair) {
    const auto lhs_a = unpack_lane<T>(lhs_bits, (pair * 2) * sizeof(T));
    const auto lhs_b = unpack_lane<T>(lhs_bits, (pair * 2 + 1) * sizeof(T));
    const auto lhs_rhs = subtract ? static_cast<T>(-lhs_b) : lhs_b;
    const auto lhs_result = subtract ? static_cast<T>(lhs_a - lhs_b) : static_cast<T>(lhs_a + lhs_b);
    exceptions |= classify_result(lhs_result, lhs_a, lhs_rhs);
    exceptions |= precision_from_binary(lhs_a, lhs_rhs, lhs_result);
    pack_lane(out, pair * sizeof(T), lhs_result);

    const auto rhs_a = unpack_lane<T>(rhs_bits, (pair * 2) * sizeof(T));
    const auto rhs_b = unpack_lane<T>(rhs_bits, (pair * 2 + 1) * sizeof(T));
    const auto rhs_rhs = subtract ? static_cast<T>(-rhs_b) : rhs_b;
    const auto rhs_result = subtract ? static_cast<T>(rhs_a - rhs_b) : static_cast<T>(rhs_a + rhs_b);
    exceptions |= classify_result(rhs_result, rhs_a, rhs_rhs);
    exceptions |= precision_from_binary(rhs_a, rhs_rhs, rhs_result);
    pack_lane(out, (pair + half_lanes) * sizeof(T), rhs_result);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}


}  // namespace

#define KUBERA_PACKED_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return packed_binary<type>(ctx, fn); }
#define KUBERA_SCALAR_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return scalar_binary<type>(ctx, fn); }
#define KUBERA_PACKED_UNARY(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return packed_unary<type>(ctx, fn); }
#define KUBERA_SCALAR_UNARY(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return scalar_unary<type>(ctx, fn); }
#define KUBERA_VEX_PACKED_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_packed_binary<type>(ctx, fn, true); }
#define KUBERA_VEX_SCALAR_BIN(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_scalar_binary<type>(ctx, fn, true); }
#define KUBERA_VEX_PACKED_UNARY(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_packed_unary<type>(ctx, fn, true); }
#define KUBERA_VEX_SCALAR_UNARY(name, type, fn) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return vex_scalar_unary<type>(ctx, fn, true); }
#define KUBERA_VEX_LOGIC(name, op) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return packed_logic(ctx, 1, 2, op, true); }
#define KUBERA_VEX_COMPARE(name, type, quiet) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return scalar_compare<type>(ctx, quiet); }

KUBERA_PACKED_BIN(ADDPS_XMM_XMMM128, float, [](float a, float b) { return a + b; })
KUBERA_PACKED_BIN(ADDPD_XMM_XMMM128, double, [](double a, double b) { return a + b; })
KUBERA_PACKED_BIN(SUBPS_XMM_XMMM128, float, [](float a, float b) { return a - b; })
KUBERA_PACKED_BIN(SUBPD_XMM_XMMM128, double, [](double a, double b) { return a - b; })
KUBERA_PACKED_BIN(MULPS_XMM_XMMM128, float, [](float a, float b) { return a * b; })
KUBERA_PACKED_BIN(MULPD_XMM_XMMM128, double, [](double a, double b) { return a * b; })
KUBERA_PACKED_BIN(DIVPS_XMM_XMMM128, float, [](float a, float b) { return a / b; })
KUBERA_PACKED_BIN(DIVPD_XMM_XMMM128, double, [](double a, double b) { return a / b; })
KUBERA_PACKED_BIN(MINPS_XMM_XMMM128, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_PACKED_BIN(MINPD_XMM_XMMM128, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_PACKED_BIN(MAXPS_XMM_XMMM128, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_PACKED_BIN(MAXPD_XMM_XMMM128, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_PACKED_UNARY(SQRTPS_XMM_XMMM128, float, [](float a) { return std::sqrt(a); })
KUBERA_PACKED_UNARY(SQRTPD_XMM_XMMM128, double, [](double a) { return std::sqrt(a); })
ExecutionResult handle_code_ADDSUBPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_addsub<float>(ctx); }
ExecutionResult handle_code_ADDSUBPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_addsub<double>(ctx); }
ExecutionResult handle_code_HADDPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_horizontal<float>(ctx, false); }
ExecutionResult handle_code_HADDPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_horizontal<double>(ctx, false); }
ExecutionResult handle_code_HSUBPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_horizontal<float>(ctx, true); }
ExecutionResult handle_code_HSUBPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_horizontal<double>(ctx, true); }

KUBERA_SCALAR_BIN(ADDSS_XMM_XMMM32, float, [](float a, float b) { return a + b; })
KUBERA_SCALAR_BIN(ADDSD_XMM_XMMM64, double, [](double a, double b) { return a + b; })
KUBERA_SCALAR_BIN(SUBSS_XMM_XMMM32, float, [](float a, float b) { return a - b; })
KUBERA_SCALAR_BIN(SUBSD_XMM_XMMM64, double, [](double a, double b) { return a - b; })
KUBERA_SCALAR_BIN(MULSS_XMM_XMMM32, float, [](float a, float b) { return a * b; })
KUBERA_SCALAR_BIN(MULSD_XMM_XMMM64, double, [](double a, double b) { return a * b; })
KUBERA_SCALAR_BIN(DIVSS_XMM_XMMM32, float, [](float a, float b) { return a / b; })
KUBERA_SCALAR_BIN(DIVSD_XMM_XMMM64, double, [](double a, double b) { return a / b; })
KUBERA_SCALAR_BIN(MINSS_XMM_XMMM32, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_SCALAR_BIN(MINSD_XMM_XMMM64, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_SCALAR_BIN(MAXSS_XMM_XMMM32, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_SCALAR_BIN(MAXSD_XMM_XMMM64, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_SCALAR_UNARY(SQRTSS_XMM_XMMM32, float, [](float a) { return std::sqrt(a); })
KUBERA_SCALAR_UNARY(SQRTSD_XMM_XMMM64, double, [](double a) { return std::sqrt(a); })

ExecutionResult handle_code_ANDPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 0); }
ExecutionResult handle_code_ANDPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 0); }
ExecutionResult handle_code_ANDNPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 3); }
ExecutionResult handle_code_ANDNPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 3); }
ExecutionResult handle_code_ORPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 1); }
ExecutionResult handle_code_ORPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 1); }
ExecutionResult handle_code_XORPS_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 2); }
ExecutionResult handle_code_XORPD_XMM_XMMM128(ExecutionContext& ctx) { return packed_logic(ctx, 0, 1, 2); }

ExecutionResult handle_code_COMISS_XMM_XMMM32(ExecutionContext& ctx) { return scalar_compare<float>(ctx, false); }
ExecutionResult handle_code_UCOMISS_XMM_XMMM32(ExecutionContext& ctx) { return scalar_compare<float>(ctx, true); }
ExecutionResult handle_code_COMISD_XMM_XMMM64(ExecutionContext& ctx) { return scalar_compare<double>(ctx, false); }
ExecutionResult handle_code_UCOMISD_XMM_XMMM64(ExecutionContext& ctx) { return scalar_compare<double>(ctx, true); }

ExecutionResult handle_code_CMPPD_XMM_XMMM128_IMM8(ExecutionContext& ctx) { return packed_compare_mask<double>(ctx); }
ExecutionResult handle_code_CMPSD_XMM_XMMM64_IMM8(ExecutionContext& ctx) { return scalar_compare_mask<double>(ctx); }

ExecutionResult handle_code_CVTPS2PD_XMM_XMMM64(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<float>(src_bits, lane * sizeof(float));
    const auto [result_value, lane_exceptions] = convert_fp_to_fp<double>(ctx.state, input);
    exceptions |= lane_exceptions;
    pack_lane(out, lane * sizeof(double), result_value);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTPD2PS_XMM_XMMM128(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, 16, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<double>(src_bits, lane * sizeof(double));
    const auto [result_value, lane_exceptions] = convert_fp_to_fp<float>(ctx.state, input);
    exceptions |= lane_exceptions;
    pack_lane(out, lane * sizeof(float), result_value);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTDQ2PD_XMM_XMMM64(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, 8, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<std::int32_t>(src_bits, lane * sizeof(std::int32_t));
    pack_lane(out, lane * sizeof(double), static_cast<double>(input));
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTPD2DQ_XMM_XMMM128(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, 16, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<double>(src_bits, lane * sizeof(double));
    const auto [bits, lane_exceptions] = convert_fp_to_int_bits<std::int32_t>(ctx.state, input, false);
    exceptions |= lane_exceptions;
    pack_lane(out, lane * sizeof(std::int32_t), static_cast<std::uint32_t>(bits));
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTTPD2DQ_XMM_XMMM128(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, 16, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<double>(src_bits, lane * sizeof(double));
    const auto [bits, lane_exceptions] = convert_fp_to_int_bits<std::int32_t>(ctx.state, input, true);
    exceptions |= lane_exceptions;
    pack_lane(out, lane * sizeof(std::int32_t), static_cast<std::uint32_t>(bits));
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTSS2SD_XMM_XMMM32(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto dst_bits = read_vec(ctx.state, dst_reg);
  const auto src_bits = read_operand(ctx, 1, sizeof(float), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto input = unpack_lane<float>(src_bits, 0);
  const auto [result_value, exceptions] = convert_fp_to_fp<double>(ctx.state, input);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = dst_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTSD2SS_XMM_XMMM64(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto dst_bits = read_vec(ctx.state, dst_reg);
  const auto src_bits = read_operand(ctx, 1, sizeof(double), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto input = unpack_lane<double>(src_bits, 0);
  const auto [result_value, exceptions] = convert_fp_to_fp<float>(ctx.state, input);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = dst_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTSI2SS_XMM_RM32(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto dst_bits = read_vec(ctx.state, dst_reg);
  const auto input = static_cast<std::int32_t>(detail::read_operand(ctx, 1, 4, &ok));
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto [result_value, exceptions] = convert_int_to_fp<float>(ctx.state, input);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = dst_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTSI2SS_XMM_RM64(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto dst_bits = read_vec(ctx.state, dst_reg);
  const auto input = static_cast<std::int64_t>(detail::read_operand(ctx, 1, 8, &ok));
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto [result_value, exceptions] = convert_int_to_fp<float>(ctx.state, input);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = dst_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}


ExecutionResult handle_code_CVTSI2SD_XMM_RM32(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto dst_bits = read_vec(ctx.state, dst_reg);
  const auto input = static_cast<std::int32_t>(detail::read_operand(ctx, 1, 4, &ok));
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto [result_value, exceptions] = convert_int_to_fp<double>(ctx.state, input);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = dst_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTSI2SD_XMM_RM64(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto dst_bits = read_vec(ctx.state, dst_reg);
  const auto input = static_cast<std::int64_t>(detail::read_operand(ctx, 1, 8, &ok));
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto [result_value, exceptions] = convert_int_to_fp<double>(ctx.state, input);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  big_uint out = dst_bits;
  pack_lane(out, 0, result_value);
  write_vec(ctx.state, dst_reg, out, false);
  return {};
}

ExecutionResult handle_code_CVTSD2SI_R32_XMMM64(ExecutionContext& ctx) {
  bool ok = false;
  const auto src_bits = read_operand(ctx, 1, sizeof(double), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto input = unpack_lane<double>(src_bits, 0);
  const auto [bits, exceptions] = convert_fp_to_int_bits<std::int32_t>(ctx.state, input, false);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), bits, 4);
  return {};
}

ExecutionResult handle_code_CVTSD2SI_R64_XMMM64(ExecutionContext& ctx) {
  bool ok = false;
  const auto src_bits = read_operand(ctx, 1, sizeof(double), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto input = unpack_lane<double>(src_bits, 0);
  const auto [bits, exceptions] = convert_fp_to_int_bits<std::int64_t>(ctx.state, input, false);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), bits, 8);
  return {};
}

ExecutionResult handle_code_CVTTSD2SI_R32_XMMM64(ExecutionContext& ctx) {
  bool ok = false;
  const auto src_bits = read_operand(ctx, 1, sizeof(double), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto input = unpack_lane<double>(src_bits, 0);
  const auto [bits, exceptions] = convert_fp_to_int_bits<std::int32_t>(ctx.state, input, true);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), bits, 4);
  return {};
}

ExecutionResult handle_code_CVTTSD2SI_R64_XMMM64(ExecutionContext& ctx) {
  bool ok = false;
  const auto src_bits = read_operand(ctx, 1, sizeof(double), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto input = unpack_lane<double>(src_bits, 0);
  const auto [bits, exceptions] = convert_fp_to_int_bits<std::int64_t>(ctx.state, input, true);
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), bits, 8);
  return {};
}

ExecutionResult handle_code_CVTPI2PD_XMM_MMM64(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = true;
  std::uint64_t src = 0;
  if (ctx.instr.op_kind(1) == iced_x86::OpKind::REGISTER && is_mmx_register(ctx.instr.op_register(1))) {
    src = ctx.state.mmx_get(mmx_index(ctx.instr.op_register(1)));
  } else {
    src = detail::read_operand(ctx, 1, 8, &ok);
  }
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  pack_lane(out, 0, static_cast<double>(static_cast<std::int32_t>(src & 0xFFFFFFFFu)));
  pack_lane(out, sizeof(double), static_cast<double>(static_cast<std::int32_t>((src >> 32) & 0xFFFFFFFFu)));
  write_vec(ctx.state, ctx.instr.op_register(0), out, false);
  return {};
}

ExecutionResult handle_code_CVTPD2PI_MM_XMMM128(ExecutionContext& ctx) {
  bool ok = false;
  const auto src_bits = read_operand(ctx, 1, 16, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t packed = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<double>(src_bits, lane * sizeof(double));
    const auto [bits, lane_exceptions] = convert_fp_to_int_bits<std::int32_t>(ctx.state, input, false);
    exceptions |= lane_exceptions;
    packed |= (bits & 0xFFFFFFFFull) << (lane * 32);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.mmx_set(mmx_index(ctx.instr.op_register(0)), packed);
  return {};
}

ExecutionResult handle_code_CVTTPD2PI_MM_XMMM128(ExecutionContext& ctx) {
  bool ok = false;
  const auto src_bits = read_operand(ctx, 1, 16, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  std::uint64_t packed = 0;
  std::uint32_t exceptions = 0;
  for (std::size_t lane = 0; lane < 2; ++lane) {
    const auto input = unpack_lane<double>(src_bits, lane * sizeof(double));
    const auto [bits, lane_exceptions] = convert_fp_to_int_bits<std::int32_t>(ctx.state, input, true);
    exceptions |= lane_exceptions;
    packed |= (bits & 0xFFFFFFFFull) << (lane * 32);
  }
  if (exceptions != 0) {
    auto result = sse_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.mmx_set(mmx_index(ctx.instr.op_register(0)), packed);
  return {};
}

KUBERA_VEX_PACKED_BIN(VEX_VADDPS_XMM_XMM_XMMM128, float, [](float a, float b) { return a + b; })
KUBERA_VEX_PACKED_BIN(VEX_VADDPD_XMM_XMM_XMMM128, double, [](double a, double b) { return a + b; })
KUBERA_VEX_PACKED_BIN(VEX_VADDPS_YMM_YMM_YMMM256, float, [](float a, float b) { return a + b; })
KUBERA_VEX_PACKED_BIN(VEX_VADDPD_YMM_YMM_YMMM256, double, [](double a, double b) { return a + b; })
KUBERA_VEX_PACKED_BIN(VEX_VSUBPS_XMM_XMM_XMMM128, float, [](float a, float b) { return a - b; })
KUBERA_VEX_PACKED_BIN(VEX_VSUBPD_XMM_XMM_XMMM128, double, [](double a, double b) { return a - b; })
KUBERA_VEX_PACKED_BIN(VEX_VSUBPS_YMM_YMM_YMMM256, float, [](float a, float b) { return a - b; })
KUBERA_VEX_PACKED_BIN(VEX_VSUBPD_YMM_YMM_YMMM256, double, [](double a, double b) { return a - b; })
KUBERA_VEX_PACKED_BIN(VEX_VMULPS_XMM_XMM_XMMM128, float, [](float a, float b) { return a * b; })
KUBERA_VEX_PACKED_BIN(VEX_VMULPD_XMM_XMM_XMMM128, double, [](double a, double b) { return a * b; })
KUBERA_VEX_PACKED_BIN(VEX_VMULPS_YMM_YMM_YMMM256, float, [](float a, float b) { return a * b; })
KUBERA_VEX_PACKED_BIN(VEX_VMULPD_YMM_YMM_YMMM256, double, [](double a, double b) { return a * b; })
KUBERA_VEX_PACKED_BIN(VEX_VDIVPS_XMM_XMM_XMMM128, float, [](float a, float b) { return a / b; })
KUBERA_VEX_PACKED_BIN(VEX_VDIVPD_XMM_XMM_XMMM128, double, [](double a, double b) { return a / b; })
KUBERA_VEX_PACKED_BIN(VEX_VDIVPS_YMM_YMM_YMMM256, float, [](float a, float b) { return a / b; })
KUBERA_VEX_PACKED_BIN(VEX_VDIVPD_YMM_YMM_YMMM256, double, [](double a, double b) { return a / b; })
KUBERA_VEX_PACKED_BIN(VEX_VMINPS_XMM_XMM_XMMM128, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMINPD_XMM_XMM_XMMM128, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMINPS_YMM_YMM_YMMM256, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMINPD_YMM_YMM_YMMM256, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMAXPS_XMM_XMM_XMMM128, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMAXPD_XMM_XMM_XMMM128, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMAXPS_YMM_YMM_YMMM256, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(VEX_VMAXPD_YMM_YMM_YMMM256, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_UNARY(VEX_VSQRTPS_XMM_XMM_XMMM128, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(VEX_VSQRTPD_XMM_XMM_XMMM128, double, [](double a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(VEX_VSQRTPS_YMM_YMM_YMMM256, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(VEX_VSQRTPD_YMM_YMM_YMMM256, double, [](double a) { return std::sqrt(a); })

KUBERA_VEX_SCALAR_BIN(VEX_VADDSS_XMM_XMM_XMMM32, float, [](float a, float b) { return a + b; })
KUBERA_VEX_SCALAR_BIN(VEX_VADDSD_XMM_XMM_XMMM64, double, [](double a, double b) { return a + b; })
KUBERA_VEX_SCALAR_BIN(VEX_VSUBSS_XMM_XMM_XMMM32, float, [](float a, float b) { return a - b; })
KUBERA_VEX_SCALAR_BIN(VEX_VSUBSD_XMM_XMM_XMMM64, double, [](double a, double b) { return a - b; })
KUBERA_VEX_SCALAR_BIN(VEX_VMULSS_XMM_XMM_XMMM32, float, [](float a, float b) { return a * b; })
KUBERA_VEX_SCALAR_BIN(VEX_VMULSD_XMM_XMM_XMMM64, double, [](double a, double b) { return a * b; })
KUBERA_VEX_SCALAR_BIN(VEX_VDIVSS_XMM_XMM_XMMM32, float, [](float a, float b) { return a / b; })
KUBERA_VEX_SCALAR_BIN(VEX_VDIVSD_XMM_XMM_XMMM64, double, [](double a, double b) { return a / b; })
KUBERA_VEX_SCALAR_BIN(VEX_VMINSS_XMM_XMM_XMMM32, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_SCALAR_BIN(VEX_VMINSD_XMM_XMM_XMMM64, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_SCALAR_BIN(VEX_VMAXSS_XMM_XMM_XMMM32, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_SCALAR_BIN(VEX_VMAXSD_XMM_XMM_XMMM64, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_SCALAR_UNARY(VEX_VSQRTSS_XMM_XMM_XMMM32, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_SCALAR_UNARY(VEX_VSQRTSD_XMM_XMM_XMMM64, double, [](double a) { return std::sqrt(a); })

KUBERA_VEX_LOGIC(VEX_VANDPS_XMM_XMM_XMMM128, 0)
KUBERA_VEX_LOGIC(VEX_VANDPD_XMM_XMM_XMMM128, 0)
KUBERA_VEX_LOGIC(VEX_VANDNPS_XMM_XMM_XMMM128, 3)
KUBERA_VEX_LOGIC(VEX_VANDNPD_XMM_XMM_XMMM128, 3)
KUBERA_VEX_LOGIC(VEX_VORPS_XMM_XMM_XMMM128, 1)
KUBERA_VEX_LOGIC(VEX_VORPD_XMM_XMM_XMMM128, 1)
KUBERA_VEX_LOGIC(VEX_VXORPS_XMM_XMM_XMMM128, 2)
KUBERA_VEX_LOGIC(VEX_VXORPD_XMM_XMM_XMMM128, 2)
KUBERA_VEX_LOGIC(VEX_VANDPS_YMM_YMM_YMMM256, 0)
KUBERA_VEX_LOGIC(VEX_VANDPD_YMM_YMM_YMMM256, 0)
KUBERA_VEX_LOGIC(VEX_VANDNPS_YMM_YMM_YMMM256, 3)
KUBERA_VEX_LOGIC(VEX_VANDNPD_YMM_YMM_YMMM256, 3)
KUBERA_VEX_LOGIC(VEX_VORPS_YMM_YMM_YMMM256, 1)
KUBERA_VEX_LOGIC(VEX_VORPD_YMM_YMM_YMMM256, 1)
KUBERA_VEX_LOGIC(VEX_VXORPS_YMM_YMM_YMMM256, 2)
KUBERA_VEX_LOGIC(VEX_VXORPD_YMM_YMM_YMMM256, 2)

KUBERA_VEX_COMPARE(VEX_VCOMISS_XMM_XMM_XMMM32, float, false)
KUBERA_VEX_COMPARE(VEX_VUCOMISS_XMM_XMM_XMMM32, float, true)
KUBERA_VEX_COMPARE(VEX_VCOMISD_XMM_XMM_XMMM64, double, false)
KUBERA_VEX_COMPARE(VEX_VUCOMISD_XMM_XMM_XMMM64, double, true)

KUBERA_VEX_PACKED_BIN(EVEX_VADDPS_XMM_K1Z_XMM_XMMM128, float, [](float a, float b) { return a + b; })
KUBERA_VEX_PACKED_BIN(EVEX_VADDPD_XMM_K1Z_XMM_XMMM128, double, [](double a, double b) { return a + b; })
KUBERA_VEX_PACKED_BIN(EVEX_VADDPS_YMM_K1Z_YMM_YMMM256, float, [](float a, float b) { return a + b; })
KUBERA_VEX_PACKED_BIN(EVEX_VADDPD_YMM_K1Z_YMM_YMMM256, double, [](double a, double b) { return a + b; })
KUBERA_VEX_PACKED_BIN(EVEX_VADDPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a, float b) { return a + b; })
KUBERA_VEX_PACKED_BIN(EVEX_VADDPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a, double b) { return a + b; })
KUBERA_VEX_PACKED_BIN(EVEX_VSUBPS_XMM_K1Z_XMM_XMMM128, float, [](float a, float b) { return a - b; })
KUBERA_VEX_PACKED_BIN(EVEX_VSUBPD_XMM_K1Z_XMM_XMMM128, double, [](double a, double b) { return a - b; })
KUBERA_VEX_PACKED_BIN(EVEX_VSUBPS_YMM_K1Z_YMM_YMMM256, float, [](float a, float b) { return a - b; })
KUBERA_VEX_PACKED_BIN(EVEX_VSUBPD_YMM_K1Z_YMM_YMMM256, double, [](double a, double b) { return a - b; })
KUBERA_VEX_PACKED_BIN(EVEX_VSUBPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a, float b) { return a - b; })
KUBERA_VEX_PACKED_BIN(EVEX_VSUBPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a, double b) { return a - b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMULPS_XMM_K1Z_XMM_XMMM128, float, [](float a, float b) { return a * b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMULPD_XMM_K1Z_XMM_XMMM128, double, [](double a, double b) { return a * b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMULPS_YMM_K1Z_YMM_YMMM256, float, [](float a, float b) { return a * b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMULPD_YMM_K1Z_YMM_YMMM256, double, [](double a, double b) { return a * b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMULPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a, float b) { return a * b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMULPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a, double b) { return a * b; })
KUBERA_VEX_PACKED_BIN(EVEX_VDIVPS_XMM_K1Z_XMM_XMMM128, float, [](float a, float b) { return a / b; })
KUBERA_VEX_PACKED_BIN(EVEX_VDIVPD_XMM_K1Z_XMM_XMMM128, double, [](double a, double b) { return a / b; })
KUBERA_VEX_PACKED_BIN(EVEX_VDIVPS_YMM_K1Z_YMM_YMMM256, float, [](float a, float b) { return a / b; })
KUBERA_VEX_PACKED_BIN(EVEX_VDIVPD_YMM_K1Z_YMM_YMMM256, double, [](double a, double b) { return a / b; })
KUBERA_VEX_PACKED_BIN(EVEX_VDIVPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a, float b) { return a / b; })
KUBERA_VEX_PACKED_BIN(EVEX_VDIVPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a, double b) { return a / b; })
KUBERA_VEX_PACKED_BIN(EVEX_VMINPS_XMM_K1Z_XMM_XMMM128, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMINPD_XMM_K1Z_XMM_XMMM128, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMINPS_YMM_K1Z_YMM_YMMM256, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMINPD_YMM_K1Z_YMM_YMMM256, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMINPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMINPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMAXPS_XMM_K1Z_XMM_XMMM128, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMAXPD_XMM_K1Z_XMM_XMMM128, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMAXPS_YMM_K1Z_YMM_YMMM256, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMAXPD_YMM_K1Z_YMM_YMMM256, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMAXPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_BIN(EVEX_VMAXPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_PACKED_UNARY(EVEX_VSQRTPS_XMM_K1Z_XMM_XMMM128, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(EVEX_VSQRTPD_XMM_K1Z_XMM_XMMM128, double, [](double a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(EVEX_VSQRTPS_YMM_K1Z_YMM_YMMM256, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(EVEX_VSQRTPD_YMM_K1Z_YMM_YMMM256, double, [](double a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(EVEX_VSQRTPS_ZMM_K1Z_ZMM_ZMMM512, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_PACKED_UNARY(EVEX_VSQRTPD_ZMM_K1Z_ZMM_ZMMM512, double, [](double a) { return std::sqrt(a); })

KUBERA_VEX_SCALAR_BIN(EVEX_VADDSS_XMM_K1Z_XMM_XMMM32, float, [](float a, float b) { return a + b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VADDSD_XMM_K1Z_XMM_XMMM64, double, [](double a, double b) { return a + b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VSUBSS_XMM_K1Z_XMM_XMMM32, float, [](float a, float b) { return a - b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VSUBSD_XMM_K1Z_XMM_XMMM64, double, [](double a, double b) { return a - b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VMULSS_XMM_K1Z_XMM_XMMM32, float, [](float a, float b) { return a * b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VMULSD_XMM_K1Z_XMM_XMMM64, double, [](double a, double b) { return a * b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VDIVSS_XMM_K1Z_XMM_XMMM32, float, [](float a, float b) { return a / b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VDIVSD_XMM_K1Z_XMM_XMMM64, double, [](double a, double b) { return a / b; })
KUBERA_VEX_SCALAR_BIN(EVEX_VMINSS_XMM_K1Z_XMM_XMMM32, float, [](float a, float b) { return std::fmin(a, b); })
KUBERA_VEX_SCALAR_BIN(EVEX_VMINSD_XMM_K1Z_XMM_XMMM64, double, [](double a, double b) { return std::fmin(a, b); })
KUBERA_VEX_SCALAR_BIN(EVEX_VMAXSS_XMM_K1Z_XMM_XMMM32, float, [](float a, float b) { return std::fmax(a, b); })
KUBERA_VEX_SCALAR_BIN(EVEX_VMAXSD_XMM_K1Z_XMM_XMMM64, double, [](double a, double b) { return std::fmax(a, b); })
KUBERA_VEX_SCALAR_UNARY(EVEX_VSQRTSS_XMM_K1Z_XMM_XMMM32, float, [](float a) { return std::sqrt(a); })
KUBERA_VEX_SCALAR_UNARY(EVEX_VSQRTSD_XMM_K1Z_XMM_XMMM64, double, [](double a) { return std::sqrt(a); })

KUBERA_VEX_LOGIC(EVEX_VANDPS_XMM_K1Z_XMM_XMMM128, 0)
KUBERA_VEX_LOGIC(EVEX_VANDPD_XMM_K1Z_XMM_XMMM128, 0)
KUBERA_VEX_LOGIC(EVEX_VANDNPS_XMM_K1Z_XMM_XMMM128, 3)
KUBERA_VEX_LOGIC(EVEX_VANDNPD_XMM_K1Z_XMM_XMMM128, 3)
KUBERA_VEX_LOGIC(EVEX_VORPS_XMM_K1Z_XMM_XMMM128, 1)
KUBERA_VEX_LOGIC(EVEX_VORPD_XMM_K1Z_XMM_XMMM128, 1)
KUBERA_VEX_LOGIC(EVEX_VXORPS_XMM_K1Z_XMM_XMMM128, 2)
KUBERA_VEX_LOGIC(EVEX_VXORPD_XMM_K1Z_XMM_XMMM128, 2)
KUBERA_VEX_LOGIC(EVEX_VANDPS_YMM_K1Z_YMM_YMMM256, 0)
KUBERA_VEX_LOGIC(EVEX_VANDPD_YMM_K1Z_YMM_YMMM256, 0)
KUBERA_VEX_LOGIC(EVEX_VANDNPS_YMM_K1Z_YMM_YMMM256, 3)
KUBERA_VEX_LOGIC(EVEX_VANDNPD_YMM_K1Z_YMM_YMMM256, 3)
KUBERA_VEX_LOGIC(EVEX_VORPS_YMM_K1Z_YMM_YMMM256, 1)
KUBERA_VEX_LOGIC(EVEX_VORPD_YMM_K1Z_YMM_YMMM256, 1)
KUBERA_VEX_LOGIC(EVEX_VXORPS_YMM_K1Z_YMM_YMMM256, 2)
KUBERA_VEX_LOGIC(EVEX_VXORPD_YMM_K1Z_YMM_YMMM256, 2)
KUBERA_VEX_LOGIC(EVEX_VANDPS_ZMM_K1Z_ZMM_ZMMM512, 0)
KUBERA_VEX_LOGIC(EVEX_VANDPD_ZMM_K1Z_ZMM_ZMMM512, 0)
KUBERA_VEX_LOGIC(EVEX_VANDNPS_ZMM_K1Z_ZMM_ZMMM512, 3)
KUBERA_VEX_LOGIC(EVEX_VANDNPD_ZMM_K1Z_ZMM_ZMMM512, 3)
KUBERA_VEX_LOGIC(EVEX_VORPS_ZMM_K1Z_ZMM_ZMMM512, 1)
KUBERA_VEX_LOGIC(EVEX_VORPD_ZMM_K1Z_ZMM_ZMMM512, 1)
KUBERA_VEX_LOGIC(EVEX_VXORPS_ZMM_K1Z_ZMM_ZMMM512, 2)
KUBERA_VEX_LOGIC(EVEX_VXORPD_ZMM_K1Z_ZMM_ZMMM512, 2)

KUBERA_VEX_COMPARE(EVEX_VCOMISS_XMM_K1Z_XMM_XMMM32, float, false)
KUBERA_VEX_COMPARE(EVEX_VUCOMISS_XMM_K1Z_XMM_XMMM32, float, true)
KUBERA_VEX_COMPARE(EVEX_VCOMISD_XMM_K1Z_XMM_XMMM64, double, false)
KUBERA_VEX_COMPARE(EVEX_VUCOMISD_XMM_K1Z_XMM_XMMM64, double, true)

#undef KUBERA_PACKED_BIN
#undef KUBERA_SCALAR_BIN
#undef KUBERA_PACKED_UNARY
#undef KUBERA_SCALAR_UNARY
#undef KUBERA_VEX_PACKED_BIN
#undef KUBERA_VEX_SCALAR_BIN
#undef KUBERA_VEX_PACKED_UNARY
#undef KUBERA_VEX_SCALAR_UNARY
#undef KUBERA_VEX_LOGIC
#undef KUBERA_VEX_COMPARE

}  // namespace seven::handlers



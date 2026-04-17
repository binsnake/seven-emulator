#include <array>
#include <cstdint>
#include <cstring>

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
  if (bytes >= sizeof(big_uint)) { big_uint _all_ones(0); return ~_all_ones; }
  return (big_uint(1) << (bytes * 8)) - 1;
}

big_uint read_vec(const CpuState& state, iced_x86::Register reg) {
  return state.vectors[vector_index(reg)].value & mask(vector_width(reg));
}

void write_vec(CpuState& state, iced_x86::Register reg, big_uint value, bool zero_upper = false) {
  auto& slot = state.vectors[vector_index(reg)].value;
  auto m = mask(vector_width(reg));
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
T load_lane(const big_uint& value, std::size_t byte_offset) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> ((byte_offset + i) * 8)) & 0xFFu);
  }
  T out{};
  std::memcpy(&out, bytes.data(), sizeof(T));
  return out;
}

template <typename T>
void store_lane(big_uint& value, std::size_t byte_offset, T lane_value) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &lane_value, sizeof(T));
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    const auto bit_offset = (byte_offset + i) * 8;
    value &= ~(big_uint(0xFFu) << bit_offset);
    value |= (big_uint(bytes[i]) << bit_offset);
  }
}

template <typename Dst, typename Src, typename Fn>
big_uint pack_lane_pair(const big_uint& lhs, const big_uint& rhs, std::size_t lane_offset, Fn&& fn) {
  constexpr std::size_t lhs_count = 16 / sizeof(Src);
  constexpr std::size_t dst_count = 16 / sizeof(Dst);
  std::array<Src, lhs_count> a{};
  std::array<Src, lhs_count> b{};
  for (std::size_t i = 0; i < lhs_count; ++i) {
    a[i] = load_lane<Src>(lhs, lane_offset + i * sizeof(Src));
    b[i] = load_lane<Src>(rhs, lane_offset + i * sizeof(Src));
  }
  const auto out = fn(a, b);
  big_uint result = 0;
  for (std::size_t i = 0; i < dst_count; ++i) {
    store_lane(result, i * sizeof(Dst), out[i]);
  }
  return result;
}

template <typename Dst, typename Src, typename Fn>
ExecutionResult legacy_pack(ExecutionContext& ctx, Fn&& fn) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, kXmmBytes, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  out |= pack_lane_pair<Dst, Src>(lhs_bits, rhs_bits, 0, fn);
  write_vec(ctx.state, dst_reg, out);
  return {};
}

template <typename Dst, typename Src, typename Fn>
ExecutionResult vex_pack(ExecutionContext& ctx, Fn&& fn) {
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
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += kXmmBytes) {
    out |= (pack_lane_pair<Dst, Src>(lhs_bits, rhs_bits, lane, fn) << (lane * 8));
  }
  write_vec(ctx.state, dst_reg, out, true);
  return {};
}

template <typename Src, typename Fn>
ExecutionResult legacy_unpack(ExecutionContext& ctx, Fn&& fn) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, kXmmBytes, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  out |= fn(lhs_bits, rhs_bits, 0);
  write_vec(ctx.state, dst_reg, out);
  return {};
}

template <typename Src, typename Fn>
ExecutionResult vex_unpack(ExecutionContext& ctx, Fn&& fn) {
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
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += kXmmBytes) {
    out |= (fn(lhs_bits, rhs_bits, lane) << (lane * 8));
  }
  write_vec(ctx.state, dst_reg, out, true);
  return {};
}

template <typename T>
T sat_pack_signed(int64_t value) {
  const auto max = static_cast<int64_t>(std::numeric_limits<T>::max());
  const auto min = static_cast<int64_t>(std::numeric_limits<T>::min());
  if (value > max) return std::numeric_limits<T>::max();
  if (value < min) return std::numeric_limits<T>::min();
  return static_cast<T>(value);
}

template <typename T>
T sat_pack_unsigned(int64_t value) {
  if (value < 0) return T{0};
  const auto max = static_cast<int64_t>(std::numeric_limits<T>::max());
  if (value > max) return std::numeric_limits<T>::max();
  return static_cast<T>(value);
}

}  // namespace

ExecutionResult handle_code_PACKSSWB_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_pack<std::int8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int8_t, 16>{
      sat_pack_signed<std::int8_t>(lhs[0]), sat_pack_signed<std::int8_t>(lhs[1]),
      sat_pack_signed<std::int8_t>(lhs[2]), sat_pack_signed<std::int8_t>(lhs[3]),
      sat_pack_signed<std::int8_t>(lhs[4]), sat_pack_signed<std::int8_t>(lhs[5]),
      sat_pack_signed<std::int8_t>(lhs[6]), sat_pack_signed<std::int8_t>(lhs[7]),
      sat_pack_signed<std::int8_t>(rhs[0]), sat_pack_signed<std::int8_t>(rhs[1]),
      sat_pack_signed<std::int8_t>(rhs[2]), sat_pack_signed<std::int8_t>(rhs[3]),
      sat_pack_signed<std::int8_t>(rhs[4]), sat_pack_signed<std::int8_t>(rhs[5]),
      sat_pack_signed<std::int8_t>(rhs[6]), sat_pack_signed<std::int8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_PACKSSDW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_pack<std::int16_t, std::int32_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int16_t, 8>{
      sat_pack_signed<std::int16_t>(lhs[0]), sat_pack_signed<std::int16_t>(lhs[1]),
      sat_pack_signed<std::int16_t>(lhs[2]), sat_pack_signed<std::int16_t>(lhs[3]),
      sat_pack_signed<std::int16_t>(rhs[0]), sat_pack_signed<std::int16_t>(rhs[1]),
      sat_pack_signed<std::int16_t>(rhs[2]), sat_pack_signed<std::int16_t>(rhs[3]),
    };
  });
}

ExecutionResult handle_code_PACKUSWB_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_pack<std::uint8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::uint8_t, 16>{
      sat_pack_unsigned<std::uint8_t>(lhs[0]), sat_pack_unsigned<std::uint8_t>(lhs[1]),
      sat_pack_unsigned<std::uint8_t>(lhs[2]), sat_pack_unsigned<std::uint8_t>(lhs[3]),
      sat_pack_unsigned<std::uint8_t>(lhs[4]), sat_pack_unsigned<std::uint8_t>(lhs[5]),
      sat_pack_unsigned<std::uint8_t>(lhs[6]), sat_pack_unsigned<std::uint8_t>(lhs[7]),
      sat_pack_unsigned<std::uint8_t>(rhs[0]), sat_pack_unsigned<std::uint8_t>(rhs[1]),
      sat_pack_unsigned<std::uint8_t>(rhs[2]), sat_pack_unsigned<std::uint8_t>(rhs[3]),
      sat_pack_unsigned<std::uint8_t>(rhs[4]), sat_pack_unsigned<std::uint8_t>(rhs[5]),
      sat_pack_unsigned<std::uint8_t>(rhs[6]), sat_pack_unsigned<std::uint8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_PUNPCKLBW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_PUNPCKHBW_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + 8 + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + 8 + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_PUNPCKLWD_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_PUNPCKHWD_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + 8 + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + 8 + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_PUNPCKLDQ_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_PUNPCKHDQ_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + 8 + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + 8 + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_PUNPCKLQDQ_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint64_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    const auto a = load_lane<std::uint64_t>(lhs, lane);
    const auto b = load_lane<std::uint64_t>(rhs, lane);
    store_lane(out, 0, a);
    store_lane(out, 8, b);
    return out;
  });
}

ExecutionResult handle_code_PUNPCKHQDQ_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unpack<std::uint64_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    const auto a = load_lane<std::uint64_t>(lhs, lane + 8);
    const auto b = load_lane<std::uint64_t>(rhs, lane + 8);
    store_lane(out, 0, a);
    store_lane(out, 8, b);
    return out;
  });
}

ExecutionResult handle_code_VEX_VPACKSSWB_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_pack<std::int8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int8_t, 16>{
      sat_pack_signed<std::int8_t>(lhs[0]), sat_pack_signed<std::int8_t>(lhs[1]),
      sat_pack_signed<std::int8_t>(lhs[2]), sat_pack_signed<std::int8_t>(lhs[3]),
      sat_pack_signed<std::int8_t>(lhs[4]), sat_pack_signed<std::int8_t>(lhs[5]),
      sat_pack_signed<std::int8_t>(lhs[6]), sat_pack_signed<std::int8_t>(lhs[7]),
      sat_pack_signed<std::int8_t>(rhs[0]), sat_pack_signed<std::int8_t>(rhs[1]),
      sat_pack_signed<std::int8_t>(rhs[2]), sat_pack_signed<std::int8_t>(rhs[3]),
      sat_pack_signed<std::int8_t>(rhs[4]), sat_pack_signed<std::int8_t>(rhs[5]),
      sat_pack_signed<std::int8_t>(rhs[6]), sat_pack_signed<std::int8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_VEX_VPACKSSDW_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_pack<std::int16_t, std::int32_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int16_t, 8>{
      sat_pack_signed<std::int16_t>(lhs[0]), sat_pack_signed<std::int16_t>(lhs[1]),
      sat_pack_signed<std::int16_t>(lhs[2]), sat_pack_signed<std::int16_t>(lhs[3]),
      sat_pack_signed<std::int16_t>(rhs[0]), sat_pack_signed<std::int16_t>(rhs[1]),
      sat_pack_signed<std::int16_t>(rhs[2]), sat_pack_signed<std::int16_t>(rhs[3]),
    };
  });
}

ExecutionResult handle_code_VEX_VPACKUSWB_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_pack<std::uint8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::uint8_t, 16>{
      sat_pack_unsigned<std::uint8_t>(lhs[0]), sat_pack_unsigned<std::uint8_t>(lhs[1]),
      sat_pack_unsigned<std::uint8_t>(lhs[2]), sat_pack_unsigned<std::uint8_t>(lhs[3]),
      sat_pack_unsigned<std::uint8_t>(lhs[4]), sat_pack_unsigned<std::uint8_t>(lhs[5]),
      sat_pack_unsigned<std::uint8_t>(lhs[6]), sat_pack_unsigned<std::uint8_t>(lhs[7]),
      sat_pack_unsigned<std::uint8_t>(rhs[0]), sat_pack_unsigned<std::uint8_t>(rhs[1]),
      sat_pack_unsigned<std::uint8_t>(rhs[2]), sat_pack_unsigned<std::uint8_t>(rhs[3]),
      sat_pack_unsigned<std::uint8_t>(rhs[4]), sat_pack_unsigned<std::uint8_t>(rhs[5]),
      sat_pack_unsigned<std::uint8_t>(rhs[6]), sat_pack_unsigned<std::uint8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_VEX_VPUNPCKLBW_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKHBW_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + 8 + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + 8 + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKLWD_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKHWD_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + 8 + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + 8 + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKLDQ_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKHDQ_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + 8 + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + 8 + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPACKSSWB_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_pack<std::int8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int8_t, 16>{
      sat_pack_signed<std::int8_t>(lhs[0]), sat_pack_signed<std::int8_t>(lhs[1]),
      sat_pack_signed<std::int8_t>(lhs[2]), sat_pack_signed<std::int8_t>(lhs[3]),
      sat_pack_signed<std::int8_t>(lhs[4]), sat_pack_signed<std::int8_t>(lhs[5]),
      sat_pack_signed<std::int8_t>(lhs[6]), sat_pack_signed<std::int8_t>(lhs[7]),
      sat_pack_signed<std::int8_t>(rhs[0]), sat_pack_signed<std::int8_t>(rhs[1]),
      sat_pack_signed<std::int8_t>(rhs[2]), sat_pack_signed<std::int8_t>(rhs[3]),
      sat_pack_signed<std::int8_t>(rhs[4]), sat_pack_signed<std::int8_t>(rhs[5]),
      sat_pack_signed<std::int8_t>(rhs[6]), sat_pack_signed<std::int8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_VEX_VPACKSSDW_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_pack<std::int16_t, std::int32_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int16_t, 8>{
      sat_pack_signed<std::int16_t>(lhs[0]), sat_pack_signed<std::int16_t>(lhs[1]),
      sat_pack_signed<std::int16_t>(lhs[2]), sat_pack_signed<std::int16_t>(lhs[3]),
      sat_pack_signed<std::int16_t>(rhs[0]), sat_pack_signed<std::int16_t>(rhs[1]),
      sat_pack_signed<std::int16_t>(rhs[2]), sat_pack_signed<std::int16_t>(rhs[3]),
    };
  });
}

ExecutionResult handle_code_VEX_VPACKUSWB_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_pack<std::uint8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::uint8_t, 16>{
      sat_pack_unsigned<std::uint8_t>(lhs[0]), sat_pack_unsigned<std::uint8_t>(lhs[1]),
      sat_pack_unsigned<std::uint8_t>(lhs[2]), sat_pack_unsigned<std::uint8_t>(lhs[3]),
      sat_pack_unsigned<std::uint8_t>(lhs[4]), sat_pack_unsigned<std::uint8_t>(lhs[5]),
      sat_pack_unsigned<std::uint8_t>(lhs[6]), sat_pack_unsigned<std::uint8_t>(lhs[7]),
      sat_pack_unsigned<std::uint8_t>(rhs[0]), sat_pack_unsigned<std::uint8_t>(rhs[1]),
      sat_pack_unsigned<std::uint8_t>(rhs[2]), sat_pack_unsigned<std::uint8_t>(rhs[3]),
      sat_pack_unsigned<std::uint8_t>(rhs[4]), sat_pack_unsigned<std::uint8_t>(rhs[5]),
      sat_pack_unsigned<std::uint8_t>(rhs[6]), sat_pack_unsigned<std::uint8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_VEX_VPUNPCKLBW_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKHBW_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + 8 + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + 8 + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKLWD_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKHWD_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + 8 + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + 8 + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKLDQ_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_VEX_VPUNPCKHDQ_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + 8 + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + 8 + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}
ExecutionResult handle_code_EVEX_VPACKSSWB_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_pack<std::int8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int8_t, 16>{
      sat_pack_signed<std::int8_t>(lhs[0]), sat_pack_signed<std::int8_t>(lhs[1]),
      sat_pack_signed<std::int8_t>(lhs[2]), sat_pack_signed<std::int8_t>(lhs[3]),
      sat_pack_signed<std::int8_t>(lhs[4]), sat_pack_signed<std::int8_t>(lhs[5]),
      sat_pack_signed<std::int8_t>(lhs[6]), sat_pack_signed<std::int8_t>(lhs[7]),
      sat_pack_signed<std::int8_t>(rhs[0]), sat_pack_signed<std::int8_t>(rhs[1]),
      sat_pack_signed<std::int8_t>(rhs[2]), sat_pack_signed<std::int8_t>(rhs[3]),
      sat_pack_signed<std::int8_t>(rhs[4]), sat_pack_signed<std::int8_t>(rhs[5]),
      sat_pack_signed<std::int8_t>(rhs[6]), sat_pack_signed<std::int8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPACKSSDW_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_pack<std::int16_t, std::int32_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int16_t, 8>{
      sat_pack_signed<std::int16_t>(lhs[0]), sat_pack_signed<std::int16_t>(lhs[1]),
      sat_pack_signed<std::int16_t>(lhs[2]), sat_pack_signed<std::int16_t>(lhs[3]),
      sat_pack_signed<std::int16_t>(rhs[0]), sat_pack_signed<std::int16_t>(rhs[1]),
      sat_pack_signed<std::int16_t>(rhs[2]), sat_pack_signed<std::int16_t>(rhs[3]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPACKUSWB_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_pack<std::uint8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::uint8_t, 16>{
      sat_pack_unsigned<std::uint8_t>(lhs[0]), sat_pack_unsigned<std::uint8_t>(lhs[1]),
      sat_pack_unsigned<std::uint8_t>(lhs[2]), sat_pack_unsigned<std::uint8_t>(lhs[3]),
      sat_pack_unsigned<std::uint8_t>(lhs[4]), sat_pack_unsigned<std::uint8_t>(lhs[5]),
      sat_pack_unsigned<std::uint8_t>(lhs[6]), sat_pack_unsigned<std::uint8_t>(lhs[7]),
      sat_pack_unsigned<std::uint8_t>(rhs[0]), sat_pack_unsigned<std::uint8_t>(rhs[1]),
      sat_pack_unsigned<std::uint8_t>(rhs[2]), sat_pack_unsigned<std::uint8_t>(rhs[3]),
      sat_pack_unsigned<std::uint8_t>(rhs[4]), sat_pack_unsigned<std::uint8_t>(rhs[5]),
      sat_pack_unsigned<std::uint8_t>(rhs[6]), sat_pack_unsigned<std::uint8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLBW_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHBW_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + 8 + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + 8 + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLWD_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHWD_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + 8 + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + 8 + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLDQ_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHDQ_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + 8 + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + 8 + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPACKSSWB_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_pack<std::int8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int8_t, 16>{
      sat_pack_signed<std::int8_t>(lhs[0]), sat_pack_signed<std::int8_t>(lhs[1]),
      sat_pack_signed<std::int8_t>(lhs[2]), sat_pack_signed<std::int8_t>(lhs[3]),
      sat_pack_signed<std::int8_t>(lhs[4]), sat_pack_signed<std::int8_t>(lhs[5]),
      sat_pack_signed<std::int8_t>(lhs[6]), sat_pack_signed<std::int8_t>(lhs[7]),
      sat_pack_signed<std::int8_t>(rhs[0]), sat_pack_signed<std::int8_t>(rhs[1]),
      sat_pack_signed<std::int8_t>(rhs[2]), sat_pack_signed<std::int8_t>(rhs[3]),
      sat_pack_signed<std::int8_t>(rhs[4]), sat_pack_signed<std::int8_t>(rhs[5]),
      sat_pack_signed<std::int8_t>(rhs[6]), sat_pack_signed<std::int8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPACKSSDW_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_pack<std::int16_t, std::int32_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int16_t, 8>{
      sat_pack_signed<std::int16_t>(lhs[0]), sat_pack_signed<std::int16_t>(lhs[1]),
      sat_pack_signed<std::int16_t>(lhs[2]), sat_pack_signed<std::int16_t>(lhs[3]),
      sat_pack_signed<std::int16_t>(rhs[0]), sat_pack_signed<std::int16_t>(rhs[1]),
      sat_pack_signed<std::int16_t>(rhs[2]), sat_pack_signed<std::int16_t>(rhs[3]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPACKUSWB_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_pack<std::uint8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::uint8_t, 16>{
      sat_pack_unsigned<std::uint8_t>(lhs[0]), sat_pack_unsigned<std::uint8_t>(lhs[1]),
      sat_pack_unsigned<std::uint8_t>(lhs[2]), sat_pack_unsigned<std::uint8_t>(lhs[3]),
      sat_pack_unsigned<std::uint8_t>(lhs[4]), sat_pack_unsigned<std::uint8_t>(lhs[5]),
      sat_pack_unsigned<std::uint8_t>(lhs[6]), sat_pack_unsigned<std::uint8_t>(lhs[7]),
      sat_pack_unsigned<std::uint8_t>(rhs[0]), sat_pack_unsigned<std::uint8_t>(rhs[1]),
      sat_pack_unsigned<std::uint8_t>(rhs[2]), sat_pack_unsigned<std::uint8_t>(rhs[3]),
      sat_pack_unsigned<std::uint8_t>(rhs[4]), sat_pack_unsigned<std::uint8_t>(rhs[5]),
      sat_pack_unsigned<std::uint8_t>(rhs[6]), sat_pack_unsigned<std::uint8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLBW_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHBW_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + 8 + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + 8 + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLWD_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHWD_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + 8 + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + 8 + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLDQ_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHDQ_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + 8 + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + 8 + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}
ExecutionResult handle_code_EVEX_VPACKSSWB_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_pack<std::int8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int8_t, 16>{
      sat_pack_signed<std::int8_t>(lhs[0]), sat_pack_signed<std::int8_t>(lhs[1]),
      sat_pack_signed<std::int8_t>(lhs[2]), sat_pack_signed<std::int8_t>(lhs[3]),
      sat_pack_signed<std::int8_t>(lhs[4]), sat_pack_signed<std::int8_t>(lhs[5]),
      sat_pack_signed<std::int8_t>(lhs[6]), sat_pack_signed<std::int8_t>(lhs[7]),
      sat_pack_signed<std::int8_t>(rhs[0]), sat_pack_signed<std::int8_t>(rhs[1]),
      sat_pack_signed<std::int8_t>(rhs[2]), sat_pack_signed<std::int8_t>(rhs[3]),
      sat_pack_signed<std::int8_t>(rhs[4]), sat_pack_signed<std::int8_t>(rhs[5]),
      sat_pack_signed<std::int8_t>(rhs[6]), sat_pack_signed<std::int8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPACKSSDW_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_pack<std::int16_t, std::int32_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::int16_t, 8>{
      sat_pack_signed<std::int16_t>(lhs[0]), sat_pack_signed<std::int16_t>(lhs[1]),
      sat_pack_signed<std::int16_t>(lhs[2]), sat_pack_signed<std::int16_t>(lhs[3]),
      sat_pack_signed<std::int16_t>(rhs[0]), sat_pack_signed<std::int16_t>(rhs[1]),
      sat_pack_signed<std::int16_t>(rhs[2]), sat_pack_signed<std::int16_t>(rhs[3]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPACKUSWB_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_pack<std::uint8_t, std::int16_t>(ctx, [](const auto& lhs, const auto& rhs) {
    return std::array<std::uint8_t, 16>{
      sat_pack_unsigned<std::uint8_t>(lhs[0]), sat_pack_unsigned<std::uint8_t>(lhs[1]),
      sat_pack_unsigned<std::uint8_t>(lhs[2]), sat_pack_unsigned<std::uint8_t>(lhs[3]),
      sat_pack_unsigned<std::uint8_t>(lhs[4]), sat_pack_unsigned<std::uint8_t>(lhs[5]),
      sat_pack_unsigned<std::uint8_t>(lhs[6]), sat_pack_unsigned<std::uint8_t>(lhs[7]),
      sat_pack_unsigned<std::uint8_t>(rhs[0]), sat_pack_unsigned<std::uint8_t>(rhs[1]),
      sat_pack_unsigned<std::uint8_t>(rhs[2]), sat_pack_unsigned<std::uint8_t>(rhs[3]),
      sat_pack_unsigned<std::uint8_t>(rhs[4]), sat_pack_unsigned<std::uint8_t>(rhs[5]),
      sat_pack_unsigned<std::uint8_t>(rhs[6]), sat_pack_unsigned<std::uint8_t>(rhs[7]),
    };
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLBW_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHBW_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unpack<std::uint8_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const auto a = load_lane<std::uint8_t>(lhs, lane + 8 + i);
      const auto b = load_lane<std::uint8_t>(rhs, lane + 8 + i);
      store_lane(out, i * 2, a);
      store_lane(out, (i * 2) + 1, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLWD_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHWD_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unpack<std::uint16_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto a = load_lane<std::uint16_t>(lhs, lane + 8 + i * 2);
      const auto b = load_lane<std::uint16_t>(rhs, lane + 8 + i * 2);
      store_lane(out, i * 4, a);
      store_lane(out, (i * 4) + 2, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKLDQ_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

ExecutionResult handle_code_EVEX_VPUNPCKHDQ_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unpack<std::uint32_t>(ctx, [](const auto& lhs, const auto& rhs, std::size_t lane) {
    big_uint out = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto a = load_lane<std::uint32_t>(lhs, lane + 8 + i * 4);
      const auto b = load_lane<std::uint32_t>(rhs, lane + 8 + i * 4);
      store_lane(out, i * 8, a);
      store_lane(out, (i * 8) + 4, b);
    }
    return out;
  });
}

}  // namespace seven::handlers



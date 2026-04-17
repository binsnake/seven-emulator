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

template <typename T, std::size_t N>
std::array<T, N> load_elements(const big_uint& value, std::size_t lane_offset_bytes) {
  std::array<T, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    for (std::size_t j = 0; j < sizeof(T); ++j) {
      bytes[j] = static_cast<std::uint8_t>((value >> ((lane_offset_bytes + (i * sizeof(T)) + j) * 8)) & 0xFFu);
    }
    std::memcpy(&out[i], bytes.data(), sizeof(T));
  }
  return out;
}

template <typename T, std::size_t N>
void store_elements(big_uint& value, std::size_t lane_offset_bytes, const std::array<T, N>& elements) {
  for (std::size_t i = 0; i < N; ++i) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &elements[i], sizeof(T));
    for (std::size_t j = 0; j < sizeof(T); ++j) {
      const auto bit_offset = ((lane_offset_bytes + (i * sizeof(T)) + j) * 8);
      value &= ~(big_uint(0xFFu) << bit_offset);
      value |= (big_uint(bytes[j]) << bit_offset);
    }
  }
}

template <typename T, std::size_t N, typename Fn>
ExecutionResult legacy_binary_lanewise(ExecutionContext& ctx, std::size_t lane_bytes, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto lhs_bits = read_vec(ctx.state, dst_reg);
  const auto rhs_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = lhs_bits;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += lane_bytes) {
    const auto lhs = load_elements<T, N>(lhs_bits, lane);
    const auto rhs = load_elements<T, N>(rhs_bits, lane);
    const auto result = fn(lhs, rhs);
    store_elements<T, N>(out, lane, result);
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, std::size_t N, typename Fn>
ExecutionResult vex_binary_lanewise(ExecutionContext& ctx, std::size_t lane_bytes, Fn&& fn, bool zero_upper = true) {
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
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += lane_bytes) {
    const auto lhs = load_elements<T, N>(lhs_bits, lane);
    const auto rhs = load_elements<T, N>(rhs_bits, lane);
    const auto result = fn(lhs, rhs);
    store_elements<T, N>(out, lane, result);
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, std::size_t N, typename Fn>
ExecutionResult legacy_unary_lanewise(ExecutionContext& ctx, std::size_t lane_bytes, Fn&& fn, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += lane_bytes) {
    const auto src = load_elements<T, N>(src_bits, lane);
    const auto result = fn(src);
    store_elements<T, N>(out, lane, result);
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

template <typename T, std::size_t N, typename Fn>
ExecutionResult vex_unary_lanewise(ExecutionContext& ctx, std::size_t lane_bytes, Fn&& fn, bool zero_upper = true) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, vector_width(dst_reg), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  big_uint out = 0;
  for (std::size_t lane = 0; lane < vector_width(dst_reg); lane += lane_bytes) {
    const auto src = load_elements<T, N>(src_bits, lane);
    const auto result = fn(src);
    store_elements<T, N>(out, lane, result);
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

ExecutionResult duplicate_low_double(ExecutionContext& ctx, bool zero_upper = false) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || !is_vector_register(ctx.instr.op_register(0))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  bool ok = false;
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_bits = read_operand(ctx, 1, sizeof(double), &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto src = load_elements<double, 1>(src_bits, 0);
  big_uint out = 0;
  const auto width = vector_width(dst_reg);
  for (std::size_t lane = 0; lane < width; lane += kXmmBytes) {
    store_elements<double, 2>(out, lane, std::array<double, 2>{src[0], src[0]});
  }
  write_vec(ctx.state, dst_reg, out, zero_upper);
  return {};
}

}  // namespace

ExecutionResult handle_code_UNPCKLPS_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[0], rhs[0], lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_UNPCKLPD_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[0], rhs[0]};
  });
}

ExecutionResult handle_code_UNPCKHPS_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[2], rhs[2], lhs[3], rhs[3]};
  });
}

ExecutionResult handle_code_UNPCKHPD_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_SHUFPS_XMM_XMMM128(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return legacy_binary_lanewise<float, 4>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{
      lhs[imm & 0x3u],
      lhs[(imm >> 2) & 0x3u],
      rhs[(imm >> 4) & 0x3u],
      rhs[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_SHUFPD_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return legacy_binary_lanewise<double, 2>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{
      lhs[(imm >> 0) & 0x1u],
      rhs[(imm >> 1) & 0x1u],
    };
  });
}

ExecutionResult handle_code_PSHUFD_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return legacy_unary_lanewise<std::uint32_t, 4>(ctx, kXmmBytes, [imm](const auto& src) {
    return std::array<std::uint32_t, 4>{
      src[(imm >> 0) & 0x3u],
      src[(imm >> 2) & 0x3u],
      src[(imm >> 4) & 0x3u],
      src[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_PSHUFLW_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return legacy_unary_lanewise<std::uint16_t, 8>(ctx, kXmmBytes, [imm](const auto& src) {
    return std::array<std::uint16_t, 8>{
      src[(imm >> 0) & 0x3u],
      src[(imm >> 2) & 0x3u],
      src[(imm >> 4) & 0x3u],
      src[(imm >> 6) & 0x3u],
      src[4],
      src[5],
      src[6],
      src[7],
    };
  });
}

ExecutionResult handle_code_PSHUFHW_XMM_XMMM128_IMM8(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return legacy_unary_lanewise<std::uint16_t, 8>(ctx, kXmmBytes, [imm](const auto& src) {
    return std::array<std::uint16_t, 8>{
      src[0],
      src[1],
      src[2],
      src[3],
      src[4 + ((imm >> 0) & 0x3u)],
      src[4 + ((imm >> 2) & 0x3u)],
      src[4 + ((imm >> 4) & 0x3u)],
      src[4 + ((imm >> 6) & 0x3u)],
    };
  });
}

ExecutionResult handle_code_MOVSLDUP_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[0], src[0], src[2], src[2]};
  });
}

ExecutionResult handle_code_MOVSHDUP_XMM_XMMM128(ExecutionContext& ctx) {
  return legacy_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[1], src[1], src[3], src[3]};
  });
}

ExecutionResult handle_code_MOVDDUP_XMM_XMMM64(ExecutionContext& ctx) {
  return duplicate_low_double(ctx);
}

ExecutionResult handle_code_VEX_VUNPCKLPS_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[0], rhs[0], lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_VEX_VUNPCKLPD_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[0], rhs[0]};
  });
}

ExecutionResult handle_code_VEX_VUNPCKHPS_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[2], rhs[2], lhs[3], rhs[3]};
  });
}

ExecutionResult handle_code_VEX_VUNPCKHPD_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_VEX_VSHUFPS_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{
      lhs[imm & 0x3u],
      lhs[(imm >> 2) & 0x3u],
      rhs[(imm >> 4) & 0x3u],
      rhs[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_VEX_VSHUFPD_XMM_XMM_XMMM128(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{
      lhs[(imm >> 0) & 0x1u],
      rhs[(imm >> 1) & 0x1u],
    };
  });
}

ExecutionResult handle_code_VEX_VMOVSLDUP_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[0], src[0], src[2], src[2]};
  });
}

ExecutionResult handle_code_VEX_VMOVSHDUP_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[1], src[1], src[3], src[3]};
  });
}

ExecutionResult handle_code_VEX_VMOVDDUP_XMM_XMMM64(ExecutionContext& ctx) {
  return duplicate_low_double(ctx, true);
}

ExecutionResult handle_code_VEX_VUNPCKLPS_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[0], rhs[0], lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_VEX_VUNPCKLPD_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[0], rhs[0]};
  });
}

ExecutionResult handle_code_VEX_VUNPCKHPS_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[2], rhs[2], lhs[3], rhs[3]};
  });
}

ExecutionResult handle_code_VEX_VUNPCKHPD_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_VEX_VSHUFPS_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{
      lhs[imm & 0x3u],
      lhs[(imm >> 2) & 0x3u],
      rhs[(imm >> 4) & 0x3u],
      rhs[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_VEX_VSHUFPD_YMM_YMM_YMMM256(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{
      lhs[(imm >> 0) & 0x1u],
      rhs[(imm >> 1) & 0x1u],
    };
  });
}

ExecutionResult handle_code_VEX_VMOVSLDUP_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[0], src[0], src[2], src[2]};
  });
}

ExecutionResult handle_code_VEX_VMOVSHDUP_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[1], src[1], src[3], src[3]};
  });
}

ExecutionResult handle_code_VEX_VMOVDDUP_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<double, 2>{src[0], src[0]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKLPS_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[0], rhs[0], lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKLPD_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[0], rhs[0]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKHPS_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[2], rhs[2], lhs[3], rhs[3]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKHPD_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_EVEX_VSHUFPS_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{
      lhs[imm & 0x3u],
      lhs[(imm >> 2) & 0x3u],
      rhs[(imm >> 4) & 0x3u],
      rhs[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_EVEX_VSHUFPD_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{
      lhs[(imm >> 0) & 0x1u],
      rhs[(imm >> 1) & 0x1u],
    };
  });
}

ExecutionResult handle_code_EVEX_VMOVSLDUP_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[0], src[0], src[2], src[2]};
  });
}

ExecutionResult handle_code_EVEX_VMOVSHDUP_XMM_K1Z_XMM_XMMM128(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[1], src[1], src[3], src[3]};
  });
}

ExecutionResult handle_code_EVEX_VMOVDDUP_XMM_K1Z_XMM_XMMM64(ExecutionContext& ctx) {
  return duplicate_low_double(ctx, true);
}

ExecutionResult handle_code_EVEX_VUNPCKLPS_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[0], rhs[0], lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKLPD_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[0], rhs[0]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKHPS_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[2], rhs[2], lhs[3], rhs[3]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKHPD_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_EVEX_VSHUFPS_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{
      lhs[imm & 0x3u],
      lhs[(imm >> 2) & 0x3u],
      rhs[(imm >> 4) & 0x3u],
      rhs[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_EVEX_VSHUFPD_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{
      lhs[(imm >> 0) & 0x1u],
      rhs[(imm >> 1) & 0x1u],
    };
  });
}

ExecutionResult handle_code_EVEX_VMOVSLDUP_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[0], src[0], src[2], src[2]};
  });
}

ExecutionResult handle_code_EVEX_VMOVSHDUP_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[1], src[1], src[3], src[3]};
  });
}

ExecutionResult handle_code_EVEX_VMOVDDUP_YMM_K1Z_YMM_YMMM256(ExecutionContext& ctx) {
  return vex_unary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<double, 2>{src[0], src[0]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKLPS_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[0], rhs[0], lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKLPD_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[0], rhs[0]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKHPS_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{lhs[2], rhs[2], lhs[3], rhs[3]};
  });
}

ExecutionResult handle_code_EVEX_VUNPCKHPD_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{lhs[1], rhs[1]};
  });
}

ExecutionResult handle_code_EVEX_VSHUFPS_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<float, 4>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<float, 4>{
      lhs[imm & 0x3u],
      lhs[(imm >> 2) & 0x3u],
      rhs[(imm >> 4) & 0x3u],
      rhs[(imm >> 6) & 0x3u],
    };
  });
}

ExecutionResult handle_code_EVEX_VSHUFPD_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  const auto imm = ctx.instr.immediate8();
  return vex_binary_lanewise<double, 2>(ctx, kXmmBytes, [imm](const auto& lhs, const auto& rhs) {
    return std::array<double, 2>{
      lhs[(imm >> 0) & 0x1u],
      rhs[(imm >> 1) & 0x1u],
    };
  });
}

ExecutionResult handle_code_EVEX_VMOVSLDUP_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[0], src[0], src[2], src[2]};
  });
}

ExecutionResult handle_code_EVEX_VMOVSHDUP_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unary_lanewise<float, 4>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<float, 4>{src[1], src[1], src[3], src[3]};
  });
}

ExecutionResult handle_code_EVEX_VMOVDDUP_ZMM_K1Z_ZMM_ZMMM512(ExecutionContext& ctx) {
  return vex_unary_lanewise<double, 2>(ctx, kXmmBytes, [](const auto& src) {
    return std::array<double, 2>{src[0], src[0]};
  });
}

}  // namespace seven::handlers



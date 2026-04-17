#include <array>

#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

using big_uint = seven::SimdUint;

constexpr std::size_t kVectorRegisterCount = 32;
constexpr std::size_t kXmmWidth = 16;
constexpr std::size_t kYmmWidth = 32;
constexpr std::size_t kZmmWidth = 64;

bool is_vector_register(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  return (value >= xmm0 && value < xmm0 + kVectorRegisterCount) || (value >= ymm0 && value < ymm0 + kVectorRegisterCount) || (value >= zmm0 && value < zmm0 + kVectorRegisterCount);
}

std::size_t vector_index(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= xmm0 && value < xmm0 + kVectorRegisterCount) return static_cast<std::size_t>(value - xmm0);
  if (value >= ymm0 && value < ymm0 + kVectorRegisterCount) return static_cast<std::size_t>(value - ymm0);
  if (value >= zmm0 && value < zmm0 + kVectorRegisterCount) return static_cast<std::size_t>(value - zmm0);
  return 0;
}

std::size_t vector_width(iced_x86::Register reg) {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  if (value >= zmm0 && value < zmm0 + kVectorRegisterCount) return kZmmWidth;
  if (value >= ymm0 && value < ymm0 + kVectorRegisterCount) return kYmmWidth;
  if (value >= xmm0 && value < xmm0 + kVectorRegisterCount) return kXmmWidth;
  return kXmmWidth;
}

big_uint mask(std::size_t width) {
  if (width >= kZmmWidth) { big_uint _all_ones(0); return ~_all_ones; }
  return (big_uint(1) << (width * 8)) - 1;
}

big_uint read_mem(ExecutionContext& ctx, std::uint64_t address, std::size_t width, bool* ok) {
  std::array<std::uint8_t, kZmmWidth> bytes{};
  if (!ctx.memory.read(address, bytes.data(), width)) {
    if (ok) *ok = false;
    return 0;
  }
  if (ok) *ok = true;
  big_uint value = 0;
  for (std::size_t i = 0; i < width; ++i) value |= (big_uint(bytes[i]) << (8 * i));
  return value;
}

bool write_mem(ExecutionContext& ctx, std::uint64_t address, big_uint value, std::size_t width) {
  std::array<std::uint8_t, kZmmWidth> bytes{};
  for (std::size_t i = 0; i < width; ++i) bytes[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFu);
  return ctx.memory.write(address, bytes.data(), width);
}

big_uint read_vec(CpuState& state, iced_x86::Register reg) {
  return state.vectors[vector_index(reg)].value;
}

void write_vec(CpuState& state, iced_x86::Register reg, big_uint value, std::size_t width, bool zero_upper) {
  auto& slot = state.vectors[vector_index(reg)].value;
  auto m = mask(width);
  if (zero_upper) {
    slot = value & m;
  } else {
    slot = (slot & ~m) | (value & m);
  }
}

std::size_t infer_width(ExecutionContext& ctx, std::uint32_t dst_index, std::uint32_t src_index) {
  if (ctx.instr.op_kind(dst_index) == iced_x86::OpKind::REGISTER && is_vector_register(ctx.instr.op_register(dst_index))) {
    return vector_width(ctx.instr.op_register(dst_index));
  }
  if (ctx.instr.op_kind(src_index) == iced_x86::OpKind::REGISTER && is_vector_register(ctx.instr.op_register(src_index))) {
    return vector_width(ctx.instr.op_register(src_index));
  }
  const auto width = detail::operand_width(ctx.instr, dst_index);
  return width != 0 ? width : detail::operand_width(ctx.instr, src_index);
}

big_uint read_any(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, bool* ok) {
  if (ok) *ok = true;
  const auto kind = ctx.instr.op_kind(operand_index);
  if (kind == iced_x86::OpKind::REGISTER) {
    const auto reg = ctx.instr.op_register(operand_index);
    if (is_vector_register(reg)) return read_vec(ctx.state, reg) & mask(width);
    return big_uint(detail::read_register(ctx.state, reg)) & mask(width);
  }
  if (kind == iced_x86::OpKind::MEMORY) return read_mem(ctx, detail::memory_address(ctx), width, ok);
  return big_uint(detail::immediate_value(ctx.instr, operand_index)) & mask(width);
}

bool write_any(ExecutionContext& ctx, std::uint32_t operand_index, big_uint value, std::size_t width, bool zero_upper = false) {
  value &= mask(width);
  const auto kind = ctx.instr.op_kind(operand_index);
  if (kind == iced_x86::OpKind::REGISTER) {
    const auto reg = ctx.instr.op_register(operand_index);
    if (is_vector_register(reg)) write_vec(ctx.state, reg, value, width, zero_upper);
    else detail::write_register(ctx.state, reg, static_cast<std::uint64_t>(value), width);
    return true;
  }
  if (kind == iced_x86::OpKind::MEMORY) return write_mem(ctx, detail::memory_address(ctx), value, width);
  return false;
}

ExecutionResult full_move(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, bool zero_upper) {
  bool ok = false;
  const auto width = infer_width(ctx, dst, src);
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  if (!write_any(ctx, dst, value, width, zero_upper)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult low_move(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, std::size_t width, bool zero_upper) {
  bool ok = false;
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  if (!write_any(ctx, dst, value, width, zero_upper)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult low_move_legacy_scalar_load(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, std::size_t width) {
  bool ok = false;
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const bool src_is_memory = ctx.instr.op_kind(src) == iced_x86::OpKind::MEMORY;
  if (!write_any(ctx, dst, value, width, src_is_memory)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult merge_low_move(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t merge, std::uint32_t src, std::size_t width, bool zero_upper) {
  bool ok = false;
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto dst_reg = ctx.instr.op_register(dst);
  const auto merge_value = read_vec(ctx.state, ctx.instr.op_register(merge));
  const auto updated = (merge_value & ~mask(width)) | (value & mask(width));
  write_vec(ctx.state, dst_reg, updated, vector_width(dst_reg), zero_upper);
  return {};
}

ExecutionResult merge_high_move(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t merge, std::uint32_t src, std::size_t width, bool zero_upper) {
  bool ok = false;
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto dst_reg = ctx.instr.op_register(dst);
  const auto merge_value = read_vec(ctx.state, ctx.instr.op_register(merge));
  auto shifted = mask(width) << (width * 8);
  const auto updated = (merge_value & ~shifted) | ((value << (width * 8)) & shifted);
  write_vec(ctx.state, dst_reg, updated, vector_width(dst_reg), zero_upper);
  return {};
}

ExecutionResult gpr_to_xmm(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, std::size_t width) {
  bool ok = false;
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  if (!write_any(ctx, dst, value, width, true)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult xmm_to_gpr(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, std::size_t width) {
  bool ok = false;
  const auto value = read_any(ctx, src, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  if (!write_any(ctx, dst, value, width)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult pextrw_to_gpr_or_mem(ExecutionContext& ctx) {
  const auto src_reg = ctx.instr.op_register(1);
  const auto lane = static_cast<std::size_t>(detail::immediate_value(ctx.instr, 2) & 0x7u);
  const auto word = (read_vec(ctx.state, src_reg) >> (lane * 16)) & 0xFFFFu;
  if (ctx.instr.op_kind(0) == iced_x86::OpKind::REGISTER) {
    const auto dst_reg = ctx.instr.op_register(0);
    const auto dst_width = detail::register_width(dst_reg);
    detail::write_register(ctx.state, dst_reg, static_cast<std::uint64_t>(word), dst_width);
    return {};
  }
  if (!write_mem(ctx, detail::memory_address(ctx), word, 2)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}


ExecutionResult high_lane_to_gpr(ExecutionContext& ctx, std::uint32_t dst, std::uint32_t src, std::size_t width) {
  const auto src_reg = ctx.instr.op_register(src);
  const auto value = (read_vec(ctx.state, src_reg) >> (width * 8)) & mask(width);
  if (!write_any(ctx, dst, value, width)) return detail::memory_fault(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult movlhps_legacy(ExecutionContext& ctx) {
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_reg = ctx.instr.op_register(1);
  const auto dst_value = read_vec(ctx.state, dst_reg);
  const auto src_low = read_vec(ctx.state, src_reg) & mask(8);
  const auto updated = (dst_value & mask(8)) | (src_low << 64);
  write_vec(ctx.state, dst_reg, updated, vector_width(dst_reg), false);
  return {};
}

ExecutionResult movhlps_legacy(ExecutionContext& ctx) {
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src_reg = ctx.instr.op_register(1);
  const auto dst_value = read_vec(ctx.state, dst_reg);
  const auto src_high = (read_vec(ctx.state, src_reg) >> 64) & mask(8);
  const auto updated = (dst_value & (mask(8) << 64)) | src_high;
  write_vec(ctx.state, dst_reg, updated, vector_width(dst_reg), false);
  return {};
}

ExecutionResult vmovlhps(ExecutionContext& ctx) {
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src1_reg = ctx.instr.op_register(1);
  const auto src2_reg = ctx.instr.op_register(2);
  const auto src1_low = read_vec(ctx.state, src1_reg) & mask(8);
  const auto src2_low = read_vec(ctx.state, src2_reg) & mask(8);
  const auto updated = src1_low | (src2_low << 64);
  write_vec(ctx.state, dst_reg, updated, vector_width(dst_reg), true);
  return {};
}

ExecutionResult vmovhlps(ExecutionContext& ctx) {
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src1_reg = ctx.instr.op_register(1);
  const auto src2_reg = ctx.instr.op_register(2);
  const auto src1_high = (read_vec(ctx.state, src1_reg) >> 64) & mask(8);
  const auto src2_high = (read_vec(ctx.state, src2_reg) >> 64) & mask(8);
  const auto updated = src2_high | (src1_high << 64);
  write_vec(ctx.state, dst_reg, updated, vector_width(dst_reg), true);
  return {};
}

}  // namespace

#define KUBERA_FULL_MOVE(name, zero_upper) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return full_move(ctx, 0, 1, zero_upper); }
#define KUBERA_LOW_MOVE(name, width, zero_upper) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return low_move(ctx, 0, 1, width, zero_upper); }
#define KUBERA_MERGE_LOW_MOVE(name, width, zero_upper) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return merge_low_move(ctx, 0, 1, 2, width, zero_upper); }
#define KUBERA_MERGE_HIGH_MOVE(name, width, zero_upper) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return merge_high_move(ctx, 0, 1, 2, width, zero_upper); }
#define KUBERA_GPR_TO_XMM(name, width) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return gpr_to_xmm(ctx, 0, 1, width); }
#define KUBERA_XMM_TO_GPR(name, width) ExecutionResult handle_code_##name(ExecutionContext& ctx) { return xmm_to_gpr(ctx, 0, 1, width); }

KUBERA_FULL_MOVE(MOVAPS_XMM_XMMM128, false)
KUBERA_FULL_MOVE(MOVAPD_XMM_XMMM128, false)
KUBERA_FULL_MOVE(MOVUPS_XMM_XMMM128, false)
KUBERA_FULL_MOVE(MOVUPD_XMM_XMMM128, false)
ExecutionResult handle_code_MOVSS_XMM_XMMM32(ExecutionContext& ctx) { return low_move_legacy_scalar_load(ctx, 0, 1, 4); }
ExecutionResult handle_code_MOVSD_XMM_XMMM64(ExecutionContext& ctx) { return low_move_legacy_scalar_load(ctx, 0, 1, 8); }
KUBERA_FULL_MOVE(MOVAPS_XMMM128_XMM, false)
KUBERA_FULL_MOVE(MOVAPD_XMMM128_XMM, false)
KUBERA_FULL_MOVE(MOVUPS_XMMM128_XMM, false)
KUBERA_FULL_MOVE(MOVUPD_XMMM128_XMM, false)
KUBERA_LOW_MOVE(MOVSS_XMMM32_XMM, 4, false)
KUBERA_LOW_MOVE(MOVSD_XMMM64_XMM, 8, false)
KUBERA_FULL_MOVE(MOVDQA_XMM_XMMM128, false)
KUBERA_FULL_MOVE(MOVDQU_XMM_XMMM128, false)
ExecutionResult handle_code_LDDQU_XMM_M128(ExecutionContext& ctx) { return full_move(ctx, 0, 1, false); }
KUBERA_FULL_MOVE(MOVDQA_XMMM128_XMM, false)
KUBERA_FULL_MOVE(MOVDQU_XMMM128_XMM, false)
KUBERA_LOW_MOVE(MOVLPS_XMM_M64, 8, false)
KUBERA_LOW_MOVE(MOVLPD_XMM_M64, 8, false)
KUBERA_MERGE_HIGH_MOVE(MOVHPS_XMM_M64, 8, false)
KUBERA_MERGE_HIGH_MOVE(MOVHPD_XMM_M64, 8, false)
ExecutionResult handle_code_MOVLHPS_XMM_XMM(ExecutionContext& ctx) { return movlhps_legacy(ctx); }
ExecutionResult handle_code_MOVHLPS_XMM_XMM(ExecutionContext& ctx) { return movhlps_legacy(ctx); }
KUBERA_XMM_TO_GPR(MOVLPS_M64_XMM, 8)
KUBERA_XMM_TO_GPR(MOVLPD_M64_XMM, 8)
ExecutionResult handle_code_MOVHPS_M64_XMM(ExecutionContext& ctx) { return high_lane_to_gpr(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVHPD_M64_XMM(ExecutionContext& ctx) { return high_lane_to_gpr(ctx, 0, 1, 8); }
KUBERA_GPR_TO_XMM(MOVD_XMM_RM32, 4)
KUBERA_GPR_TO_XMM(MOVQ_XMM_RM64, 8)
KUBERA_XMM_TO_GPR(MOVD_RM32_XMM, 4)
KUBERA_XMM_TO_GPR(MOVQ_RM64_XMM, 8)
ExecutionResult handle_code_PEXTRW_R32_XMM_IMM8(ExecutionContext& ctx) { return pextrw_to_gpr_or_mem(ctx); }
ExecutionResult handle_code_PEXTRW_R64_XMM_IMM8(ExecutionContext& ctx) { return pextrw_to_gpr_or_mem(ctx); }
ExecutionResult handle_code_PEXTRW_R32M16_XMM_IMM8(ExecutionContext& ctx) { return pextrw_to_gpr_or_mem(ctx); }
ExecutionResult handle_code_PEXTRW_R64M16_XMM_IMM8(ExecutionContext& ctx) { return pextrw_to_gpr_or_mem(ctx); }
KUBERA_LOW_MOVE(MOVQ_XMM_XMMM64, 8, true)
KUBERA_LOW_MOVE(MOVQ_XMMM64_XMM, 8, true)
KUBERA_XMM_TO_GPR(MOVNTPS_M128_XMM, 16)
KUBERA_XMM_TO_GPR(MOVNTPD_M128_XMM, 16)
KUBERA_XMM_TO_GPR(MOVNTDQ_M128_XMM, 16)
KUBERA_FULL_MOVE(MOVNTDQA_XMM_M128, false)

KUBERA_FULL_MOVE(VEX_VMOVAPS_XMM_XMMM128, true)
KUBERA_FULL_MOVE(VEX_VMOVAPS_YMM_YMMM256, true)
KUBERA_FULL_MOVE(VEX_VMOVAPS_XMMM128_XMM, true)
KUBERA_FULL_MOVE(VEX_VMOVAPS_YMMM256_YMM, true)
KUBERA_FULL_MOVE(VEX_VMOVAPD_XMM_XMMM128, true)
KUBERA_FULL_MOVE(VEX_VMOVAPD_YMM_YMMM256, true)
KUBERA_FULL_MOVE(VEX_VMOVAPD_XMMM128_XMM, true)
KUBERA_FULL_MOVE(VEX_VMOVAPD_YMMM256_YMM, true)
KUBERA_FULL_MOVE(VEX_VMOVUPS_XMM_XMMM128, true)
KUBERA_FULL_MOVE(VEX_VMOVUPS_YMM_YMMM256, true)
KUBERA_FULL_MOVE(VEX_VMOVUPS_XMMM128_XMM, true)
KUBERA_FULL_MOVE(VEX_VMOVUPS_YMMM256_YMM, true)
KUBERA_FULL_MOVE(VEX_VMOVUPD_XMM_XMMM128, true)
KUBERA_FULL_MOVE(VEX_VMOVUPD_YMM_YMMM256, true)
KUBERA_FULL_MOVE(VEX_VMOVUPD_XMMM128_XMM, true)
KUBERA_FULL_MOVE(VEX_VMOVUPD_YMMM256_YMM, true)
KUBERA_MERGE_LOW_MOVE(VEX_VMOVSS_XMM_XMM_XMM, 4, true)
KUBERA_MERGE_LOW_MOVE(VEX_VMOVSS_XMM_XMM_XMM_0_F11, 4, true)
KUBERA_LOW_MOVE(VEX_VMOVSS_XMM_M32, 4, true)
KUBERA_LOW_MOVE(VEX_VMOVSS_M32_XMM, 4, false)
KUBERA_MERGE_LOW_MOVE(VEX_VMOVSD_XMM_XMM_XMM, 8, true)
KUBERA_MERGE_LOW_MOVE(VEX_VMOVSD_XMM_XMM_XMM_0_F11, 8, true)
KUBERA_LOW_MOVE(VEX_VMOVSD_XMM_M64, 8, true)
KUBERA_LOW_MOVE(VEX_VMOVSD_M64_XMM, 8, false)
KUBERA_FULL_MOVE(VEX_VMOVDQA_XMM_XMMM128, true)
KUBERA_FULL_MOVE(VEX_VMOVDQA_YMM_YMMM256, true)
KUBERA_FULL_MOVE(VEX_VMOVDQA_XMMM128_XMM, true)
KUBERA_FULL_MOVE(VEX_VMOVDQA_YMMM256_YMM, true)
KUBERA_FULL_MOVE(VEX_VMOVDQU_XMM_XMMM128, true)
KUBERA_FULL_MOVE(VEX_VMOVDQU_YMM_YMMM256, true)
KUBERA_FULL_MOVE(VEX_VMOVDQU_XMMM128_XMM, true)
KUBERA_FULL_MOVE(VEX_VMOVDQU_YMMM256_YMM, true)
KUBERA_GPR_TO_XMM(VEX_VMOVD_XMM_RM32, 4)
KUBERA_GPR_TO_XMM(VEX_VMOVQ_XMM_RM64, 8)
KUBERA_XMM_TO_GPR(VEX_VMOVD_RM32_XMM, 4)
KUBERA_XMM_TO_GPR(VEX_VMOVQ_RM64_XMM, 8)
KUBERA_LOW_MOVE(VEX_VMOVQ_XMM_XMMM64, 8, true)
KUBERA_LOW_MOVE(VEX_VMOVQ_XMMM64_XMM, 8, true)
KUBERA_MERGE_HIGH_MOVE(VEX_VMOVLPS_XMM_XMM_M64, 8, true)
KUBERA_XMM_TO_GPR(VEX_VMOVLPS_M64_XMM, 8)
KUBERA_MERGE_HIGH_MOVE(VEX_VMOVLPD_XMM_XMM_M64, 8, true)
KUBERA_XMM_TO_GPR(VEX_VMOVLPD_M64_XMM, 8)
KUBERA_MERGE_HIGH_MOVE(VEX_VMOVHPS_XMM_XMM_M64, 8, true)
ExecutionResult handle_code_VEX_VMOVLHPS_XMM_XMM_XMM(ExecutionContext& ctx) { return vmovlhps(ctx); }
ExecutionResult handle_code_VEX_VMOVHLPS_XMM_XMM_XMM(ExecutionContext& ctx) { return vmovhlps(ctx); }
ExecutionResult handle_code_VEX_VMOVHPS_M64_XMM(ExecutionContext& ctx) { return high_lane_to_gpr(ctx, 0, 1, 8); }
KUBERA_MERGE_HIGH_MOVE(VEX_VMOVHPD_XMM_XMM_M64, 8, true)
ExecutionResult handle_code_VEX_VMOVHPD_M64_XMM(ExecutionContext& ctx) { return high_lane_to_gpr(ctx, 0, 1, 8); }
KUBERA_XMM_TO_GPR(VEX_VMOVNTPS_M128_XMM, 16)
KUBERA_XMM_TO_GPR(VEX_VMOVNTPS_M256_YMM, 32)
KUBERA_XMM_TO_GPR(VEX_VMOVNTPD_M128_XMM, 16)
KUBERA_XMM_TO_GPR(VEX_VMOVNTPD_M256_YMM, 32)
KUBERA_XMM_TO_GPR(VEX_VMOVNTDQ_M128_XMM, 16)
KUBERA_XMM_TO_GPR(VEX_VMOVNTDQ_M256_YMM, 32)
KUBERA_FULL_MOVE(VEX_VMOVNTDQA_XMM_M128, true)
KUBERA_FULL_MOVE(VEX_VMOVNTDQA_YMM_M256, true)

KUBERA_FULL_MOVE(EVEX_VMOVAPS_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPS_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPS_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPS_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPS_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPS_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPD_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPD_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPD_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPD_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPD_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVAPD_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPS_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPS_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPS_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPS_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPS_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPS_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPD_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPD_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPD_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPD_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPD_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVUPD_ZMMM512_K1Z_ZMM, true)
KUBERA_MERGE_LOW_MOVE(EVEX_VMOVSS_XMM_K1Z_XMM_XMM, 4, true)
KUBERA_MERGE_LOW_MOVE(EVEX_VMOVSS_XMM_K1Z_XMM_XMM_0_F11, 4, true)
KUBERA_LOW_MOVE(EVEX_VMOVSS_XMM_K1Z_M32, 4, true)
KUBERA_LOW_MOVE(EVEX_VMOVSS_M32_K1_XMM, 4, false)
KUBERA_MERGE_LOW_MOVE(EVEX_VMOVSD_XMM_K1Z_XMM_XMM, 8, true)
KUBERA_MERGE_LOW_MOVE(EVEX_VMOVSD_XMM_K1Z_XMM_XMM_0_F11, 8, true)
KUBERA_LOW_MOVE(EVEX_VMOVSD_XMM_K1Z_M64, 8, true)
KUBERA_LOW_MOVE(EVEX_VMOVSD_M64_K1_XMM, 8, false)
KUBERA_FULL_MOVE(EVEX_VMOVDQA32_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA32_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA32_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA32_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA32_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA32_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA64_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA64_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA64_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA64_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA64_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQA64_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU32_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU32_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU32_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU32_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU32_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU32_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU64_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU64_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU64_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU64_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU64_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU64_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU8_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU8_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU8_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU8_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU8_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU8_ZMMM512_K1Z_ZMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU16_XMM_K1Z_XMMM128, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU16_YMM_K1Z_YMMM256, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU16_ZMM_K1Z_ZMMM512, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU16_XMMM128_K1Z_XMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU16_YMMM256_K1Z_YMM, true)
KUBERA_FULL_MOVE(EVEX_VMOVDQU16_ZMMM512_K1Z_ZMM, true)
KUBERA_GPR_TO_XMM(EVEX_VMOVD_XMM_RM32, 4)
KUBERA_GPR_TO_XMM(EVEX_VMOVQ_XMM_RM64, 8)
KUBERA_XMM_TO_GPR(EVEX_VMOVD_RM32_XMM, 4)
KUBERA_XMM_TO_GPR(EVEX_VMOVQ_RM64_XMM, 8)
KUBERA_LOW_MOVE(EVEX_VMOVQ_XMM_XMMM64, 8, true)
KUBERA_LOW_MOVE(EVEX_VMOVQ_XMMM64_XMM, 8, true)
KUBERA_MERGE_HIGH_MOVE(EVEX_VMOVLPS_XMM_XMM_M64, 8, true)
KUBERA_XMM_TO_GPR(EVEX_VMOVLPS_M64_XMM, 8)
KUBERA_MERGE_HIGH_MOVE(EVEX_VMOVLPD_XMM_XMM_M64, 8, true)
KUBERA_XMM_TO_GPR(EVEX_VMOVLPD_M64_XMM, 8)
KUBERA_MERGE_HIGH_MOVE(EVEX_VMOVHPS_XMM_XMM_M64, 8, true)
ExecutionResult handle_code_EVEX_VMOVLHPS_XMM_XMM_XMM(ExecutionContext& ctx) { return vmovlhps(ctx); }
ExecutionResult handle_code_EVEX_VMOVHLPS_XMM_XMM_XMM(ExecutionContext& ctx) { return vmovhlps(ctx); }
ExecutionResult handle_code_EVEX_VMOVHPS_M64_XMM(ExecutionContext& ctx) { return high_lane_to_gpr(ctx, 0, 1, 8); }
KUBERA_MERGE_HIGH_MOVE(EVEX_VMOVHPD_XMM_XMM_M64, 8, true)
ExecutionResult handle_code_EVEX_VMOVHPD_M64_XMM(ExecutionContext& ctx) { return high_lane_to_gpr(ctx, 0, 1, 8); }
KUBERA_XMM_TO_GPR(EVEX_VMOVNTPS_M128_XMM, 16)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTPS_M256_YMM, 32)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTPS_M512_ZMM, 64)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTPD_M128_XMM, 16)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTPD_M256_YMM, 32)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTPD_M512_ZMM, 64)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTDQ_M128_XMM, 16)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTDQ_M256_YMM, 32)
KUBERA_XMM_TO_GPR(EVEX_VMOVNTDQ_M512_ZMM, 64)
KUBERA_FULL_MOVE(EVEX_VMOVNTDQA_XMM_M128, true)
KUBERA_FULL_MOVE(EVEX_VMOVNTDQA_YMM_M256, true)
KUBERA_FULL_MOVE(EVEX_VMOVNTDQA_ZMM_M512, true)

#undef KUBERA_FULL_MOVE
#undef KUBERA_LOW_MOVE
#undef KUBERA_MERGE_LOW_MOVE
#undef KUBERA_MERGE_HIGH_MOVE
#undef KUBERA_GPR_TO_XMM
#undef KUBERA_XMM_TO_GPR

}  // namespace seven::handlers



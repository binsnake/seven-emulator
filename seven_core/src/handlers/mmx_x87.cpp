#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "seven/handler_helpers.hpp"
#include "seven/x87_helpers.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

namespace {
using big_uint = seven::SimdUint;

big_uint mask(std::size_t width) {
  return (big_uint(1) << (width * 8)) - 1;
}


std::size_t xmm_index(iced_x86::Register reg) {
  return static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::XMM0));
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

ExecutionResult validate_memory_span(ExecutionContext& ctx, std::uint64_t base, std::size_t size, seven::MemoryAccessKind kind) {
  std::size_t offset = 0;
  while (offset < size) {
    const auto address = base + offset;
    const auto page_offset = static_cast<std::size_t>(address % seven::Memory::kPageSize);
    const auto chunk = std::min(size - offset, seven::Memory::kPageSize - page_offset);
    bool allowed = false;
    if (kind == seven::MemoryAccessKind::data_write) {
      allowed = ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write));
    } else {
      allowed = ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::read)) ||
                ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::write)) ||
                ctx.memory.has_permissions(address, chunk, static_cast<seven::MemoryPermissionMask>(seven::MemoryPermission::execute));
    }
    if (!ctx.memory.is_mapped(address, chunk) || !allowed) {
      return detail::memory_fault(ctx, address);
    }
    offset += chunk;
  }
  return {};
}


std::uint64_t read_mmx(CpuState& state, iced_x86::Register reg);

ExecutionResult pmovmskb(ExecutionContext& ctx, std::size_t src_bytes, bool use_mmx, std::size_t dst_bytes) {
  SimdUint src = 0;
  if (use_mmx) {
    src = SimdUint(read_mmx(ctx.state, ctx.instr.op_register(1)));
  } else {
    const auto src_reg = ctx.instr.op_register(1);
    src = ctx.state.vectors[vector_index(src_reg)].value;
  }

  std::uint64_t mask = 0;
  for (std::size_t i = 0; i < src_bytes; ++i) {
    const std::uint64_t msb = ((src >> ((i * 8) + 7)) & SimdUint(1)) != SimdUint(0) ? 1u : 0u;
    mask |= (msb << i);
  }

  if (!detail::write_operand(ctx, 0, mask, dst_bytes)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult movmskpd(ExecutionContext& ctx, std::size_t dst_bytes) {
  const auto src_reg = ctx.instr.op_register(1);
  const auto src = ctx.state.vectors[vector_index(src_reg)].value;
  const auto mask = static_cast<std::uint64_t>(((src >> 63) & 1u) | (((src >> 127) & 1u) << 1));
  if (!detail::write_operand(ctx, 0, mask, dst_bytes)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

std::uint64_t read_mmx(CpuState& state, iced_x86::Register reg) {
  return state.mmx_get(static_cast<std::size_t>(static_cast<std::uint32_t>(reg) - static_cast<std::uint32_t>(iced_x86::Register::MM0)));
}

ExecutionResult store_masked_bytes(ExecutionContext& ctx, std::uint32_t data_index, std::uint32_t mask_index, std::size_t width, bool use_xmm) {
  const auto dest = detail::read_register(ctx.state, iced_x86::Register::RDI);
  std::array<std::uint8_t, 64> bytes{};
  std::array<std::uint8_t, 64> mask{};

  if (use_xmm) {
    const auto data_reg = ctx.instr.op_register(data_index);
    const auto mask_reg = ctx.instr.op_register(mask_index);
    const auto data = ctx.state.vectors[xmm_index(data_reg)].value;
    const auto mask_value = ctx.state.vectors[xmm_index(mask_reg)].value;
    for (std::size_t i = 0; i < width; ++i) {
      bytes[i] = static_cast<std::uint8_t>((data >> (8 * i)) & 0xFFu);
      mask[i] = static_cast<std::uint8_t>((mask_value >> (8 * i)) & 0xFFu);
    }
  } else {
    const auto data = read_mmx(ctx.state, ctx.instr.op_register(data_index));
    const auto mask_value = read_mmx(ctx.state, ctx.instr.op_register(mask_index));
    for (std::size_t i = 0; i < width; ++i) {
      bytes[i] = static_cast<std::uint8_t>((data >> (8 * i)) & 0xFFu);
      mask[i] = static_cast<std::uint8_t>((mask_value >> (8 * i)) & 0xFFu);
    }
  }

  for (std::size_t i = 0; i < width; ++i) {
    if ((mask[i] & 0x80u) != 0) {
      if (!ctx.memory.write(dest + i, &bytes[i], 1)) {
        return detail::memory_fault(ctx, dest + i);
      }
    }
  }
  return {};
}

ExecutionResult mmx_move(ExecutionContext& ctx, std::uint32_t dst_index, std::uint32_t src_index, std::size_t width) {
  bool ok = false;
  const auto value = detail::read_operand(ctx, src_index, width, &ok);
  if (!ok) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!detail::write_operand(ctx, dst_index, value, width)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

void emms(CpuState& state) {
  state.x87_tags.fill(0x3);
}

uint8_t x87_ftw(const CpuState& state) {
  uint8_t ftw = 0;
  for (std::size_t phys = 0; phys < 8; ++phys) {
    if (state.x87_tags[phys] != 0x3) {
      ftw |= static_cast<uint8_t>(1u << phys);
    }
  }
  return ftw;
}

void store_x87_env(ExecutionContext& ctx, std::uint64_t base) {
  const std::uint16_t fcw = ctx.state.get_x87_control_word();
  const std::uint16_t fsw = ctx.state.get_x87_status_word();
  const std::uint16_t ftw = static_cast<std::uint16_t>(x87_ftw(ctx.state));
  const std::uint16_t zero16 = 0;
  const std::uint64_t zero64 = 0;
  ctx.memory.write(base + 0, &fcw, 2);
  ctx.memory.write(base + 2, &fsw, 2);
  ctx.memory.write(base + 4, &ftw, 2);
  ctx.memory.write(base + 6, &zero16, 2);
  ctx.memory.write(base + 8, &zero64, 8);
  ctx.memory.write(base + 16, &zero64, 8);
  ctx.memory.write(base + 24, &zero16, 2);
}

ExecutionResult load_x87_env(ExecutionContext& ctx, std::uint64_t base) {
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint16_t ftw = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 2)) return detail::memory_fault(ctx, base + 4);
  if ((fcw & 0xE0C0u) != 0) return detail::memory_fault(ctx, base);
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  for (std::size_t i = 0; i < 8; ++i) {
    ctx.state.x87_tags[ctx.state.x87_phys_index(i)] = ((ftw >> i) & 1u) ? 0x0 : 0x3;
  }
  return {};
}

void write_fxsave_st(ExecutionContext& ctx, std::uint64_t base, std::size_t phys, X87Scalar value) {
  std::array<std::uint8_t, 16> raw{};
  std::memcpy(raw.data(), &value, std::min(sizeof(value), raw.size()));
  const auto slot = base + 32 + (phys * 16);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    ctx.memory.write(slot + i, &raw[i], 1);
  }
}

X87Scalar read_fxsave_st(ExecutionContext& ctx, std::uint64_t base, std::size_t phys) {
  std::array<std::uint8_t, 16> raw{};
  const auto slot = base + 32 + (phys * 16);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    ctx.memory.read(slot + i, &raw[i], 1);
  }
  X87Scalar value = 0;
  std::memcpy(&value, raw.data(), std::min(sizeof(value), raw.size()));
  return value;
}

void write_fpu_state(ExecutionContext& ctx, std::uint64_t base, std::size_t offset) {
  const std::uint16_t fcw = ctx.state.get_x87_control_word();
  const std::uint16_t fsw = ctx.state.get_x87_status_word();
  const std::uint8_t ftw = x87_ftw(ctx.state);
  const std::uint64_t zero64 = 0;
  ctx.memory.write(base + offset + 0, &fcw, 2);
  ctx.memory.write(base + offset + 2, &fsw, 2);
  ctx.memory.write(base + offset + 4, &ftw, 1);
  ctx.memory.write(base + offset + 5, &zero64, 1);
  ctx.memory.write(base + offset + 6, &zero64, 2);
  ctx.memory.write(base + offset + 8, &zero64, 8);
  ctx.memory.write(base + offset + 16, &zero64, 8);
}

ExecutionResult fsave(ExecutionContext& ctx, std::size_t env_size) {
  (void)env_size;
  const auto base = detail::memory_address(ctx);
  if ((base & 0x7) != 0) {
    return detail::memory_fault(ctx, base);
  }
  write_fpu_state(ctx, base, 0);
  for (std::size_t i = 0; i < 8; ++i) {
    write_fxsave_st(ctx, base, i, ctx.state.x87_get(i));
  }
  ctx.state.x87_reset();
  return {};
}

ExecutionResult frstor(ExecutionContext& ctx, std::size_t env_size) {
  (void)env_size;
  const auto base = detail::memory_address(ctx);
  if ((base & 0x7) != 0) {
    return detail::memory_fault(ctx, base);
  }
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint8_t ftw = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  for (std::size_t i = 0; i < 8; ++i) {
    if ((ftw >> i) & 1u) {
      ctx.state.x87_set(i, read_fxsave_st(ctx, base, i));
    } else {
      ctx.state.x87_mark_empty(i);
    }
  }
  return {};
}

ExecutionResult fxsave(ExecutionContext& ctx, bool /*is64*/) {
  const auto base = detail::memory_address(ctx);
  if ((base & 0xF) != 0) {
    return detail::memory_fault(ctx, base);
  }
  if (const auto span = validate_memory_span(ctx, base, 0x200, seven::MemoryAccessKind::data_write); !span.ok()) return span;
  if (!ctx.memory.write(base + 0, &ctx.state.x87_control_word, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.write(base + 2, &ctx.state.x87_status_word, 2)) return detail::memory_fault(ctx, base + 2);
  const auto ftw = x87_ftw(ctx.state);
  if (!ctx.memory.write(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  const std::uint8_t zero8 = 0;
  if (!ctx.memory.write(base + 5, &zero8, 1)) return detail::memory_fault(ctx, base + 5);
  const std::uint16_t fop = 0;
  if (!ctx.memory.write(base + 6, &fop, 2)) return detail::memory_fault(ctx, base + 6);
  const std::uint64_t zero64 = 0;
  if (!ctx.memory.write(base + 8, &zero64, 8)) return detail::memory_fault(ctx, base + 8);
  if (!ctx.memory.write(base + 16, &zero64, 8)) return detail::memory_fault(ctx, base + 16);
  if (!ctx.memory.write(base + 24, &ctx.state.mxcsr, 4)) return detail::memory_fault(ctx, base + 24);
  const std::uint32_t mxcsr_mask = 0xFFFFu;
  if (!ctx.memory.write(base + 28, &mxcsr_mask, 4)) return detail::memory_fault(ctx, base + 28);

  for (std::size_t phys = 0; phys < 8; ++phys) {
    write_fxsave_st(ctx, base, phys, ctx.state.x87_stack[phys]);
  }
  for (std::size_t i = 0; i < 16; ++i) {
    const auto value = ctx.state.vectors[i].value;
    std::array<std::uint8_t, 16> raw{};
    for (std::size_t b = 0; b < raw.size(); ++b) {
      raw[b] = static_cast<std::uint8_t>((value >> (8 * b)) & 0xFFu);
    }
    const auto slot = base + 160 + (i * 16);
    for (std::size_t b = 0; b < raw.size(); ++b) {
      if (!ctx.memory.write(slot + b, &raw[b], 1)) return detail::memory_fault(ctx, slot + b);
    }
  }
  for (std::size_t offset = 416; offset < 512; ++offset) {
    if (!ctx.memory.write(base + offset, &zero8, 1)) return detail::memory_fault(ctx, base + offset);
  }
  return {};
}

ExecutionResult fxrstor(ExecutionContext& ctx, bool /*is64*/) {
  const auto base = detail::memory_address(ctx);
  if ((base & 0xF) != 0) {
    return detail::memory_fault(ctx, base);
  }
  if (const auto span = validate_memory_span(ctx, base, 0x200, seven::MemoryAccessKind::data_read); !span.ok()) return span;
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint8_t ftw = 0;
  std::uint32_t mxcsr = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  if (!ctx.memory.read(base + 24, &mxcsr, 4)) return detail::memory_fault(ctx, base + 24);
  fcw &= static_cast<std::uint16_t>(~0xE0C0u);
  mxcsr &= 0x0000FFFFu;
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  ctx.state.mxcsr = mxcsr;
  for (std::size_t phys = 0; phys < 8; ++phys) {
    if ((ftw >> phys) & 1u) {
      ctx.state.x87_stack[phys] = read_fxsave_st(ctx, base, phys);
      ctx.state.x87_tags[phys] = (ctx.state.x87_stack[phys] == 0) ? 0x1 : 0x0;
    } else {
      ctx.state.x87_tags[phys] = 0x3;
    }
  }
  for (std::size_t i = 0; i < 16; ++i) {
    std::array<std::uint8_t, 16> raw{};
    const auto slot = base + 160 + (i * 16);
    for (std::size_t b = 0; b < raw.size(); ++b) {
      if (!ctx.memory.read(slot + b, &raw[b], 1)) return detail::memory_fault(ctx, slot + b);
    }
    seven::SimdUint value = 0;
    for (std::size_t b = 0; b < raw.size(); ++b) {
      value |= (seven::SimdUint(raw[b]) << (8 * b));
    }
    ctx.state.vectors[i].value = value;
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_MOVD_MM_RM32(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 4); }
ExecutionResult handle_code_MOVQ_MM_RM64(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVD_RM32_MM(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 4); }
ExecutionResult handle_code_MOVQ_RM64_MM(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVQ_MM_MMM64(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }
ExecutionResult handle_code_MOVQ_MMM64_MM(ExecutionContext& ctx) { return mmx_move(ctx, 0, 1, 8); }

ExecutionResult handle_code_FNINIT(ExecutionContext& ctx) {
  ctx.state.x87_reset();
  return {};
}

ExecutionResult handle_code_FINIT(ExecutionContext& ctx) {
  ctx.state.x87_reset();
  return {};
}

ExecutionResult handle_code_EMMS(ExecutionContext& ctx) {
  emms(ctx.state);
  return {};
}

ExecutionResult handle_code_FEMMS(ExecutionContext& ctx) {
  emms(ctx.state);
  return {};
}

ExecutionResult handle_code_FNCLEX(ExecutionContext& ctx) {
  ctx.state.x87_status_word &= ~std::uint16_t(0xBFu);
  return {};
}

ExecutionResult handle_code_FLD1(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(1)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDL2T(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(3.32192809488736234787)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDL2E(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(1.44269504088896340736)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDPI(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(3.14159265358979323846)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDLG2(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(0.30102999566398119521)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDLN2(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(0.69314718055994530942)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDZ(ExecutionContext& ctx) {
  if (!ctx.state.x87_push(0)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FCHS(ExecutionContext& ctx) {
  return x87_unary_st0(ctx, [](X87Scalar v) { return -v; });
}

ExecutionResult handle_code_FABS(ExecutionContext& ctx) {
  return x87_unary_st0(ctx, [](X87Scalar v) { return boost::multiprecision::abs(v); });
}

ExecutionResult handle_code_FSQRT(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  if (value < 0) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = boost::multiprecision::sqrt(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  if (auto r = x87_precision_if_changed(ctx, value, result); !r.ok()) return r;
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FRNDINT(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar rounded = x87_round_to_control(ctx.state, value);
  if (rounded != value) {
    auto result = x87_exception(ctx, kX87ExceptionPrecision);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, rounded);
  return {};
}

ExecutionResult handle_code_FSIN(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar result = boost::multiprecision::sin(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FCOS(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar result = boost::multiprecision::cos(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FSINCOS(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar x = ctx.state.x87_get(0);
  const X87Scalar cosine = boost::multiprecision::cos(x);
  const X87Scalar sine = boost::multiprecision::sin(x);
  if (x != 0 && (cosine == 0 || sine == 0)) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = static_cast<std::uint16_t>(x87_classify_result(cosine, x, 0) | x87_classify_result(sine, x, 0)); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, cosine);
  if (!ctx.state.x87_push(sine)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FPTAN(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  const X87Scalar result = boost::multiprecision::tan(value);
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  if (!ctx.state.x87_push(1)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FPATAN(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  ctx.state.x87_set(1, boost::multiprecision::atan2(y, x));
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_F2XM1(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow(ctx);
  const X87Scalar value = ctx.state.x87_get(0);
  if (value < -1 || value > 1) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = boost::multiprecision::pow(2, value) - 1;
  if (value != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, value, 0); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FYL2X(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  if (x <= 0) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = y * boost::multiprecision::log2(x);
  if (y != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, y, x); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_FYL2XP1(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  if (x <= -1) return x87_exception(ctx, kX87ExceptionInvalid);
  const X87Scalar result = y * boost::multiprecision::log2(x + 1);
  if (y != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, y, x + 1); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  if (!ctx.state.x87_pop()) return x87_stack_underflow(ctx);
  return {};
}

ExecutionResult handle_code_FSCALE(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar a = ctx.state.x87_get(0);
  const X87Scalar b = ctx.state.x87_get(1);
  const X87Scalar result = boost::multiprecision::ldexp(a, static_cast<int>(boost::multiprecision::trunc(b)));
  if (a != 0 && result == 0) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = x87_classify_result(result, a, b); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(0, result);
  return {};
}

ExecutionResult handle_code_FLDENV_M14BYTE(ExecutionContext& ctx) {
  return load_x87_env(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_FLDENV_M28BYTE(ExecutionContext& ctx) {
  return load_x87_env(ctx, detail::memory_address(ctx));
}

ExecutionResult handle_code_FLD_M32FP(ExecutionContext& ctx) {
  return x87_push_from_memory(ctx, 4);
}

ExecutionResult handle_code_FLD_M64FP(ExecutionContext& ctx) {
  return x87_push_from_memory(ctx, 8);
}

ExecutionResult handle_code_FLD_M80FP(ExecutionContext& ctx) {
  return x87_push_from_memory(ctx, 10);
}

ExecutionResult handle_code_FLD_STI(ExecutionContext& ctx) {
  if (ctx.instr.op_kind(0) != iced_x86::OpKind::REGISTER || ctx.instr.op_kind(1) != iced_x86::OpKind::REGISTER) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if (!ctx.state.x87_push(ctx.state.x87_get(x87_st_index(ctx.instr.op_register(0))))) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult handle_code_FNSTENV_M14BYTE(ExecutionContext& ctx) {
  store_x87_env(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult handle_code_FSTENV_M14BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTENV_M14BYTE(ctx);
}

ExecutionResult handle_code_FNSTENV_M28BYTE(ExecutionContext& ctx) {
  store_x87_env(ctx, detail::memory_address(ctx));
  return {};
}

ExecutionResult handle_code_FSTENV_M28BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTENV_M28BYTE(ctx);
}

ExecutionResult handle_code_FLDCW_M2BYTE(ExecutionContext& ctx) {
  std::uint16_t value = 0;
  if (!detail::read_operand_checked(ctx, 0, 2, value).ok()) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  if ((value & 0xE0C0u) != 0) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  ctx.state.set_x87_control_word(value);
  return {};
}

ExecutionResult handle_code_FNSTCW_M2BYTE(ExecutionContext& ctx) {
  if (!detail::write_operand(ctx, 0, ctx.state.get_x87_control_word(), 2)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult handle_code_FST_M32FP(ExecutionContext& ctx) {
  return x87_store_to_memory(ctx, 4, false);
}

ExecutionResult handle_code_FST_M64FP(ExecutionContext& ctx) {
  return x87_store_to_memory(ctx, 8, false);
}

ExecutionResult handle_code_FST_STI(ExecutionContext& ctx) {
  return x87_reg_move(ctx, 0, 1, false);
}

ExecutionResult handle_code_FNSTSW_M2BYTE(ExecutionContext& ctx) {
  if (!detail::write_operand(ctx, 0, ctx.state.get_x87_status_word(), 2)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  return {};
}

ExecutionResult handle_code_FSTSW_M2BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTSW_M2BYTE(ctx);
}

ExecutionResult handle_code_FNSTSW_AX(ExecutionContext& ctx) {
  detail::write_register(ctx.state, iced_x86::Register::AX, ctx.state.get_x87_status_word(), 2);
  return {};
}

ExecutionResult handle_code_FSTSW_AX(ExecutionContext& ctx) {
  return handle_code_FNSTSW_AX(ctx);
}

ExecutionResult handle_code_FXSAVE_M512BYTE(ExecutionContext& ctx) {
  return fxsave(ctx, false);
}

ExecutionResult handle_code_FXSAVE64_M512BYTE(ExecutionContext& ctx) {
  return fxsave(ctx, true);
}

ExecutionResult handle_code_FXRSTOR_M512BYTE(ExecutionContext& ctx) {
  return fxrstor(ctx, false);
}

ExecutionResult handle_code_FXRSTOR64_M512BYTE(ExecutionContext& ctx) {
  return fxrstor(ctx, true);
}

ExecutionResult handle_code_FSAVE_M94BYTE(ExecutionContext& ctx) {
  return fsave(ctx, 94);
}

ExecutionResult handle_code_FSAVE_M108BYTE(ExecutionContext& ctx) {
  return fsave(ctx, 108);
}

ExecutionResult handle_code_FRSTOR_M94BYTE(ExecutionContext& ctx) {
  return frstor(ctx, 94);
}

ExecutionResult handle_code_FRSTOR_M108BYTE(ExecutionContext& ctx) {
  return frstor(ctx, 108);
}

ExecutionResult handle_code_FILD_M16INT(ExecutionContext& ctx) {
  return x87_load_integer(ctx, 2);
}

ExecutionResult handle_code_FILD_M32INT(ExecutionContext& ctx) {
  return x87_load_integer(ctx, 4);
}

ExecutionResult handle_code_FILD_M64INT(ExecutionContext& ctx) {
  return x87_load_integer(ctx, 8);
}

ExecutionResult handle_code_FIST_M16INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 2, false, false);
}

ExecutionResult handle_code_FIST_M32INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 4, false, false);
}

ExecutionResult handle_code_FIST_M64INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 8, false, false);
}

ExecutionResult handle_code_FISTP_M16INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 2, true, false);
}

ExecutionResult handle_code_FISTP_M32INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 4, true, false);
}

ExecutionResult handle_code_FISTP_M64INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 8, true, false);
}

ExecutionResult handle_code_FISTTP_M16INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 2, true, true);
}

ExecutionResult handle_code_FISTTP_M32INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 4, true, true);
}

ExecutionResult handle_code_FISTTP_M64INT(ExecutionContext& ctx) {
  return x87_store_integer(ctx, 8, true, true);
}

ExecutionResult handle_code_MASKMOVQ_R_DI_MM_MM(ExecutionContext& ctx) {
  return store_masked_bytes(ctx, 1, 2, 8, false);
}

ExecutionResult handle_code_MASKMOVDQU_R_DI_XMM_XMM(ExecutionContext& ctx) {
  return store_masked_bytes(ctx, 1, 2, 16, true);
}

ExecutionResult handle_code_VEX_VMASKMOVDQU_R_DI_XMM_XMM(ExecutionContext& ctx) {
  return store_masked_bytes(ctx, 1, 2, 16, true);
}

ExecutionResult handle_code_PMOVMSKB_R32_MM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 8, true, 4);
}

ExecutionResult handle_code_PMOVMSKB_R64_MM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 8, true, 8);
}

ExecutionResult handle_code_PMOVMSKB_R32_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 4);
}

ExecutionResult handle_code_PMOVMSKB_R64_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 8);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R32_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 4);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R64_XMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 16, false, 8);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R32_YMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 32, false, 4);
}

ExecutionResult handle_code_VEX_VPMOVMSKB_R64_YMM(ExecutionContext& ctx) {
  return pmovmskb(ctx, 32, false, 8);
}
ExecutionResult handle_code_MOVMSKPD_R32_XMM(ExecutionContext& ctx) {
  return movmskpd(ctx, 4);
}
ExecutionResult handle_code_MOVMSKPD_R64_XMM(ExecutionContext& ctx) {
  return movmskpd(ctx, 8);
}
ExecutionResult handle_code_MOVQ2DQ_XMM_MM(ExecutionContext& ctx) {
  const auto dst_reg = ctx.instr.op_register(0);
  const auto src = read_mmx(ctx.state, ctx.instr.op_register(1));
  auto& slot = ctx.state.vectors[vector_index(dst_reg)].value;
  slot = (slot & ~mask(16)) | big_uint(src);
  return {};
}
ExecutionResult handle_code_MOVDQ2Q_MM_XMM(ExecutionContext& ctx) {
  const auto src_reg = ctx.instr.op_register(1);
  const auto value = static_cast<std::uint64_t>(ctx.state.vectors[vector_index(src_reg)].value & mask(8));
  ctx.state.mmx_set(static_cast<std::size_t>(static_cast<std::uint32_t>(ctx.instr.op_register(0)) - static_cast<std::uint32_t>(iced_x86::Register::MM0)), value);
  return {};
}

}  // namespace seven::handlers






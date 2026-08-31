#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "seven/handler_helpers.hpp"
#include "seven/x87_helpers.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

namespace {
using big_uint = seven::SimdUint;

big_uint mask(std::size_t width) {
  return (big_uint(1) << (width * 8)) - 1;
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
  // Callers check the whole image before touching any of it, but base + offset is plain uint64, so a
  // span near the top of the address space wrapped and validated page 0 before faulting mid-store.
  if (size != 0 && base + (static_cast<std::uint64_t>(size) - 1u) < base) {
    return detail::memory_fault(ctx, base);
  }
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
    const auto data = ctx.state.vectors[vector_index(data_reg)].value;
    const auto mask_value = ctx.state.vectors[vector_index(mask_reg)].value;
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

// These come in a 16-bit form with a 14-byte environment and a 32-bit form with a 28-byte one, laid
// out at different offsets; FNSAVE adds eight 10-byte slots for 94 and 108 total. The tag word is
// per physical register while the data slots are top-relative, and crossing the two loses registers.
constexpr std::size_t kX87Slots = 8;
constexpr std::size_t kX87SlotBytes = 10;

struct X87EnvLayout {
  std::size_t fcw;
  std::size_t fsw;
  std::size_t ftw;
};

constexpr X87EnvLayout x87_env_layout(std::size_t env_size) {
  // The 32-bit form widens each of these to four bytes, with the upper half reserved.
  return env_size == 14 ? X87EnvLayout{0, 2, 4} : X87EnvLayout{0, 4, 8};
}

std::uint16_t x87_full_tag_word(const CpuState& state) {
  std::uint16_t ftw = 0;
  for (std::size_t phys = 0; phys < kX87Slots; ++phys) {
    ftw |= static_cast<std::uint16_t>((state.x87_tags[phys] & 0x3u) << (phys * 2));
  }
  return ftw;
}

void put_le16(std::uint8_t* dst, std::uint16_t value) {
  dst[0] = static_cast<std::uint8_t>(value);
  dst[1] = static_cast<std::uint8_t>(value >> 8);
}

std::uint16_t get_le16(const std::uint8_t* src) {
  return static_cast<std::uint16_t>(src[0] | (static_cast<std::uint16_t>(src[1]) << 8));
}

ExecutionResult store_x87_env(ExecutionContext& ctx, std::uint64_t base, std::size_t env_size) {
  std::array<std::uint8_t, 28> image{};
  // 14 or 28 by construction, but fsave derives it by subtracting the register file from the image
  // size, and that subtraction would wrap to an enormous size_t if the image were ever smaller.
  env_size = std::min(env_size, image.size());
  const auto layout = x87_env_layout(env_size);
  put_le16(image.data() + layout.fcw, ctx.state.get_x87_control_word());
  put_le16(image.data() + layout.fsw, ctx.state.get_x87_status_word());
  put_le16(image.data() + layout.ftw, x87_full_tag_word(ctx.state));
  // FIP/FCS/FDP/FDS stay zero; this emulator does not track the x87 instruction or data pointers.
  if (!ctx.memory.write(base, image.data(), env_size)) return detail::memory_fault(ctx, base);
  // Storing the environment masks every exception, so the handler about to walk it cannot be
  // interrupted by one of its own. The image above was built first and keeps the old control word.
  // FSAVE reaches the same state through the reset it does afterwards.
  ctx.state.set_x87_control_word(
      static_cast<std::uint16_t>(ctx.state.get_x87_control_word() | 0x3Fu));
  return {};
}

ExecutionResult load_x87_env(ExecutionContext& ctx, std::uint64_t base, std::size_t env_size) {
  std::array<std::uint8_t, 28> image{};
  env_size = std::min(env_size, image.size());
  if (!ctx.memory.read(base, image.data(), env_size)) return detail::memory_fault(ctx, base);
  const auto layout = x87_env_layout(env_size);
  // No reserved-bit check on the control word. FLDENV loads whatever is there rather than faulting
  // on the value, and the old check reported it as a page fault at that.
  ctx.state.set_x87_control_word(get_le16(image.data() + layout.fcw));
  ctx.state.set_x87_status_word(get_le16(image.data() + layout.fsw));
  const auto ftw = get_le16(image.data() + layout.ftw);
  for (std::size_t phys = 0; phys < kX87Slots; ++phys) {
    ctx.state.x87_tags[phys] = static_cast<std::uint8_t>((ftw >> (phys * 2)) & 0x3u);
  }
  return {};
}

ExecutionResult write_fxsave_st(ExecutionContext& ctx, std::uint64_t base, std::size_t phys, X87Scalar value) {
  std::array<std::uint8_t, 16> raw{};
  x87_encoding::encode_ext80(value, raw.data());
  const auto slot = base + 32 + (phys * 16);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (!ctx.memory.write(slot + i, &raw[i], 1)) return detail::memory_fault(ctx, slot + i);
  }
  return {};
}

ExecutionResult read_fxsave_st(ExecutionContext& ctx, std::uint64_t base, std::size_t phys, X87Scalar& out) {
  std::array<std::uint8_t, 16> raw{};
  const auto slot = base + 32 + (phys * 16);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (!ctx.memory.read(slot + i, &raw[i], 1)) return detail::memory_fault(ctx, slot + i);
  }
  out = x87_encoding::decode_ext80(raw.data());
  return {};
}

ExecutionResult write_fpu_state(ExecutionContext& ctx, std::uint64_t base, std::size_t offset) {
  const std::uint16_t fcw = ctx.state.get_x87_control_word();
  const std::uint16_t fsw = ctx.state.get_x87_status_word();
  const std::uint8_t ftw = x87_ftw(ctx.state);
  const std::uint64_t zero64 = 0;
  if (!ctx.memory.write(base + offset + 0, &fcw, 2)) return detail::memory_fault(ctx, base + offset + 0);
  if (!ctx.memory.write(base + offset + 2, &fsw, 2)) return detail::memory_fault(ctx, base + offset + 2);
  if (!ctx.memory.write(base + offset + 4, &ftw, 1)) return detail::memory_fault(ctx, base + offset + 4);
  if (!ctx.memory.write(base + offset + 5, &zero64, 1)) return detail::memory_fault(ctx, base + offset + 5);
  if (!ctx.memory.write(base + offset + 6, &zero64, 2)) return detail::memory_fault(ctx, base + offset + 6);
  if (!ctx.memory.write(base + offset + 8, &zero64, 8)) return detail::memory_fault(ctx, base + offset + 8);
  if (!ctx.memory.write(base + offset + 16, &zero64, 8)) return detail::memory_fault(ctx, base + offset + 16);
  return {};
}

ExecutionResult fsave(ExecutionContext& ctx, std::size_t image_size) {
  const auto base = detail::memory_address(ctx);
  const auto env_size = image_size - (kX87Slots * kX87SlotBytes);
  // No alignment requirement on hardware. Validate the whole image first, since hardware faults
  // before any of the store and the x87_reset() below must not wipe state for a save that failed.
  if (const auto span = validate_memory_span(ctx, base, image_size, seven::MemoryAccessKind::data_write); !span.ok()) {
    return span;
  }
  if (const auto r = store_x87_env(ctx, base, env_size); !r.ok()) return r;
  for (std::size_t i = 0; i < kX87Slots; ++i) {
    std::array<std::uint8_t, kX87SlotBytes> raw{};
    x87_encoding::encode_ext80(ctx.state.x87_get(i), raw.data());
    const auto slot = base + env_size + (i * kX87SlotBytes);
    if (!ctx.memory.write(slot, raw.data(), raw.size())) return detail::memory_fault(ctx, slot);
  }
  ctx.state.x87_reset();
  return {};
}

ExecutionResult frstor(ExecutionContext& ctx, std::size_t image_size) {
  const auto base = detail::memory_address(ctx);
  const auto env_size = image_size - (kX87Slots * kX87SlotBytes);
  if (const auto span = validate_memory_span(ctx, base, image_size, seven::MemoryAccessKind::data_read); !span.ok()) {
    return span;
  }
  // The environment has to land first: it carries TOP, which is what makes the top-relative slot
  // indexing below resolve to the right physical registers.
  if (const auto r = load_x87_env(ctx, base, env_size); !r.ok()) return r;
  for (std::size_t i = 0; i < kX87Slots; ++i) {
    std::array<std::uint8_t, kX87SlotBytes> raw{};
    const auto slot = base + env_size + (i * kX87SlotBytes);
    if (!ctx.memory.read(slot, raw.data(), raw.size())) return detail::memory_fault(ctx, slot);
    // Straight into the stack rather than through x87_set, which would stamp the tag valid and
    // undo the tag word the environment just restored.
    ctx.state.x87_stack[ctx.state.x87_phys_index(i)] = x87_encoding::decode_ext80(raw.data());
  }
  return {};
}

ExecutionResult fxsave(ExecutionContext& ctx, bool /*is64*/) {
  const auto base = detail::memory_address(ctx);
  // A misaligned FXSAVE area is #GP(0), not a page fault, the same as the legacy SSE m128 forms.
  if (auto fault = detail::require_aligned_memory_operand(ctx, 0, 0xFULL)) return *fault;
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

  // Slot i holds ST(i). The tag word written above is indexed by physical register instead, which is
  // the pairing fsave/frstor already use; doing both by physical index put every register in the
  // wrong slot whenever TOP was not 0, and only showed up outside a save/restore round trip.
  for (std::size_t i = 0; i < 8; ++i) {
    if (const auto r = write_fxsave_st(ctx, base, i, ctx.state.x87_get(i)); !r.ok()) return r;
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
  if (auto fault = detail::require_aligned_memory_operand(ctx, 0, 0xFULL)) return *fault;
  if (const auto span = validate_memory_span(ctx, base, 0x200, seven::MemoryAccessKind::data_read); !span.ok()) return span;
  std::uint16_t fcw = 0;
  std::uint16_t fsw = 0;
  std::uint8_t ftw = 0;
  std::uint32_t mxcsr = 0;
  if (!ctx.memory.read(base + 0, &fcw, 2)) return detail::memory_fault(ctx, base + 0);
  if (!ctx.memory.read(base + 2, &fsw, 2)) return detail::memory_fault(ctx, base + 2);
  if (!ctx.memory.read(base + 4, &ftw, 1)) return detail::memory_fault(ctx, base + 4);
  if (!ctx.memory.read(base + 24, &mxcsr, 4)) return detail::memory_fault(ctx, base + 24);
  // Loaded as-is, same as FLDENV and FLDCW: clearing the reserved bits here meant the control word
  // did not survive its own fxsave/fxrstor.
  mxcsr &= 0x0000FFFFu;
  ctx.state.set_x87_control_word(fcw);
  ctx.state.set_x87_status_word(fsw);
  ctx.state.mxcsr = mxcsr;
  // Slot i is ST(i), the tag bits are physical -- see fxsave. The status word above carried TOP, so
  // x87_phys_index resolves against the image's own top rather than the one we started with.
  for (std::size_t i = 0; i < 8; ++i) {
    const auto phys = ctx.state.x87_phys_index(i);
    if ((ftw >> phys) & 1u) {
      X87Scalar loaded{};
      if (const auto r = read_fxsave_st(ctx, base, i, loaded); !r.ok()) return r;
      ctx.state.x87_stack[phys] = loaded;
      ctx.state.x87_tags[phys] = seven::x87_tag_of(loaded);
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
  // The stack fault bit and the busy bit are part of what FNCLEX clears. Leaving SF set meant a
  // guest that cleared after a stack fault still read one back on the next FNSTSW.
  ctx.state.x87_status_word &= ~std::uint16_t(0x80FFu);
  return {};
}

ExecutionResult handle_code_FLD1(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(1)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDL2T(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(x87_constant(ctx.state, 0x4000u, 0xD49A784BCD1B8AFEull, false))) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDL2E(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(x87_constant(ctx.state, 0x3FFFu, 0xB8AA3B295C17F0BBull, true))) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDPI(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(x87_constant(ctx.state, 0x4000u, 0xC90FDAA22168C234ull, true))) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDLG2(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(x87_constant(ctx.state, 0x3FFDu, 0x9A209A84FBCFF798ull, true))) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDLN2(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(x87_constant(ctx.state, 0x3FFEu, 0xB17217F7D1CF79ABull, true))) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDZ(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (!ctx.state.x87_push(0)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FCHS(ExecutionContext& ctx) {
  return x87_unary_st0(ctx, [](X87Scalar v) { return -v; });
}

ExecutionResult handle_code_FABS(ExecutionContext& ctx) {
  return x87_unary_st0(ctx, [](X87Scalar v) { return seven::abs(v); });
}

// extF80_sqrt raises #IA on a negative operand itself and reports its own inexactness, so none of
// this needs guessing. The old version asked whether sqrt(x) differed from x and called that a
// precision loss, which is true of almost every square root there is.
ExecutionResult handle_code_FSQRT(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  return x87_finish(ctx, 0, x87_evaluate(ctx.state, value, 0, [](X87Scalar v, X87Scalar) {
                      return seven::sqrt(v);
                    }));
}

ExecutionResult handle_code_FRNDINT(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  // An encoding hardware refuses to interpret is #IA with the indefinite for an answer, and it is
  // decided before any rounding happens. Rounding it first reported an inexact result instead.
  if (seven::isunsupported(value)) {
    auto result = x87_exception(ctx, kX87ExceptionInvalid);
    if (!result.ok()) return result;
    ctx.state.x87_set(0, x87_indefinite());
    return {};
  }
  if (seven::isnan(value) || seven::isinf(value)) {
    ctx.state.x87_set(0, value);
    return {};
  }
  std::uint16_t exceptions = 0;
  if (x87_is_denormal_operand(value)) exceptions |= kX87ExceptionDenormal;
  const X87Scalar rounded = x87_round_to_control(ctx.state, value);
  if (rounded != value) {
    exceptions |= kX87ExceptionPrecision;
    x87_set_c1(ctx, rounded != seven::trunc(value));
  }
  if (exceptions != 0) {
    auto result = x87_exception(ctx, exceptions);
    if (!result.ok()) return result;
  }
  ctx.state.x87_set(0, rounded);
  return {};
}

ExecutionResult handle_code_FSIN(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  if (auto answer = x87_reject_operand(ctx, value); answer.has_value()) return *answer;
  if (x87_trig_argument_out_of_range(value)) {
    x87_set_c2(ctx, true);
    return {};
  }
  x87_set_c2(ctx, false);
  const X87Scalar result = x87_tiny_argument(value) ? value : seven::sin(value);
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
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  if (auto answer = x87_reject_operand(ctx, value); answer.has_value()) return *answer;
  if (x87_trig_argument_out_of_range(value)) {
    x87_set_c2(ctx, true);
    return {};
  }
  x87_set_c2(ctx, false);
  const X87Scalar result = x87_tiny_argument(value) ? X87Scalar(1) : seven::cos(value);
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
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar x = ctx.state.x87_get(0);
  if (auto answer = x87_reject_operand_and_push(ctx, x); answer.has_value()) return *answer;
  if (x87_trig_argument_out_of_range(x)) {
    x87_set_c2(ctx, true);
    return {};
  }
  x87_set_c2(ctx, false);
  const bool tiny = x87_tiny_argument(x);
  const X87Scalar cosine = tiny ? X87Scalar(1) : seven::cos(x);
  const X87Scalar sine = tiny ? x : seven::sin(x);
  if (x != 0 && (cosine == 0 || sine == 0)) {
    auto r = x87_exception(ctx, static_cast<std::uint16_t>(kX87ExceptionUnderflow | kX87ExceptionPrecision));
    if (!r.ok()) return r;
  }
  if (const auto exceptions = static_cast<std::uint16_t>(x87_classify_result(cosine, x, 0) | x87_classify_result(sine, x, 0)); exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  // The sine replaces ST(0) and the cosine is what gets pushed, so the cosine ends up on top. This
  // was the other way round, which every caller reading ST(0) as the cosine got wrong.
  ctx.state.x87_set(0, sine);
  if (!ctx.state.x87_push(cosine)) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FPTAN(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  if (auto answer = x87_reject_operand_and_push(ctx, value); answer.has_value()) return *answer;
  if (x87_trig_argument_out_of_range(value)) {
    x87_set_c2(ctx, true);
    return {};
  }
  x87_set_c2(ctx, false);
  const X87Scalar result = x87_tiny_argument(value) ? value : seven::tan(value);
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
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  X87Scalar result{};
  std::uint16_t exceptions = 0;
  if (!x87_special_result(x, y, result, exceptions)) {
    exceptions = x87_operand_exceptions(x, y);
    result = seven::atan2(y, x);
  }
  if (exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  x87_forced_pop(ctx.state);
  return {};
}

ExecutionResult handle_code_F2XM1(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  if (auto answer = x87_reject_operand(ctx, value); answer.has_value()) return *answer;
  // The SDM calls the result undefined outside [-1, 1], but silicon just evaluates 2^x - 1, so
  // F2XM1 of -infinity answers -1. Below 2^-64 the answer is x*ln2, which a host double flattens
  // to zero across a whole denormal range.
  const X87Scalar result = x87_tiny_argument(value)
                               ? value * x87_ln2()
                               : seven::pow(X87Scalar(2), value) - X87Scalar(1);
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
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  X87Scalar result{};
  std::uint16_t exceptions = 0;
  if (!x87_special_result(x, y, result, exceptions)) {
    exceptions = x87_operand_exceptions(x, y);
    // log2(0) is -infinity, which is a divide-by-zero on the x87, not an invalid operand. Only the
    // 0 * infinity case that ST(1) = 0 produces is genuinely invalid.
    if (x == 0 && y != 0) {
      exceptions |= kX87ExceptionZeroDiv;
      const auto infinity = std::numeric_limits<X87Scalar>::infinity();
      result = seven::signbit(y) ? infinity : -infinity;
    } else if (seven::signbit(x) || x == 0) {
      exceptions |= kX87ExceptionInvalid;
      result = x87_indefinite();
    } else {
      result = y * seven::log2(x);
      exceptions |= x87_classify_result(result, y, x);
      if (y != 0 && result == 0) exceptions |= kX87ExceptionUnderflow | kX87ExceptionPrecision;
    }
  }
  // A masked exception does not cancel the instruction: hardware still writes ST(1) and still pops.
  // Returning from the #IA path without doing either left the guest's stack where it started.
  if (exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  x87_forced_pop(ctx.state);
  return {};
}

ExecutionResult handle_code_FYL2XP1(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow(ctx);
  const X87Scalar y = ctx.state.x87_get(1);
  const X87Scalar x = ctx.state.x87_get(0);
  X87Scalar result{};
  std::uint16_t exceptions = 0;
  if (!x87_special_result(x, y, result, exceptions)) {
    exceptions = x87_operand_exceptions(x, y);
    if (x <= X87Scalar(-1)) {
      exceptions |= kX87ExceptionInvalid;
      result = x87_indefinite();
    } else if (x87_tiny_argument(x)) {
      // log2(1 + x) collapses to x / ln2 below 2^-64, and this is the range FYL2XP1 exists for.
      result = y * (x / x87_ln2());
    } else {
      result = y * seven::log2(x + 1);
      exceptions |= x87_classify_result(result, y, x + 1);
      if (y != 0 && result == 0) exceptions |= kX87ExceptionUnderflow | kX87ExceptionPrecision;
    }
  }
  if (exceptions != 0) {
    auto r = x87_exception(ctx, exceptions);
    if (!r.ok()) return r;
  }
  ctx.state.x87_set(1, result);
  x87_forced_pop(ctx.state);
  return {};
}

ExecutionResult handle_code_FSCALE(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0) || ctx.state.x87_is_empty(1)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar a = ctx.state.x87_get(0);
  const X87Scalar b = ctx.state.x87_get(1);
  {
    X87Scalar answer{};
    std::uint16_t special = 0;
    if (x87_special_result(a, b, answer, special)) {
      if (special != 0) {
        auto r = x87_exception(ctx, special);
        if (!r.ok()) return r;
      }
      ctx.state.x87_set(0, answer);
      return {};
    }
  }
  // ST(1) is whatever the guest left there and the narrowing below is only defined inside int's
  // range. ldexp saturates long before the clamp bites, so +inf and 2^70 both answer as hardware
  // does, where the raw cast landed on 0 and returned ST(0) unchanged.
  const X87Scalar truncated = seven::trunc(b);
  constexpr int kMaxShift = 0x7FFFFFFF;
  int shift = 0;
  if (truncated > X87Scalar(kMaxShift)) {
    shift = kMaxShift;
  } else if (truncated < X87Scalar(-kMaxShift)) {
    shift = -kMaxShift;
  } else {
    shift = static_cast<int>(truncated);
  }
  const X87Scalar result = seven::ldexp(a, shift);
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

// D9 F4. Splits ST(0) into its unbiased exponent and its significand, leaving the exponent in ST(1)
// and pushing the significand. It had no handler at all, so every FXTRACT stopped the guest with
// unsupported_instruction.
ExecutionResult handle_code_FXTRACT(ExecutionContext& ctx) {
  x87_set_c1(ctx, false);
  if (ctx.state.x87_is_empty(0)) return x87_stack_underflow_into(ctx, 0);
  const X87Scalar value = ctx.state.x87_get(0);
  const auto infinity = std::numeric_limits<X87Scalar>::infinity();
  // Same as FRNDINT: the encoding is judged before the split is attempted, and an unnormal leaves the
  // indefinite in both halves.
  if (seven::isunsupported(value)) {
    auto r = x87_exception(ctx, kX87ExceptionInvalid);
    if (!r.ok()) return r;
    ctx.state.x87_set(0, x87_indefinite());
    if (!ctx.state.x87_push(x87_indefinite())) return x87_stack_overflow(ctx);
    return {};
  }
  if (x87_is_denormal_operand(value)) {
    auto r = x87_exception(ctx, kX87ExceptionDenormal);
    if (!r.ok()) return r;
  }
  if (value == 0) {
    auto r = x87_exception(ctx, kX87ExceptionZeroDiv);
    if (!r.ok()) return r;
    ctx.state.x87_set(0, -infinity);
    if (!ctx.state.x87_push(value)) return x87_stack_overflow(ctx);
    return {};
  }
  if (seven::isnan(value)) {
    const X87Scalar answer = seven::issnan(value) ? seven::quiet(value) : value;
    if (seven::issnan(value)) {
      auto r = x87_exception(ctx, kX87ExceptionInvalid);
      if (!r.ok()) return r;
    }
    ctx.state.x87_set(0, answer);
    if (!ctx.state.x87_push(answer)) return x87_stack_overflow(ctx);
    return {};
  }
  if (seven::isinf(value)) {
    ctx.state.x87_set(0, infinity);
    if (!ctx.state.x87_push(value)) return x87_stack_overflow(ctx);
    return {};
  }
  int exponent = 0;
  const X87Scalar mantissa = seven::frexp(value, &exponent);
  // frexp normalizes to [0.5, 1); the x87 wants the significand in [1, 2), hence the shared step.
  ctx.state.x87_set(0, X87Scalar(static_cast<std::int64_t>(exponent) - 1));
  if (!ctx.state.x87_push(seven::ldexp(mantissa, 1))) return x87_stack_overflow(ctx);
  return {};
}

ExecutionResult handle_code_FLDENV_M14BYTE(ExecutionContext& ctx) {
  return load_x87_env(ctx, detail::memory_address(ctx), 14);
}

ExecutionResult handle_code_FLDENV_M28BYTE(ExecutionContext& ctx) {
  return load_x87_env(ctx, detail::memory_address(ctx), 28);
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
  x87_set_c1(ctx, false);
  if (!x87_operand_is_st(ctx, 0)) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  // FLD ST(i) reads a register like any other operand, so an empty source is a stack underflow and
  // the indefinite is what gets pushed. Pushing the stale contents instead handed the guest a value
  // it had already popped, with no #IS to say anything had gone wrong.
  const auto src = x87_st_index(ctx.instr.op_register(0));
  if (ctx.state.x87_is_empty(src)) {
    auto fault = x87_stack_underflow(ctx);
    if (!fault.ok()) return fault;
    if (!ctx.state.x87_push(x87_indefinite())) return x87_stack_overflow(ctx);
    return {};
  }
  if (!ctx.state.x87_push(ctx.state.x87_get(src))) {
    return x87_stack_overflow(ctx);
  }
  return {};
}

ExecutionResult handle_code_FNSTENV_M14BYTE(ExecutionContext& ctx) {
  return store_x87_env(ctx, detail::memory_address(ctx), 14);
}

ExecutionResult handle_code_FSTENV_M14BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTENV_M14BYTE(ctx);
}

ExecutionResult handle_code_FNSTENV_M28BYTE(ExecutionContext& ctx) {
  return store_x87_env(ctx, detail::memory_address(ctx), 28);
}

ExecutionResult handle_code_FSTENV_M28BYTE(ExecutionContext& ctx) {
  return handle_code_FNSTENV_M28BYTE(ctx);
}

ExecutionResult handle_code_FLDCW_M2BYTE(ExecutionContext& ctx) {
  std::uint16_t value = 0;
  if (!detail::read_operand_checked(ctx, 0, 2, value).ok()) {
    return detail::memory_fault(ctx, detail::memory_address(ctx));
  }
  // No reserved-bit check, for the reason FLDENV's load already gives: hardware ignores those bits
  // rather than faulting on them, and 0x037F -- the value the FPU resets to, so the one every
  // fnstcw/fldcw pair round-trips -- has bit 6 set and was being rejected.
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
  // FST ST(i) names only the destination; the source is always ST(0).
  return x87_store_st0_to_sti(ctx, false);
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

// DD /6 decodes to these, not to the FSAVE_* codes above -- see handled_codes.def.
ExecutionResult handle_code_FNSAVE_M94BYTE(ExecutionContext& ctx) {
  return fsave(ctx, 94);
}

ExecutionResult handle_code_FNSAVE_M108BYTE(ExecutionContext& ctx) {
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






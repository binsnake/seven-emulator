#include "seven/handler_helpers.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace seven::handlers {
namespace {

constexpr std::uint64_t bit_width(std::size_t size) {
  return size * 8ull;
}

constexpr std::uint64_t width_mask(std::size_t size) {
  return (size >= 8) ? ~0ull : ((1ull << bit_width(size)) - 1ull);
}

std::uint64_t bextr(std::uint64_t source, std::uint64_t control, std::size_t size) {
  const auto bits = bit_width(size);
  const auto start = control & 0xFFull;
  const auto len = (control >> 8) & 0xFFull;
  if (start >= bits) {
    return 0;
  }

  const auto extract_len = (start + len >= bits) ? (bits - start) : len;
  std::uint64_t result = 0;
  for (std::uint64_t i = 0; i < extract_len; ++i) {
    if ((source >> (start + i)) & 1ull) {
      result |= (1ull << i);
    }
  }
  return result;
}

std::uint64_t pext(std::uint64_t source, std::uint64_t mask, std::size_t size) {
  const auto bits = bit_width(size);
  std::uint64_t result = 0;
  std::uint64_t destination_bit = 0;
  for (std::uint64_t i = 0; i < bits; ++i) {
    if ((mask >> i) & 1ull) {
      if ((source >> i) & 1ull) {
        result |= (1ull << destination_bit);
      }
      ++destination_bit;
    }
  }
  return result;
}

std::uint64_t pdep(std::uint64_t source, std::uint64_t mask, std::size_t size) {
  const auto bits = bit_width(size);
  std::uint64_t result = 0;
  std::uint64_t source_bit = 0;
  for (std::uint64_t i = 0; i < bits; ++i) {
    if ((mask >> i) & 1ull) {
      if ((source >> source_bit) & 1ull) {
        result |= (1ull << i);
      }
      ++source_bit;
    }
  }
  return result;
}

std::uint64_t mulx_high(std::uint64_t lhs, std::uint64_t rhs, std::size_t size) {
  if (size == 4) {
    const std::uint64_t product = lhs * rhs;
    return product >> 32;
  }
#if defined(_MSC_VER) && defined(_M_X64)
  return __umulh(lhs, rhs);
#else
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
  return static_cast<std::uint64_t>(product >> 64);
#endif
}

std::uint64_t mulx_low(std::uint64_t lhs, std::uint64_t rhs, std::size_t size) {
  if (size == 4) {
    const std::uint64_t product = lhs * rhs;
    return product & 0xFFFFFFFFull;
  }
#if defined(_MSC_VER) && defined(_M_X64)
  return lhs * rhs;
#else
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
  return static_cast<std::uint64_t>(product & 0xFFFFFFFFFFFFFFFFull);
#endif
}

std::uint64_t shift_amount(std::size_t size, std::uint64_t value) {
  return value & (bit_width(size) - 1ull);
}

std::uint64_t rotate_right(std::uint64_t value, std::uint64_t count, std::size_t size) {
  const auto bits = bit_width(size);
  const auto masked_count = count & (bits - 1ull);
  if (masked_count == 0) {
    return value;
  }
  return (value >> masked_count) | (value << (bits - masked_count));
}

std::uint64_t arithmetic_shift_right(std::uint64_t value, std::uint64_t count, std::size_t size) {
  const auto masked_count = shift_amount(size, count);
  if (masked_count == 0) {
    return value;
  }
  const auto sign_extended = detail::sign_extend(value, size);
  return static_cast<std::uint64_t>((static_cast<std::int64_t>(sign_extended) >> masked_count)) & width_mask(size);
}

std::uint64_t shift_left(std::uint64_t value, std::uint64_t count, std::size_t size) {
  const auto masked_count = shift_amount(size, count);
  return (value << masked_count) & width_mask(size);
}

std::uint64_t shift_right(std::uint64_t value, std::uint64_t count, std::size_t size) {
  const auto masked_count = shift_amount(size, count);
  return (value >> masked_count) & width_mask(size);
}

ExecutionResult handle_code_blsi(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = src & (~src + 1ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & (1ull << (bit_width(size) - 1ull))) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagZF, result == 0);
  detail::set_flag(ctx.state.rflags, kFlagCF, src != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagOF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagPF, false);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_blsr(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = src & (src - 1ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & (1ull << (bit_width(size) - 1ull))) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagZF, result == 0);
  detail::set_flag(ctx.state.rflags, kFlagCF, src == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagOF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagPF, false);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_blsmsk(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = src ^ (src - 1ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & (1ull << (bit_width(size) - 1ull))) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  detail::set_flag(ctx.state.rflags, kFlagCF, src == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagOF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagPF, false);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_andn(std::uint8_t size, ExecutionContext& ctx) {
  bool src1_ok = false;
  const auto src1 = detail::read_operand(ctx, 1, size, &src1_ok);
  if (!src1_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool src2_ok = false;
  const auto src2 = detail::read_operand(ctx, 2, size, &src2_ok);
  if (!src2_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = (~src1) & src2;
  detail::set_logic_flags(ctx.state, result, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_bzhi(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool idx_ok = false;
  const auto raw_index = detail::read_operand(ctx, 2, size, &idx_ok);
  if (!idx_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = bit_width(size);
  const auto index = raw_index & 0xFFull;
  auto result = src;
  if (index == 0) {
    result = 0;
  } else if (index < bits) {
    result &= ((1ull << static_cast<unsigned>(index)) - 1ull);
  }
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & (1ull << (bits - 1ull))) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagZF, result == 0);
  detail::set_flag(ctx.state.rflags, kFlagCF, index >= bits);
  detail::set_flag(ctx.state.rflags, kFlagOF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagPF, false);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_bextr(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool control_ok = false;
  const auto control = detail::read_operand(ctx, 2, size, &control_ok);
  if (!control_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = bextr(src, control, size);
  detail::set_flag(ctx.state.rflags, kFlagZF, result == 0);
  detail::set_flag(ctx.state.rflags, kFlagCF, false);
  detail::set_flag(ctx.state.rflags, kFlagOF, false);
  detail::set_flag(ctx.state.rflags, kFlagSF, false);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagPF, false);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_mulx(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src2 = detail::read_operand(ctx, 2, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto src1 = detail::read_register(ctx.state, size == 4 ? iced_x86::Register::EDX : iced_x86::Register::RDX);
  const auto lo = mulx_low(src1, src2, size);
  const auto hi = mulx_high(src1, src2, size);
  // libLISA writes the low destination before the high destination.
  if (!detail::write_operand(ctx, 1, lo, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!detail::write_operand(ctx, 0, hi, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_pdep(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool mask_ok = false;
  const auto mask = detail::read_operand(ctx, 2, size, &mask_ok);
  if (!mask_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = pdep(src, mask, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_pext(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool mask_ok = false;
  const auto mask = detail::read_operand(ctx, 2, size, &mask_ok);
  if (!mask_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = pext(src, mask, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_rorx(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto count = ctx.instr.immediate8();
  const auto result = rotate_right(src, count, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_sarx(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool count_ok = false;
  const auto count = detail::read_operand(ctx, 2, size, &count_ok);
  if (!count_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = arithmetic_shift_right(src, count, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_shlx(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool count_ok = false;
  const auto count = detail::read_operand(ctx, 2, size, &count_ok);
  if (!count_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = shift_left(src, count, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_shrx(std::uint8_t size, ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, size, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  bool count_ok = false;
  const auto count = detail::read_operand(ctx, 2, size, &count_ok);
  if (!count_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto result = shift_right(src, count, size);
  if (!detail::write_operand(ctx, 0, result, size)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}

ExecutionResult handle_code_VEX_BEXTR_R32_RM32_R32(ExecutionContext& ctx) { return handle_code_bextr(4, ctx); }
ExecutionResult handle_code_VEX_BEXTR_R64_RM64_R64(ExecutionContext& ctx) { return handle_code_bextr(8, ctx); }
ExecutionResult handle_code_VEX_BLSI_R32_RM32(ExecutionContext& ctx) { return handle_code_blsi(4, ctx); }
ExecutionResult handle_code_VEX_BLSI_R64_RM64(ExecutionContext& ctx) { return handle_code_blsi(8, ctx); }
ExecutionResult handle_code_VEX_ANDN_R32_R32_RM32(ExecutionContext& ctx) { return handle_code_andn(4, ctx); }
ExecutionResult handle_code_VEX_ANDN_R64_R64_RM64(ExecutionContext& ctx) { return handle_code_andn(8, ctx); }
ExecutionResult handle_code_VEX_BLSMSK_R32_RM32(ExecutionContext& ctx) { return handle_code_blsmsk(4, ctx); }
ExecutionResult handle_code_VEX_BLSMSK_R64_RM64(ExecutionContext& ctx) { return handle_code_blsmsk(8, ctx); }
ExecutionResult handle_code_VEX_BLSR_R32_RM32(ExecutionContext& ctx) { return handle_code_blsr(4, ctx); }
ExecutionResult handle_code_VEX_BLSR_R64_RM64(ExecutionContext& ctx) { return handle_code_blsr(8, ctx); }
ExecutionResult handle_code_VEX_BZHI_R32_RM32_R32(ExecutionContext& ctx) { return handle_code_bzhi(4, ctx); }
ExecutionResult handle_code_VEX_BZHI_R64_RM64_R64(ExecutionContext& ctx) { return handle_code_bzhi(8, ctx); }
ExecutionResult handle_code_VEX_MULX_R32_R32_RM32(ExecutionContext& ctx) { return handle_code_mulx(4, ctx); }
ExecutionResult handle_code_VEX_MULX_R64_R64_RM64(ExecutionContext& ctx) { return handle_code_mulx(8, ctx); }
ExecutionResult handle_code_VEX_PDEP_R32_R32_RM32(ExecutionContext& ctx) { return handle_code_pdep(4, ctx); }
ExecutionResult handle_code_VEX_PDEP_R64_R64_RM64(ExecutionContext& ctx) { return handle_code_pdep(8, ctx); }
ExecutionResult handle_code_VEX_PEXT_R32_R32_RM32(ExecutionContext& ctx) { return handle_code_pext(4, ctx); }
ExecutionResult handle_code_VEX_PEXT_R64_R64_RM64(ExecutionContext& ctx) { return handle_code_pext(8, ctx); }
ExecutionResult handle_code_VEX_RORX_R32_RM32_IMM8(ExecutionContext& ctx) { return handle_code_rorx(4, ctx); }
ExecutionResult handle_code_VEX_RORX_R64_RM64_IMM8(ExecutionContext& ctx) { return handle_code_rorx(8, ctx); }
ExecutionResult handle_code_VEX_SARX_R32_RM32_R32(ExecutionContext& ctx) { return handle_code_sarx(4, ctx); }
ExecutionResult handle_code_VEX_SARX_R64_RM64_R64(ExecutionContext& ctx) { return handle_code_sarx(8, ctx); }
ExecutionResult handle_code_VEX_SHLX_R32_RM32_R32(ExecutionContext& ctx) { return handle_code_shlx(4, ctx); }
ExecutionResult handle_code_VEX_SHLX_R64_RM64_R64(ExecutionContext& ctx) { return handle_code_shlx(8, ctx); }
ExecutionResult handle_code_VEX_SHRX_R32_RM32_R32(ExecutionContext& ctx) { return handle_code_shrx(4, ctx); }
ExecutionResult handle_code_VEX_SHRX_R64_RM64_R64(ExecutionContext& ctx) { return handle_code_shrx(8, ctx); }

}  // namespace seven::handlers


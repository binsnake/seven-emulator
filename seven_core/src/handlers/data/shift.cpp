#include "seven/handler_helpers.hpp"

namespace seven::handlers {
namespace {
enum class ShiftKind {
  left,
  logical_right,
  arithmetic_right,
};

enum class ShiftCountSource {
  one,
  cl,
  imm8,
};

[[nodiscard]] constexpr std::uint64_t shift_count_mask(std::size_t width) noexcept {
  return width == 8 ? 63ull : 31ull;
}

[[nodiscard]] std::uint64_t read_shift_count(ExecutionContext& ctx, ShiftCountSource source, std::size_t width) {
  switch (source) {
    case ShiftCountSource::one:
      return 1ull;
    case ShiftCountSource::cl:
      return detail::read_register(ctx.state, iced_x86::Register::CL) & shift_count_mask(width);
    case ShiftCountSource::imm8:
      return ctx.instr.immediate8() & shift_count_mask(width);
  }

  return 0ull;
}

ExecutionResult shift_rm(ExecutionContext& ctx, std::size_t width, ShiftKind kind, ShiftCountSource source) {
  std::uint64_t value = 0;
  if (auto result = detail::read_operand_checked(ctx, 0, width, value); !result.ok()) {
    return result;
  }

  const auto count = read_shift_count(ctx, source, width);
  if (count == 0) {
    return {};
  }

  const auto bits = static_cast<unsigned>(width * 8);
  const auto mask = mask_for_width(width);
  const auto sign = sign_bit_for_width(width);
  const auto original = value & mask;
  const bool old_msb = (original & sign) != 0ull;
  const bool old_next_msb = (original & (sign >> 1)) != 0ull;

  std::uint64_t result_value = 0;
  bool cf = false;

  switch (kind) {
    case ShiftKind::left:
      if (count < bits) {
        result_value = (original << count) & mask;
        cf = ((original >> (bits - static_cast<unsigned>(count))) & 1ull) != 0ull;
      } else if (count == bits) {
        result_value = 0ull;
        cf = (original & 1ull) != 0ull;
      } else {
        result_value = 0ull;
        cf = false;
      }
      break;
    case ShiftKind::logical_right:
      if (count < bits) {
        result_value = original >> count;
        cf = ((original >> (count - 1ull)) & 1ull) != 0ull;
      } else if (count == bits) {
        result_value = 0ull;
        cf = old_msb;
      } else {
        result_value = 0ull;
        cf = false;
      }
      break;
    case ShiftKind::arithmetic_right:
      if (count < bits) {
        result_value = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(detail::sign_extend(original, width)) >> static_cast<unsigned>(count)) &
                       mask;
        cf = ((original >> (count - 1ull)) & 1ull) != 0ull;
      } else {
        result_value = old_msb ? mask : 0ull;
        cf = old_msb;
      }
      break;
  }

  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  detail::set_flag(ctx.state.rflags, kFlagZF, result_value == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result_value & sign) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(static_cast<std::uint8_t>(result_value & 0xFFull)));

  // libLISA captures the concrete CPU behavior for non-zero counts here.
  switch (kind) {
    case ShiftKind::left:
      detail::set_flag(ctx.state.rflags, kFlagOF, old_msb ^ old_next_msb);
      break;
    case ShiftKind::logical_right:
      detail::set_flag(ctx.state.rflags, kFlagOF, old_msb);
      break;
    case ShiftKind::arithmetic_right:
      detail::set_flag(ctx.state.rflags, kFlagOF, false);
      break;
  }

  return detail::write_operand_checked(ctx, 0, result_value, width);
}

}  // namespace

#define SEVEN_DEFINE_SHIFT_HANDLER(code, width, kind, source) \
ExecutionResult handle_code_##code(ExecutionContext& ctx) { \
  return shift_rm(ctx, width, ShiftKind::kind, ShiftCountSource::source); \
}

SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM8_1, 1, left, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM8_CL, 1, left, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM8_IMM8, 1, left, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM16_1, 2, left, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM16_CL, 2, left, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM16_IMM8, 2, left, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM32_1, 4, left, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM32_CL, 4, left, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM32_IMM8, 4, left, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM64_1, 8, left, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM64_CL, 8, left, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHL_RM64_IMM8, 8, left, imm8)

SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM8_1, 1, logical_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM8_CL, 1, logical_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM8_IMM8, 1, logical_right, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM16_1, 2, logical_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM16_CL, 2, logical_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM16_IMM8, 2, logical_right, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM32_1, 4, logical_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM32_CL, 4, logical_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM32_IMM8, 4, logical_right, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM64_1, 8, logical_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM64_CL, 8, logical_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SHR_RM64_IMM8, 8, logical_right, imm8)

SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM8_1, 1, arithmetic_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM8_CL, 1, arithmetic_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM8_IMM8, 1, arithmetic_right, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM16_1, 2, arithmetic_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM16_CL, 2, arithmetic_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM16_IMM8, 2, arithmetic_right, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM32_1, 4, arithmetic_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM32_CL, 4, arithmetic_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM32_IMM8, 4, arithmetic_right, imm8)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM64_1, 8, arithmetic_right, one)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM64_CL, 8, arithmetic_right, cl)
SEVEN_DEFINE_SHIFT_HANDLER(SAR_RM64_IMM8, 8, arithmetic_right, imm8)

#undef SEVEN_DEFINE_SHIFT_HANDLER


namespace {

[[nodiscard]] constexpr std::uint64_t width_mask(std::size_t width) noexcept {
  return width >= 8 ? ~0ull : ((1ull << (width * 8)) - 1ull);
}

[[nodiscard]] constexpr std::uint64_t msb_mask(std::size_t width) noexcept {
  return 1ull << ((width * 8) - 1);
}

ExecutionResult rotate_left(ExecutionContext& ctx, std::size_t width, std::uint64_t count) {
  bool ok = false;
  const auto value = detail::read_operand(ctx, 0, width, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = static_cast<unsigned>(width * 8);
  const auto shift = static_cast<unsigned>(count & (bits - 1));
  if (shift == 0) {
    return {};
  }
  const auto m = width_mask(width);
  const auto result = ((value << shift) | (value >> (bits - shift))) & m;
  const auto cf = (result & 1ull) != 0ull;
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  if (shift == 1) {
    const auto of = ((result & msb_mask(width)) != 0ull) ^ cf;
    detail::set_flag(ctx.state.rflags, kFlagOF, of);
  }
  if (!detail::write_operand(ctx, 0, result, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult rotate_right(ExecutionContext& ctx, std::size_t width, std::uint64_t count) {
  bool ok = false;
  const auto value = detail::read_operand(ctx, 0, width, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = static_cast<unsigned>(width * 8);
  const auto shift = static_cast<unsigned>(count & (bits - 1));
  if (shift == 0) {
    return {};
  }
  const auto m = width_mask(width);
  const auto result = ((value >> shift) | (value << (bits - shift))) & m;
  const auto cf = ((result & msb_mask(width)) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  if (shift == 1) {
    const auto msb = (result & msb_mask(width)) != 0ull;
    const auto second_msb = (result & (msb_mask(width) >> 1)) != 0ull;
    detail::set_flag(ctx.state.rflags, kFlagOF, msb ^ second_msb);
  }
  if (!detail::write_operand(ctx, 0, result, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult rotate_carry_left(ExecutionContext& ctx, std::size_t width, std::uint64_t count) {
  bool ok = false;
  auto value = detail::read_operand(ctx, 0, width, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = static_cast<unsigned>(width * 8);
  const auto raw = static_cast<unsigned>(count & (bits == 64 ? 0x3Fu : 0x1Fu));
  const auto shift = raw % (bits + 1);
  if (shift == 0) {
    return {};
  }

  const auto m = width_mask(width);
  bool cf = (ctx.state.rflags & kFlagCF) != 0;
  value &= m;
  const auto original = value;
  for (unsigned i = 0; i < shift; ++i) {
    const auto new_cf = ((value & msb_mask(width)) != 0ull);
    value = ((value << 1) & m) | (cf ? 1ull : 0ull);
    cf = new_cf;
  }
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  const auto of = ((original & msb_mask(width)) != 0ull) ^ ((original & (msb_mask(width) >> 1)) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagOF, of);
  if (!detail::write_operand(ctx, 0, value, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult rotate_carry_right(ExecutionContext& ctx, std::size_t width, std::uint64_t count) {
  bool ok = false;
  auto value = detail::read_operand(ctx, 0, width, &ok);
  if (!ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = static_cast<unsigned>(width * 8);
  const auto raw = static_cast<unsigned>(count & (bits == 64 ? 0x3Fu : 0x1Fu));
  const auto shift = raw % (bits + 1);
  if (shift == 0) {
    return {};
  }

  const auto m = width_mask(width);
  const bool old_cf = (ctx.state.rflags & kFlagCF) != 0;
  bool cf = old_cf;
  value &= m;
  const auto original = value;
  for (unsigned i = 0; i < shift; ++i) {
    const auto new_cf = (value & 1ull) != 0ull;
    value = (value >> 1) | (cf ? msb_mask(width) : 0ull);
    value &= m;
    cf = new_cf;
  }
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  const auto of = ((original & msb_mask(width)) != 0ull) ^ old_cf;
  detail::set_flag(ctx.state.rflags, kFlagOF, of);
  if (!detail::write_operand(ctx, 0, value, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult shld(ExecutionContext& ctx, std::size_t width, std::uint64_t count) {
  bool dst_ok = false;
  const auto dst = detail::read_operand(ctx, 0, width, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = static_cast<unsigned>(width * 8);
  const auto raw = static_cast<unsigned>(count & (bits == 64 ? 0x3Fu : 0x1Fu));
  const auto shift = raw % bits;
  if (shift == 0) {
    return {};
  }

  const auto src = detail::read_register(ctx.state, ctx.instr.op_register(1)) & width_mask(width);
  const auto result = ((dst << shift) | (src >> (bits - shift))) & width_mask(width);
  const auto cf = ((dst >> (bits - shift)) & 1ull) != 0ull;
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  if (shift == 1) {
    const auto of = ((result & msb_mask(width)) != 0ull) ^ cf;
    detail::set_flag(ctx.state.rflags, kFlagOF, of);
  } else {
    detail::set_flag(ctx.state.rflags, kFlagOF, false);
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, (result & width_mask(width)) == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & msb_mask(width)) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(static_cast<std::uint8_t>(result & 0xFFull)));
  if (!detail::write_operand(ctx, 0, result, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult shrd(ExecutionContext& ctx, std::size_t width, std::uint64_t count) {
  bool dst_ok = false;
  const auto dst = detail::read_operand(ctx, 0, width, &dst_ok);
  if (!dst_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto bits = static_cast<unsigned>(width * 8);
  const auto raw = static_cast<unsigned>(count & (bits == 64 ? 0x3Fu : 0x1Fu));
  const auto shift = raw % bits;
  if (shift == 0) {
    return {};
  }

  const auto src = detail::read_register(ctx.state, ctx.instr.op_register(1)) & width_mask(width);
  const auto result = ((dst >> shift) | (src << (bits - shift))) & width_mask(width);
  const auto cf = ((dst >> (shift - 1)) & 1ull) != 0ull;
  detail::set_flag(ctx.state.rflags, kFlagCF, cf);
  detail::set_flag(ctx.state.rflags, kFlagAF, false);
  if (shift == 1) {
    const auto of = ((dst & msb_mask(width)) != 0ull) ^ ((result & msb_mask(width)) != 0ull);
    detail::set_flag(ctx.state.rflags, kFlagOF, of);
  } else {
    detail::set_flag(ctx.state.rflags, kFlagOF, false);
  }
  detail::set_flag(ctx.state.rflags, kFlagZF, (result & width_mask(width)) == 0ull);
  detail::set_flag(ctx.state.rflags, kFlagSF, (result & msb_mask(width)) != 0ull);
  detail::set_flag(ctx.state.rflags, kFlagPF, detail::even_parity(static_cast<std::uint8_t>(result & 0xFFull)));
  if (!detail::write_operand(ctx, 0, result, width)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace

ExecutionResult handle_code_SAL_RM8_1(ExecutionContext& ctx) { return handle_code_SHL_RM8_1(ctx); }
ExecutionResult handle_code_SAL_RM16_1(ExecutionContext& ctx) { return handle_code_SHL_RM16_1(ctx); }
ExecutionResult handle_code_SAL_RM32_1(ExecutionContext& ctx) { return handle_code_SHL_RM32_1(ctx); }
ExecutionResult handle_code_SAL_RM64_1(ExecutionContext& ctx) { return handle_code_SHL_RM64_1(ctx); }
ExecutionResult handle_code_SAL_RM8_CL(ExecutionContext& ctx) { return handle_code_SHL_RM8_CL(ctx); }
ExecutionResult handle_code_SAL_RM16_CL(ExecutionContext& ctx) { return handle_code_SHL_RM16_CL(ctx); }
ExecutionResult handle_code_SAL_RM32_CL(ExecutionContext& ctx) { return handle_code_SHL_RM32_CL(ctx); }
ExecutionResult handle_code_SAL_RM64_CL(ExecutionContext& ctx) { return handle_code_SHL_RM64_CL(ctx); }
ExecutionResult handle_code_SAL_RM8_IMM8(ExecutionContext& ctx) { return handle_code_SHL_RM8_IMM8(ctx); }
ExecutionResult handle_code_SAL_RM16_IMM8(ExecutionContext& ctx) { return handle_code_SHL_RM16_IMM8(ctx); }
ExecutionResult handle_code_SAL_RM32_IMM8(ExecutionContext& ctx) { return handle_code_SHL_RM32_IMM8(ctx); }
ExecutionResult handle_code_SAL_RM64_IMM8(ExecutionContext& ctx) { return handle_code_SHL_RM64_IMM8(ctx); }

ExecutionResult handle_code_ROL_RM8_1(ExecutionContext& ctx) { return rotate_left(ctx, 1, 1); }
ExecutionResult handle_code_ROL_RM16_1(ExecutionContext& ctx) { return rotate_left(ctx, 2, 1); }
ExecutionResult handle_code_ROL_RM32_1(ExecutionContext& ctx) { return rotate_left(ctx, 4, 1); }
ExecutionResult handle_code_ROL_RM64_1(ExecutionContext& ctx) { return rotate_left(ctx, 8, 1); }
ExecutionResult handle_code_ROL_RM8_CL(ExecutionContext& ctx) { return rotate_left(ctx, 1, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROL_RM16_CL(ExecutionContext& ctx) { return rotate_left(ctx, 2, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROL_RM32_CL(ExecutionContext& ctx) { return rotate_left(ctx, 4, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROL_RM64_CL(ExecutionContext& ctx) { return rotate_left(ctx, 8, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROL_RM8_IMM8(ExecutionContext& ctx) { return rotate_left(ctx, 1, ctx.instr.immediate8()); }
ExecutionResult handle_code_ROL_RM16_IMM8(ExecutionContext& ctx) { return rotate_left(ctx, 2, ctx.instr.immediate8()); }
ExecutionResult handle_code_ROL_RM32_IMM8(ExecutionContext& ctx) { return rotate_left(ctx, 4, ctx.instr.immediate8()); }
ExecutionResult handle_code_ROL_RM64_IMM8(ExecutionContext& ctx) { return rotate_left(ctx, 8, ctx.instr.immediate8()); }

ExecutionResult handle_code_ROR_RM8_1(ExecutionContext& ctx) { return rotate_right(ctx, 1, 1); }
ExecutionResult handle_code_ROR_RM16_1(ExecutionContext& ctx) { return rotate_right(ctx, 2, 1); }
ExecutionResult handle_code_ROR_RM32_1(ExecutionContext& ctx) { return rotate_right(ctx, 4, 1); }
ExecutionResult handle_code_ROR_RM64_1(ExecutionContext& ctx) { return rotate_right(ctx, 8, 1); }
ExecutionResult handle_code_ROR_RM8_CL(ExecutionContext& ctx) { return rotate_right(ctx, 1, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROR_RM16_CL(ExecutionContext& ctx) { return rotate_right(ctx, 2, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROR_RM32_CL(ExecutionContext& ctx) { return rotate_right(ctx, 4, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROR_RM64_CL(ExecutionContext& ctx) { return rotate_right(ctx, 8, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_ROR_RM8_IMM8(ExecutionContext& ctx) { return rotate_right(ctx, 1, ctx.instr.immediate8()); }
ExecutionResult handle_code_ROR_RM16_IMM8(ExecutionContext& ctx) { return rotate_right(ctx, 2, ctx.instr.immediate8()); }
ExecutionResult handle_code_ROR_RM32_IMM8(ExecutionContext& ctx) { return rotate_right(ctx, 4, ctx.instr.immediate8()); }
ExecutionResult handle_code_ROR_RM64_IMM8(ExecutionContext& ctx) { return rotate_right(ctx, 8, ctx.instr.immediate8()); }

ExecutionResult handle_code_RCL_RM8_1(ExecutionContext& ctx) { return rotate_carry_left(ctx, 1, 1); }
ExecutionResult handle_code_RCL_RM16_1(ExecutionContext& ctx) { return rotate_carry_left(ctx, 2, 1); }
ExecutionResult handle_code_RCL_RM32_1(ExecutionContext& ctx) { return rotate_carry_left(ctx, 4, 1); }
ExecutionResult handle_code_RCL_RM64_1(ExecutionContext& ctx) { return rotate_carry_left(ctx, 8, 1); }
ExecutionResult handle_code_RCL_RM8_CL(ExecutionContext& ctx) { return rotate_carry_left(ctx, 1, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCL_RM16_CL(ExecutionContext& ctx) { return rotate_carry_left(ctx, 2, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCL_RM32_CL(ExecutionContext& ctx) { return rotate_carry_left(ctx, 4, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCL_RM64_CL(ExecutionContext& ctx) { return rotate_carry_left(ctx, 8, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCL_RM8_IMM8(ExecutionContext& ctx) { return rotate_carry_left(ctx, 1, ctx.instr.immediate8()); }
ExecutionResult handle_code_RCL_RM16_IMM8(ExecutionContext& ctx) { return rotate_carry_left(ctx, 2, ctx.instr.immediate8()); }
ExecutionResult handle_code_RCL_RM32_IMM8(ExecutionContext& ctx) { return rotate_carry_left(ctx, 4, ctx.instr.immediate8()); }
ExecutionResult handle_code_RCL_RM64_IMM8(ExecutionContext& ctx) { return rotate_carry_left(ctx, 8, ctx.instr.immediate8()); }

ExecutionResult handle_code_RCR_RM8_1(ExecutionContext& ctx) { return rotate_carry_right(ctx, 1, 1); }
ExecutionResult handle_code_RCR_RM16_1(ExecutionContext& ctx) { return rotate_carry_right(ctx, 2, 1); }
ExecutionResult handle_code_RCR_RM32_1(ExecutionContext& ctx) { return rotate_carry_right(ctx, 4, 1); }
ExecutionResult handle_code_RCR_RM64_1(ExecutionContext& ctx) { return rotate_carry_right(ctx, 8, 1); }
ExecutionResult handle_code_RCR_RM8_CL(ExecutionContext& ctx) { return rotate_carry_right(ctx, 1, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCR_RM16_CL(ExecutionContext& ctx) { return rotate_carry_right(ctx, 2, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCR_RM32_CL(ExecutionContext& ctx) { return rotate_carry_right(ctx, 4, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCR_RM64_CL(ExecutionContext& ctx) { return rotate_carry_right(ctx, 8, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_RCR_RM8_IMM8(ExecutionContext& ctx) { return rotate_carry_right(ctx, 1, ctx.instr.immediate8()); }
ExecutionResult handle_code_RCR_RM16_IMM8(ExecutionContext& ctx) { return rotate_carry_right(ctx, 2, ctx.instr.immediate8()); }
ExecutionResult handle_code_RCR_RM32_IMM8(ExecutionContext& ctx) { return rotate_carry_right(ctx, 4, ctx.instr.immediate8()); }
ExecutionResult handle_code_RCR_RM64_IMM8(ExecutionContext& ctx) { return rotate_carry_right(ctx, 8, ctx.instr.immediate8()); }

ExecutionResult handle_code_SHLD_RM16_R16_IMM8(ExecutionContext& ctx) { return shld(ctx, 2, ctx.instr.immediate8()); }
ExecutionResult handle_code_SHLD_RM32_R32_IMM8(ExecutionContext& ctx) { return shld(ctx, 4, ctx.instr.immediate8()); }
ExecutionResult handle_code_SHLD_RM64_R64_IMM8(ExecutionContext& ctx) { return shld(ctx, 8, ctx.instr.immediate8()); }
ExecutionResult handle_code_SHLD_RM16_R16_CL(ExecutionContext& ctx) { return shld(ctx, 2, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_SHLD_RM32_R32_CL(ExecutionContext& ctx) { return shld(ctx, 4, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_SHLD_RM64_R64_CL(ExecutionContext& ctx) { return shld(ctx, 8, detail::read_register(ctx.state, iced_x86::Register::CL)); }

ExecutionResult handle_code_SHRD_RM16_R16_IMM8(ExecutionContext& ctx) { return shrd(ctx, 2, ctx.instr.immediate8()); }
ExecutionResult handle_code_SHRD_RM32_R32_IMM8(ExecutionContext& ctx) { return shrd(ctx, 4, ctx.instr.immediate8()); }
ExecutionResult handle_code_SHRD_RM64_R64_IMM8(ExecutionContext& ctx) { return shrd(ctx, 8, ctx.instr.immediate8()); }
ExecutionResult handle_code_SHRD_RM16_R16_CL(ExecutionContext& ctx) { return shrd(ctx, 2, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_SHRD_RM32_R32_CL(ExecutionContext& ctx) { return shrd(ctx, 4, detail::read_register(ctx.state, iced_x86::Register::CL)); }
ExecutionResult handle_code_SHRD_RM64_R64_CL(ExecutionContext& ctx) { return shrd(ctx, 8, detail::read_register(ctx.state, iced_x86::Register::CL)); }

}  // namespace seven::handlers



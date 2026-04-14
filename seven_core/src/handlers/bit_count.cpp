#include "seven/handler_helpers.hpp"

#include <bit>

namespace seven::handlers {
namespace {

ExecutionResult read_rm64(ExecutionContext& ctx, std::uint64_t& value) {
  if (auto result = detail::read_operand_checked(ctx, 1, 8, value); !result.ok()) {
    return result;
  }
  return {};
}

void clear_count_flags(CpuState& state) {
  detail::set_flag(state.rflags, kFlagCF, false);
  detail::set_flag(state.rflags, kFlagOF, false);
  detail::set_flag(state.rflags, kFlagSF, false);
  detail::set_flag(state.rflags, kFlagAF, false);
  detail::set_flag(state.rflags, kFlagPF, false);
}

iced_x86::Register crc32_dest_register(iced_x86::Register reg, std::size_t width) {
  switch (reg) {
    case iced_x86::Register::RAX: case iced_x86::Register::EAX: case iced_x86::Register::AX: case iced_x86::Register::AL:
      return width == 8 ? iced_x86::Register::RAX : iced_x86::Register::EAX;
    case iced_x86::Register::RCX: case iced_x86::Register::ECX: case iced_x86::Register::CX: case iced_x86::Register::CL:
      return width == 8 ? iced_x86::Register::RCX : iced_x86::Register::ECX;
    case iced_x86::Register::RDX: case iced_x86::Register::EDX: case iced_x86::Register::DX: case iced_x86::Register::DL:
      return width == 8 ? iced_x86::Register::RDX : iced_x86::Register::EDX;
    case iced_x86::Register::RBX: case iced_x86::Register::EBX: case iced_x86::Register::BX: case iced_x86::Register::BL:
      return width == 8 ? iced_x86::Register::RBX : iced_x86::Register::EBX;
    case iced_x86::Register::RSP: case iced_x86::Register::ESP: case iced_x86::Register::SP: case iced_x86::Register::SPL:
      return width == 8 ? iced_x86::Register::RSP : iced_x86::Register::ESP;
    case iced_x86::Register::RBP: case iced_x86::Register::EBP: case iced_x86::Register::BP: case iced_x86::Register::BPL:
      return width == 8 ? iced_x86::Register::RBP : iced_x86::Register::EBP;
    case iced_x86::Register::RSI: case iced_x86::Register::ESI: case iced_x86::Register::SI: case iced_x86::Register::SIL:
      return width == 8 ? iced_x86::Register::RSI : iced_x86::Register::ESI;
    case iced_x86::Register::RDI: case iced_x86::Register::EDI: case iced_x86::Register::DI: case iced_x86::Register::DIL:
      return width == 8 ? iced_x86::Register::RDI : iced_x86::Register::EDI;
    case iced_x86::Register::R8: case iced_x86::Register::R8_D: case iced_x86::Register::R8_W: case iced_x86::Register::R8_L:
      return width == 8 ? iced_x86::Register::R8 : iced_x86::Register::R8_D;
    case iced_x86::Register::R9: case iced_x86::Register::R9_D: case iced_x86::Register::R9_W: case iced_x86::Register::R9_L:
      return width == 8 ? iced_x86::Register::R9 : iced_x86::Register::R9_D;
    case iced_x86::Register::R10: case iced_x86::Register::R10_D: case iced_x86::Register::R10_W: case iced_x86::Register::R10_L:
      return width == 8 ? iced_x86::Register::R10 : iced_x86::Register::R10_D;
    case iced_x86::Register::R11: case iced_x86::Register::R11_D: case iced_x86::Register::R11_W: case iced_x86::Register::R11_L:
      return width == 8 ? iced_x86::Register::R11 : iced_x86::Register::R11_D;
    case iced_x86::Register::R12: case iced_x86::Register::R12_D: case iced_x86::Register::R12_W: case iced_x86::Register::R12_L:
      return width == 8 ? iced_x86::Register::R12 : iced_x86::Register::R12_D;
    case iced_x86::Register::R13: case iced_x86::Register::R13_D: case iced_x86::Register::R13_W: case iced_x86::Register::R13_L:
      return width == 8 ? iced_x86::Register::R13 : iced_x86::Register::R13_D;
    case iced_x86::Register::R14: case iced_x86::Register::R14_D: case iced_x86::Register::R14_W: case iced_x86::Register::R14_L:
      return width == 8 ? iced_x86::Register::R14 : iced_x86::Register::R14_D;
    case iced_x86::Register::R15: case iced_x86::Register::R15_D: case iced_x86::Register::R15_W: case iced_x86::Register::R15_L:
      return width == 8 ? iced_x86::Register::R15 : iced_x86::Register::R15_D;
    default: return reg;
  }
}

std::uint32_t crc32c_update(std::uint32_t crc, std::uint64_t value, std::size_t width) {
  constexpr std::uint32_t kPoly = 0x82F63B78u;
  for (std::size_t i = 0; i < width; ++i) {
    crc ^= static_cast<std::uint8_t>((value >> (i * 8)) & 0xFFu);
    for (unsigned bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1u) != 0u ? kPoly : 0u);
    }
  }
  return crc;
}

ExecutionResult crc32_impl(ExecutionContext& ctx) {
  const auto width = detail::operand_width(ctx.instr, 1);
  const auto dest_width = ctx.instr.code() == iced_x86::Code::CRC32_R64_RM8 || ctx.instr.code() == iced_x86::Code::CRC32_R64_RM64 ? 8u : 4u;
  const auto dest_reg = crc32_dest_register(ctx.instr.op_register(0), dest_width);
  bool ok = false;
  const auto src = detail::read_operand(ctx, 1, width, &ok);
  if (!ok) return detail::memory_fault(ctx, detail::memory_address(ctx));
  const auto seed = static_cast<std::uint32_t>(detail::read_register(ctx.state, dest_reg));
  const auto crc = static_cast<std::uint64_t>(crc32c_update(seed, src, width));
  detail::write_register(ctx.state, dest_reg, crc);
  return {};
}

}  // namespace

ExecutionResult handle_code_POPCNT_R64_RM64(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = read_rm64(ctx, value); !result.ok()) {
    return result;
  }
  const auto count = static_cast<std::uint64_t>(std::popcount(value));
  detail::write_register(ctx.state, ctx.instr.op_register(0), count, 8);
  clear_count_flags(ctx.state);
  detail::set_flag(ctx.state.rflags, kFlagZF, value == 0);
  return {};
}

ExecutionResult handle_code_TZCNT_R64_RM64(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = read_rm64(ctx, value); !result.ok()) {
    return result;
  }
  const auto count = static_cast<std::uint64_t>(std::countr_zero(value));
  detail::write_register(ctx.state, ctx.instr.op_register(0), count, 8);
  clear_count_flags(ctx.state);
  detail::set_flag(ctx.state.rflags, kFlagCF, value == 0);
  detail::set_flag(ctx.state.rflags, kFlagZF, count == 0);
  return {};
}

ExecutionResult handle_code_LZCNT_R64_RM64(ExecutionContext& ctx) {
  std::uint64_t value = 0;
  if (auto result = read_rm64(ctx, value); !result.ok()) {
    return result;
  }
  const auto count = static_cast<std::uint64_t>(std::countl_zero(value));
  detail::write_register(ctx.state, ctx.instr.op_register(0), count, 8);
  clear_count_flags(ctx.state);
  detail::set_flag(ctx.state.rflags, kFlagCF, value == 0);
  detail::set_flag(ctx.state.rflags, kFlagZF, count == 0);
  return {};
}
ExecutionResult handle_code_CRC32_R32_RM8(ExecutionContext& ctx) { return crc32_impl(ctx); }
ExecutionResult handle_code_CRC32_R64_RM8(ExecutionContext& ctx) { return crc32_impl(ctx); }
ExecutionResult handle_code_CRC32_R32_RM16(ExecutionContext& ctx) { return crc32_impl(ctx); }
ExecutionResult handle_code_CRC32_R32_RM32(ExecutionContext& ctx) { return crc32_impl(ctx); }
ExecutionResult handle_code_CRC32_R64_RM64(ExecutionContext& ctx) { return crc32_impl(ctx); }


}  // namespace seven::handlers

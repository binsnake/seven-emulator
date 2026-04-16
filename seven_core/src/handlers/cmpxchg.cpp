#include <array>
#include <cstddef>

#include "seven/handler_helpers.hpp"

namespace {
[[nodiscard]] bool trace_cmpxchg16b_enabled() noexcept {
  static const bool enabled = [] {
    if (const char* v = std::getenv("SEVEN_TRACE_CMPXCHG16B")) {
      return v[0] != '\0' && v[0] != '0';
    }
    return false;
  }();
  return enabled;
}
}


namespace seven::handlers {

namespace {

inline void set_cmpxchg_flags(CpuState& state, std::uint64_t acc, std::uint64_t dst, std::size_t width, bool success) {
  const auto mask = mask_for_width(width);
  const auto ua = acc & mask;
  const auto ub = dst & mask;
  const auto res = (ua - ub) & mask;
  const auto sa = static_cast<std::int64_t>(detail::sign_extend(ua, width));
  const auto sb = static_cast<std::int64_t>(detail::sign_extend(ub, width));
  const auto sres = static_cast<std::int64_t>(detail::sign_extend(res, width));

  detail::set_flag(state.rflags, kFlagCF, ua < ub);
  detail::set_flag(state.rflags, kFlagZF, success);
  detail::set_flag(state.rflags, kFlagPF, detail::even_parity(static_cast<std::uint8_t>(res & 0xFFull)));
  detail::set_flag(state.rflags, kFlagAF, ((ua ^ ub ^ res) & 0x10ull) != 0ull);
  detail::set_flag(state.rflags, kFlagSF, sres < 0);
  detail::set_flag(state.rflags, kFlagOF, (sa >= 0 && sb < 0 && sres < 0) || (sa < 0 && sb >= 0 && sres >= 0));
}

}  // namespace

ExecutionResult handle_code_CMPXCHG_RM8_R8(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 1, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::AL);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 1)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    set_cmpxchg_flags(ctx.state, accumulator, lhs, 1, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::AL, lhs, 1);
  set_cmpxchg_flags(ctx.state, accumulator, lhs, 1, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG_RM16_R16(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 2, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::AX);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 2)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    set_cmpxchg_flags(ctx.state, accumulator, lhs, 2, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::AX, lhs, 2);
  set_cmpxchg_flags(ctx.state, accumulator, lhs, 2, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG_RM32_R32(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 4, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::EAX);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 4)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    set_cmpxchg_flags(ctx.state, accumulator, lhs, 4, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::EAX, lhs, 4);
  set_cmpxchg_flags(ctx.state, accumulator, lhs, 4, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG_RM64_R64(ExecutionContext& ctx) {
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  const auto rhs = detail::read_register(ctx.state, ctx.instr.op_register(1));
  const auto accumulator = detail::read_register(ctx.state, iced_x86::Register::RAX);
  if (lhs == accumulator) {
    if (!detail::write_operand(ctx, 0, rhs, 8)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
    }
    set_cmpxchg_flags(ctx.state, accumulator, lhs, 8, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::RAX, lhs, 8);
  set_cmpxchg_flags(ctx.state, accumulator, lhs, 8, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG8B_M64(ExecutionContext& ctx) {
  const auto address = detail::memory_address(ctx);
  if ((address & 0x7ull) != 0ull) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, address, 0}, ctx.instr.code()};
  }
  bool lhs_ok = false;
  const auto lhs = detail::read_operand(ctx, 0, 8, &lhs_ok);
  if (!lhs_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }
  const auto desired = (detail::read_register(ctx.state, iced_x86::Register::EDX) << 32) |
                      (detail::read_register(ctx.state, iced_x86::Register::EAX) & 0xFFFFFFFFull);
  if (lhs == desired) {
    const auto source = (detail::read_register(ctx.state, iced_x86::Register::ECX) << 32) |
                        (detail::read_register(ctx.state, iced_x86::Register::EBX) & 0xFFFFFFFFull);
    if (!detail::write_operand(ctx, 0, source, 8)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
    }
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::EAX, lhs & 0xFFFFFFFFull, 4);
  detail::write_register(ctx.state, iced_x86::Register::EDX, (lhs >> 32) & 0xFFFFFFFFull, 4);
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  return {};
}

ExecutionResult handle_code_CMPXCHG16B_M128(ExecutionContext& ctx) {
  const auto address = detail::memory_address(ctx);
  if ((address & 0xFULL) != 0ull) {
    return {StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, address, 0}, ctx.instr.code()};
  }
  std::array<std::uint8_t, 16> lhs_bytes{};
  if (!ctx.memory.read(address, lhs_bytes.data(), 16)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
  }

  seven::SimdUint lhs = 0;
  for (std::size_t i = 0; i < lhs_bytes.size(); ++i) {
    lhs |= (seven::SimdUint(lhs_bytes[i]) << (8 * i));
  }
  if (trace_cmpxchg16b_enabled()) {
    std::fprintf(stderr,
                 "[seven-cmpxchg16b] rip=0x%llx addr=0x%llx mem_lo=0x%llx mem_hi=0x%llx rax=0x%llx rdx=0x%llx rbx=0x%llx rcx=0x%llx\n",
                 static_cast<unsigned long long>(ctx.state.rip),
                 static_cast<unsigned long long>(address),
                 static_cast<unsigned long long>(lhs),
                 static_cast<unsigned long long>(lhs >> 64),
                 static_cast<unsigned long long>(detail::read_register(ctx.state, iced_x86::Register::RAX)),
                 static_cast<unsigned long long>(detail::read_register(ctx.state, iced_x86::Register::RDX)),
                 static_cast<unsigned long long>(detail::read_register(ctx.state, iced_x86::Register::RBX)),
                 static_cast<unsigned long long>(detail::read_register(ctx.state, iced_x86::Register::RCX)));
  }
  const auto desired = (seven::SimdUint(detail::read_register(ctx.state, iced_x86::Register::RDX)) << 64) |
                      seven::SimdUint(detail::read_register(ctx.state, iced_x86::Register::RAX));
  if (lhs == desired) {
    seven::SimdUint source = (seven::SimdUint(detail::read_register(ctx.state, iced_x86::Register::RCX)) << 64) |
                       seven::SimdUint(detail::read_register(ctx.state, iced_x86::Register::RBX));
    std::array<std::uint8_t, 16> source_bytes{};
    for (std::size_t i = 0; i < source_bytes.size(); ++i) {
      source_bytes[i] = static_cast<std::uint8_t>(source >> (8 * i));
    }
    if (!ctx.memory.write(address, source_bytes.data(), 16)) {
      return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
    }
    detail::set_flag(ctx.state.rflags, kFlagZF, true);
    return {};
  }
  detail::write_register(ctx.state, iced_x86::Register::RAX, static_cast<std::uint64_t>(lhs), 8);
  detail::write_register(ctx.state, iced_x86::Register::RDX, static_cast<std::uint64_t>(lhs >> 64), 8);
  detail::set_flag(ctx.state.rflags, kFlagZF, false);
  return {};
}

}  // namespace seven::handlers



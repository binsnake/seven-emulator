#include "seven/executor.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <stdexcept>
#include <string>

#include <iced_x86/code.hpp>
#include <iced_x86/decoder.hpp>

namespace seven {
namespace handlers {
ExecutionResult handle_code_VEX_ANDN_R32_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_ANDN_R64_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BEXTR_R32_RM32_R32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BEXTR_R64_RM64_R64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BLSI_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BLSI_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BLSMSK_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BLSMSK_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BLSR_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BLSR_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BZHI_R32_RM32_R32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_BZHI_R64_RM64_R64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_MULX_R32_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_MULX_R64_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_PDEP_R32_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_PDEP_R64_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_PEXT_R32_R32_RM32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_PEXT_R64_R64_RM64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_RORX_R32_RM32_IMM8(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_RORX_R64_RM64_IMM8(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_SARX_R32_RM32_R32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_SARX_R64_RM64_R64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_SHLX_R32_RM32_R32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_SHLX_R64_RM64_R64(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_SHRX_R32_RM32_R32(ExecutionContext& ctx);
ExecutionResult handle_code_VEX_SHRX_R64_RM64_R64(ExecutionContext& ctx);
}  // namespace handlers

Executor::Executor() = default;

constexpr std::size_t Executor::stop_reason_to_index(StopReason reason) noexcept {
  return static_cast<std::size_t>(reason);
}

void Executor::reset_stats() {
  total_steps_ = 0;
  total_retired_ = 0;
  std::fill(code_execution_counts_.begin(), code_execution_counts_.end(), 0);
  std::fill(stop_reason_counts_.begin(), stop_reason_counts_.end(), 0);
}

const std::vector<std::uint64_t>& Executor::code_execution_counts() const noexcept {
  return code_execution_counts_;
}

const std::vector<std::uint64_t>& Executor::stop_reason_counts() const noexcept {
  return stop_reason_counts_;
}

std::uint64_t Executor::total_steps() const noexcept {
  return total_steps_;
}

std::uint64_t Executor::total_retired() const noexcept {
  return total_retired_;
}

ExecutionResult Executor::step(CpuState& state, Memory& memory) {
  ++total_steps_;
  std::array<std::uint8_t, iced_x86::IcedConstants::_MAX_INSTRUCTION_LENGTH> bytes{};
  if (!memory.read(state.rip, bytes.data(), bytes.size())) {
    ++stop_reason_counts_[stop_reason_to_index(StopReason::page_fault)];
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, state.rip, 0}, std::nullopt};
  }

  iced_x86::Decoder decoder(64, std::span<const std::uint8_t>(bytes.data(), bytes.size()), state.rip);
  const auto decoded = decoder.decode();
  if (!decoded.has_value()) {
    ++stop_reason_counts_[stop_reason_to_index(StopReason::decode_error)];
    return {StopReason::decode_error, 0, ExceptionInfo{StopReason::decode_error, state.rip, 0}, std::nullopt};
  }

  const auto& instr = decoded.value();
  if (instr.code() == iced_x86::Code::INVALID) {
    ++stop_reason_counts_[stop_reason_to_index(StopReason::invalid_opcode)];
    return {StopReason::invalid_opcode, 0, ExceptionInfo{StopReason::invalid_opcode, state.rip, 0}, instr.code()};
  }

  ExecutionContext ctx{state, memory, instr, state.rip + instr.length(), false};
  ExecutionResult result{};
  const auto code = instr.code();
  switch (code) {
    case iced_x86::Code::VEX_ANDN_R32_R32_RM32: result = handlers::handle_code_VEX_ANDN_R32_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_ANDN_R64_R64_RM64: result = handlers::handle_code_VEX_ANDN_R64_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_BEXTR_R32_RM32_R32: result = handlers::handle_code_VEX_BEXTR_R32_RM32_R32(ctx); break;
    case iced_x86::Code::VEX_BEXTR_R64_RM64_R64: result = handlers::handle_code_VEX_BEXTR_R64_RM64_R64(ctx); break;
    case iced_x86::Code::VEX_BLSI_R32_RM32: result = handlers::handle_code_VEX_BLSI_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_BLSI_R64_RM64: result = handlers::handle_code_VEX_BLSI_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_BLSMSK_R32_RM32: result = handlers::handle_code_VEX_BLSMSK_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_BLSMSK_R64_RM64: result = handlers::handle_code_VEX_BLSMSK_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_BLSR_R32_RM32: result = handlers::handle_code_VEX_BLSR_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_BLSR_R64_RM64: result = handlers::handle_code_VEX_BLSR_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_BZHI_R32_RM32_R32: result = handlers::handle_code_VEX_BZHI_R32_RM32_R32(ctx); break;
    case iced_x86::Code::VEX_BZHI_R64_RM64_R64: result = handlers::handle_code_VEX_BZHI_R64_RM64_R64(ctx); break;
    case iced_x86::Code::VEX_MULX_R32_R32_RM32: result = handlers::handle_code_VEX_MULX_R32_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_MULX_R64_R64_RM64: result = handlers::handle_code_VEX_MULX_R64_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_PDEP_R32_R32_RM32: result = handlers::handle_code_VEX_PDEP_R32_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_PDEP_R64_R64_RM64: result = handlers::handle_code_VEX_PDEP_R64_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_PEXT_R32_R32_RM32: result = handlers::handle_code_VEX_PEXT_R32_R32_RM32(ctx); break;
    case iced_x86::Code::VEX_PEXT_R64_R64_RM64: result = handlers::handle_code_VEX_PEXT_R64_R64_RM64(ctx); break;
    case iced_x86::Code::VEX_RORX_R32_RM32_IMM8: result = handlers::handle_code_VEX_RORX_R32_RM32_IMM8(ctx); break;
    case iced_x86::Code::VEX_RORX_R64_RM64_IMM8: result = handlers::handle_code_VEX_RORX_R64_RM64_IMM8(ctx); break;
    case iced_x86::Code::VEX_SARX_R32_RM32_R32: result = handlers::handle_code_VEX_SARX_R32_RM32_R32(ctx); break;
    case iced_x86::Code::VEX_SARX_R64_RM64_R64: result = handlers::handle_code_VEX_SARX_R64_RM64_R64(ctx); break;
    case iced_x86::Code::VEX_SHLX_R32_RM32_R32: result = handlers::handle_code_VEX_SHLX_R32_RM32_R32(ctx); break;
    case iced_x86::Code::VEX_SHLX_R64_RM64_R64: result = handlers::handle_code_VEX_SHLX_R64_RM64_R64(ctx); break;
    case iced_x86::Code::VEX_SHRX_R32_RM32_R32: result = handlers::handle_code_VEX_SHRX_R32_RM32_R32(ctx); break;
    case iced_x86::Code::VEX_SHRX_R64_RM64_R64: result = handlers::handle_code_VEX_SHRX_R64_RM64_R64(ctx); break;
    default:
      result = unsupported(ctx);
      break;
  }

  result.code = code;
  if (result.reason == StopReason::none) {
    if (static_cast<std::size_t>(code) < code_execution_counts_.size()) {
      ++code_execution_counts_[static_cast<std::size_t>(code)];
    }
    ++total_retired_;
    state.rip = ctx.control_flow_taken ? state.rip : ctx.next_rip;
    result.retired = 1;
  } else {
    ++stop_reason_counts_[stop_reason_to_index(result.reason)];
  }
  return result;
}

ExecutionResult Executor::run(CpuState& state, Memory& memory, std::size_t max_instructions) {
  ExecutionResult last{};
  for (std::size_t i = 0; i < max_instructions; ++i) {
    last = step(state, memory);
    if (last.reason != StopReason::none) {
      last.retired += i;
      return last;
    }
  }
  ++stop_reason_counts_[stop_reason_to_index(StopReason::execution_limit)];
  return {StopReason::execution_limit, max_instructions, std::nullopt, std::nullopt};
}

std::size_t Executor::supported_code_count() const noexcept {
  return 26;
}

ExecutionResult Executor::unsupported(ExecutionContext& ctx) {
  return {StopReason::unsupported_instruction, 0, ExceptionInfo{StopReason::unsupported_instruction, ctx.state.rip, 0}, std::nullopt};
}

std::vector<std::uint8_t> parse_hex_bytes(std::string_view text) {
  std::vector<std::uint8_t> bytes;
  std::string compact;
  compact.reserve(text.size());
  for (const auto ch : text) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      compact.push_back(static_cast<char>(ch));
    }
  }
  if ((compact.size() % 2) != 0) {
    throw std::runtime_error("hex byte string must contain an even number of digits");
  }
  for (std::size_t i = 0; i < compact.size(); i += 2) {
    unsigned value = 0;
    const auto pair = compact.substr(i, 2);
    const auto result = std::from_chars(pair.data(), pair.data() + pair.size(), value, 16);
    if (result.ec != std::errc{}) {
      throw std::runtime_error("invalid hex byte string");
    }
    bytes.push_back(static_cast<std::uint8_t>(value));
  }
  return bytes;
}

}  // namespace seven


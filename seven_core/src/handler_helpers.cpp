#include "seven/handler_helpers.hpp"

#include <iced_x86/instruction.hpp>
#include <iced_x86/memory_size_info.hpp>
#include <iced_x86/op_kind.hpp>
#include <iced_x86/register.hpp>

namespace seven {
namespace detail {

namespace {

constexpr std::uint32_t kMsrStar = 0xC0000081u;
constexpr std::uint32_t kMsrLStar = 0xC0000082u;
constexpr std::uint32_t kMsrCStar = 0xC0000083u;
constexpr std::uint32_t kMsrFMask = 0xC0000084u;
constexpr std::uint32_t kMsrKernelGsBase = 0xC0000102u;
constexpr std::uint32_t kMsrSysenterCs = 0x174u;
constexpr std::uint32_t kMsrSysenterEsp = 0x175u;
constexpr std::uint32_t kMsrSysenterEip = 0x176u;

std::uint64_t read_msr_unchecked(CpuState& state, std::uint32_t index) {
  const auto it = state.msr.find(index);
  if (it != state.msr.end()) {
    return it->second;
  }
  switch (index) {
    case kMsrStar:
    case kMsrLStar:
    case kMsrCStar:
    case kMsrFMask:
    case kMsrKernelGsBase:
    case kMsrSysenterCs:
    case kMsrSysenterEsp:
    case kMsrSysenterEip:
      return 0u;
    default:
      return 0u;
  }
}

}  // namespace

std::uint64_t read_msr(CpuState& state, std::uint32_t index) {
  return read_msr_unchecked(state, index);
}

void write_msr(CpuState& state, std::uint32_t index, std::uint64_t value) {
  state.msr[index] = value;
}

std::uint64_t read_xcr(CpuState& state, std::uint32_t index) {
  if (index < state.xcr.size()) {
    return state.xcr[index];
  }
  return 0u;
}

void write_xcr(CpuState& state, std::uint32_t index, std::uint64_t value) {
  if (index < state.xcr.size()) {
    state.xcr[index] = value;
  }
}

void set_flag(std::uint64_t& rflags, std::uint64_t bit, bool value) {
  if (value) {
    rflags |= bit;
  } else {
    rflags &= ~bit;
  }
}

std::uint64_t truncate(std::uint64_t value, std::size_t width) {
  return value & mask_for_width(width);
}

bool even_parity(std::uint8_t value) {
  return seven::even_parity(value);
}

std::uint64_t sign_extend(std::uint64_t value, std::size_t width) {
  return seven::sign_extend(value, width);
}

ExecutionResult memory_fault(ExecutionContext& ctx, std::uint64_t address) {
  return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, address, 0}, ctx.instr.code()};
}

ExecutionResult read_memory_checked(ExecutionContext& ctx, std::uint64_t address, void* value, std::size_t width) {
  if (!ctx.memory.read(address, value, width)) {
    return memory_fault(ctx, address);
  }
  return {};
}

ExecutionResult write_memory_checked(ExecutionContext& ctx, std::uint64_t address, const void* value, std::size_t width) {
  if (!ctx.memory.write(address, value, width)) {
    return memory_fault(ctx, address);
  }
  return {};
}

ExecutionResult read_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, std::uint64_t& value) {
  bool ok = false;
  value = read_operand(ctx, operand_index, width, &ok);
  if (!ok) {
    return memory_fault(ctx, memory_address(ctx));
  }
  return {};
}

ExecutionResult write_operand_checked(ExecutionContext& ctx, std::uint32_t operand_index, std::uint64_t value, std::size_t width) {
  if (!write_operand(ctx, operand_index, value, width)) {
    return memory_fault(ctx, memory_address(ctx));
  }
  return {};
}

std::uint64_t debug_data_breakpoint_hits(CpuState& state, std::uint64_t address, std::size_t size, bool is_read, bool is_write) noexcept {
  auto dr_len_from_encoding = [](std::uint64_t len) noexcept -> std::size_t {
    switch (len & 0x3u) {
      case 0: return 1;
      case 1: return 2;
      case 2: return 8;
      default: return 4;
    }
  };
  auto ranges_overlap = [](std::uint64_t a_base, std::size_t a_size, std::uint64_t b_base, std::size_t b_size) noexcept {
    if (a_size == 0 || b_size == 0) return false;
    return a_base < (b_base + b_size) && b_base < (a_base + a_size);
  };

  std::uint64_t hit_bits = 0;
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) continue;
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw == 0u || rw == 2u) continue;
    const bool access_matches = (rw == 1u && is_write) || (rw == 3u && (is_read || is_write));
    if (!access_matches) continue;
    const auto watch_base = mask_linear_address(state, state.dr[i]);
    const auto watch_size = dr_len_from_encoding((dr7 >> (18 + i * 4)) & 0x3u);
    if (ranges_overlap(watch_base, watch_size, address, size)) hit_bits |= (1ull << i);
  }
  return hit_bits;
}


ExecutionResult dispatch_interrupt(ExecutionContext& ctx, std::uint8_t vector, std::uint64_t return_rip,
                                   std::optional<std::uint32_t> error_code, bool push_rf_in_frame) {
  const auto gp_fault = [&]() -> ExecutionResult {
    return {StopReason::general_protection, 0,
            ExceptionInfo{StopReason::general_protection, ctx.state.rip, vector}, ctx.instr.code()};
  };

  const auto idt_has_entry = [&](std::size_t entry_size) -> bool {
    const std::uint64_t end_offset = (static_cast<std::uint64_t>(vector) + 1u) * entry_size - 1u;
    return end_offset <= ctx.state.idtr.limit;
  };

  const auto push_width = [&](std::uint64_t value, std::size_t width) -> ExecutionResult {
    ctx.state.gpr[4] = mask_stack_pointer(ctx.state, ctx.state.gpr[4] - width);
    switch (width) {
      case 2: {
        const auto v = static_cast<std::uint16_t>(value);
        return write_memory_checked(ctx, ctx.state.gpr[4], v);
      }
      case 4: {
        const auto v = static_cast<std::uint32_t>(value);
        return write_memory_checked(ctx, ctx.state.gpr[4], v);
      }
      case 8:
        return write_memory_checked(ctx, ctx.state.gpr[4], value);
      default:
        return gp_fault();
    }
  };

  std::uint64_t target_rip = 0;
  std::uint16_t target_cs = 0;
  std::uint8_t gate_type = 0;

  switch (ctx.state.mode) {
    case ExecutionMode::real16: {
      constexpr std::size_t kEntrySize = 4;
      if (!idt_has_entry(kEntrySize)) {
        return gp_fault();
      }
      const auto entry_base = ctx.state.idtr.base + static_cast<std::uint64_t>(vector) * kEntrySize;
      std::uint16_t ip = 0;
      if (auto result = read_memory_checked(ctx, entry_base, ip); !result.ok()) {
        return result;
      }
      if (auto result = read_memory_checked(ctx, entry_base + 2, target_cs); !result.ok()) {
        return result;
      }
      target_rip = ip;
      gate_type = 0x0E;
      break;
    }
    case ExecutionMode::compat32: {
      constexpr std::size_t kEntrySize = 8;
      if (!idt_has_entry(kEntrySize)) {
        return gp_fault();
      }
      const auto entry_base = ctx.state.idtr.base + static_cast<std::uint64_t>(vector) * kEntrySize;
      std::uint64_t descriptor = 0;
      if (auto result = read_memory_checked(ctx, entry_base, descriptor); !result.ok()) {
        return result;
      }
      target_rip = ((descriptor & 0xFFFFull) | (((descriptor >> 48) & 0xFFFFull) << 16)) & 0xFFFFFFFFull;
      target_cs = static_cast<std::uint16_t>((descriptor >> 16) & 0xFFFFull);
      const std::uint8_t type_attr = static_cast<std::uint8_t>((descriptor >> 40) & 0xFFu);
      const bool present = (type_attr & 0x80u) != 0;
      gate_type = static_cast<std::uint8_t>(type_attr & 0x0Fu);
      if (!present || (gate_type != 0x0Eu && gate_type != 0x0Fu)) {
        return gp_fault();
      }
      break;
    }
    case ExecutionMode::long64:
    default: {
      constexpr std::size_t kEntrySize = 16;
      if (!idt_has_entry(kEntrySize)) {
        return gp_fault();
      }
      const auto entry_base = ctx.state.idtr.base + static_cast<std::uint64_t>(vector) * kEntrySize;
      std::uint64_t desc_lo = 0;
      std::uint64_t desc_hi = 0;
      if (auto result = read_memory_checked(ctx, entry_base, desc_lo); !result.ok()) {
        return result;
      }
      if (auto result = read_memory_checked(ctx, entry_base + 8, desc_hi); !result.ok()) {
        return result;
      }
      target_rip = (desc_lo & 0xFFFFull) |
                   (((desc_lo >> 48) & 0xFFFFull) << 16) |
                   ((desc_hi & 0xFFFFFFFFull) << 32);
      target_cs = static_cast<std::uint16_t>((desc_lo >> 16) & 0xFFFFull);
      const std::uint8_t type_attr = static_cast<std::uint8_t>((desc_lo >> 40) & 0xFFu);
      const bool present = (type_attr & 0x80u) != 0;
      gate_type = static_cast<std::uint8_t>(type_attr & 0x0Fu);
      if (!present || (gate_type != 0x0Eu && gate_type != 0x0Fu)) {
        return gp_fault();
      }
      break;
    }
  }

  const auto return_width = instruction_pointer_width(ctx.state.mode);
  const auto flags_width = instruction_pointer_width(ctx.state.mode);
  const auto pushed_rflags = push_rf_in_frame ? (ctx.state.rflags | kFlagRF) : ctx.state.rflags;
  if (auto result = push_width(pushed_rflags, flags_width); !result.ok()) {
    return result;
  }
  if (auto result = push_width(ctx.state.sreg[1], 2); !result.ok()) {
    return result;
  }
  if (auto result = push_width(return_rip, return_width); !result.ok()) {
    return result;
  }
  if (error_code.has_value()) {
    if (auto result = push_width(error_code.value(), flags_width); !result.ok()) {
      return result;
    }
  }

  if (gate_type == 0x0E) {
    set_flag(ctx.state.rflags, kFlagIF, false);
  }
  ctx.state.rflags &= ~kFlagTF;

  ctx.state.sreg[1] = target_cs;
  ctx.state.rip = mask_instruction_pointer(ctx.state, target_rip);
  ctx.control_flow_taken = true;
  return {};
}

std::size_t register_width(iced_x86::Register reg) {
  switch (reg) {
    case iced_x86::Register::AL: case iced_x86::Register::CL: case iced_x86::Register::DL: case iced_x86::Register::BL:
    case iced_x86::Register::AH: case iced_x86::Register::CH: case iced_x86::Register::DH: case iced_x86::Register::BH:
    case iced_x86::Register::SPL: case iced_x86::Register::BPL: case iced_x86::Register::SIL: case iced_x86::Register::DIL:
    case iced_x86::Register::R8_L: case iced_x86::Register::R9_L: case iced_x86::Register::R10_L: case iced_x86::Register::R11_L:
    case iced_x86::Register::R12_L: case iced_x86::Register::R13_L: case iced_x86::Register::R14_L: case iced_x86::Register::R15_L:
      return 1;
    case iced_x86::Register::AX: case iced_x86::Register::CX: case iced_x86::Register::DX: case iced_x86::Register::BX:
    case iced_x86::Register::SP: case iced_x86::Register::BP: case iced_x86::Register::SI: case iced_x86::Register::DI:
    case iced_x86::Register::R8_W: case iced_x86::Register::R9_W: case iced_x86::Register::R10_W: case iced_x86::Register::R11_W:
    case iced_x86::Register::R12_W: case iced_x86::Register::R13_W: case iced_x86::Register::R14_W: case iced_x86::Register::R15_W:
      return 2;
    case iced_x86::Register::EAX: case iced_x86::Register::ECX: case iced_x86::Register::EDX: case iced_x86::Register::EBX:
    case iced_x86::Register::ESP: case iced_x86::Register::EBP: case iced_x86::Register::ESI: case iced_x86::Register::EDI:
    case iced_x86::Register::R8_D: case iced_x86::Register::R9_D: case iced_x86::Register::R10_D: case iced_x86::Register::R11_D:
    case iced_x86::Register::R12_D: case iced_x86::Register::R13_D: case iced_x86::Register::R14_D: case iced_x86::Register::R15_D:
      return 4;
    case iced_x86::Register::ES:
    case iced_x86::Register::CS:
    case iced_x86::Register::SS:
    case iced_x86::Register::DS:
    case iced_x86::Register::FS:
    case iced_x86::Register::GS:
      return 2;
    case iced_x86::Register::MM0: case iced_x86::Register::MM1: case iced_x86::Register::MM2: case iced_x86::Register::MM3:
    case iced_x86::Register::MM4: case iced_x86::Register::MM5: case iced_x86::Register::MM6: case iced_x86::Register::MM7:
      return 8;
    default:
      return 8;
  }
}

std::size_t operand_width(const iced_x86::Instruction& instr, std::uint32_t operand_index) {
  const auto kind = instr.op_kind(operand_index);
  if (kind == iced_x86::OpKind::REGISTER) {
    return register_width(instr.op_register(operand_index));
  }
  if (kind == iced_x86::OpKind::MEMORY) {
    return iced_x86::memory_size_ext::get_size(static_cast<iced_x86::MemorySize>(instr.memory_size()));
  }
  switch (kind) {
    case iced_x86::OpKind::IMMEDIATE8:
    case iced_x86::OpKind::IMMEDIATE8_2ND:
    case iced_x86::OpKind::IMMEDIATE8TO16:
    case iced_x86::OpKind::IMMEDIATE8TO32:
    case iced_x86::OpKind::IMMEDIATE8TO64:
      return 1;
    case iced_x86::OpKind::IMMEDIATE16:
      return 2;
    case iced_x86::OpKind::IMMEDIATE32:
    case iced_x86::OpKind::IMMEDIATE32TO64:
      return 4;
    case iced_x86::OpKind::IMMEDIATE64:
      return 8;
    default:
      return 8;
  }
}

std::uint64_t memory_address(ExecutionContext& ctx) {
  if (ctx.instr.is_ip_rel_memory_operand()) {
    return mask_linear_address(ctx.state, ctx.instr.ip_rel_memory_address());
  }
  std::uint64_t address = 0;
  if (ctx.instr.memory_base() != iced_x86::Register::NONE) {
    address += read_register(ctx.state, ctx.instr.memory_base());
  }
  if (ctx.instr.memory_index() != iced_x86::Register::NONE) {
    address += read_register(ctx.state, ctx.instr.memory_index()) * ctx.instr.memory_index_scale();
  }
  address += ctx.instr.memory_displacement64();
  if (ctx.instr.segment_prefix() == iced_x86::Register::FS) {
    address += ctx.state.fs_base;
  } else if (ctx.instr.segment_prefix() == iced_x86::Register::GS) {
    address += ctx.state.gs_base;
  }
  return mask_linear_address(ctx.state, address);
}

std::uint64_t read_register(CpuState& state, iced_x86::Register reg) {
  auto seg_index = [](iced_x86::Register sreg) -> std::size_t {
    switch (sreg) {
      case iced_x86::Register::ES: return 0;
      case iced_x86::Register::CS: return 1;
      case iced_x86::Register::SS: return 2;
      case iced_x86::Register::DS: return 3;
      case iced_x86::Register::FS: return 4;
      case iced_x86::Register::GS: return 5;
      default: return 0;
    }
  };
  auto ctrl_index = [](iced_x86::Register creg) -> std::size_t {
    return static_cast<std::size_t>(creg) - static_cast<std::size_t>(iced_x86::Register::CR0);
  };
  auto dbg_index = [](iced_x86::Register dreg) -> std::size_t {
    return static_cast<std::size_t>(dreg) - static_cast<std::size_t>(iced_x86::Register::DR0);
  };
  auto test_index = [](iced_x86::Register treg) -> std::size_t {
    return static_cast<std::size_t>(treg) - static_cast<std::size_t>(iced_x86::Register::TR0);
  };
  if (reg >= iced_x86::Register::ES && reg <= iced_x86::Register::GS) {
    return state.sreg[seg_index(reg)];
  }
  if (reg >= iced_x86::Register::CR0 && reg <= iced_x86::Register::CR15) {
    return state.cr[ctrl_index(reg)];
  }
  if (reg >= iced_x86::Register::DR0 && reg <= iced_x86::Register::DR15) {
    return state.dr[dbg_index(reg)];
  }
  if (reg >= iced_x86::Register::TR0 && reg <= iced_x86::Register::TR7) {
    return state.tr[test_index(reg)];
  }
  if (reg >= iced_x86::Register::MM0 && reg <= iced_x86::Register::MM7) {
    return state.mmx_get(static_cast<std::size_t>(reg) - static_cast<std::size_t>(iced_x86::Register::MM0));
  }

  auto reg_index = [](iced_x86::Register full) -> std::size_t {
    switch (full) {
      case iced_x86::Register::RAX: case iced_x86::Register::EAX: case iced_x86::Register::AX: case iced_x86::Register::AL: case iced_x86::Register::AH:
        return 0;
      case iced_x86::Register::RCX: case iced_x86::Register::ECX: case iced_x86::Register::CX: case iced_x86::Register::CL: case iced_x86::Register::CH:
        return 1;
      case iced_x86::Register::RDX: case iced_x86::Register::EDX: case iced_x86::Register::DX: case iced_x86::Register::DL: case iced_x86::Register::DH:
        return 2;
      case iced_x86::Register::RBX: case iced_x86::Register::EBX: case iced_x86::Register::BX: case iced_x86::Register::BL: case iced_x86::Register::BH:
        return 3;
      case iced_x86::Register::RSP: case iced_x86::Register::ESP: case iced_x86::Register::SP: case iced_x86::Register::SPL:
        return 4;
      case iced_x86::Register::RBP: case iced_x86::Register::EBP: case iced_x86::Register::BP: case iced_x86::Register::BPL:
        return 5;
      case iced_x86::Register::RSI: case iced_x86::Register::ESI: case iced_x86::Register::SI: case iced_x86::Register::SIL:
        return 6;
      case iced_x86::Register::RDI: case iced_x86::Register::EDI: case iced_x86::Register::DI: case iced_x86::Register::DIL:
        return 7;
      case iced_x86::Register::R8: case iced_x86::Register::R8_D: case iced_x86::Register::R8_W: case iced_x86::Register::R8_L: return 8;
      case iced_x86::Register::R9: case iced_x86::Register::R9_D: case iced_x86::Register::R9_W: case iced_x86::Register::R9_L: return 9;
      case iced_x86::Register::R10: case iced_x86::Register::R10_D: case iced_x86::Register::R10_W: case iced_x86::Register::R10_L: return 10;
      case iced_x86::Register::R11: case iced_x86::Register::R11_D: case iced_x86::Register::R11_W: case iced_x86::Register::R11_L: return 11;
      case iced_x86::Register::R12: case iced_x86::Register::R12_D: case iced_x86::Register::R12_W: case iced_x86::Register::R12_L: return 12;
      case iced_x86::Register::R13: case iced_x86::Register::R13_D: case iced_x86::Register::R13_W: case iced_x86::Register::R13_L: return 13;
      case iced_x86::Register::R14: case iced_x86::Register::R14_D: case iced_x86::Register::R14_W: case iced_x86::Register::R14_L: return 14;
      case iced_x86::Register::R15: case iced_x86::Register::R15_D: case iced_x86::Register::R15_W: case iced_x86::Register::R15_L: return 15;
      default: return 0;
    }
  };

  const auto raw = state.gpr[reg_index(reg)];
  switch (register_width(reg)) {
    case 1:
      if (reg == iced_x86::Register::AH || reg == iced_x86::Register::BH || reg == iced_x86::Register::CH || reg == iced_x86::Register::DH) {
        return (raw >> 8) & 0xFF;
      }
      return raw & 0xFF;
    case 2:
      return raw & 0xFFFF;
    case 4:
      return raw & 0xFFFFFFFFu;
    default:
      return raw;
  }
}

void write_register(CpuState& state, iced_x86::Register reg, std::uint64_t value, std::size_t width_override) {
  auto seg_index = [](iced_x86::Register sreg) -> std::size_t {
    switch (sreg) {
      case iced_x86::Register::ES: return 0;
      case iced_x86::Register::CS: return 1;
      case iced_x86::Register::SS: return 2;
      case iced_x86::Register::DS: return 3;
      case iced_x86::Register::FS: return 4;
      case iced_x86::Register::GS: return 5;
      default: return 0;
    }
  };
  auto ctrl_index = [](iced_x86::Register creg) -> std::size_t {
    return static_cast<std::size_t>(creg) - static_cast<std::size_t>(iced_x86::Register::CR0);
  };
  auto dbg_index = [](iced_x86::Register dreg) -> std::size_t {
    return static_cast<std::size_t>(dreg) - static_cast<std::size_t>(iced_x86::Register::DR0);
  };
  auto test_index = [](iced_x86::Register treg) -> std::size_t {
    return static_cast<std::size_t>(treg) - static_cast<std::size_t>(iced_x86::Register::TR0);
  };
  if (reg >= iced_x86::Register::ES && reg <= iced_x86::Register::GS) {
    state.sreg[seg_index(reg)] = static_cast<std::uint16_t>(value & 0xFFFFu);
    return;
  }
  if (reg >= iced_x86::Register::CR0 && reg <= iced_x86::Register::CR15) {
    state.cr[ctrl_index(reg)] = value;
    return;
  }
  if (reg >= iced_x86::Register::DR0 && reg <= iced_x86::Register::DR15) {
    state.dr[dbg_index(reg)] = value;
    return;
  }
  if (reg >= iced_x86::Register::TR0 && reg <= iced_x86::Register::TR7) {
    state.tr[test_index(reg)] = value;
    return;
  }
  if (reg >= iced_x86::Register::MM0 && reg <= iced_x86::Register::MM7) {
    state.mmx_set(static_cast<std::size_t>(reg) - static_cast<std::size_t>(iced_x86::Register::MM0), value);
    return;
  }

  auto reg_index = [](iced_x86::Register full) -> std::size_t {
    switch (full) {
      case iced_x86::Register::RAX: case iced_x86::Register::EAX: case iced_x86::Register::AX: case iced_x86::Register::AL: case iced_x86::Register::AH: return 0;
      case iced_x86::Register::RCX: case iced_x86::Register::ECX: case iced_x86::Register::CX: case iced_x86::Register::CL: case iced_x86::Register::CH: return 1;
      case iced_x86::Register::RDX: case iced_x86::Register::EDX: case iced_x86::Register::DX: case iced_x86::Register::DL: case iced_x86::Register::DH: return 2;
      case iced_x86::Register::RBX: case iced_x86::Register::EBX: case iced_x86::Register::BX: case iced_x86::Register::BL: case iced_x86::Register::BH: return 3;
      case iced_x86::Register::RSP: case iced_x86::Register::ESP: case iced_x86::Register::SP: case iced_x86::Register::SPL: return 4;
      case iced_x86::Register::RBP: case iced_x86::Register::EBP: case iced_x86::Register::BP: case iced_x86::Register::BPL: return 5;
      case iced_x86::Register::RSI: case iced_x86::Register::ESI: case iced_x86::Register::SI: case iced_x86::Register::SIL: return 6;
      case iced_x86::Register::RDI: case iced_x86::Register::EDI: case iced_x86::Register::DI: case iced_x86::Register::DIL: return 7;
      case iced_x86::Register::R8: case iced_x86::Register::R8_D: case iced_x86::Register::R8_W: case iced_x86::Register::R8_L: return 8;
      case iced_x86::Register::R9: case iced_x86::Register::R9_D: case iced_x86::Register::R9_W: case iced_x86::Register::R9_L: return 9;
      case iced_x86::Register::R10: case iced_x86::Register::R10_D: case iced_x86::Register::R10_W: case iced_x86::Register::R10_L: return 10;
      case iced_x86::Register::R11: case iced_x86::Register::R11_D: case iced_x86::Register::R11_W: case iced_x86::Register::R11_L: return 11;
      case iced_x86::Register::R12: case iced_x86::Register::R12_D: case iced_x86::Register::R12_W: case iced_x86::Register::R12_L: return 12;
      case iced_x86::Register::R13: case iced_x86::Register::R13_D: case iced_x86::Register::R13_W: case iced_x86::Register::R13_L: return 13;
      case iced_x86::Register::R14: case iced_x86::Register::R14_D: case iced_x86::Register::R14_W: case iced_x86::Register::R14_L: return 14;
      case iced_x86::Register::R15: case iced_x86::Register::R15_D: case iced_x86::Register::R15_W: case iced_x86::Register::R15_L: return 15;
      default:
        return 0;
    }
  };

  auto& raw = state.gpr[reg_index(reg)];
  const auto reg_width = width_override != 0 ? width_override : register_width(reg);
  switch (reg_width) {
    case 1:
      if (reg == iced_x86::Register::AH || reg == iced_x86::Register::BH || reg == iced_x86::Register::CH || reg == iced_x86::Register::DH) {
        raw = (raw & ~0xFF00ull) | ((value & 0xFFull) << 8);
      } else {
        raw = (raw & ~0xFFull) | (value & 0xFFull);
      }
      break;
    case 2:
      raw = (raw & ~0xFFFFull) | (value & 0xFFFFull);
      break;
    case 4:
      raw = value & 0xFFFFFFFFull;
      break;
    default:
      raw = value;
      break;
  }
}

std::uint64_t immediate_value(const iced_x86::Instruction& instr, std::uint32_t operand_index) {
  switch (instr.op_kind(operand_index)) {
    case iced_x86::OpKind::IMMEDIATE8: return instr.immediate8();
    case iced_x86::OpKind::IMMEDIATE8_2ND: return instr.immediate8_2nd();
    case iced_x86::OpKind::IMMEDIATE16: return instr.immediate16();
    case iced_x86::OpKind::IMMEDIATE32: return instr.immediate32();
    case iced_x86::OpKind::IMMEDIATE64: return instr.immediate64();
    case iced_x86::OpKind::IMMEDIATE8TO16: return static_cast<std::uint64_t>(instr.immediate8to16());
    case iced_x86::OpKind::IMMEDIATE8TO32: return static_cast<std::uint64_t>(instr.immediate8to32());
    case iced_x86::OpKind::IMMEDIATE8TO64: return static_cast<std::uint64_t>(instr.immediate8to64());
    case iced_x86::OpKind::IMMEDIATE32TO64: return static_cast<std::uint64_t>(instr.immediate32to64());
    case iced_x86::OpKind::NEAR_BRANCH16: return instr.near_branch16();
    case iced_x86::OpKind::NEAR_BRANCH32: return instr.near_branch32();
    case iced_x86::OpKind::NEAR_BRANCH64: return instr.near_branch64();
    default:
      return 0;
  }
}

std::uint64_t read_operand(ExecutionContext& ctx, std::uint32_t operand_index, std::size_t width, bool* ok) {
  if (ok) {
    *ok = true;
  }
  const auto kind = ctx.instr.op_kind(operand_index);
  if (kind == iced_x86::OpKind::REGISTER) {
    return read_register(ctx.state, ctx.instr.op_register(operand_index));
  }
  if (kind == iced_x86::OpKind::MEMORY) {
    const auto address = memory_address(ctx);
    std::uint64_t value = 0;
    if (!ctx.memory.read(address, &value, width)) {
      if (ok) {
        *ok = false;
      }
      return 0;
    }
    return truncate(value, width);
  }
  return truncate(immediate_value(ctx.instr, operand_index), width);
}

bool write_operand(ExecutionContext& ctx, std::uint32_t operand_index, std::uint64_t value, std::size_t width) {
  value = truncate(value, width);
  const auto kind = ctx.instr.op_kind(operand_index);
  if (kind == iced_x86::OpKind::REGISTER) {
    write_register(ctx.state, ctx.instr.op_register(operand_index), value);
    return true;
  }
  if (kind == iced_x86::OpKind::MEMORY) {
    return ctx.memory.write(memory_address(ctx), &value, width);
  }
  return false;
}

ExecutionResult divide_fault(ExecutionContext& ctx) {
  return {StopReason::divide_error, 0, ExceptionInfo{StopReason::divide_error, ctx.state.rip, 0}, ctx.instr.code()};
}

void set_logic_flags(CpuState& state, std::uint64_t value, std::size_t width) {
  value = truncate(value, width);
  set_flag(state.rflags, kFlagCF, false);
  set_flag(state.rflags, kFlagOF, false);
  set_flag(state.rflags, kFlagAF, false);
  set_flag(state.rflags, kFlagZF, value == 0);
  set_flag(state.rflags, kFlagSF, (value & sign_bit_for_width(width)) != 0);
  set_flag(state.rflags, kFlagPF, even_parity(static_cast<std::uint8_t>(value & 0xFF)));
}
void set_multiply_flags(CpuState& state, std::uint64_t value, std::size_t width, bool overflow) {
  value = truncate(value, width);
  set_flag(state.rflags, kFlagCF, overflow);
  set_flag(state.rflags, kFlagOF, overflow);
  set_flag(state.rflags, kFlagAF, false);
  // libLISA reports multiply families clear ZF on this target rather than recomputing it from the result.
  set_flag(state.rflags, kFlagZF, false);
  set_flag(state.rflags, kFlagSF, (value & sign_bit_for_width(width)) != 0);
  set_flag(state.rflags, kFlagPF, even_parity(static_cast<std::uint8_t>(value & 0xFF)));
}


void set_add_flags(CpuState& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result, std::size_t width, bool carry_in) {
  const auto mask = mask_for_width(width);
  lhs &= mask;
  rhs &= mask;
  result &= mask;
  const auto partial = lhs + rhs;
  const auto final_result = partial + (carry_in ? 1ull : 0ull);
  bool carry = partial < lhs;
  if (carry_in) {
    carry = carry || (final_result == 0);
  }
  if (width < 8) {
    carry = carry || ((final_result & ~mask) != 0);
  }
  set_flag(state.rflags, kFlagCF, carry);
  set_flag(state.rflags, kFlagAF, ((lhs ^ rhs ^ result) & 0x10) != 0);
  set_flag(state.rflags, kFlagZF, result == 0);
  set_flag(state.rflags, kFlagSF, (result & sign_bit_for_width(width)) != 0);
  set_flag(state.rflags, kFlagPF, even_parity(static_cast<std::uint8_t>(result & 0xFF)));
  const auto sign = sign_bit_for_width(width);
  set_flag(state.rflags, kFlagOF, ((~(lhs ^ rhs) & (lhs ^ result)) & sign) != 0);
}

void set_sub_flags(CpuState& state, std::uint64_t lhs, std::uint64_t rhs, std::uint64_t result, std::size_t width, bool borrow_in) {
  const auto mask = mask_for_width(width);
  lhs &= mask;
  rhs &= mask;
  result &= mask;
  const auto borrow = static_cast<std::uint64_t>(borrow_in ? 1 : 0);
  set_flag(state.rflags, kFlagCF, lhs < (rhs + borrow));
  set_flag(state.rflags, kFlagAF, ((lhs ^ rhs ^ result) & 0x10) != 0);
  set_flag(state.rflags, kFlagZF, result == 0);
  set_flag(state.rflags, kFlagSF, (result & sign_bit_for_width(width)) != 0);
  set_flag(state.rflags, kFlagPF, even_parity(static_cast<std::uint8_t>(result & 0xFF)));
  const auto sign = sign_bit_for_width(width);
  set_flag(state.rflags, kFlagOF, (((lhs ^ rhs) & (lhs ^ result)) & sign) != 0);
}

}  // namespace detail
}  // namespace seven


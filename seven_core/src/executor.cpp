#include "seven/executor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include <iced_x86/code.hpp>
#include <iced_x86/decoder.hpp>
#include <iced_x86/instruction_info.hpp>
#include <iced_x86/memory_size_info.hpp>

#include "seven/handler_helpers.hpp"
#include "seven/handlers_fwd.hpp"

namespace seven {

namespace {

[[nodiscard]] bool env_flag_set(const char* name) noexcept {
  if (const char* v = std::getenv(name)) {
    return v[0] != '\0' && v[0] != '0';
  }
  return false;
}

}  // namespace

Executor::Executor()
    : code_execution_counts_(kCodeCount, 0),
      stop_reason_counts_(kStopReasonCount, 0) {
  trace_semantics_ = env_flag_set("SEVEN_TRACE_SEMANTICS");
  trace_openkey_probe_ = env_flag_set("SEVEN_TRACE_OPENKEY");
  trace_strrchr_ = env_flag_set("SEVEN_TRACE_STRRCHR");
  collect_code_stats_ = env_flag_set("SEVEN_COLLECT_CODE_STATS");
}

namespace {

constexpr bool kEnableAvx = SEVEN_ENABLE_AVX != 0;
constexpr bool kEnableAvx512 = SEVEN_ENABLE_AVX512 != 0;
constexpr std::size_t kMaxFaultRetries = 8;

constexpr std::size_t kVectorRegisterCount = 32;
constexpr std::size_t kXmmWidth = 16;
constexpr std::size_t kYmmWidth = 32;
constexpr std::size_t kZmmWidth = 64;

[[nodiscard]] std::size_t vector_width_for_register(iced_x86::Register reg) noexcept {
  const auto value = static_cast<std::uint32_t>(reg);
  const auto xmm0 = static_cast<std::uint32_t>(iced_x86::Register::XMM0);
  const auto ymm0 = static_cast<std::uint32_t>(iced_x86::Register::YMM0);
  const auto zmm0 = static_cast<std::uint32_t>(iced_x86::Register::ZMM0);
  if (value >= zmm0 && value < zmm0 + kVectorRegisterCount) return kZmmWidth;
  if (value >= ymm0 && value < ymm0 + kVectorRegisterCount) return kYmmWidth;
  if (value >= xmm0 && value < xmm0 + kVectorRegisterCount) return kXmmWidth;
  return 0;
}

[[nodiscard]] bool simd_profile_allows(const iced_x86::Instruction& instr) noexcept {
  const auto encoding = iced_x86::InstructionExtensions::encoding(instr);
  if (encoding == iced_x86::EncodingKind::EVEX && !kEnableAvx512) {
    return false;
  }
  if (encoding == iced_x86::EncodingKind::VEX && !kEnableAvx) {
    return false;
  }
  for (std::uint32_t i = 0; i < instr.op_count(); ++i) {
    if (instr.op_kind(i) == iced_x86::OpKind::REGISTER &&
        vector_width_for_register(instr.op_register(i)) > kVectorBytes) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] iced_x86::Code normalize_reported_code(iced_x86::Code code) noexcept {
  switch (code) {
    case iced_x86::Code::PUSHD_IMM8:
      return iced_x86::Code::PUSHQ_IMM8;
    case iced_x86::Code::PUSHD_IMM32:
      return iced_x86::Code::PUSHQ_IMM32;
    default:
      return code;
  }
}

[[nodiscard]] std::optional<TrapKind> trap_kind_for_code(iced_x86::Code code) noexcept {
  switch (code) {
    case iced_x86::Code::SYSCALL:
      return TrapKind::syscall;
    case iced_x86::Code::CPUID:
      return TrapKind::cpuid;
    case iced_x86::Code::RDTSC:
      return TrapKind::rdtsc;
    case iced_x86::Code::RDTSCP:
      return TrapKind::rdtscp;
    case iced_x86::Code::INT1:
    case iced_x86::Code::INT3:
    case iced_x86::Code::INT_IMM8:
    case iced_x86::Code::INTO:
      return TrapKind::interrupt;
    default:
      return std::nullopt;
  }
}

void maybe_trace_strrchr(const CpuState& state, const Memory& memory) noexcept {
  if (state.rip == 0x180126AF0ull) {
    std::array<std::uint8_t, 48> s{};
    const bool ok = memory.read(state.gpr[1], s.data(), s.size(), MemoryAccessKind::data_read);
    std::fprintf(stderr,
        "[seven-strrchr] enter rcx=0x%llx dl=0x%02llx ok=%u bytes=",
        static_cast<unsigned long long>(state.gpr[1]),
        static_cast<unsigned long long>(state.gpr[2] & 0xFFull),
        ok ? 1u : 0u);
    if (ok) {
      for (std::size_t i = 0; i < s.size(); ++i) {
        std::fprintf(stderr, "%02x", static_cast<unsigned>(s[i]));
      }
    }
    std::fprintf(stderr, "\n");
  } else if (state.rip == 0x180126BECull) {
    std::fprintf(stderr,
        "[seven-strrchr] exit rax=0x%llx r9=0x%llx r10=0x%llx\n",
        static_cast<unsigned long long>(state.gpr[0]),
        static_cast<unsigned long long>(state.gpr[9]),
        static_cast<unsigned long long>(state.gpr[10]));
  }
}

void maybe_trace_openkey_probe(const CpuState& state, const Memory& memory) noexcept {
  if (state.rip != 0x180161cd2ull) {
    return;
  }
  const std::uint32_t sysno = static_cast<std::uint32_t>(state.gpr[0] & 0xFFFFFFFFull);
  if (sysno != 0x12u) {
    return;
  }

  std::uint64_t ret_addr = 0;
  const bool have_ret = memory.read(state.gpr[4], &ret_addr, sizeof(ret_addr), MemoryAccessKind::data_read);
  if (!have_ret || ret_addr != 0x1800ac33cull) {
    return;
  }

  struct ObjAttrs64 {
    std::uint32_t length;
    std::uint32_t pad0;
    std::uint64_t root_dir;
    std::uint64_t object_name;
    std::uint32_t attributes;
    std::uint32_t pad1;
    std::uint64_t security_descriptor;
    std::uint64_t security_qos;
  } attrs{};

  const std::uint64_t key_handle_ptr = state.gpr[1];
  const std::uint64_t desired_access = state.gpr[2];
  const std::uint64_t obj_attr_ptr = state.gpr[8];
  const bool have_attrs = memory.read(obj_attr_ptr, &attrs, sizeof(attrs), MemoryAccessKind::data_read);

  struct UnicodeString64 {
    std::uint16_t length;
    std::uint16_t max_length;
    std::uint32_t pad0;
    std::uint64_t buffer;
  } us{};
  bool have_us = false;
  if (have_attrs && attrs.object_name != 0) {
    have_us = memory.read(attrs.object_name, &us, sizeof(us), MemoryAccessKind::data_read);
  }

  std::array<char16_t, 260> name_buf{};
  std::size_t name_chars = 0;
  bool have_name = false;
  if (have_us && us.buffer != 0 && us.length > 0) {
    name_chars = std::min<std::size_t>(name_buf.size() - 1, static_cast<std::size_t>(us.length / 2u));
    have_name = memory.read(us.buffer, name_buf.data(), name_chars * sizeof(char16_t), MemoryAccessKind::data_read);
  }

  std::fprintf(stderr,
      "[seven-openkey] rip=0x%llx ret=0x%llx rcx=0x%llx rdx=0x%llx r8=0x%llx attrs_ok=%u attrs_len=0x%x root=0x%llx name_ptr=0x%llx attrs=0x%x us_ok=%u us_len=0x%x us_max=0x%x us_buf=0x%llx\n",
      static_cast<unsigned long long>(state.rip),
      static_cast<unsigned long long>(ret_addr),
      static_cast<unsigned long long>(key_handle_ptr),
      static_cast<unsigned long long>(desired_access),
      static_cast<unsigned long long>(obj_attr_ptr),
      have_attrs ? 1u : 0u,
      have_attrs ? attrs.length : 0u,
      static_cast<unsigned long long>(have_attrs ? attrs.root_dir : 0ull),
      static_cast<unsigned long long>(have_attrs ? attrs.object_name : 0ull),
      have_attrs ? attrs.attributes : 0u,
      have_us ? 1u : 0u,
      have_us ? us.length : 0u,
      have_us ? us.max_length : 0u,
      static_cast<unsigned long long>(have_us ? us.buffer : 0ull));

  if (have_name && name_chars != 0) {
    std::fprintf(stderr, "[seven-openkey] name_utf16:");
    for (std::size_t i = 0; i < name_chars; ++i) {
      std::fprintf(stderr, " %04x", static_cast<unsigned>(name_buf[i]));
    }
    std::fprintf(stderr, "\n");
  }
}

void maybe_trace_semantics(const CpuState& state, const iced_x86::Instruction& instr, const Memory& memory) noexcept {
  constexpr std::uint64_t kCxxThrowStart = 0x1059751E0ull;
  constexpr std::uint64_t kCxxThrowEnd = 0x105975288ull;
  if (!(state.rip >= kCxxThrowStart && state.rip < kCxxThrowEnd)) {
    return;
  }
  std::uint64_t call_target = 0;
  const bool call_target_ok = memory.read(0x105984080ull, &call_target, sizeof(call_target), MemoryAccessKind::data_read);
  std::uint64_t stack0 = 0;
  std::uint64_t stack1 = 0;
  std::uint64_t stack_c0 = 0;
  std::uint64_t stack_c8 = 0;
  const bool stack0_ok = memory.read(state.gpr[4], &stack0, sizeof(stack0), MemoryAccessKind::data_read);
  const bool stack1_ok = memory.read(state.gpr[4] + 8, &stack1, sizeof(stack1), MemoryAccessKind::data_read);
  const bool stack_c0_ok = memory.read(state.gpr[4] + 0xC0, &stack_c0, sizeof(stack_c0), MemoryAccessKind::data_read);
  const bool stack_c8_ok = memory.read(state.gpr[4] + 0xC8, &stack_c8, sizeof(stack_c8), MemoryAccessKind::data_read);
  std::fprintf(stderr,
      "[seven-trace] rip=0x%llx code=%u len=%u rax=0x%llx rbx=0x%llx rcx=0x%llx rdx=0x%llx rsp=0x%llx [rsp]=%s0x%llx [rsp+8]=%s0x%llx [rsp+c0]=%s0x%llx [rsp+c8]=%s0x%llx iat=%s0x%llx rbp=0x%llx rsi=0x%llx rdi=0x%llx r8=0x%llx r9=0x%llx cf=%u zf=%u sf=%u of=%u\n",
      static_cast<unsigned long long>(state.rip),
      static_cast<unsigned>(instr.code()),
      static_cast<unsigned>(instr.length()),
      static_cast<unsigned long long>(state.gpr[0]),
      static_cast<unsigned long long>(state.gpr[3]),
      static_cast<unsigned long long>(state.gpr[1]),
      static_cast<unsigned long long>(state.gpr[2]),
      static_cast<unsigned long long>(state.gpr[4]),
      stack0_ok ? "" : "err:",
      static_cast<unsigned long long>(stack0),
      stack1_ok ? "" : "err:",
      static_cast<unsigned long long>(stack1),
      stack_c0_ok ? "" : "err:",
      static_cast<unsigned long long>(stack_c0),
      stack_c8_ok ? "" : "err:",
      static_cast<unsigned long long>(stack_c8),
      call_target_ok ? "" : "err:",
      static_cast<unsigned long long>(call_target),
      static_cast<unsigned long long>(state.gpr[5]),
      static_cast<unsigned long long>(state.gpr[6]),
      static_cast<unsigned long long>(state.gpr[7]),
      static_cast<unsigned long long>(state.gpr[8]),
      static_cast<unsigned long long>(state.gpr[9]),
      (state.rflags & kFlagCF) ? 1u : 0u,
      (state.rflags & kFlagZF) ? 1u : 0u,
      (state.rflags & kFlagSF) ? 1u : 0u,
      (state.rflags & kFlagOF) ? 1u : 0u);
}

struct DebugMemoryAccess {
  std::uint64_t address = 0;
  std::size_t size = 0;
  bool is_read = false;
  bool is_write = false;
};

[[nodiscard]] bool op_access_reads(iced_x86::OpAccess access) noexcept {
  switch (access) {
    case iced_x86::OpAccess::READ:
    case iced_x86::OpAccess::COND_READ:
    case iced_x86::OpAccess::READ_WRITE:
    case iced_x86::OpAccess::READ_COND_WRITE:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool op_access_writes(iced_x86::OpAccess access) noexcept {
  switch (access) {
    case iced_x86::OpAccess::WRITE:
    case iced_x86::OpAccess::COND_WRITE:
    case iced_x86::OpAccess::READ_WRITE:
    case iced_x86::OpAccess::READ_COND_WRITE:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] std::size_t dr_len_from_encoding(std::uint64_t len_bits) noexcept {
  switch (len_bits & 0x3u) {
    case 0u: return 1;
    case 1u: return 2;
    case 2u: return 8;
    case 3u:
    default:
      return 4;
  }
}

[[nodiscard]] bool ranges_overlap(std::uint64_t a_base, std::size_t a_size, std::uint64_t b_base, std::size_t b_size) noexcept {
  if (a_size == 0 || b_size == 0) {
    return false;
  }
  const auto a_end = a_base + static_cast<std::uint64_t>(a_size - 1);
  const auto b_end = b_base + static_cast<std::uint64_t>(b_size - 1);
  return !(a_end < b_base || b_end < a_base);
}

[[nodiscard]] std::vector<DebugMemoryAccess> collect_debug_memory_accesses(CpuState& state, const iced_x86::Instruction& instr) {
  std::vector<DebugMemoryAccess> accesses;
  iced_x86::InstructionInfoFactory info_factory;
  const auto& info = info_factory.info(instr);
  for (const auto& used_mem : info.used_memory()) {
    const auto access = used_mem.access;
    const bool is_read = op_access_reads(access);
    const bool is_write = op_access_writes(access);
    if (!is_read && !is_write) {
      continue;
    }

    std::uint64_t address = 0;
    if (used_mem.base != iced_x86::Register::NONE) {
      address += detail::read_register(state, used_mem.base);
    }
    if (used_mem.index != iced_x86::Register::NONE) {
      address += detail::read_register(state, used_mem.index) * used_mem.scale;
    }
    address += used_mem.displacement;
    if (used_mem.segment == iced_x86::Register::FS) {
      address += state.fs_base;
    } else if (used_mem.segment == iced_x86::Register::GS) {
      address += state.gs_base;
    }
    address = mask_linear_address(state, address);

    const auto size = std::max<std::size_t>(1u, iced_x86::memory_size_ext::get_size(used_mem.memory_size));
    accesses.push_back(DebugMemoryAccess{address, size, is_read, is_write});
  }
  return accesses;
}

[[nodiscard]] bool has_enabled_execute_breakpoints(const CpuState& state) noexcept {
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) {
      continue;
    }
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw == 0u) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool has_enabled_data_breakpoints(const CpuState& state) noexcept {
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) {
      continue;
    }
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw != 0u) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::uint64_t collect_execute_breakpoint_hits(CpuState& state, std::uint64_t instruction_rip) noexcept {
  std::uint64_t hit_bits = 0;
  const std::uint64_t dr7 = state.dr[7];
  for (std::uint32_t i = 0; i < 4; ++i) {
    const bool enabled = ((dr7 >> (i * 2)) & 0x3u) != 0;
    if (!enabled) {
      continue;
    }
    const std::uint64_t rw = (dr7 >> (16 + i * 4)) & 0x3u;
    if (rw != 0u) {
      continue;
    }
    const auto watch = mask_linear_address(state, state.dr[i]);
    if (watch == instruction_rip) {
      hit_bits |= (1ull << i);
    }
  }
  return hit_bits;
}

[[nodiscard]] std::uint64_t collect_data_breakpoint_hits(CpuState& state, const std::vector<DebugMemoryAccess>& accesses) noexcept {
  std::uint64_t hit_bits = 0;
  for (const auto& access : accesses) {
    hit_bits |= detail::debug_data_breakpoint_hits(state, access.address, access.size, access.is_read, access.is_write);
  }
  return hit_bits;
}

}  // namespace

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

void Executor::request_stop() noexcept {
  stop_requested_ = true;
}

void Executor::clear_stop_request() noexcept {
  stop_requested_ = false;
}

bool Executor::stop_requested() const noexcept {
  return stop_requested_;
}

bool Executor::has_violation() const noexcept {
  return has_violation_;
}

void Executor::clear_violation() noexcept {
  has_violation_ = false;
  violation_ip_ = 0;
  violation_address_ = 0;
  violation_reason_ = StopReason::none;
}

std::uint64_t Executor::violation_ip() const noexcept {
  return violation_ip_;
}

std::uint64_t Executor::violation_address() const noexcept {
  return violation_address_;
}

StopReason Executor::violation_reason() const noexcept {
  return violation_reason_;
}

ExecutionResult Executor::step(CpuState& state, Memory& memory) {
  clear_violation();
  if (collect_code_stats_) { ++total_steps_; }
  if (stop_requested_) {
    const ExecutionResult stopped{StopReason::stop_requested, 0, std::nullopt, std::nullopt};
    ++stop_reason_counts_[stop_reason_to_index(stopped.reason)];
    notify_stop_hooks(state, memory, stopped, state.rip);
    return stopped;
  }
  state.rip = mask_instruction_pointer(state, state.rip);
  state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
  const auto instruction_start_rip = state.rip;
  const auto record_violation = [&](const ExecutionResult& result, std::uint64_t fault_address) {
    if (result.reason == StopReason::none ||
        result.reason == StopReason::halted ||
        result.reason == StopReason::execution_limit ||
        result.reason == StopReason::stop_requested) {
      return;
    }
    has_violation_ = true;
    violation_reason_ = result.reason;
    violation_ip_ = instruction_start_rip;
    violation_address_ = fault_address;
  };
  const auto fault_address_of = [&](const ExecutionResult& result, std::uint64_t fallback) -> std::uint64_t {
    if (result.exception.has_value()) {
      return result.exception->address;
    }
    return fallback;
  };

  for (std::size_t attempt = 0; attempt < kMaxFaultRetries; ++attempt) {
    const auto try_recover_fault = [&](const ExecutionResult& fault, std::uint64_t fault_address) -> bool {
      const auto action = run_fault_hooks(FaultHookEvent{state, memory, fault, instruction_start_rip, fault_address});
      if (trace_semantics_) {
        std::fprintf(stderr,
            "[seven-fault] rip=0x%llx mode=%u reason=%u addr=0x%llx action=%u\n",
            static_cast<unsigned long long>(instruction_start_rip),
            static_cast<unsigned>(state.mode),
            static_cast<unsigned>(fault.reason),
            static_cast<unsigned long long>(fault_address),
            static_cast<unsigned>(action));
      }
      if (action == FaultHookAction::retry) {
        return true;
      }
      if (action == FaultHookAction::restart_instruction) {
        state.rip = instruction_start_rip;
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        return true;
      }
      return false;
    };

    const auto code_epoch = memory.code_epoch();
    const bool can_use_decode_cache = !memory.has_fetch_access_hooks() && std::getenv("SEVEN_DISABLE_DECODE_CACHE") == nullptr;
    const auto cache_index = static_cast<std::size_t>((state.rip >> 1) & (kDecodeCacheSize - 1));
    auto& cache_entry = (*decode_cache_)[cache_index];
    const bool cache_hit = can_use_decode_cache &&
                           cache_entry.valid &&
                           cache_entry.rip == state.rip &&
                           cache_entry.code_epoch == code_epoch &&
                           cache_entry.mode == state.mode;

    if (!cache_hit) [[unlikely]] {
      std::array<std::uint8_t, iced_x86::IcedConstants::_MAX_INSTRUCTION_LENGTH> bytes{};
      const std::uint8_t* decode_bytes = bytes.data();
      std::size_t decode_size = bytes.size();
      bool fetched = false;
      if (can_use_decode_cache) {
        const auto page_base = state.rip & ~static_cast<std::uint64_t>(Memory::kPageSize - 1);
        const auto page_offset = static_cast<std::size_t>(state.rip - page_base);
        if (page_offset + bytes.size() <= Memory::kPageSize) {
          auto& page_cache = (*code_page_cache_)[static_cast<std::size_t>((page_base >> 12) & (kCodePageCacheSize - 1))];
          if (!page_cache.valid || page_cache.page_base != page_base || page_cache.code_epoch != code_epoch) {
            if (memory.read_code_page(page_base, page_cache.bytes.data())) {
              page_cache.page_base = page_base;
              page_cache.code_epoch = code_epoch;
              page_cache.valid = true;
            } else {
              page_cache.valid = false;
            }
          }
          if (page_cache.valid && page_cache.page_base == page_base && page_cache.code_epoch == code_epoch) {
            decode_bytes = page_cache.bytes.data() + page_offset;
            decode_size = Memory::kPageSize - page_offset;
            fetched = true;
          }
        }
      }
      if (!fetched) {
        if (!memory.read(state.rip, bytes.data(), bytes.size(), MemoryAccessKind::instruction_fetch)) {
          const ExecutionResult fault{StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, state.rip, 0}, std::nullopt};
          if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
            continue;
          }
          record_violation(fault, fault_address_of(fault, state.rip));
          ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
          notify_stop_hooks(state, memory, fault, state.rip);
          return fault;
        }
      }

      iced_x86::Decoder decoder(
          decoder_bitness(state.mode),
          std::span<const std::uint8_t>(decode_bytes, decode_size),
          state.rip,
          iced_x86::DecoderOptions::NO_INVALID_CHECK);
      const auto decoded = decoder.decode();
      if (!decoded.has_value()) {
        if (trace_semantics_) {
          if (decode_bytes != bytes.data()) {
            std::memcpy(bytes.data(), decode_bytes, bytes.size());
          }
          std::fprintf(stderr,
              "[seven-decode-fail] rip=0x%llx mode=%u bytes=%02x %02x %02x %02x %02x %02x %02x %02x\n",
              static_cast<unsigned long long>(state.rip),
              static_cast<unsigned>(state.mode),
              static_cast<unsigned>(bytes[0]),
              static_cast<unsigned>(bytes[1]),
              static_cast<unsigned>(bytes[2]),
              static_cast<unsigned>(bytes[3]),
              static_cast<unsigned>(bytes[4]),
              static_cast<unsigned>(bytes[5]),
              static_cast<unsigned>(bytes[6]),
              static_cast<unsigned>(bytes[7]));
        }
        const ExecutionResult fault{StopReason::decode_error, 0, ExceptionInfo{StopReason::decode_error, state.rip, 0}, std::nullopt};
        if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }

      cache_entry.rip = state.rip;
      cache_entry.code_epoch = code_epoch;
      cache_entry.mode = state.mode;
      cache_entry.instr = decoded.value();
      cache_entry.simd_allowed = simd_profile_allows(cache_entry.instr);
      cache_entry.reported_code = normalize_reported_code(cache_entry.instr.code());
      const auto trap = trap_kind_for_code(cache_entry.instr.code());
      cache_entry.trap_kind = trap.has_value() ? static_cast<std::uint8_t>(*trap) : 0xFFu;
      cache_entry.instruction_length = std::max<std::uint32_t>(1u, cache_entry.instr.length());
      cache_entry.valid = can_use_decode_cache;
    }

    const auto& instr = cache_entry.instr;
    const auto next_rip = mask_instruction_pointer(state, state.rip + cache_entry.instruction_length);
    const auto reported_code = cache_entry.reported_code;
    const std::uint64_t dr7 = state.dr[7];
    const bool check_execute_breakpoints = dr7 != 0 && has_enabled_execute_breakpoints(state);
    const bool check_data_breakpoints = dr7 != 0 && has_enabled_data_breakpoints(state);
    const auto debug_memory_accesses = check_data_breakpoints
        ? collect_debug_memory_accesses(state, instr)
        : std::vector<DebugMemoryAccess>{};

    if (instr.code() == iced_x86::Code::INVALID) {
      TrapHookContext trap_ctx{state, memory, instr, next_rip, TrapKind::invalid_opcode};
      const auto trap_result = run_trap_hooks(trap_ctx);
      if (trap_result.action == TrapHookAction::handled) {
        if (collect_code_stats_) { ++total_retired_; }
        state.rip = next_rip;
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        return {StopReason::none, 1, std::nullopt, iced_x86::Code::INVALID};
      }
      if (trap_result.action == TrapHookAction::stop) {
        const auto fault = trap_result.stop_result.value_or(
            ExecutionResult{StopReason::invalid_opcode, 0, ExceptionInfo{StopReason::invalid_opcode, state.rip, 0}, iced_x86::Code::INVALID});
        if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }
      const ExecutionResult fault{StopReason::invalid_opcode, 0, ExceptionInfo{StopReason::invalid_opcode, state.rip, 0}, iced_x86::Code::INVALID};
      if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
        continue;
      }
      record_violation(fault, fault_address_of(fault, state.rip));
      ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
      notify_stop_hooks(state, memory, fault, state.rip);
      return fault;
    }

    if (trace_strrchr_) {
      maybe_trace_strrchr(state, memory);
    }

    if (!cache_entry.simd_allowed) {
      const ExecutionResult fault{StopReason::unsupported_instruction, 0, ExceptionInfo{StopReason::unsupported_instruction, state.rip, 0}, instr.code()};
      if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
        continue;
      }
      record_violation(fault, fault_address_of(fault, state.rip));
      ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
      notify_stop_hooks(state, memory, fault, state.rip);
      return fault;
    }

    const bool rf_suppressed = (state.rflags & kFlagRF) != 0;
    if (rf_suppressed) {
      state.rflags &= ~kFlagRF;
    }
    const bool debug_suppressed = state.debug_suppression != 0;
    if (debug_suppressed) {
      state.debug_suppression = 0;
    }
    const bool tf_active = (state.rflags & kFlagTF) != 0;
    if (!rf_suppressed && !debug_suppressed && check_execute_breakpoints) {
      const auto exec_hit_bits = collect_execute_breakpoint_hits(state, instruction_start_rip);
      if (exec_hit_bits != 0) {
        state.dr[6] |= exec_hit_bits;
        ExecutionContext db_ctx{state, memory, instr, next_rip, false};
        const auto db_result = detail::dispatch_interrupt(db_ctx, 1u, instruction_start_rip);
        if (db_result.reason != StopReason::none) {
          if (try_recover_fault(db_result, fault_address_of(db_result, state.rip))) {
            continue;
          }
          record_violation(db_result, fault_address_of(db_result, state.rip));
          ++stop_reason_counts_[stop_reason_to_index(db_result.reason)];
          notify_stop_hooks(state, memory, db_result, state.rip);
          return db_result;
        }
        state.rip = mask_instruction_pointer(state, state.rip);
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        return {StopReason::none, 0, std::nullopt, reported_code};
      }
    }

    if (cache_entry.trap_kind != 0xFFu) {
      const auto trap_kind = static_cast<TrapKind>(cache_entry.trap_kind);
      if (trace_openkey_probe_) {
        maybe_trace_openkey_probe(state, memory);
      }
      TrapHookContext trap_ctx{state, memory, instr, next_rip, trap_kind};
      const auto trap_result = run_trap_hooks(trap_ctx);
      if (trap_result.action == TrapHookAction::handled) {
        if (state.rip == instruction_start_rip) {
          state.rip = next_rip;
        } else {
          state.rip = mask_instruction_pointer(state, state.rip);
        }
        state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
        if (collect_code_stats_ && static_cast<std::size_t>(instr.code()) < code_execution_counts_.size()) {
          ++code_execution_counts_[static_cast<std::size_t>(instr.code())];
        }
        if (collect_code_stats_) { ++total_retired_; }
        return {StopReason::none, 1, std::nullopt, reported_code};
      }
      if (trap_result.action == TrapHookAction::stop) {
        const auto fault = trap_result.stop_result.value_or(
            ExecutionResult{StopReason::general_protection, 0, ExceptionInfo{StopReason::general_protection, state.rip, 0}, instr.code()});
        if (try_recover_fault(fault, fault_address_of(fault, state.rip))) {
          continue;
        }
        record_violation(fault, fault_address_of(fault, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
        notify_stop_hooks(state, memory, fault, state.rip);
        return fault;
      }
    }

    ExecutionContext ctx{state, memory, instr, next_rip, false};
    if (has_execution_hooks_) {
      for (auto& [id, hook] : execution_hooks_) {
        (void)id;
        hook(state.rip);
      }
    }
    if (has_execution_address_hooks_) {
      const auto exec_addr_it = execution_address_hooks_.find(state.rip);
      if (exec_addr_it != execution_address_hooks_.end()) {
        for (auto& [id, hook] : exec_addr_it->second) {
          (void)id;
          hook(state.rip);
        }
      }
    }
    const bool need_instruction_hooks = has_instruction_hooks_ || has_code_hooks_;
    InstructionHookContext hook_ctx{state, memory, instr, next_rip};
    ExecutionResult hook_stop_result{};
    const auto hook_action = need_instruction_hooks
        ? run_instruction_hooks(hook_ctx, hook_stop_result)
        : InstructionHookAction::continue_to_core;
    if (hook_action == InstructionHookAction::stop) {
      if (!hook_stop_result.code.has_value()) {
        hook_stop_result.code = reported_code;
      }
      if (hook_stop_result.reason != StopReason::none) {
        if (try_recover_fault(hook_stop_result, fault_address_of(hook_stop_result, state.rip))) {
          continue;
        }
        record_violation(hook_stop_result, fault_address_of(hook_stop_result, state.rip));
        ++stop_reason_counts_[stop_reason_to_index(hook_stop_result.reason)];
        notify_stop_hooks(state, memory, hook_stop_result, state.rip);
      }
      return hook_stop_result;
    }
    if (hook_action == InstructionHookAction::skip_core) {
      if (state.rip == instruction_start_rip) {
        state.rip = next_rip;
      } else {
        state.rip = mask_instruction_pointer(state, state.rip);
      }
      state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
      if (collect_code_stats_ && static_cast<std::size_t>(instr.code()) < code_execution_counts_.size()) {
        ++code_execution_counts_[static_cast<std::size_t>(instr.code())];
      }
      if (collect_code_stats_) { ++total_retired_; }
      return {StopReason::none, 1, std::nullopt, reported_code};
    }

    ExecutionResult result{};
    const auto code = instr.code();
    switch (code) {
#define KUBERA_CODE(code) \
    case iced_x86::Code::code: result = handlers::handle_code_##code(ctx); break;
#include "seven/handled_codes.def"
#undef KUBERA_CODE
      default:
        result = unsupported(ctx);
        break;
    }
    result.code = reported_code;
    if (result.reason == StopReason::none) {
      if (collect_code_stats_ && static_cast<std::size_t>(code) < code_execution_counts_.size()) {
        ++code_execution_counts_[static_cast<std::size_t>(code)];
      }
      if (collect_code_stats_) { ++total_retired_; }
      const bool current_instruction_set_shadow = state.debug_suppression != 0;
      state.rip = ctx.control_flow_taken ? mask_instruction_pointer(state, state.rip) : ctx.next_rip;
      state.gpr[4] = mask_stack_pointer(state, state.gpr[4]);
      result.retired = 1;
      if (!current_instruction_set_shadow && state.pending_debug_hit_bits == 0 && !check_data_breakpoints &&
          !state.pending_single_step && !tf_active) {
        return result;
      }
      const auto current_data_hit_bits = ctx.debug_hit_bits != 0
                                             ? ctx.debug_hit_bits
                                             : collect_data_breakpoint_hits(state, debug_memory_accesses);
      if (current_instruction_set_shadow) {
        state.pending_debug_hit_bits |= current_data_hit_bits;
        return result;
      }

      const auto data_hit_bits = state.pending_debug_hit_bits | current_data_hit_bits;
      const bool tf_hit = state.pending_single_step || (!debug_suppressed && tf_active);
      if (data_hit_bits != 0 || tf_hit) {
        if (tf_hit) {
          state.dr[6] |= (1ull << 14);
          state.pending_single_step = false;
        }
        state.dr[6] |= data_hit_bits;
        state.pending_debug_hit_bits = 0;
        const auto return_rip = state.rip;
        ExecutionContext db_ctx{state, memory, instr, return_rip, false};
        const auto db_result = detail::dispatch_interrupt(db_ctx, 1u, return_rip, std::nullopt, ctx.push_rf_for_debug);
        if (db_result.reason != StopReason::none) {
          record_violation(db_result, fault_address_of(db_result, state.rip));
          ++stop_reason_counts_[stop_reason_to_index(db_result.reason)];
          notify_stop_hooks(state, memory, db_result, state.rip);
          return db_result;
        }
      }
      return result;
    }
    if (try_recover_fault(result, fault_address_of(result, state.rip))) {
      continue;
    }
    if (debug_suppressed) {
      state.pending_single_step = false;
      state.pending_debug_hit_bits = 0;
    }
    record_violation(result, fault_address_of(result, state.rip));
    ++stop_reason_counts_[stop_reason_to_index(result.reason)];
    notify_stop_hooks(state, memory, result, state.rip);
    return result;
  }

  const ExecutionResult fault{StopReason::execution_limit, 0, std::nullopt, std::nullopt};
  ++stop_reason_counts_[stop_reason_to_index(fault.reason)];
  notify_stop_hooks(state, memory, fault, state.rip);
  return fault;
}

ExecutionResult Executor::run(CpuState& state, Memory& memory, std::size_t max_instructions) {
  ExecutionResult last{};
  for (std::size_t i = 0; i < max_instructions; ++i) {
    if (stop_requested_) {
      const ExecutionResult stopped{StopReason::stop_requested, i, std::nullopt, std::nullopt};
      ++stop_reason_counts_[stop_reason_to_index(stopped.reason)];
      notify_stop_hooks(state, memory, stopped, state.rip);
      return stopped;
    }
    last = step(state, memory);
    if (last.reason != StopReason::none) {
      last.retired += i;
      return last;
    }
  }
  ++stop_reason_counts_[stop_reason_to_index(StopReason::execution_limit)];
  const ExecutionResult limit{StopReason::execution_limit, max_instructions, std::nullopt, std::nullopt};
  notify_stop_hooks(state, memory, limit, state.rip);
  return limit;
}

Executor::HookId Executor::add_instruction_hook(InstructionHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      instruction_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    instruction_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_code_hook(iced_x86::Code code, InstructionHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, code, hook = std::move(hook)]() mutable {
      code_hooks_[code].emplace_back(id, std::move(hook));
    });
  } else {
    code_hooks_[code].emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_execution_hook(std::function<void(std::uint64_t)> hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      execution_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    execution_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_execution_hook(std::uint64_t address, std::function<void(std::uint64_t)> hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, address, hook = std::move(hook)]() mutable {
      execution_address_hooks_[address].emplace_back(id, std::move(hook));
    });
  } else {
    execution_address_hooks_[address].emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_stop_hook(StopHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      stop_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    stop_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_fault_hook(FaultHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, hook = std::move(hook)]() mutable {
      fault_hooks_.emplace_back(id, std::move(hook));
    });
  } else {
    fault_hooks_.emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

Executor::HookId Executor::add_trap_hook(TrapKind kind, TrapHook hook) {
  const auto id = next_hook_id_++;
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this, id, kind, hook = std::move(hook)]() mutable {
      trap_hooks_[kind].emplace_back(id, std::move(hook));
    });
  } else {
    trap_hooks_[kind].emplace_back(id, std::move(hook));
    refresh_hook_flags();
  }
  return id;
}

bool Executor::remove_hook(HookId id) {
  auto remove_now = [this, id]() {
    for (auto it = execution_hooks_.begin(); it != execution_hooks_.end(); ++it) {
      if (it->first == id) {
        execution_hooks_.erase(it);
        return true;
      }
    }
    for (auto map_it = execution_address_hooks_.begin(); map_it != execution_address_hooks_.end(); ++map_it) {
      auto& hooks = map_it->second;
      for (auto it = hooks.begin(); it != hooks.end(); ++it) {
        if (it->first == id) {
          hooks.erase(it);
          if (hooks.empty()) {
            execution_address_hooks_.erase(map_it);
          }
          return true;
        }
      }
    }
    for (auto it = instruction_hooks_.begin(); it != instruction_hooks_.end(); ++it) {
      if (it->first == id) {
        instruction_hooks_.erase(it);
        return true;
      }
    }
    for (auto& [code, hooks] : code_hooks_) {
      (void)code;
      for (auto it = hooks.begin(); it != hooks.end(); ++it) {
        if (it->first == id) {
          hooks.erase(it);
          return true;
        }
      }
    }
    for (auto it = stop_hooks_.begin(); it != stop_hooks_.end(); ++it) {
      if (it->first == id) {
        stop_hooks_.erase(it);
        return true;
      }
    }
    for (auto it = fault_hooks_.begin(); it != fault_hooks_.end(); ++it) {
      if (it->first == id) {
        fault_hooks_.erase(it);
        return true;
      }
    }
    for (auto& [kind, hooks] : trap_hooks_) {
      (void)kind;
      for (auto it = hooks.begin(); it != hooks.end(); ++it) {
        if (it->first == id) {
          hooks.erase(it);
          return true;
        }
      }
    }
    return false;
  };

  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([remove_now]() mutable { (void)remove_now(); });
    return true;
  }
  const bool removed = remove_now();
  refresh_hook_flags();
  return removed;
}

void Executor::clear_hooks() {
  if (dispatching_hooks_) {
    pending_hook_mutations_.push_back([this]() {
      execution_hooks_.clear();
      execution_address_hooks_.clear();
      instruction_hooks_.clear();
      code_hooks_.clear();
      stop_hooks_.clear();
      fault_hooks_.clear();
      trap_hooks_.clear();
      pending_hook_mutations_.clear();
    });
    return;
  }
  execution_hooks_.clear();
  execution_address_hooks_.clear();
  instruction_hooks_.clear();
  code_hooks_.clear();
  stop_hooks_.clear();
  fault_hooks_.clear();
  trap_hooks_.clear();
  pending_hook_mutations_.clear();
  refresh_hook_flags();
}

InstructionHookAction Executor::run_instruction_hooks(InstructionHookContext& ctx, ExecutionResult& stop_result) {
  if (!pending_hook_mutations_.empty()) {
    apply_pending_hook_mutations();
  }

  const auto it = code_hooks_.find(ctx.instr.code());
  const bool has_code_hooks = it != code_hooks_.end() && !it->second.empty();
  if (instruction_hooks_.empty() && !has_code_hooks) {
    return InstructionHookAction::continue_to_core;
  }

  dispatching_hooks_ = true;
  for (auto& [id, hook] : instruction_hooks_) {
    (void)id;
    const auto result = hook(ctx);
    if (result.action == InstructionHookAction::stop) {
      dispatching_hooks_ = false;
      apply_pending_hook_mutations();
      stop_result = result.stop_result.value_or(ExecutionResult{StopReason::unsupported_instruction, 0, std::nullopt, ctx.instr.code()});
      return result.action;
    }
    if (result.action == InstructionHookAction::skip_core) {
      dispatching_hooks_ = false;
      apply_pending_hook_mutations();
      return result.action;
    }
  }
  if (has_code_hooks) {
    for (auto& [id, hook] : it->second) {
      (void)id;
      const auto result = hook(ctx);
      if (result.action == InstructionHookAction::stop) {
        dispatching_hooks_ = false;
        apply_pending_hook_mutations();
        stop_result = result.stop_result.value_or(ExecutionResult{StopReason::unsupported_instruction, 0, std::nullopt, ctx.instr.code()});
        return result.action;
      }
      if (result.action == InstructionHookAction::skip_core) {
        dispatching_hooks_ = false;
        apply_pending_hook_mutations();
        return result.action;
      }
    }
  }
  dispatching_hooks_ = false;
  apply_pending_hook_mutations();
  return InstructionHookAction::continue_to_core;
}

TrapHookResult Executor::run_trap_hooks(TrapHookContext& ctx) {
  const auto it = trap_hooks_.find(ctx.kind);
  if (it == trap_hooks_.end()) {
    return {};
  }
  dispatching_hooks_ = true;
  for (auto& [id, hook] : it->second) {
    (void)id;
    const auto result = hook(ctx);
    if (result.action != TrapHookAction::continue_to_core) {
      dispatching_hooks_ = false;
      apply_pending_hook_mutations();
      return result;
    }
  }
  dispatching_hooks_ = false;
  apply_pending_hook_mutations();
  return {};
}

FaultHookAction Executor::run_fault_hooks(const FaultHookEvent& event) {
  if (fault_hooks_.empty()) {
    return FaultHookAction::stop;
  }
  dispatching_hooks_ = true;
  for (auto& [id, hook] : fault_hooks_) {
    (void)id;
    const auto action = hook(event);
    if (action != FaultHookAction::stop) {
      dispatching_hooks_ = false;
      apply_pending_hook_mutations();
      return action;
    }
  }
  dispatching_hooks_ = false;
  apply_pending_hook_mutations();
  return FaultHookAction::stop;
}

void Executor::notify_stop_hooks(CpuState& state, Memory& memory, const ExecutionResult& result, std::uint64_t fault_address) const {
  if (stop_hooks_.empty()) {
    return;
  }
  auto& self = const_cast<Executor&>(*this);
  self.dispatching_hooks_ = true;
  const StopHookEvent event{state, memory, result, fault_address};
  for (const auto& [id, hook] : stop_hooks_) {
    (void)id;
    hook(event);
  }
  self.dispatching_hooks_ = false;
  self.apply_pending_hook_mutations();
}

void Executor::apply_pending_hook_mutations() {
  if (pending_hook_mutations_.empty()) {
    return;
  }
  auto queued = std::move(pending_hook_mutations_);
  pending_hook_mutations_.clear();
  for (auto& mutation : queued) {
    mutation();
  }
  refresh_hook_flags();
}

void Executor::refresh_hook_flags() noexcept {
  has_instruction_hooks_ = !instruction_hooks_.empty();
  has_code_hooks_ = false;
  for (const auto& [code, hooks] : code_hooks_) {
    (void)code;
    if (!hooks.empty()) {
      has_code_hooks_ = true;
      break;
    }
  }
  has_execution_hooks_ = !execution_hooks_.empty();
  has_execution_address_hooks_ = !execution_address_hooks_.empty();
  has_stop_hooks_ = !stop_hooks_.empty();
  has_fault_hooks_ = !fault_hooks_.empty();
  has_trap_hooks_ = false;
  for (const auto& [kind, hooks] : trap_hooks_) {
    (void)kind;
    if (!hooks.empty()) {
      has_trap_hooks_ = true;
      break;
    }
  }
}

std::size_t Executor::supported_code_count() const noexcept {
#define KUBERA_CODE(code) +1
  static constexpr std::size_t kSupportedCodeCount = 0
#include "seven/handled_codes.def"
      ;
#undef KUBERA_CODE
  return kSupportedCodeCount;
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

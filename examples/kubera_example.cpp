#include <cctype>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "seven/compat.hpp"

void write_bytes(seven::Memory& memory, std::uint64_t base, const std::vector<std::uint8_t>& bytes) {
  memory.map(base, bytes.size());
  memory.write(base, bytes.data(), bytes.size());
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
  std::uint64_t parsed = 0;
  int base = 10;
  std::string_view trim = text;
  while (!trim.empty() && std::isspace(static_cast<unsigned char>(trim.front()))) {
    trim.remove_prefix(1);
  }
  while (!trim.empty() && std::isspace(static_cast<unsigned char>(trim.back()))) {
    trim.remove_suffix(1);
  }
  if (trim.empty()) {
    return false;
  }
  if (trim.size() > 2 && trim[0] == '0' && (trim[1] == 'x' || trim[1] == 'X')) {
    base = 16;
    trim = trim.substr(2);
  }
  const auto result = std::from_chars(trim.data(), trim.data() + trim.size(), parsed, base);
  return result.ec == std::errc{} && result.ptr == trim.data() + trim.size();
}

void usage(std::string_view app) {
  std::cout << "Usage:\n";
  std::cout << "  " << app << " [--steps N] [--rip 0x1000] [--rsp 0x2000] [--mode long|compat|real] [--trace]\n";
  std::cout << "      [--hex <bytes>|--file <path>]\n";
  std::cout << "If --hex/--file is omitted, a default MOV+ADD sample program is used.\n";
}

std::string stop_reason_name(seven::StopReason reason) {
  switch (reason) {
    case seven::StopReason::none:
      return "none";
    case seven::StopReason::halted:
      return "halted";
    case seven::StopReason::invalid_opcode:
      return "invalid_opcode";
    case seven::StopReason::unsupported_instruction:
      return "unsupported_instruction";
    case seven::StopReason::page_fault:
      return "page_fault";
    case seven::StopReason::general_protection:
      return "general_protection";
    case seven::StopReason::decode_error:
      return "decode_error";
    case seven::StopReason::execution_limit:
      return "execution_limit";
    default:
      return "unknown";
  }
}

bool load_hex(const std::vector<std::string>& args, std::size_t start, std::vector<std::uint8_t>& code) {
  std::ostringstream stream;
  for (std::size_t i = start; i < args.size(); ++i) {
    if (!stream.str().empty()) {
      stream << ' ';
    }
    stream << args[i];
  }
  try {
    code = seven::parse_hex_bytes(stream.str());
  } catch (...) {
    return false;
  }
  return !code.empty();
}

bool load_file(const std::string& path, std::vector<std::uint8_t>& code) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return false;
  }
  const auto size = static_cast<std::size_t>(input.tellg());
  input.seekg(0, std::ios::beg);
  code.resize(size);
  if (size > 0 && !input.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size))) {
    return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  constexpr std::uint64_t kDefaultRip = 0x1000;
  constexpr std::uint64_t kDefaultRsp = 0x2000;
  constexpr std::size_t kDefaultSteps = 128;
  constexpr std::uint64_t kDefaultRflags = 0x202;

  std::vector<std::string> args(argv + 1, argv + argc);
  bool trace = false;
  bool use_hex = false;
  bool use_file = false;
  std::vector<std::uint8_t> code = {
      0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x48, 0x83, 0xC0, 0x01
  };
  std::uint64_t entry = kDefaultRip;
  std::uint64_t rsp = kDefaultRsp;
  std::size_t max_steps = kDefaultSteps;
  seven::ExecutionMode mode = seven::ExecutionMode::long64;

  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--steps") {
      if (i + 1 >= args.size()) {
        usage(argv[0]);
        return 1;
      }
      std::uint64_t parsed = 0;
      if (!parse_u64(args[i + 1], parsed) || parsed == 0) {
        std::cout << "Invalid --steps value: " << args[i + 1] << "\n";
        return 1;
      }
      max_steps = static_cast<std::size_t>(parsed);
      ++i;
    } else if (args[i] == "--rip") {
      if (i + 1 >= args.size()) {
        usage(argv[0]);
        return 1;
      }
      if (!parse_u64(args[i + 1], entry)) {
        std::cout << "Invalid --rip value: " << args[i + 1] << "\n";
        return 1;
      }
      ++i;
    } else if (args[i] == "--rsp") {
      if (i + 1 >= args.size()) {
        usage(argv[0]);
        return 1;
      }
      if (!parse_u64(args[i + 1], rsp)) {
        std::cout << "Invalid --rsp value: " << args[i + 1] << "\n";
        return 1;
      }
      ++i;
    } else if (args[i] == "--trace") {
      trace = true;
    } else if (args[i] == "--mode") {
      if (i + 1 >= args.size()) {
        usage(argv[0]);
        return 1;
      }
      if (args[i + 1] == "long") {
        mode = seven::ExecutionMode::long64;
      } else if (args[i + 1] == "compat") {
        mode = seven::ExecutionMode::compat32;
      } else if (args[i + 1] == "real") {
        mode = seven::ExecutionMode::real16;
      } else {
        std::cout << "Invalid --mode value: " << args[i + 1] << "\n";
        return 1;
      }
      ++i;
    } else if (args[i] == "--hex") {
      if (i + 1 >= args.size()) {
        usage(argv[0]);
        return 1;
      }
      if (!load_hex(args, i + 1, code)) {
        std::cout << "Invalid hex sequence.\n";
        return 1;
      }
      use_hex = true;
      break;
    } else if (args[i] == "--file") {
      if (i + 1 >= args.size()) {
        usage(argv[0]);
        return 1;
      }
      if (!load_file(args[i + 1], code)) {
        std::cout << "Failed to read file: " << args[i + 1] << "\n";
        return 1;
      }
      use_file = true;
      ++i;
    } else if (args[i] == "--help" || args[i] == "-h") {
      usage(argv[0]);
      return 0;
    } else if (!use_hex && !use_file && args[i].size() > 0 && args[i][0] != '-') {
      code.clear();
      const std::vector<std::string> hex_args(args.begin() + static_cast<std::ptrdiff_t>(i), args.end());
      if (!load_hex(hex_args, 0, code)) {
        std::cout << "Invalid positional hex input: " << args[i] << "\n";
        return 1;
      }
      break;
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  if (code.empty()) {
    std::cout << "No instruction bytes provided.\n";
    usage(argv[0]);
    return 1;
  }

  seven::StandaloneMachine machine;
  machine.state.mode = mode;
  machine.state.rip = entry;
  machine.state.rflags = kDefaultRflags;
  machine.state.gpr[4] = rsp;

  const auto stack_base = (rsp > 0x1000 ? rsp - 0x1000 : 0);
  machine.memory.map(entry, code.size() + 0x100);
  machine.memory.map(stack_base, 0x1000);
  write_bytes(machine.memory, machine.state.rip, code);

  for (std::size_t step = 0; step < max_steps; ++step) {
    const auto result = machine.executor.step(machine.state, machine.memory);
    if (trace) {
      std::cout << "step=" << std::dec << step << " rip=0x" << std::hex << machine.state.rip << " reason="
                << stop_reason_name(result.reason) << "\n";
    }
    if (result.reason != seven::StopReason::none) {
      std::cout << "Stopped: reason=" << stop_reason_name(result.reason) << " retired=" << std::dec << result.retired << "\n";
      std::cout << "final_rip=0x" << std::hex << machine.state.rip << "\n";
      std::cout << "rax=0x" << std::hex << machine.state.gpr[0] << " rsp=0x" << machine.state.gpr[4] << "\n";
      return static_cast<int>(result.reason != seven::StopReason::none);
    }
  }

  std::cout << "Stopped: reason=" << stop_reason_name(seven::StopReason::execution_limit) << " retired=" << max_steps << "\n";
  std::cout << "final_rip=0x" << std::hex << machine.state.rip << "\n";
  std::cout << "rax=0x" << std::hex << machine.state.gpr[0] << " rsp=0x" << machine.state.gpr[4] << "\n";
  return 0;
}


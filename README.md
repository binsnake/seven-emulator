# seven

An x86 emulator library written in C++23. Decodes and executes x86 machine code in long (64-bit), compatibility (32-bit), and real (16-bit) modes. SIMD support is configurable between AVX, AVX2, and AVX512 profiles.

## Building

**Requirements:**
- CMake 3.24+
- C++23 compiler (MSVC 2022, GCC 13+, Clang 16+)
- Boost.Multiprecision headers (for wide integer and x87 float types)

```sh
cmake -B build -DSEVEN_SIMD_PROFILE=AVX512 -DBOOST_ROOT=/path/to/boost
cmake --build build
```

**CMake options:**

| Option | Default | Description |
|-|-|-|
| `SEVEN_SIMD_PROFILE` | `AVX512` | SIMD register width: `AVX` (128-bit), `AVX2` (256-bit), `AVX512` (512-bit) |
| `SEVEN_ENABLE_LTO` | `ON` | Enable link-time optimization |
| `SEVEN_ENABLE_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |

Google Test is fetched automatically if not found on the system.

## Library API

The primary targets are:
- **`seven_core`** — the emulator static library
- **`iced_x86`** — the bundled x86 decoder (linked transitively)

All public types live in the `seven` namespace. Include `seven/compat.hpp` for the `StandaloneMachine` convenience wrapper, or include individual headers for fine-grained use.

### Quick start

```cpp
#include "seven/compat.hpp"

// MOV RAX, 5 / ADD RAX, 1
std::vector<std::uint8_t> code = {
    0x48, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x48, 0x83, 0xC0, 0x01
};

seven::StandaloneMachine machine;
machine.state.mode   = seven::ExecutionMode::long64;
machine.state.rip    = 0x1000;
machine.state.rflags = 0x202;
machine.state.gpr[4] = 0x2000; // RSP

machine.memory.map(0x1000, code.size() + 0x100);
machine.memory.map(0x1000, 0x1000); // stack
machine.memory.write(0x1000, code.data(), code.size());

auto result = machine.executor.run(machine.state, machine.memory, /*max_steps=*/128);
// machine.state.gpr[0] == 6 (RAX)
```

### Execution modes

```cpp
seven::ExecutionMode::long64    // 64-bit long mode (default)
seven::ExecutionMode::compat32  // 32-bit compatibility mode
seven::ExecutionMode::real16    // 16-bit real mode
```

### Step vs. run

```cpp
// Single instruction
seven::ExecutionResult r = executor.step(state, memory);

// Up to N instructions
seven::ExecutionResult r = executor.run(state, memory, /*max_instructions=*/1000);

// Stop from another thread
executor.request_stop();
```

### Stop reasons

```cpp
enum class StopReason : std::uint8_t {
    none,
    halted,
    invalid_opcode,
    unsupported_instruction,
    floating_point_exception,
    page_fault,
    divide_error,
    general_protection,
    decode_error,
    execution_limit,
    stop_requested,
};
```

### Hooks

```cpp
// Fire before every instruction
auto id = executor.add_instruction_hook([](seven::InstructionHookContext& ctx) {
    return seven::InstructionHookAction::proceed;
});

// Fire only for a specific opcode
executor.add_code_hook(iced_x86::Code::Syscall, handler);

// Fire at a specific RIP
executor.add_execution_hook(0x401000, [](std::uint64_t rip) { /* ... */ });

// Fire on faults
executor.add_fault_hook([](const seven::FaultHookEvent& e) {
    return seven::FaultHookAction::propagate;
});

executor.remove_hook(id);
executor.clear_hooks();
```

### Statistics

```cpp
executor.total_steps();           // decode attempts
executor.total_retired();         // successfully executed instructions
executor.code_execution_counts(); // per-opcode counters (indexed by iced_x86::Code)
executor.stop_reason_counts();    // per-StopReason counters
executor.reset_stats();
```

### CPU state

`CpuState` exposes the full architectural state:

```cpp
state.gpr[0..15]      // RAX–R15
state.rip
state.rflags
state.vectors[0..31]  // XMM/YMM/ZMM (width set by SIMD profile)
state.opmask[0..7]    // AVX-512 k registers
state.mmx[0..7]
state.x87_stack[0..7] // 80-bit floats via Boost.Multiprecision
state.sreg[0..5]      // ES, CS, SS, DS, FS, GS
state.cr[0..15]
state.dr[0..15]
state.msr             // std::unordered_map<uint32_t, uint64_t>
state.fs_base / gs_base
state.gdtr / idtr
```

## CLI tool

`seven_example` executes raw x86 bytes from the command line:

```
seven_example [--steps N] [--rip ADDR] [--rsp ADDR] [--mode long|compat|real] [--trace]
              [--hex <bytes> | --file <path>]
```

**Examples:**

```sh
# Execute hex bytes (MOV RAX,5 / ADD RAX,1) in long mode
seven_example --hex "48 B8 05 00 00 00 00 00 00 00 48 83 C0 01"

# Load raw binary and trace each step
seven_example --file shellcode.bin --trace --steps 64

# Run in 32-bit compatibility mode with a custom entry point
seven_example --mode compat --rip 0x8048000 --file code.bin

# Positional hex shorthand
seven_example 48 31 C0 C3
```

Output reports the stop reason, instruction count, final RIP, RAX, and RSP.

## Instruction coverage

1,721 instruction encodings are handled, spanning:

| Category | Instructions |
|-|-|
| Arithmetic | ADD, SUB, ADC, SBB, MUL, IMUL, DIV, IDIV, CMP, NEG, INC, DEC, XADD, DAA, DAS, AAA, AAS, AAM, AAD, AADD |
| Shift / rotate | SHL, SHR, SAR, ROL, ROR, RCL, RCR, SHLD, SHRD |
| Data movement | MOV, MOVZX, MOVSX, MOVSXD, MOVBE, MOVDIR64B, XCHG, XLAT, LEA, PUSH, POP |
| String ops | MOVS, CMPS, SCAS, LODS, STOS (with REP/REPNE) |
| Control flow | JMP, CALL, RET, Jcc (all conditions), LOOP/LOOPE/LOOPNE, ENTER, LEAVE |
| Bit manipulation | BSF, BSR, BSWAP, BT, BTC, BTR, BTS, POPCNT, LZCNT, TZCNT |
| BMI / BMI2 | ANDN, BEXTR, BLSI, BLSMSK, BLSR, BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX, SHRX |
| SIMD (SSE–AVX-512) | Integer, floating-point, pack/unpack, shuffle, move variants |
| x87 FPU | Arithmetic, compare, load/store, transcendental, control |
| MMX | Integer SIMD on mm0–mm7 |
| System | SYSCALL, SYSENTER/SYSEXIT, SYSRET, RDMSR, WRMSR, RDTSC, RDTSCP, RDPMC, CPUID, SWAPGS, CLTS, INVD, WBINVD, XSETBV |
| Synchronization | LOCK prefix, CMPXCHG, CMPXCHG8B/16B, XADD, PAUSE, LFENCE, MFENCE, SFENCE |
| Misc | NOP (all forms), CLFLUSH, CLFLUSHOPT, CLWB, PREFETCH, UD0/UD1/UD2, BOUND, ARPL, CBW/CWDE/CDQE, CWD/CDQ/CQO, CMOV, SETcc |

## Testing

```sh
ctest --test-dir build
```

Test suites:
- **`seven_tests`** — scalar instructions, SIMD, debug/trace behavior
- **`seven_bmi2_tests`** — comprehensive BMI2 instruction tests

## Dependencies

| Dependency | Source |
|-|-|
| [iced-x86](https://github.com/icedland/iced) | Bundled in `iced_x86/` |
| [Boost.Multiprecision](https://www.boost.org/libs/multiprecision/) | External (headers only) |
| [Google Test 1.14](https://github.com/google/googletest) | Fetched by CMake if absent |

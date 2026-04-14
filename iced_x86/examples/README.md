# Examples

This directory contains small programs covering the primary `iced_x86` components.

## Included examples

- `decoder_stream.cpp` decode a stream and format each instruction.
- `decoder_error_handling.cpp` handle decode failures safely.
- `encoder_roundtrip.cpp` decode/encode byte-exact roundtrip.
- `encoder_from_scratch.cpp` build an instruction and encode it.
- `formatter_styles.cpp` compare Intel/MASM/NASM/GAS/Fast formatters.
- `instruction_info.cpp` inspect register/memory access and flow metadata.
- `opcode_info.cpp` query opcode metadata from `OpCodeInfo`.
- `block_encoder_relocate.cpp` re-encode instruction blocks with metadata.
- `code_assembler_labels.cpp` use `CodeAssembler` with named/anonymous labels.

## Build examples

```powershell
cmake -S . -B build -DICEDCPP_BUILD_EXAMPLES=ON
cmake --build build --config Debug
```

Example binaries are generated as:

- `build/bin/icedcpp_example_<name>.exe` on Windows

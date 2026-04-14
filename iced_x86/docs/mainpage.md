# icedcpp API Documentation

`icedcpp` packages the `iced_x86` C++ implementation with CMake-first build and test workflows.

## Modules

- Decoder: instruction decoding and error reporting.
- Encoder: machine-code emission and relocation-aware branch encoding.
- Formatters: Intel, AT&T/GAS, MASM, NASM, and fast formatter variants.
- Metadata: opcode, register, and memory-size introspection helpers.
- Examples: runnable samples in `examples/` covering all primary components.

## Include Root

Use headers from:

- `include/iced_x86/`

Internal implementation headers are available under:

- `include/iced_x86/internal/`

## Build Docs

Enable docs in CMake and run one of the docs targets:

```powershell
cmake -S . -B build -DICEDCPP_BUILD_DOCS=ON
cmake --build build --config Debug --target docs
```

Useful targets:

- `docs_html` generate HTML only
- `docs_xml` generate XML output
- `docs_clean` remove generated documentation

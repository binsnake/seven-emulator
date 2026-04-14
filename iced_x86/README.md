# icedcpp

`icedcpp` is a C++ adaptation of the Rust [`iced_x86`](https://github.com/icedland/iced) project.

It provides decoder, encoder, and formatter APIs with a CMake-first build, shared-library support, and a large regression test suite focused on encoding correctness.

## Status

- Work in progress.
- API and structure are stabilizing, but expect some iteration.
- Current performance target is pragmatic: typically between Zydis and the original Rust implementation, depending on workload and build settings.

## Quick Usage

Decode and format one instruction:

```cpp
#include "iced_x86/decoder.hpp"
#include "iced_x86/intel_formatter.hpp"
#include "iced_x86/formatter_output.hpp"

#include <cstdint>
#include <string>
#include <vector>

int main() {
    const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0xD8}; // mov rax, rbx
    iced_x86::Decoder decoder(64, bytes, 0x1000);
    auto decoded = decoder.decode();
    if (!decoded.has_value()) {
        return 1;
    }

    std::string text;
    iced_x86::StringFormatterOutput out(text);
    iced_x86::IntelFormatter fmt;
    fmt.format(decoded.value(), out);

    // text == "mov rax,rbx" (formatter style/options dependent)
    return 0;
}
```

Encode roundtrip:

```cpp
#include "iced_x86/decoder.hpp"
#include "iced_x86/encoder.hpp"

#include <cstdint>
#include <vector>

int main() {
    const std::vector<std::uint8_t> bytes = {0x90}; // nop
    iced_x86::Decoder decoder(64, bytes, 0x2000);
    auto decoded = decoder.decode();
    if (!decoded.has_value()) return 1;

    iced_x86::Encoder encoder(64);
    auto encoded = encoder.encode(decoded.value(), 0x2000);
    if (!encoded.has_value()) return 2;

    auto out = encoder.take_buffer(); // out contains encoded bytes
    return out == bytes ? 0 : 3;
}
```

## Build

```powershell
cmake -S . -B build -DICEDCPP_BUILD_SHARED=ON -DICEDCPP_BUILD_TESTS=ON -DICEDCPP_BUILD_EXAMPLES=ON
cmake --build build --config Debug
```

Artifacts:

- `build/bin` executables and shared libraries
- `build/lib` import/static libraries

## Test

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

## Examples

Examples are available under `examples/` and cover decoder, encoder, formatters, metadata APIs, block encoding, and `CodeAssembler` labels.

Build and run one:

```powershell
cmake --build build --config Debug --target icedcpp_example_decoder_stream
.\build\bin\icedcpp_example_decoder_stream.exe
```

## Documentation (Doxygen)

```powershell
cmake -S . -B build -DICEDCPP_BUILD_DOCS=ON
cmake --build build --config Debug --target docs
```

Targets:

- `docs` generate HTML + XML
- `docs_html` generate HTML only
- `docs_xml` generate XML only
- `docs_clean` remove generated docs output

Outputs:

- `build/docs/out/html/index.html`
- `build/docs/out/xml/`
- `build/docs/doxygen-warnings.log`

## GitHub Pages Publishing

This repo includes a workflow at `.github/workflows/docs-pages.yml` that builds Doxygen HTML and deploys it to GitHub Pages.

One-time setup in GitHub:

1. Open `Settings -> Pages`.
2. Set `Source` to `GitHub Actions`.
3. Push to `main`/`master` (or run the workflow manually from `Actions`).

Published URL pattern:

- `https://<owner>.github.io/<repo>/`

Release packaging:

```powershell
powershell -ExecutionPolicy Bypass -File docs\package_release_docs.ps1 -Version 0.1.1
```

Packaged docs are written to:

- `docs/release/<version>/`
- `docs/release/latest/`

## Layout

- `include/` public and internal headers (`include/iced_x86`, `include/iced_x86/internal`)
- `src/iced_x86/src/` library implementation
- `tests/iced_x86/` test suite
- `docs/` Doxygen config and release packaging helpers
- `examples/` runnable API examples across core components
- `src/` top-level CMake entrypoints
- `bin/` reserved for tools/examples

## Install

```powershell
cmake --install build --config Debug --prefix install
```

Installed layout:

- `install/include/iced_x86/*.hpp`
- `install/include/iced_x86/internal/*.hpp`
- `install/lib/*`
- `install/bin/*`

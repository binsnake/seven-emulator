#include "iced_x86/decoder.hpp"
#include "iced_x86/fast_formatter.hpp"
#include "iced_x86/fast_string_output.hpp"
#include "iced_x86/formatter_output.hpp"
#include "iced_x86/gas_formatter.hpp"
#include "iced_x86/intel_formatter.hpp"
#include "iced_x86/masm_formatter.hpp"
#include "iced_x86/nasm_formatter.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0xD8}; // mov rax,rbx
    iced_x86::Decoder decoder(64, bytes, 0x1000);
    auto decoded = decoder.decode();
    if (!decoded.has_value()) {
        std::cerr << "decode failed\n";
        return 1;
    }

    std::string intel_text;
    iced_x86::StringFormatterOutput intel_out(intel_text);
    iced_x86::IntelFormatter intel;
    intel.format(decoded.value(), intel_out);

    std::string masm_text;
    iced_x86::StringFormatterOutput masm_out(masm_text);
    iced_x86::MasmFormatter masm;
    masm.format(decoded.value(), masm_out);

    std::string nasm_text;
    iced_x86::StringFormatterOutput nasm_out(nasm_text);
    iced_x86::NasmFormatter nasm;
    nasm.format(decoded.value(), nasm_out);

    std::string gas_text;
    iced_x86::StringFormatterOutput gas_out(gas_text);
    iced_x86::GasFormatter gas;
    gas.format(decoded.value(), gas_out);

    iced_x86::FastFormatter fast;
    iced_x86::FastStringOutput fast_out;
    fast.format(decoded.value(), fast_out);

    std::cout << "Intel: " << intel_text << "\n";
    std::cout << "Masm : " << masm_text << "\n";
    std::cout << "Nasm : " << nasm_text << "\n";
    std::cout << "Gas  : " << gas_text << "\n";
    std::cout << "Fast : " << fast_out.str() << "\n";
    return 0;
}

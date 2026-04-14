#include "test_utils.hpp"

#include "iced_x86/decoder.hpp"
#include "iced_x86/fast_formatter.hpp"
#include "iced_x86/fast_string_output.hpp"
#include "iced_x86/formatter_output.hpp"
#include "iced_x86/gas_formatter.hpp"
#include "iced_x86/intel_formatter.hpp"
#include "iced_x86/masm_formatter.hpp"
#include "iced_x86/nasm_formatter.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace {

iced_x86::Instruction decode_mov_rax_rbx() {
    const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0xD8};
    iced_x86::Decoder decoder(64, bytes, 0x1000);
    auto decoded = decoder.decode();
    test::require(decoded.has_value(), "decode mov rax,rbx failed");
    return decoded.value();
}

void require_non_empty_and_contains(std::string_view text, std::string_view needle, std::string_view label) {
    test::require(!text.empty(), std::string(label) + " output is empty");
    auto haystack_lower = std::string(text);
    auto needle_lower = std::string(needle);
    std::transform(haystack_lower.begin(), haystack_lower.end(), haystack_lower.begin(), [](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    std::transform(needle_lower.begin(), needle_lower.end(), needle_lower.begin(), [](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    test::require(haystack_lower.find(needle_lower) != std::string::npos, std::string(label) + " should contain " + std::string(needle));
}

} // namespace

ICED_TEST(formatters_emit_text_for_decoded_instruction) {
    const auto instr = decode_mov_rax_rbx();

    std::string intel_text;
    iced_x86::StringFormatterOutput intel_out(intel_text);
    iced_x86::IntelFormatter intel;
    intel.format(instr, intel_out);
    require_non_empty_and_contains(intel_text, "mov", "Intel formatter");

    std::string masm_text;
    iced_x86::StringFormatterOutput masm_out(masm_text);
    iced_x86::MasmFormatter masm;
    masm.format(instr, masm_out);
    require_non_empty_and_contains(masm_text, "mov", "Masm formatter");

    std::string nasm_text;
    iced_x86::StringFormatterOutput nasm_out(nasm_text);
    iced_x86::NasmFormatter nasm;
    nasm.format(instr, nasm_out);
    require_non_empty_and_contains(nasm_text, "mov", "Nasm formatter");

    std::string gas_text;
    iced_x86::StringFormatterOutput gas_out(gas_text);
    iced_x86::GasFormatter gas;
    gas.format(instr, gas_out);
    require_non_empty_and_contains(gas_text, "mov", "Gas formatter");

    iced_x86::FastFormatter fast;
    iced_x86::FastStringOutput fast_out;
    fast.format(instr, fast_out);
    const std::string fast_text = fast_out.str();
    require_non_empty_and_contains(fast_text, "mov", "Fast formatter");
}

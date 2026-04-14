#include "iced_x86/decoder.hpp"
#include "iced_x86/intel_formatter.hpp"
#include "iced_x86/formatter_output.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::vector<std::uint8_t> bytes = {
        0x48, 0x89, 0xD8,       // mov rax,rbx
        0x48, 0x83, 0xC0, 0x10, // add rax,16
        0x90,                   // nop
        0xC3                    // ret
    };

    iced_x86::Decoder decoder(64, bytes, 0x140000000);
    iced_x86::IntelFormatter formatter;

    while (decoder.can_decode()) {
        auto decoded = decoder.decode();
        if (!decoded.has_value()) {
            std::cerr << "Decode error at ip=0x" << std::hex << decoded.error().ip << std::dec << "\n";
            return 1;
        }

        std::string text;
        iced_x86::StringFormatterOutput out(text);
        formatter.format(decoded.value(), out);

        std::cout << "ip=0x" << std::hex << decoded->ip()
                  << " len=" << std::dec << decoded->length()
                  << " text=" << text << "\n";
    }

    return 0;
}

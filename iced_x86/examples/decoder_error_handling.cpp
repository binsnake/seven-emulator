#include "iced_x86/decoder.hpp"
#include "iced_x86/decoder_error.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    const std::vector<std::uint8_t> truncated = {0xE8, 0x34, 0x12}; // incomplete rel32 call
    iced_x86::Decoder decoder(64, truncated, 0x1000);

    auto decoded = decoder.decode();
    if (decoded.has_value()) {
        std::cerr << "Unexpected success\n";
        return 1;
    }

    std::cout << "decode failed as expected\n";
    std::cout << "error code: " << static_cast<unsigned>(decoded.error().error) << "\n";
    std::cout << "error ip  : 0x" << std::hex << decoded.error().ip << std::dec << "\n";
    return 0;
}

#include "iced_x86/decoder.hpp"
#include "iced_x86/encoder.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    const std::vector<std::uint8_t> original = {
        0x48, 0x8B, 0x44, 0x24, 0x20 // mov rax,[rsp+20h]
    };
    constexpr std::uint64_t ip = 0x401000;

    iced_x86::Decoder decoder(64, original, ip);
    auto decoded = decoder.decode();
    if (!decoded.has_value()) {
        std::cerr << "decode failed\n";
        return 1;
    }

    iced_x86::Encoder encoder(64);
    auto encoded_size = encoder.encode(decoded.value(), ip);
    if (!encoded_size.has_value()) {
        std::cerr << "encode failed: " << encoded_size.error().message << "\n";
        return 1;
    }

    const auto out = encoder.take_buffer();
    std::cout << "encoded size: " << encoded_size.value() << "\n";
    std::cout << "byte-exact roundtrip: " << (out == original ? "yes" : "no") << "\n";
    return out == original ? 0 : 2;
}

#include "iced_x86/block_encoder.hpp"
#include "iced_x86/decoder.hpp"
#include "iced_x86/instruction.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    const std::vector<std::uint8_t> bytes = {
        0x90,                   // nop
        0xEB, 0x02,             // jmp short +2
        0x90,                   // nop (skipped)
        0x90,                   // nop
        0xC3                    // ret
    };

    std::vector<iced_x86::Instruction> instructions;
    iced_x86::Decoder decoder(64, bytes, 0x1000);
    while (decoder.can_decode()) {
        auto decoded = decoder.decode();
        if (!decoded.has_value()) {
            std::cerr << "decode failed\n";
            return 1;
        }
        instructions.push_back(decoded.value());
    }

    const auto encoded = iced_x86::BlockEncoder::encode(
        64,
        instructions,
        0x50000000,
        iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS |
        iced_x86::BlockEncoderOptions::RETURN_CONSTANT_OFFSETS);

    if (!encoded.has_value()) {
        std::cerr << "block encode failed: " << encoded.error() << "\n";
        return 2;
    }

    std::cout << "input instructions : " << instructions.size() << "\n";
    std::cout << "output bytes       : " << encoded->code_buffer.size() << "\n";
    std::cout << "new offsets count  : " << encoded->new_instruction_offsets.size() << "\n";
    std::cout << "constant offsets   : " << encoded->constant_offsets.size() << "\n";
    return 0;
}

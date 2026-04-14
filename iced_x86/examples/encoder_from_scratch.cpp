#include "iced_x86/code.hpp"
#include "iced_x86/decoder.hpp"
#include "iced_x86/encoder.hpp"
#include "iced_x86/instruction_create.hpp"
#include "iced_x86/intel_formatter.hpp"
#include "iced_x86/formatter_output.hpp"
#include "iced_x86/memory_operand.hpp"
#include "iced_x86/register.hpp"

#include <cstdint>
#include <iostream>
#include <string>

int main() {
    using namespace iced_x86;

    const auto mem = MemoryOperand::with_base_displ(Register::RSP, 0x20);
    const auto instr = InstructionFactory::with2(Code::MOV_R64_RM64, Register::RAX, mem);

    Encoder encoder(64);
    auto encode_res = encoder.encode(instr, 0x5000);
    if (!encode_res.has_value()) {
        std::cerr << "encode failed: " << encode_res.error().message << "\n";
        return 1;
    }
    const auto bytes = encoder.take_buffer();

    Decoder decoder(64, bytes, 0x5000);
    auto decoded = decoder.decode();
    if (!decoded.has_value()) {
        std::cerr << "decode failed\n";
        return 2;
    }

    std::string text;
    StringFormatterOutput out(text);
    IntelFormatter formatter;
    formatter.format(decoded.value(), out);

    std::cout << "encoded bytes: " << bytes.size() << "\n";
    std::cout << "decoded text : " << text << "\n";
    return 0;
}

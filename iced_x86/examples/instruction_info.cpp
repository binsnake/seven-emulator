#include "iced_x86/decoder.hpp"
#include "iced_x86/instruction_info.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    const std::vector<std::uint8_t> bytes = {0x48, 0x01, 0xD8}; // add rax,rbx
    iced_x86::Decoder decoder(64, bytes, 0x1000);
    auto decoded = decoder.decode();
    if (!decoded.has_value()) {
        std::cerr << "decode failed\n";
        return 1;
    }

    iced_x86::InstructionInfoFactory factory;
    const auto& info = factory.info(decoded.value());

    std::cout << "used registers: " << info.used_registers().size() << "\n";
    for (const auto& reg : info.used_registers()) {
        std::cout << "  reg=" << static_cast<unsigned>(reg.register_)
                  << " access=" << static_cast<unsigned>(reg.access) << "\n";
    }

    std::cout << "used memory entries: " << info.used_memory().size() << "\n";
    std::cout << "flow control: "
              << static_cast<unsigned>(iced_x86::InstructionExtensions::flow_control(decoded.value()))
              << "\n";
    std::cout << "stack instruction: "
              << (iced_x86::InstructionExtensions::is_stack_instruction(decoded.value()) ? "yes" : "no")
              << "\n";
    return 0;
}

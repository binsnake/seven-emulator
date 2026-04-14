#include "iced_x86/code.hpp"
#include "iced_x86/op_code_info.hpp"

#include <array>
#include <iostream>

int main() {
    using namespace iced_x86;

    constexpr std::array<Code, 4> sample_codes = {
        Code::MOV_RM64_R64,
        Code::MOV_R64_IMM64,
        Code::CALL_REL32_64,
        Code::JL_REL8_64
    };

    for (const auto code : sample_codes) {
        const auto& info = OpCodeInfo::get(code);
        std::cout << "code=" << static_cast<unsigned>(info.code())
                  << " mnemonic=" << static_cast<unsigned>(info.mnemonic())
                  << " op_count=" << info.op_count()
                  << " mode64=" << (info.mode64() ? "yes" : "no")
                  << " table=" << static_cast<unsigned>(info.table())
                  << "\n";
    }

    return 0;
}

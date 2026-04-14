#include "iced_x86/code_assembler.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using namespace iced_x86;

    CodeAssembler a(64);

    auto loop = a.create_label();
    auto done = a.create_label();

    a.mov(eax, 0);
    a.set_label(loop);
    a.inc(eax);
    a.cmp(eax, 5);
    a.jl(loop);
    a.jmp(done);

    a.int3(); // skipped by jump

    a.set_label(done);
    a.ret();

    // Anonymous labels: jump forward to a not-yet-defined label.
    auto forward = a.fwd();
    a.jmp(forward);
    a.int3(); // skipped
    a.anonymous_label();
    a.nop();
    a.ret();

    std::vector<std::uint8_t> bytes;
    try {
        bytes = a.assemble(0x140001000);
    } catch (const std::exception& ex) {
        std::cerr << "assemble failed: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "assembled bytes: " << bytes.size() << "\n";
    return 0;
}

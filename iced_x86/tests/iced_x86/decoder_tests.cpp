#include "test_utils.hpp"

#include "iced_x86/decoder.hpp"
#include "iced_x86/mnemonic.hpp"

#include <cstdint>
#include <vector>

ICED_TEST(decoder_decodes_instruction_stream_and_tracks_state) {
    const std::vector<std::uint8_t> stream = {
        0x48, 0x89, 0xD8, // mov rax,rbx
        0x90,             // nop
        0xEB, 0xFE        // jmp short $-0
    };

    iced_x86::Decoder decoder(64, stream, 0x1000);
    test::require(decoder.can_decode(), "decoder should have data");
    test::require_eq(static_cast<std::size_t>(0), decoder.position(), "initial decoder position");

    auto first = decoder.decode();
    test::require(first.has_value(), "decode first failed");
    test::require_eq(iced_x86::Mnemonic::MOV, first->mnemonic(), "first mnemonic");
    test::require_eq(static_cast<std::size_t>(3), decoder.position(), "position after first decode");

    auto second = decoder.decode();
    test::require(second.has_value(), "decode second failed");
    test::require_eq(iced_x86::Mnemonic::NOP, second->mnemonic(), "second mnemonic");
    test::require_eq(static_cast<std::size_t>(4), decoder.position(), "position after second decode");

    auto third = decoder.decode();
    test::require(third.has_value(), "decode third failed");
    test::require_eq(iced_x86::Mnemonic::JMP, third->mnemonic(), "third mnemonic");
    test::require_eq(stream.size(), decoder.position(), "position at end");
    test::require(!decoder.can_decode(), "decoder should be at end");
}

ICED_TEST(decoder_reports_error_on_truncated_instruction) {
    const std::vector<std::uint8_t> truncated = {0xE8, 0x01, 0x02}; // incomplete rel32
    iced_x86::Decoder decoder(64, truncated, 0x2000);
    auto decoded = decoder.decode();
    test::require(!decoded.has_value(), "decode should fail on truncated bytes");
    test::require(decoded.error().ip >= 0x2000ull, "decode error ip should be within instruction range");
    test::require(decoded.error().ip <= 0x2000ull + truncated.size(), "decode error ip should be within input range");
    test::require(decoded.error().error != iced_x86::DecoderError::NONE, "decode error code should not be NONE");
}

ICED_TEST(decoder_reconfigure_reuses_instance) {
    const std::vector<std::uint8_t> first = {0x90};
    const std::vector<std::uint8_t> second = {0x48, 0x89, 0xD8};

    iced_x86::Decoder decoder(64, first, 0x3000);
    auto first_decoded = decoder.decode();
    test::require(first_decoded.has_value(), "decode first sequence failed");
    test::require_eq(iced_x86::Mnemonic::NOP, first_decoded->mnemonic(), "first sequence mnemonic");

    decoder.reconfigure(second, 0x4000);
    test::require_eq(static_cast<std::size_t>(0), decoder.position(), "position should reset on reconfigure");
    test::require_eq(0x4000ull, decoder.ip(), "ip should update on reconfigure");
    auto second_decoded = decoder.decode();
    test::require(second_decoded.has_value(), "decode second sequence failed");
    test::require_eq(iced_x86::Mnemonic::MOV, second_decoded->mnemonic(), "second sequence mnemonic");
}

ICED_TEST(decoder_set_position_rewinds_stream) {
    const std::vector<std::uint8_t> stream = {0x90, 0x90, 0x90};
    iced_x86::Decoder decoder(64, stream, 0x5000);

    auto first = decoder.decode();
    test::require(first.has_value(), "decode first nop failed");
    test::require_eq(static_cast<std::size_t>(1), decoder.position(), "position after first nop");

    decoder.set_position(0);
    test::require_eq(static_cast<std::size_t>(0), decoder.position(), "position after rewind");

    auto rewound = decoder.decode();
    test::require(rewound.has_value(), "decode rewound nop failed");
    test::require_eq(iced_x86::Mnemonic::NOP, rewound->mnemonic(), "rewound mnemonic");
}

ICED_TEST(decoder_decodes_evex_vpaddb_masked) {
    const std::vector<std::uint8_t> bytes = {0x62, 0xF1, 0x75, 0x09, 0xFC, 0xC2};
    iced_x86::Decoder decoder(64, bytes, 0x6000);
    auto decoded = decoder.decode();
    test::require(decoded.has_value(), "decode EVEX vpaddb failed");
    test::require_eq(iced_x86::Code::EVEX_VPADDB_XMM_K1Z_XMM_XMMM128, decoded->code(), "EVEX vpaddb code");
    test::require_eq(static_cast<std::size_t>(bytes.size()), decoder.position(), "position after EVEX vpaddb");
    test::require(!decoder.can_decode(), "decoder should be at end after EVEX vpaddb");
}

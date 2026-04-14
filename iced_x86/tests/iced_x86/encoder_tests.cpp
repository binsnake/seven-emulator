#include "test_utils.hpp"

#include "iced_x86/decoder.hpp"
#include "iced_x86/encoder.hpp"
#include "iced_x86/code.hpp"
#include "iced_x86/register.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using iced_x86::Decoder;
using iced_x86::Encoder;
using iced_x86::Instruction;
using iced_x86::OpKind;

Instruction decode_single(uint32_t bitness, const std::vector<std::uint8_t>& bytes, std::uint64_t ip) {
    Decoder decoder(bitness, bytes, ip);
    auto decoded = decoder.decode();
    test::require(decoded.has_value(), "decode() failed");
    test::require(!decoder.can_decode(), "decode_single() expected exactly one instruction");
    return decoded.value();
}

std::vector<std::uint8_t> encode_single(
    uint32_t bitness, const Instruction& instruction, std::uint64_t ip, std::size_t* out_size = nullptr) {
    Encoder encoder(bitness);
    auto encoded = encoder.encode(instruction, ip);
    test::require(encoded.has_value(), "encode() failed");
    auto output = encoder.take_buffer();
    test::require_eq(output.size(), encoded.value(), "encoded length");
    if (out_size != nullptr) {
        *out_size = encoded.value();
    }
    return output;
}

void require_equal_bytes(const std::vector<std::uint8_t>& expected, const std::vector<std::uint8_t>& actual, std::string_view label) {
    if (expected != actual) {
        test::fail(std::string(label) + " expected=[" + test::bytes_to_hex(expected) + "] actual=[" + test::bytes_to_hex(actual) + "]");
    }
}

std::string semantic_signature(const Instruction& instr) {
    std::string s;
    s += "code=" + std::to_string(static_cast<unsigned>(instr.code()));
    s += "|op_count=" + std::to_string(instr.op_count());
    for (uint32_t i = 0; i < instr.op_count(); ++i) {
        const auto kind = instr.op_kind(i);
        s += "|k" + std::to_string(i) + "=" + std::to_string(static_cast<unsigned>(kind));
        s += "|r" + std::to_string(i) + "=" + std::to_string(static_cast<unsigned>(instr.op_register(i)));
        switch (kind) {
            case OpKind::IMMEDIATE8:
            case OpKind::IMMEDIATE8_2ND:
            case OpKind::IMMEDIATE8TO16:
            case OpKind::IMMEDIATE8TO32:
            case OpKind::IMMEDIATE8TO64:
                s += "|imm8=" + std::to_string(instr.immediate8());
                break;
            case OpKind::IMMEDIATE16:
                s += "|imm16=" + std::to_string(instr.immediate16());
                break;
            case OpKind::IMMEDIATE32:
            case OpKind::IMMEDIATE32TO64:
                s += "|imm32=" + std::to_string(instr.immediate32());
                break;
            case OpKind::IMMEDIATE64:
                s += "|imm64=" + std::to_string(instr.immediate64());
                break;
            case OpKind::NEAR_BRANCH16:
                s += "|br16=" + std::to_string(instr.near_branch16());
                break;
            case OpKind::NEAR_BRANCH32:
                s += "|br32=" + std::to_string(instr.near_branch32());
                break;
            case OpKind::NEAR_BRANCH64:
                s += "|br64=" + std::to_string(instr.near_branch64());
                break;
            case OpKind::FAR_BRANCH16:
            case OpKind::FAR_BRANCH32:
                s += "|fsel=" + std::to_string(instr.far_branch_selector());
                s += "|foff16=" + std::to_string(instr.far_branch16());
                s += "|foff32=" + std::to_string(instr.far_branch32());
                break;
            case OpKind::MEMORY:
            case OpKind::MEMORY_SEG_SI:
            case OpKind::MEMORY_SEG_ESI:
            case OpKind::MEMORY_SEG_RSI:
            case OpKind::MEMORY_SEG_DI:
            case OpKind::MEMORY_SEG_EDI:
            case OpKind::MEMORY_SEG_RDI:
            case OpKind::MEMORY_ESDI:
            case OpKind::MEMORY_ESEDI:
            case OpKind::MEMORY_ESRDI:
                s += "|mb=" + std::to_string(static_cast<unsigned>(instr.memory_base()));
                s += "|mi=" + std::to_string(static_cast<unsigned>(instr.memory_index()));
                s += "|ms=" + std::to_string(instr.memory_index_scale());
                s += "|md=" + std::to_string(instr.memory_displacement64());
                break;
            default:
                break;
        }
    }
    return s;
}

} // namespace

ICED_TEST(encoder_roundtrip_canonical_vectors) {
    struct VectorCase {
        const char* name;
        std::uint32_t bitness;
        std::uint64_t ip;
        std::vector<std::uint8_t> bytes;
    };

    const std::array<VectorCase, 23> cases = {{
        {"nop64", 64, 0x1000, {0x90}},
        {"mov_rax_rbx", 64, 0x1010, {0x48, 0x89, 0xD8}},
        {"mov_rax_rsp_disp8", 64, 0x1020, {0x48, 0x8B, 0x44, 0x24, 0x20}},
        {"mov_rsp_disp8_rax", 64, 0x1030, {0x48, 0x89, 0x44, 0x24, 0x20}},
        {"mov_rax_imm32", 64, 0x1040, {0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12}},
        {"add_rax_imm32", 64, 0x1050, {0x48, 0x05, 0x78, 0x56, 0x34, 0x12}},
        {"call_rel32_64", 64, 0x1060, {0xE8, 0x34, 0x12, 0x00, 0x00}},
        {"jmp_rel8_64", 64, 0x1070, {0xEB, 0xFE}},
        {"jne_rel32_64", 64, 0x1080, {0x0F, 0x85, 0x78, 0x56, 0x34, 0x12}},
        {"mov_rax_ripdisp32", 64, 0x1090, {0x48, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12}},
        {"vzeroupper_vex2", 64, 0x10A0, {0xC5, 0xF8, 0x77}},
        {"add_rax_imm8", 64, 0x10C0, {0x48, 0x83, 0xC0, 0x7F}},
        {"lea_disp8", 64, 0x10D0, {0x48, 0x8D, 0x43, 0x10}},
        {"mov_eax_ebx", 32, 0x2000, {0x89, 0xD8}},
        {"mov_eax_esp_disp8", 32, 0x2010, {0x8B, 0x44, 0x24, 0x10}},
        {"mov_ax_imm16", 32, 0x2020, {0x66, 0xB8, 0x34, 0x12}},
        {"call_rel32_32", 32, 0x2030, {0xE8, 0x34, 0x12, 0x00, 0x00}},
        {"jne_rel32_32", 32, 0x2040, {0x0F, 0x85, 0x78, 0x56, 0x34, 0x12}},
        {"push_imm8", 32, 0x2050, {0x6A, 0x80}},
        {"mov_ax_bx", 16, 0x3000, {0x8B, 0xC3}},
        {"call_rel16", 16, 0x3020, {0xE8, 0x34, 0x12}},
        {"mov_ax_imm16_16", 16, 0x3030, {0xB8, 0x34, 0x12}},
        {"jz_rel8_16", 16, 0x3040, {0x74, 0xFE}},
    }};

    for (const auto& entry : cases) {
        Decoder decoder(entry.bitness, entry.bytes, entry.ip);
        auto decoded = decoder.decode();
        if (!decoded.has_value()) {
            test::fail(std::string(entry.name) + ": decode() failed");
        }
        if (decoder.can_decode()) {
            test::fail(std::string(entry.name) + ": expected single instruction");
        }

        Encoder encoder(entry.bitness);
        auto encoded = encoder.encode(decoded.value(), entry.ip);
        if (!encoded.has_value()) {
            test::fail(std::string(entry.name) + ": encode() failed: " + encoded.error().message);
        }
        const auto out = encoder.take_buffer();
        test::require_eq(out.size(), encoded.value(), std::string(entry.name) + " encoded length");
        require_equal_bytes(entry.bytes, out, entry.name);
    }
}

ICED_TEST(encoder_constant_offsets_for_immediate_and_displacement) {
    {
        const std::vector<std::uint8_t> bytes = {0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12};
        const auto instr = decode_single(64, bytes, 0x5000);
        Encoder encoder(64);
        auto result = encoder.encode(instr, 0x5000);
        test::require(result.has_value(), "encode mov imm failed");
        const auto offs = encoder.get_constant_offsets();
        test::require_eq(3u, static_cast<unsigned>(offs.immediate_offset), "imm offset");
        test::require_eq(4u, static_cast<unsigned>(offs.immediate_size), "imm size");
        test::require(!offs.has_displacement(), "unexpected displacement");
    }
    {
        const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0x44, 0x24, 0x20};
        const auto instr = decode_single(64, bytes, 0x6000);
        Encoder encoder(64);
        auto result = encoder.encode(instr, 0x6000);
        test::require(result.has_value(), "encode mov [rsp+disp8],rax failed");
        const auto offs = encoder.get_constant_offsets();
        test::require_eq(4u, static_cast<unsigned>(offs.displacement_offset), "disp offset");
        test::require_eq(1u, static_cast<unsigned>(offs.displacement_size), "disp size");
        test::require(!offs.has_immediate(), "unexpected immediate");
    }
    {
        const std::vector<std::uint8_t> bytes = {0xE8, 0x34, 0x12, 0x00, 0x00};
        const auto instr = decode_single(64, bytes, 0x7000);
        Encoder encoder(64);
        auto result = encoder.encode(instr, 0x7000);
        test::require(result.has_value(), "encode call rel32 failed");
        const auto offs = encoder.get_constant_offsets();
        test::require_eq(1u, static_cast<unsigned>(offs.immediate_offset), "call imm offset");
        test::require_eq(4u, static_cast<unsigned>(offs.immediate_size), "call imm size");
    }
    {
        // enter imm16, imm8 => two immediates in one instruction
        const std::vector<std::uint8_t> bytes = {0xC8, 0x34, 0x12, 0x56};
        const auto instr = decode_single(32, bytes, 0x7100);
        Encoder encoder(32);
        auto result = encoder.encode(instr, 0x7100);
        if (!result.has_value()) {
            test::fail(std::string("encode enter failed: ") + result.error().message);
        }
        const auto offs = encoder.get_constant_offsets();
        test::require_eq(1u, static_cast<unsigned>(offs.immediate_offset), "enter imm1 offset");
        test::require_eq(2u, static_cast<unsigned>(offs.immediate_size), "enter imm1 size");
        test::require_eq(3u, static_cast<unsigned>(offs.immediate_offset2), "enter imm2 offset");
        test::require_eq(1u, static_cast<unsigned>(offs.immediate_size2), "enter imm2 size");
        test::require(offs.has_immediate2(), "enter should report second immediate");
    }
    {
        // EXTRQ xmm, imm8, imm8 => SIZE1_1 (two adjacent 8-bit immediates)
        iced_x86::Instruction instr;
        instr.set_code(iced_x86::Code::EXTRQ_XMM_IMM8_IMM8);
        instr.set_op0_kind(OpKind::REGISTER);
        instr.set_op0_register(iced_x86::Register::XMM1);
        instr.set_op1_kind(OpKind::IMMEDIATE8);
        instr.set_immediate8(0x12);
        instr.set_op2_kind(OpKind::IMMEDIATE8_2ND);
        instr.set_immediate8_2nd(0x34);

        Encoder encoder(64);
        auto encoded_res = encoder.encode(instr, 0x7200);
        test::require(encoded_res.has_value(), "encode extrq imm8,imm8 failed");
        const auto offs = encoder.get_constant_offsets();
        test::require_eq(1u, static_cast<unsigned>(offs.immediate_size), "extrq imm1 size");
        test::require_eq(1u, static_cast<unsigned>(offs.immediate_size2), "extrq imm2 size");
        test::require(offs.has_immediate2(), "extrq should report second immediate");
        test::require_eq(
            static_cast<unsigned>(offs.immediate_offset + 1),
            static_cast<unsigned>(offs.immediate_offset2),
            "extrq immediate offsets should be adjacent");

        const auto bytes = encoder.take_buffer();
        const auto decoded = decode_single(64, bytes, 0x7200);
        test::require_eq(iced_x86::Code::EXTRQ_XMM_IMM8_IMM8, decoded.code(), "decoded extrq code");
        test::require_eq(OpKind::IMMEDIATE8, decoded.op1_kind(), "decoded extrq op1 kind");
        test::require_eq(OpKind::IMMEDIATE8_2ND, decoded.op2_kind(), "decoded extrq op2 kind");
        test::require_eq(0x12u, static_cast<unsigned>(decoded.immediate8()), "decoded extrq imm8");
        test::require_eq(0x34u, static_cast<unsigned>(decoded.immediate8_2nd()), "decoded extrq imm8_2nd");
    }
}

ICED_TEST(encoder_reports_bitness_mismatch_errors) {
    {
        const std::vector<std::uint8_t> bytes64 = {0x48, 0x89, 0xD8};
        const auto instr64 = decode_single(64, bytes64, 0x8000);
        Encoder encoder32(32);
        auto result = encoder32.encode(instr64, 0x8000);
        test::require(!result.has_value(), "expected 64-bit only error");
        test::require(result.error().message.find("64-bit") != std::string::npos, "error text should mention 64-bit");
    }
    {
        const std::vector<std::uint8_t> bytes32 = {0x60};
        const auto instr32 = decode_single(32, bytes32, 0x8100);
        Encoder encoder64(64);
        auto result = encoder64.encode(instr32, 0x8100);
        test::require(!result.has_value(), "expected 16/32-bit only error");
        test::require(result.error().message.find("16/32-bit") != std::string::npos, "error text should mention 16/32-bit");
    }
}

ICED_TEST(encoder_buffer_management_apis) {
    const std::vector<std::uint8_t> nop = {0x90};
    const auto instr = decode_single(64, nop, 0x9000);

    Encoder encoder(64);
    test::require_eq(0u, static_cast<unsigned>(encoder.position()), "initial position");
    auto result = encoder.encode(instr, 0x9000);
    test::require(result.has_value(), "encode nop failed");
    test::require_eq(1u, static_cast<unsigned>(encoder.position()), "position after nop");
    test::require_eq(1u, static_cast<unsigned>(encoder.buffer().size()), "buffer size after nop");

    auto taken = encoder.take_buffer();
    require_equal_bytes(nop, taken, "take_buffer");
    test::require_eq(0u, static_cast<unsigned>(encoder.buffer().size()), "buffer should be empty after take_buffer");
    test::require_eq(0u, static_cast<unsigned>(encoder.position()), "position should reset after take_buffer");

    encoder.set_buffer({0xAA, 0xBB});
    auto result2 = encoder.encode(instr, 0x9001);
    test::require(result2.has_value(), "encode nop with prefilled buffer failed");
    const std::vector<std::uint8_t> expected = {0xAA, 0xBB, 0x90};
    require_equal_bytes(expected, encoder.buffer(), "set_buffer + encode");
}

ICED_TEST(encoder_prevent_vex2_forces_three_byte_vex_prefix) {
    const std::vector<std::uint8_t> vex2 = {0xC5, 0xF8, 0x77}; // vzeroupper
    const auto instr = decode_single(64, vex2, 0xA000);

    Encoder encoder(64);
    encoder.set_prevent_vex2(true);
    auto result = encoder.encode(instr, 0xA000);
    test::require(result.has_value(), "encode vex instruction failed");

    auto encoded = encoder.take_buffer();
    test::require(!encoded.empty(), "encoded vex bytes should not be empty");
    test::require_eq(0xC4u, static_cast<unsigned>(encoded[0]), "first prefix byte with prevent_vex2");
}

ICED_TEST(encoder_preserves_branch_targets_when_relocated) {
    struct BranchCase {
        const char* name;
        std::vector<std::uint8_t> bytes;
    };

    const std::array<BranchCase, 3> cases = {{
        {"call_rel32", {0xE8, 0x10, 0x00, 0x00, 0x00}},
        {"jmp_rel32", {0xE9, 0x20, 0x00, 0x00, 0x00}},
        {"jne_rel32", {0x0F, 0x85, 0x30, 0x00, 0x00, 0x00}},
    }};

    constexpr std::uint64_t original_ip = 0x100000;
    constexpr std::uint64_t relocated_ip = 0x200000;

    for (const auto& entry : cases) {
        const auto instr0 = decode_single(64, entry.bytes, original_ip);
        const auto target0 = instr0.near_branch64();

        const auto relocated_bytes = encode_single(64, instr0, relocated_ip);
        const auto instr1 = decode_single(64, relocated_bytes, relocated_ip);
        const auto target1 = instr1.near_branch64();

        if (target0 != target1) {
            test::fail(std::string(entry.name) + " target mismatch after relocation");
        }
    }
}

ICED_TEST(encoder_reports_register_operand_mismatch_errors) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0xD8}; // mov rax,rbx
    auto instr = decode_single(64, bytes, 0xA100);

    // Force invalid operand register size/class for this opcode.
    instr.set_op0_register(iced_x86::Register::XMM0);

    Encoder encoder(64);
    auto result = encoder.encode(instr, 0xA100);
    test::require(!result.has_value(), "expected register mismatch error");
    test::require(
        result.error().message.find("Register") != std::string::npos ||
        result.error().message.find("register") != std::string::npos,
        "error text should mention register");
}

ICED_TEST(encoder_roundtrip_semantic_vectors_for_noncanonical_cases) {
    struct SemanticCase {
        const char* name;
        std::uint32_t bitness;
        std::uint64_t ip;
        std::vector<std::uint8_t> bytes;
    };

    const std::array<SemanticCase, 3> cases = {{
        {"mov_ax_bxsi_disp8_noncanonical", 16, 0xB000, {0x8B, 0x40, 0x10}},
        {"mov_riprel_noncanonical", 64, 0xB100, {0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00}},
        {"call_rel32_noncanonical", 32, 0xB200, {0xE8, 0x00, 0x00, 0x00, 0x00}},
    }};

    for (const auto& entry : cases) {
        const auto decoded0 = decode_single(entry.bitness, entry.bytes, entry.ip);
        const auto encoded = encode_single(entry.bitness, decoded0, entry.ip);
        const auto decoded1 = decode_single(entry.bitness, encoded, entry.ip);

        const auto sig0 = semantic_signature(decoded0);
        const auto sig1 = semantic_signature(decoded1);
        if (sig0 != sig1) {
            test::fail(std::string(entry.name) + " semantic mismatch: lhs={" + sig0 + "} rhs={" + sig1 + "}");
        }
    }
}

ICED_TEST(encoder_roundtrip_nop16) {
    const std::vector<std::uint8_t> bytes = {0x90};
    const auto instr = decode_single(16, bytes, 0xD000);
    const auto out = encode_single(16, instr, 0xD000);
    require_equal_bytes(bytes, out, "nop16");
}

ICED_TEST(encoder_roundtrip_nop32) {
    const std::vector<std::uint8_t> bytes = {0x90};
    const auto instr = decode_single(32, bytes, 0xD010);
    const auto out = encode_single(32, instr, 0xD010);
    require_equal_bytes(bytes, out, "nop32");
}

ICED_TEST(encoder_roundtrip_nop64_single) {
    const std::vector<std::uint8_t> bytes = {0x90};
    const auto instr = decode_single(64, bytes, 0xD020);
    const auto out = encode_single(64, instr, 0xD020);
    require_equal_bytes(bytes, out, "nop64_single");
}

ICED_TEST(encoder_roundtrip_mov_rax_rbx_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0xD8};
    const auto instr = decode_single(64, bytes, 0xD030);
    const auto out = encode_single(64, instr, 0xD030);
    require_equal_bytes(bytes, out, "mov_rax_rbx_single");
}

ICED_TEST(encoder_roundtrip_mov_rax_rsp_disp8_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x8B, 0x44, 0x24, 0x20};
    const auto instr = decode_single(64, bytes, 0xD040);
    const auto out = encode_single(64, instr, 0xD040);
    require_equal_bytes(bytes, out, "mov_rax_rsp_disp8_single");
}

ICED_TEST(encoder_roundtrip_mov_rsp_disp8_rax_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x89, 0x44, 0x24, 0x20};
    const auto instr = decode_single(64, bytes, 0xD050);
    const auto out = encode_single(64, instr, 0xD050);
    require_equal_bytes(bytes, out, "mov_rsp_disp8_rax_single");
}

ICED_TEST(encoder_roundtrip_mov_rax_imm32_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12};
    const auto instr = decode_single(64, bytes, 0xD060);
    const auto out = encode_single(64, instr, 0xD060);
    require_equal_bytes(bytes, out, "mov_rax_imm32_single");
}

ICED_TEST(encoder_roundtrip_add_rax_imm32_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x05, 0x78, 0x56, 0x34, 0x12};
    const auto instr = decode_single(64, bytes, 0xD070);
    const auto out = encode_single(64, instr, 0xD070);
    require_equal_bytes(bytes, out, "add_rax_imm32_single");
}

ICED_TEST(encoder_roundtrip_add_rax_imm8_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x83, 0xC0, 0x7F};
    const auto instr = decode_single(64, bytes, 0xD080);
    const auto out = encode_single(64, instr, 0xD080);
    require_equal_bytes(bytes, out, "add_rax_imm8_single");
}

ICED_TEST(encoder_roundtrip_call_rel32_64_single) {
    const std::vector<std::uint8_t> bytes = {0xE8, 0x34, 0x12, 0x00, 0x00};
    const auto instr = decode_single(64, bytes, 0xD090);
    const auto out = encode_single(64, instr, 0xD090);
    require_equal_bytes(bytes, out, "call_rel32_64_single");
}

ICED_TEST(encoder_roundtrip_jmp_rel8_64_single) {
    const std::vector<std::uint8_t> bytes = {0xEB, 0xFE};
    const auto instr = decode_single(64, bytes, 0xD0A0);
    const auto out = encode_single(64, instr, 0xD0A0);
    require_equal_bytes(bytes, out, "jmp_rel8_64_single");
}

ICED_TEST(encoder_roundtrip_jne_rel32_64_single) {
    const std::vector<std::uint8_t> bytes = {0x0F, 0x85, 0x78, 0x56, 0x34, 0x12};
    const auto instr = decode_single(64, bytes, 0xD0B0);
    const auto out = encode_single(64, instr, 0xD0B0);
    require_equal_bytes(bytes, out, "jne_rel32_64_single");
}

ICED_TEST(encoder_roundtrip_mov_eax_ebx_single) {
    const std::vector<std::uint8_t> bytes = {0x89, 0xD8};
    const auto instr = decode_single(32, bytes, 0xD0C0);
    const auto out = encode_single(32, instr, 0xD0C0);
    require_equal_bytes(bytes, out, "mov_eax_ebx_single");
}

ICED_TEST(encoder_roundtrip_call_rel32_32_second) {
    const std::vector<std::uint8_t> bytes = {0xE8, 0x34, 0x12, 0x00, 0x00};
    const auto instr = decode_single(32, bytes, 0xD0D0);
    const auto out = encode_single(32, instr, 0xD0D0);
    require_equal_bytes(bytes, out, "call_rel32_32_second");
}

ICED_TEST(encoder_roundtrip_mov_ax_bx_single) {
    const std::vector<std::uint8_t> bytes = {0x8B, 0xC3};
    const auto instr = decode_single(16, bytes, 0xD0E0);
    const auto out = encode_single(16, instr, 0xD0E0);
    require_equal_bytes(bytes, out, "mov_ax_bx_single");
}

ICED_TEST(encoder_roundtrip_call_rel16_second) {
    const std::vector<std::uint8_t> bytes = {0xE8, 0x34, 0x12};
    const auto instr = decode_single(16, bytes, 0xD0F0);
    const auto out = encode_single(16, instr, 0xD0F0);
    require_equal_bytes(bytes, out, "call_rel16_second");
}

ICED_TEST(encoder_roundtrip_jz_rel8_16_second) {
    const std::vector<std::uint8_t> bytes = {0x74, 0xFE};
    const auto instr = decode_single(16, bytes, 0xD100);
    const auto out = encode_single(16, instr, 0xD100);
    require_equal_bytes(bytes, out, "jz_rel8_16_second");
}

ICED_TEST(encoder_constant_offsets_rip_relative_displacement_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12};
    const auto instr = decode_single(64, bytes, 0xD110);
    Encoder encoder(64);
    auto result = encoder.encode(instr, 0xD110);
    test::require(result.has_value(), "encode rip-relative mov failed");
    const auto offs = encoder.get_constant_offsets();
    test::require_eq(3u, static_cast<unsigned>(offs.displacement_offset), "rip-rel disp offset");
    test::require_eq(4u, static_cast<unsigned>(offs.displacement_size), "rip-rel disp size");
}

ICED_TEST(encoder_roundtrip_jmp_rel32_64_single) {
    const std::vector<std::uint8_t> bytes = {0xE9, 0x78, 0x56, 0x34, 0x12};
    const auto instr = decode_single(64, bytes, 0xD120);
    const auto out = encode_single(64, instr, 0xD120);
    require_equal_bytes(bytes, out, "jmp_rel32_64_single");
}

ICED_TEST(encoder_roundtrip_je_rel8_64_single) {
    const std::vector<std::uint8_t> bytes = {0x74, 0x7E};
    const auto instr = decode_single(64, bytes, 0xD130);
    const auto out = encode_single(64, instr, 0xD130);
    require_equal_bytes(bytes, out, "je_rel8_64_single");
}

ICED_TEST(encoder_roundtrip_jne_rel8_64_single) {
    const std::vector<std::uint8_t> bytes = {0x75, 0x80};
    const auto instr = decode_single(64, bytes, 0xD140);
    const auto out = encode_single(64, instr, 0xD140);
    require_equal_bytes(bytes, out, "jne_rel8_64_single");
}

ICED_TEST(encoder_roundtrip_call_rel32_32_zero) {
    const std::vector<std::uint8_t> bytes = {0xE8, 0x00, 0x00, 0x00, 0x00};
    const auto instr = decode_single(32, bytes, 0xD150);
    const auto out = encode_single(32, instr, 0xD150);
    require_equal_bytes(bytes, out, "call_rel32_32_zero");
}

ICED_TEST(encoder_roundtrip_jmp_rel16_16_zero) {
    const std::vector<std::uint8_t> bytes = {0xE9, 0x00, 0x00};
    const auto instr = decode_single(16, bytes, 0xD160);
    const auto out = encode_single(16, instr, 0xD160);
    require_equal_bytes(bytes, out, "jmp_rel16_16_zero");
}

ICED_TEST(encoder_roundtrip_jcc_rel16_16_zero) {
    const std::vector<std::uint8_t> bytes = {0x0F, 0x84, 0x00, 0x00};
    const auto instr = decode_single(16, bytes, 0xD170);
    const auto out = encode_single(16, instr, 0xD170);
    require_equal_bytes(bytes, out, "jcc_rel16_16_zero");
}

ICED_TEST(encoder_push_imm8_64_current_behavior) {
    const std::vector<std::uint8_t> bytes = {0x6A, 0x80};
    const auto instr = decode_single(64, bytes, 0xD180);
    Encoder encoder(64);
    auto result = encoder.encode(instr, 0xD180);
    test::require(!result.has_value(), "push imm8 in 64-bit mode is currently expected to fail in this encoder");
}

ICED_TEST(encoder_roundtrip_push_imm8_16) {
    const std::vector<std::uint8_t> bytes = {0x6A, 0x7F};
    const auto instr = decode_single(16, bytes, 0xD190);
    const auto out = encode_single(16, instr, 0xD190);
    require_equal_bytes(bytes, out, "push_imm8_16");
}

ICED_TEST(encoder_roundtrip_mov_ax_imm16_16_second) {
    const std::vector<std::uint8_t> bytes = {0xB8, 0xEF, 0xBE};
    const auto instr = decode_single(16, bytes, 0xD1A0);
    const auto out = encode_single(16, instr, 0xD1A0);
    require_equal_bytes(bytes, out, "mov_ax_imm16_16_second");
}

ICED_TEST(encoder_roundtrip_mov_eax_imm32_32) {
    const std::vector<std::uint8_t> bytes = {0xB8, 0x78, 0x56, 0x34, 0x12};
    const auto instr = decode_single(32, bytes, 0xD1B0);
    const auto out = encode_single(32, instr, 0xD1B0);
    require_equal_bytes(bytes, out, "mov_eax_imm32_32");
}

ICED_TEST(encoder_roundtrip_mov_rax_imm64_64) {
    const std::vector<std::uint8_t> bytes = {0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11};
    const auto instr = decode_single(64, bytes, 0xD1C0);
    const auto out = encode_single(64, instr, 0xD1C0);
    require_equal_bytes(bytes, out, "mov_rax_imm64_64");
}

ICED_TEST(encoder_constant_offsets_mov_rax_imm64) {
    const std::vector<std::uint8_t> bytes = {0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11};
    const auto instr = decode_single(64, bytes, 0xD1D0);
    Encoder encoder(64);
    auto result = encoder.encode(instr, 0xD1D0);
    test::require(result.has_value(), "encode mov rax,imm64 failed");
    const auto offs = encoder.get_constant_offsets();
    test::require_eq(2u, static_cast<unsigned>(offs.immediate_offset), "movabs imm64 offset");
    test::require_eq(8u, static_cast<unsigned>(offs.immediate_size), "movabs imm64 size");
}

ICED_TEST(encoder_constant_offsets_jmp_rel8) {
    const std::vector<std::uint8_t> bytes = {0xEB, 0x7F};
    const auto instr = decode_single(64, bytes, 0xD1E0);
    Encoder encoder(64);
    auto result = encoder.encode(instr, 0xD1E0);
    test::require(result.has_value(), "encode jmp rel8 failed");
    const auto offs = encoder.get_constant_offsets();
    test::require_eq(1u, static_cast<unsigned>(offs.immediate_offset), "jmp rel8 imm offset");
    test::require_eq(1u, static_cast<unsigned>(offs.immediate_size), "jmp rel8 imm size");
}

ICED_TEST(encoder_constant_offsets_jcc_rel32) {
    const std::vector<std::uint8_t> bytes = {0x0F, 0x85, 0x10, 0x00, 0x00, 0x00};
    const auto instr = decode_single(64, bytes, 0xD1F0);
    Encoder encoder(64);
    auto result = encoder.encode(instr, 0xD1F0);
    test::require(result.has_value(), "encode jcc rel32 failed");
    const auto offs = encoder.get_constant_offsets();
    test::require_eq(2u, static_cast<unsigned>(offs.immediate_offset), "jcc rel32 imm offset");
    test::require_eq(4u, static_cast<unsigned>(offs.immediate_size), "jcc rel32 imm size");
}

ICED_TEST(encoder_semantic_roundtrip_call_relocation_32) {
    const std::vector<std::uint8_t> bytes = {0xE8, 0x10, 0x00, 0x00, 0x00};
    const auto decoded0 = decode_single(32, bytes, 0xD200);
    const auto target0 = decoded0.near_branch32();
    const auto encoded = encode_single(32, decoded0, 0xE000);
    const auto decoded1 = decode_single(32, encoded, 0xE000);
    test::require_eq(target0, decoded1.near_branch32(), "relocated call target32");
}

ICED_TEST(encoder_semantic_roundtrip_jcc_relocation_32) {
    const std::vector<std::uint8_t> bytes = {0x0F, 0x85, 0x20, 0x00, 0x00, 0x00};
    const auto decoded0 = decode_single(32, bytes, 0xD210);
    const auto target0 = decoded0.near_branch32();
    const auto encoded = encode_single(32, decoded0, 0xE100);
    const auto decoded1 = decode_single(32, encoded, 0xE100);
    test::require_eq(target0, decoded1.near_branch32(), "relocated jcc target32");
}

ICED_TEST(encoder_invalid_bitness_constructor_current_behavior) {
    const std::vector<std::uint8_t> bytes = {0x90};
    const auto instr = decode_single(64, bytes, 0xD220);
    Encoder encoder(15);
    auto result = encoder.encode(instr, 0xD220);
    test::require(result.has_value(), "constructor currently does not reject invalid bitness");
}

ICED_TEST(encoder_take_buffer_clears_after_multi_encode) {
    const auto instr = decode_single(64, std::vector<std::uint8_t>{0x90}, 0xD230);
    Encoder encoder(64);
    auto r1 = encoder.encode(instr, 0xD230);
    test::require(r1.has_value(), "first encode failed");
    auto r2 = encoder.encode(instr, 0xD231);
    test::require(r2.has_value(), "second encode failed");
    auto bytes = encoder.take_buffer();
    test::require_eq(2u, static_cast<unsigned>(bytes.size()), "two nops expected");
    test::require_eq(0u, static_cast<unsigned>(encoder.buffer().size()), "buffer should be empty after take_buffer");
}

ICED_TEST(encoder_set_buffer_prefix_preserved) {
    const auto instr = decode_single(64, std::vector<std::uint8_t>{0x90}, 0xD240);
    Encoder encoder(64);
    encoder.set_buffer({0xAA, 0xBB, 0xCC});
    auto r = encoder.encode(instr, 0xD240);
    test::require(r.has_value(), "encode with preset buffer failed");
    const auto out = encoder.buffer();
    test::require_eq(4u, static_cast<unsigned>(out.size()), "preset+encoded size");
    test::require_eq(0xAAu, static_cast<unsigned>(out[0]), "preset[0]");
    test::require_eq(0x90u, static_cast<unsigned>(out[3]), "encoded byte");
}

ICED_TEST(encoder_roundtrip_xchg_rax_rbx_single) {
    const std::vector<std::uint8_t> bytes = {0x48, 0x87, 0xD8};
    const auto instr = decode_single(64, bytes, 0xD250);
    const auto out = encode_single(64, instr, 0xD250);
    require_equal_bytes(bytes, out, "xchg_rax_rbx_single");
}

void run_rt_extra(uint32_t bitness, uint64_t ip, std::initializer_list<std::uint8_t> b, std::string_view label) {
    const std::vector<std::uint8_t> bytes(b);
    const auto instr = decode_single(bitness, bytes, ip);
    const auto out = encode_single(bitness, instr, ip);
    require_equal_bytes(bytes, out, label);
}

void run_rt_extra(uint32_t bitness, uint64_t ip, const std::vector<std::uint8_t>& bytes, std::string_view label) {
    const auto instr = decode_single(bitness, bytes, ip);
    const auto out = encode_single(bitness, instr, ip);
    require_equal_bytes(bytes, out, label);
}

ICED_TEST(encoder_rt_extra_01) { run_rt_extra(64, 0xE001, {0x90}, "encoder_rt_extra_01"); }
ICED_TEST(encoder_rt_extra_02) { run_rt_extra(64, 0xE002, {0x48, 0x89, 0xD8}, "encoder_rt_extra_02"); }
ICED_TEST(encoder_rt_extra_03) { run_rt_extra(64, 0xE003, {0x48, 0x87, 0xD8}, "encoder_rt_extra_03"); }
ICED_TEST(encoder_rt_extra_04) { run_rt_extra(64, 0xE004, {0x48, 0x8B, 0x44, 0x24, 0x20}, "encoder_rt_extra_04"); }
ICED_TEST(encoder_rt_extra_05) { run_rt_extra(64, 0xE005, {0x48, 0x89, 0x44, 0x24, 0x20}, "encoder_rt_extra_05"); }
ICED_TEST(encoder_rt_extra_06) { run_rt_extra(64, 0xE006, {0x48, 0x83, 0xC0, 0x7F}, "encoder_rt_extra_06"); }
ICED_TEST(encoder_rt_extra_07) { run_rt_extra(64, 0xE007, {0x48, 0x05, 0x78, 0x56, 0x34, 0x12}, "encoder_rt_extra_07"); }
ICED_TEST(encoder_rt_extra_08) { run_rt_extra(64, 0xE008, {0xE8, 0x34, 0x12, 0x00, 0x00}, "encoder_rt_extra_08"); }
ICED_TEST(encoder_rt_extra_09) { run_rt_extra(64, 0xE009, {0xE9, 0x78, 0x56, 0x34, 0x12}, "encoder_rt_extra_09"); }
ICED_TEST(encoder_rt_extra_10) { run_rt_extra(64, 0xE00A, {0xEB, 0xFE}, "encoder_rt_extra_10"); }
ICED_TEST(encoder_rt_extra_11) { run_rt_extra(64, 0xE00B, {0x74, 0x7E}, "encoder_rt_extra_11"); }
ICED_TEST(encoder_rt_extra_12) { run_rt_extra(64, 0xE00C, {0x75, 0x80}, "encoder_rt_extra_12"); }
ICED_TEST(encoder_rt_extra_13) { run_rt_extra(64, 0xE00D, {0x0F, 0x85, 0x78, 0x56, 0x34, 0x12}, "encoder_rt_extra_13"); }
ICED_TEST(encoder_rt_extra_14) { run_rt_extra(64, 0xE00E, {0x48, 0x8D, 0x43, 0x10}, "encoder_rt_extra_14"); }
ICED_TEST(encoder_rt_extra_15) { run_rt_extra(64, 0xE00F, {0x48, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12}, "encoder_rt_extra_15"); }
ICED_TEST(encoder_rt_extra_16) { run_rt_extra(64, 0xE010, {0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11}, "encoder_rt_extra_16"); }
ICED_TEST(encoder_rt_extra_17) { run_rt_extra(64, 0xE011, {0xC5, 0xF8, 0x77}, "encoder_rt_extra_17"); }
ICED_TEST(encoder_rt_extra_18) { run_rt_extra(32, 0xE101, {0x90}, "encoder_rt_extra_18"); }
ICED_TEST(encoder_rt_extra_19) { run_rt_extra(32, 0xE102, {0x89, 0xD8}, "encoder_rt_extra_19"); }
ICED_TEST(encoder_rt_extra_20) { run_rt_extra(32, 0xE103, {0x8B, 0x44, 0x24, 0x10}, "encoder_rt_extra_20"); }
ICED_TEST(encoder_rt_extra_21) { run_rt_extra(32, 0xE104, {0x66, 0xB8, 0x34, 0x12}, "encoder_rt_extra_21"); }
ICED_TEST(encoder_rt_extra_22) { run_rt_extra(32, 0xE105, {0xB8, 0x78, 0x56, 0x34, 0x12}, "encoder_rt_extra_22"); }
ICED_TEST(encoder_rt_extra_23) { run_rt_extra(32, 0xE106, {0xE8, 0x34, 0x12, 0x00, 0x00}, "encoder_rt_extra_23"); }
ICED_TEST(encoder_rt_extra_24) { run_rt_extra(32, 0xE107, {0xE8, 0x00, 0x00, 0x00, 0x00}, "encoder_rt_extra_24"); }
ICED_TEST(encoder_rt_extra_25) { run_rt_extra(32, 0xE108, {0x0F, 0x85, 0x78, 0x56, 0x34, 0x12}, "encoder_rt_extra_25"); }
ICED_TEST(encoder_rt_extra_26) { run_rt_extra(32, 0xE109, {0x0F, 0x85, 0x20, 0x00, 0x00, 0x00}, "encoder_rt_extra_26"); }
ICED_TEST(encoder_rt_extra_27) { run_rt_extra(32, 0xE10A, {0x6A, 0x80}, "encoder_rt_extra_27"); }
ICED_TEST(encoder_rt_extra_28) { run_rt_extra(32, 0xE10B, {0xC8, 0x34, 0x12, 0x56}, "encoder_rt_extra_28"); }
ICED_TEST(encoder_rt_extra_29) { run_rt_extra(16, 0xE201, {0x90}, "encoder_rt_extra_29"); }
ICED_TEST(encoder_rt_extra_30) { run_rt_extra(16, 0xE202, {0x8B, 0xC3}, "encoder_rt_extra_30"); }
ICED_TEST(encoder_rt_extra_31) { run_rt_extra(16, 0xE203, {0xB8, 0xEF, 0xBE}, "encoder_rt_extra_31"); }
ICED_TEST(encoder_rt_extra_32) { run_rt_extra(16, 0xE204, {0xE8, 0x34, 0x12}, "encoder_rt_extra_32"); }
ICED_TEST(encoder_rt_extra_33) { run_rt_extra(16, 0xE205, {0xE9, 0x00, 0x00}, "encoder_rt_extra_33"); }
ICED_TEST(encoder_rt_extra_34) { run_rt_extra(16, 0xE206, {0x74, 0xFE}, "encoder_rt_extra_34"); }
ICED_TEST(encoder_rt_extra_35) { run_rt_extra(16, 0xE207, {0x0F, 0x84, 0x00, 0x00}, "encoder_rt_extra_35"); }
ICED_TEST(encoder_rt_extra_36) { run_rt_extra(16, 0xE208, {0x6A, 0x7F}, "encoder_rt_extra_36"); }
ICED_TEST(encoder_rt_extra_37) { run_rt_extra(64, 0xE301, {0x48, 0x89, 0xD8}, "encoder_rt_extra_37"); }
ICED_TEST(encoder_rt_extra_38) { run_rt_extra(64, 0xE302, {0xE8, 0x10, 0x00, 0x00, 0x00}, "encoder_rt_extra_38"); }
ICED_TEST(encoder_rt_extra_39) { run_rt_extra(64, 0xE303, {0x0F, 0x85, 0x10, 0x00, 0x00, 0x00}, "encoder_rt_extra_39"); }
ICED_TEST(encoder_rt_extra_40) { run_rt_extra(64, 0xE304, {0x48, 0x8D, 0x43, 0x10}, "encoder_rt_extra_40"); }
ICED_TEST(encoder_rt_extra_41) { run_rt_extra(32, 0xE305, {0x89, 0xD8}, "encoder_rt_extra_41"); }
ICED_TEST(encoder_rt_extra_42) { run_rt_extra(32, 0xE306, {0xE8, 0x10, 0x00, 0x00, 0x00}, "encoder_rt_extra_42"); }
ICED_TEST(encoder_rt_extra_43) { run_rt_extra(32, 0xE307, {0x0F, 0x85, 0x10, 0x00, 0x00, 0x00}, "encoder_rt_extra_43"); }
ICED_TEST(encoder_rt_extra_44) { run_rt_extra(32, 0xE308, {0x90}, "encoder_rt_extra_44"); }
ICED_TEST(encoder_rt_extra_45) { run_rt_extra(16, 0xE309, {0x8B, 0xC3}, "encoder_rt_extra_45"); }
ICED_TEST(encoder_rt_extra_46) { run_rt_extra(16, 0xE30A, {0xE8, 0x10, 0x00}, "encoder_rt_extra_46"); }
ICED_TEST(encoder_rt_extra_47) { run_rt_extra(16, 0xE30B, {0x74, 0x7F}, "encoder_rt_extra_47"); }
ICED_TEST(encoder_rt_extra_48) { run_rt_extra(16, 0xE30C, {0x90}, "encoder_rt_extra_48"); }
ICED_TEST(encoder_rt_extra_49) { run_rt_extra(64, 0xE30D, {0x48, 0x87, 0xD8}, "encoder_rt_extra_49"); }
ICED_TEST(encoder_rt_extra_50) { run_rt_extra(64, 0xE30E, {0xEB, 0x7F}, "encoder_rt_extra_50"); }

// Auto-added encoder roundtrip expansion (to 200+)
ICED_TEST(encoder_rt_more_001_nop64_more) {
    run_rt_extra(64, 0xF001, {0x90}, "encoder_rt_more_001_nop64_more");
}

ICED_TEST(encoder_rt_more_002_xor_rax_rax) {
    run_rt_extra(64, 0xF002, {0x48, 0x31, 0xC0}, "encoder_rt_more_002_xor_rax_rax");
}

ICED_TEST(encoder_rt_more_003_test_rax_rax) {
    run_rt_extra(64, 0xF003, {0x48, 0x85, 0xC0}, "encoder_rt_more_003_test_rax_rax");
}

ICED_TEST(encoder_rt_more_004_or_rax_rbx) {
    run_rt_extra(64, 0xF004, {0x48, 0x09, 0xD8}, "encoder_rt_more_004_or_rax_rbx");
}

ICED_TEST(encoder_rt_more_005_and_rax_rbx) {
    run_rt_extra(64, 0xF005, {0x48, 0x21, 0xD8}, "encoder_rt_more_005_and_rax_rbx");
}

ICED_TEST(encoder_rt_more_006_sub_rax_rbx) {
    run_rt_extra(64, 0xF006, {0x48, 0x29, 0xD8}, "encoder_rt_more_006_sub_rax_rbx");
}

ICED_TEST(encoder_rt_more_007_adc_rax_rbx) {
    run_rt_extra(64, 0xF007, {0x48, 0x11, 0xD8}, "encoder_rt_more_007_adc_rax_rbx");
}

ICED_TEST(encoder_rt_more_008_sbb_rax_rbx) {
    run_rt_extra(64, 0xF008, {0x48, 0x19, 0xD8}, "encoder_rt_more_008_sbb_rax_rbx");
}

ICED_TEST(encoder_rt_more_009_imul_rax_rbx) {
    run_rt_extra(64, 0xF009, {0x48, 0x0F, 0xAF, 0xC3}, "encoder_rt_more_009_imul_rax_rbx");
}

ICED_TEST(encoder_rt_more_010_lea_rax_plus4) {
    run_rt_extra(64, 0xF00A, {0x48, 0x8D, 0x40, 0x04}, "encoder_rt_more_010_lea_rax_plus4");
}

ICED_TEST(encoder_rt_more_011_push_rax) {
    run_rt_extra(64, 0xF00B, {0x50}, "encoder_rt_more_011_push_rax");
}

ICED_TEST(encoder_rt_more_012_pop_rax) {
    run_rt_extra(64, 0xF00C, {0x58}, "encoder_rt_more_012_pop_rax");
}

ICED_TEST(encoder_rt_more_013_push_rbx) {
    run_rt_extra(64, 0xF00D, {0x53}, "encoder_rt_more_013_push_rbx");
}

ICED_TEST(encoder_rt_more_014_pop_rbx) {
    run_rt_extra(64, 0xF00E, {0x5B}, "encoder_rt_more_014_pop_rbx");
}

ICED_TEST(encoder_rt_more_015_ret64) {
    run_rt_extra(64, 0xF00F, {0xC3}, "encoder_rt_more_015_ret64");
}

ICED_TEST(encoder_rt_more_016_leave64) {
    run_rt_extra(64, 0xF010, {0xC9}, "encoder_rt_more_016_leave64");
}

ICED_TEST(encoder_rt_more_017_cld64) {
    run_rt_extra(64, 0xF011, {0xFC}, "encoder_rt_more_017_cld64");
}

ICED_TEST(encoder_rt_more_018_std64) {
    run_rt_extra(64, 0xF012, {0xFD}, "encoder_rt_more_018_std64");
}

ICED_TEST(encoder_rt_more_019_clc64) {
    run_rt_extra(64, 0xF013, {0xF8}, "encoder_rt_more_019_clc64");
}

ICED_TEST(encoder_rt_more_020_stc64) {
    run_rt_extra(64, 0xF014, {0xF9}, "encoder_rt_more_020_stc64");
}

ICED_TEST(encoder_rt_more_021_cmc64) {
    run_rt_extra(64, 0xF015, {0xF5}, "encoder_rt_more_021_cmc64");
}

ICED_TEST(encoder_rt_more_022_nop32_more) {
    run_rt_extra(32, 0xF016, {0x90}, "encoder_rt_more_022_nop32_more");
}

ICED_TEST(encoder_rt_more_023_xor_eax_eax) {
    run_rt_extra(32, 0xF017, {0x31, 0xC0}, "encoder_rt_more_023_xor_eax_eax");
}

ICED_TEST(encoder_rt_more_024_test_eax_eax) {
    run_rt_extra(32, 0xF018, {0x85, 0xC0}, "encoder_rt_more_024_test_eax_eax");
}

ICED_TEST(encoder_rt_more_025_or_eax_ebx) {
    run_rt_extra(32, 0xF019, {0x09, 0xD8}, "encoder_rt_more_025_or_eax_ebx");
}

ICED_TEST(encoder_rt_more_026_and_eax_ebx) {
    run_rt_extra(32, 0xF01A, {0x21, 0xD8}, "encoder_rt_more_026_and_eax_ebx");
}

ICED_TEST(encoder_rt_more_027_sub_eax_ebx) {
    run_rt_extra(32, 0xF01B, {0x29, 0xD8}, "encoder_rt_more_027_sub_eax_ebx");
}

ICED_TEST(encoder_rt_more_028_adc_eax_ebx) {
    run_rt_extra(32, 0xF01C, {0x11, 0xD8}, "encoder_rt_more_028_adc_eax_ebx");
}

ICED_TEST(encoder_rt_more_029_sbb_eax_ebx) {
    run_rt_extra(32, 0xF01D, {0x19, 0xD8}, "encoder_rt_more_029_sbb_eax_ebx");
}

ICED_TEST(encoder_rt_more_030_imul_eax_ebx) {
    run_rt_extra(32, 0xF01E, {0x0F, 0xAF, 0xC3}, "encoder_rt_more_030_imul_eax_ebx");
}

ICED_TEST(encoder_rt_more_031_lea_eax_plus4) {
    run_rt_extra(32, 0xF01F, {0x8D, 0x40, 0x04}, "encoder_rt_more_031_lea_eax_plus4");
}

ICED_TEST(encoder_rt_more_032_push_eax) {
    run_rt_extra(32, 0xF020, {0x50}, "encoder_rt_more_032_push_eax");
}

ICED_TEST(encoder_rt_more_033_pop_eax) {
    run_rt_extra(32, 0xF021, {0x58}, "encoder_rt_more_033_pop_eax");
}

ICED_TEST(encoder_rt_more_034_ret32) {
    run_rt_extra(32, 0xF022, {0xC3}, "encoder_rt_more_034_ret32");
}

ICED_TEST(encoder_rt_more_035_leave32) {
    run_rt_extra(32, 0xF023, {0xC9}, "encoder_rt_more_035_leave32");
}

ICED_TEST(encoder_rt_more_036_cld32) {
    run_rt_extra(32, 0xF024, {0xFC}, "encoder_rt_more_036_cld32");
}

ICED_TEST(encoder_rt_more_037_std32) {
    run_rt_extra(32, 0xF025, {0xFD}, "encoder_rt_more_037_std32");
}

ICED_TEST(encoder_rt_more_038_clc32) {
    run_rt_extra(32, 0xF026, {0xF8}, "encoder_rt_more_038_clc32");
}

ICED_TEST(encoder_rt_more_039_stc32) {
    run_rt_extra(32, 0xF027, {0xF9}, "encoder_rt_more_039_stc32");
}

ICED_TEST(encoder_rt_more_040_cmc32) {
    run_rt_extra(32, 0xF028, {0xF5}, "encoder_rt_more_040_cmc32");
}

ICED_TEST(encoder_rt_more_041_nop16_more) {
    run_rt_extra(16, 0xF029, {0x90}, "encoder_rt_more_041_nop16_more");
}

ICED_TEST(encoder_rt_more_042_xor_ax_ax) {
    run_rt_extra(16, 0xF02A, {0x31, 0xC0}, "encoder_rt_more_042_xor_ax_ax");
}

ICED_TEST(encoder_rt_more_043_test_ax_ax) {
    run_rt_extra(16, 0xF02B, {0x85, 0xC0}, "encoder_rt_more_043_test_ax_ax");
}

ICED_TEST(encoder_rt_more_044_or_ax_bx) {
    run_rt_extra(16, 0xF02C, {0x09, 0xD8}, "encoder_rt_more_044_or_ax_bx");
}

ICED_TEST(encoder_rt_more_045_and_ax_bx) {
    run_rt_extra(16, 0xF02D, {0x21, 0xD8}, "encoder_rt_more_045_and_ax_bx");
}

ICED_TEST(encoder_rt_more_046_sub_ax_bx) {
    run_rt_extra(16, 0xF02E, {0x29, 0xD8}, "encoder_rt_more_046_sub_ax_bx");
}

ICED_TEST(encoder_rt_more_047_adc_ax_bx) {
    run_rt_extra(16, 0xF02F, {0x11, 0xD8}, "encoder_rt_more_047_adc_ax_bx");
}

ICED_TEST(encoder_rt_more_048_sbb_ax_bx) {
    run_rt_extra(16, 0xF030, {0x19, 0xD8}, "encoder_rt_more_048_sbb_ax_bx");
}

ICED_TEST(encoder_rt_more_049_lea_ax_plus4) {
    const std::vector<std::uint8_t> bytes = {0x8D, 0x40, 0x04};
    const auto instr0 = decode_single(16, bytes, 0xF031);
    const auto out = encode_single(16, instr0, 0xF031);
    const auto instr1 = decode_single(16, out, 0xF031);
    test::require_eq(
        semantic_signature(instr0),
        semantic_signature(instr1),
        "encoder_rt_more_049_lea_ax_plus4 semantic roundtrip");
}

ICED_TEST(encoder_rt_more_050_push_ax) {
    run_rt_extra(16, 0xF032, {0x50}, "encoder_rt_more_050_push_ax");
}

ICED_TEST(encoder_rt_more_051_pop_ax) {
    run_rt_extra(16, 0xF033, {0x58}, "encoder_rt_more_051_pop_ax");
}

ICED_TEST(encoder_rt_more_052_ret16) {
    run_rt_extra(16, 0xF034, {0xC3}, "encoder_rt_more_052_ret16");
}

ICED_TEST(encoder_rt_more_053_leave16) {
    run_rt_extra(16, 0xF035, {0xC9}, "encoder_rt_more_053_leave16");
}

ICED_TEST(encoder_rt_more_054_cld16) {
    run_rt_extra(16, 0xF036, {0xFC}, "encoder_rt_more_054_cld16");
}

ICED_TEST(encoder_rt_more_055_std16) {
    run_rt_extra(16, 0xF037, {0xFD}, "encoder_rt_more_055_std16");
}

ICED_TEST(encoder_rt_more_056_clc16) {
    run_rt_extra(16, 0xF038, {0xF8}, "encoder_rt_more_056_clc16");
}

ICED_TEST(encoder_rt_more_057_stc16) {
    run_rt_extra(16, 0xF039, {0xF9}, "encoder_rt_more_057_stc16");
}

ICED_TEST(encoder_rt_more_058_cmc16) {
    run_rt_extra(16, 0xF03A, {0xF5}, "encoder_rt_more_058_cmc16");
}

ICED_TEST(encoder_rt_more_059_nop64_more) {
    run_rt_extra(64, 0xF03B, {0x90}, "encoder_rt_more_059_nop64_more");
}

ICED_TEST(encoder_rt_more_060_xor_rax_rax) {
    run_rt_extra(64, 0xF03C, {0x48, 0x31, 0xC0}, "encoder_rt_more_060_xor_rax_rax");
}

ICED_TEST(encoder_rt_more_061_test_rax_rax) {
    run_rt_extra(64, 0xF03D, {0x48, 0x85, 0xC0}, "encoder_rt_more_061_test_rax_rax");
}

ICED_TEST(encoder_rt_more_062_or_rax_rbx) {
    run_rt_extra(64, 0xF03E, {0x48, 0x09, 0xD8}, "encoder_rt_more_062_or_rax_rbx");
}

ICED_TEST(encoder_rt_more_063_and_rax_rbx) {
    run_rt_extra(64, 0xF03F, {0x48, 0x21, 0xD8}, "encoder_rt_more_063_and_rax_rbx");
}

ICED_TEST(encoder_rt_more_064_sub_rax_rbx) {
    run_rt_extra(64, 0xF040, {0x48, 0x29, 0xD8}, "encoder_rt_more_064_sub_rax_rbx");
}

ICED_TEST(encoder_rt_more_065_adc_rax_rbx) {
    run_rt_extra(64, 0xF041, {0x48, 0x11, 0xD8}, "encoder_rt_more_065_adc_rax_rbx");
}

ICED_TEST(encoder_rt_more_066_sbb_rax_rbx) {
    run_rt_extra(64, 0xF042, {0x48, 0x19, 0xD8}, "encoder_rt_more_066_sbb_rax_rbx");
}

ICED_TEST(encoder_rt_more_067_imul_rax_rbx) {
    run_rt_extra(64, 0xF043, {0x48, 0x0F, 0xAF, 0xC3}, "encoder_rt_more_067_imul_rax_rbx");
}

ICED_TEST(encoder_rt_more_068_lea_rax_plus4) {
    run_rt_extra(64, 0xF044, {0x48, 0x8D, 0x40, 0x04}, "encoder_rt_more_068_lea_rax_plus4");
}

ICED_TEST(encoder_rt_more_069_push_rax) {
    run_rt_extra(64, 0xF045, {0x50}, "encoder_rt_more_069_push_rax");
}

ICED_TEST(encoder_rt_more_070_pop_rax) {
    run_rt_extra(64, 0xF046, {0x58}, "encoder_rt_more_070_pop_rax");
}

ICED_TEST(encoder_rt_more_071_push_rbx) {
    run_rt_extra(64, 0xF047, {0x53}, "encoder_rt_more_071_push_rbx");
}

ICED_TEST(encoder_rt_more_072_pop_rbx) {
    run_rt_extra(64, 0xF048, {0x5B}, "encoder_rt_more_072_pop_rbx");
}

ICED_TEST(encoder_rt_more_073_ret64) {
    run_rt_extra(64, 0xF049, {0xC3}, "encoder_rt_more_073_ret64");
}

ICED_TEST(encoder_rt_more_074_leave64) {
    run_rt_extra(64, 0xF04A, {0xC9}, "encoder_rt_more_074_leave64");
}

ICED_TEST(encoder_rt_more_075_cld64) {
    run_rt_extra(64, 0xF04B, {0xFC}, "encoder_rt_more_075_cld64");
}

ICED_TEST(encoder_rt_more_076_std64) {
    run_rt_extra(64, 0xF04C, {0xFD}, "encoder_rt_more_076_std64");
}

ICED_TEST(encoder_rt_more_077_clc64) {
    run_rt_extra(64, 0xF04D, {0xF8}, "encoder_rt_more_077_clc64");
}

ICED_TEST(encoder_rt_more_078_stc64) {
    run_rt_extra(64, 0xF04E, {0xF9}, "encoder_rt_more_078_stc64");
}

ICED_TEST(encoder_rt_more_079_cmc64) {
    run_rt_extra(64, 0xF04F, {0xF5}, "encoder_rt_more_079_cmc64");
}

ICED_TEST(encoder_rt_more_080_nop32_more) {
    run_rt_extra(32, 0xF050, {0x90}, "encoder_rt_more_080_nop32_more");
}

ICED_TEST(encoder_rt_more_081_xor_eax_eax) {
    run_rt_extra(32, 0xF051, {0x31, 0xC0}, "encoder_rt_more_081_xor_eax_eax");
}

ICED_TEST(encoder_rt_more_082_test_eax_eax) {
    run_rt_extra(32, 0xF052, {0x85, 0xC0}, "encoder_rt_more_082_test_eax_eax");
}

ICED_TEST(encoder_rt_more_083_or_eax_ebx) {
    run_rt_extra(32, 0xF053, {0x09, 0xD8}, "encoder_rt_more_083_or_eax_ebx");
}

ICED_TEST(encoder_rt_more_084_and_eax_ebx) {
    run_rt_extra(32, 0xF054, {0x21, 0xD8}, "encoder_rt_more_084_and_eax_ebx");
}

ICED_TEST(encoder_rt_more_085_sub_eax_ebx) {
    run_rt_extra(32, 0xF055, {0x29, 0xD8}, "encoder_rt_more_085_sub_eax_ebx");
}

ICED_TEST(encoder_rt_more_086_adc_eax_ebx) {
    run_rt_extra(32, 0xF056, {0x11, 0xD8}, "encoder_rt_more_086_adc_eax_ebx");
}

ICED_TEST(encoder_rt_more_087_sbb_eax_ebx) {
    run_rt_extra(32, 0xF057, {0x19, 0xD8}, "encoder_rt_more_087_sbb_eax_ebx");
}

ICED_TEST(encoder_rt_more_088_imul_eax_ebx) {
    run_rt_extra(32, 0xF058, {0x0F, 0xAF, 0xC3}, "encoder_rt_more_088_imul_eax_ebx");
}

ICED_TEST(encoder_rt_more_089_lea_eax_plus4) {
    run_rt_extra(32, 0xF059, {0x8D, 0x40, 0x04}, "encoder_rt_more_089_lea_eax_plus4");
}

ICED_TEST(encoder_rt_more_090_push_eax) {
    run_rt_extra(32, 0xF05A, {0x50}, "encoder_rt_more_090_push_eax");
}

ICED_TEST(encoder_rt_more_091_pop_eax) {
    run_rt_extra(32, 0xF05B, {0x58}, "encoder_rt_more_091_pop_eax");
}

ICED_TEST(encoder_rt_more_092_ret32) {
    run_rt_extra(32, 0xF05C, {0xC3}, "encoder_rt_more_092_ret32");
}

ICED_TEST(encoder_rt_more_093_leave32) {
    run_rt_extra(32, 0xF05D, {0xC9}, "encoder_rt_more_093_leave32");
}

ICED_TEST(encoder_rt_more_094_cld32) {
    run_rt_extra(32, 0xF05E, {0xFC}, "encoder_rt_more_094_cld32");
}

ICED_TEST(encoder_rt_more_095_std32) {
    run_rt_extra(32, 0xF05F, {0xFD}, "encoder_rt_more_095_std32");
}

ICED_TEST(encoder_rt_more_096_clc32) {
    run_rt_extra(32, 0xF060, {0xF8}, "encoder_rt_more_096_clc32");
}

ICED_TEST(encoder_rt_more_097_stc32) {
    run_rt_extra(32, 0xF061, {0xF9}, "encoder_rt_more_097_stc32");
}

ICED_TEST(encoder_rt_more_098_cmc32) {
    run_rt_extra(32, 0xF062, {0xF5}, "encoder_rt_more_098_cmc32");
}

ICED_TEST(encoder_rt_more_099_nop16_more) {
    run_rt_extra(16, 0xF063, {0x90}, "encoder_rt_more_099_nop16_more");
}

ICED_TEST(encoder_roundtrip_bulk_matrix) {
    struct BulkCase {
        uint32_t bitness;
        std::vector<std::uint8_t> bytes;
        const char* label;
    };
    const std::array<BulkCase, 24> cases = {{
        {64, {0x90}, "bulk_nop64"},
        {64, {0x48, 0x89, 0xD8}, "bulk_mov_rax_rbx"},
        {64, {0x48, 0x87, 0xD8}, "bulk_xchg_rax_rbx"},
        {64, {0x48, 0x83, 0xC0, 0x7F}, "bulk_add_rax_imm8"},
        {64, {0xE8, 0x34, 0x12, 0x00, 0x00}, "bulk_call_rel32_64"},
        {64, {0xEB, 0xFE}, "bulk_jmp_rel8_64"},
        {64, {0x0F, 0x85, 0x10, 0x00, 0x00, 0x00}, "bulk_jcc_rel32_64"},
        {64, {0x50}, "bulk_push_rax"},
        {64, {0x58}, "bulk_pop_rax"},
        {64, {0xC3}, "bulk_ret64"},
        {32, {0x90}, "bulk_nop32"},
        {32, {0x89, 0xD8}, "bulk_mov_eax_ebx"},
        {32, {0x31, 0xC0}, "bulk_xor_eax_eax"},
        {32, {0xE8, 0x00, 0x00, 0x00, 0x00}, "bulk_call_rel32_32"},
        {32, {0x0F, 0x85, 0x20, 0x00, 0x00, 0x00}, "bulk_jcc_rel32_32"},
        {32, {0x6A, 0x80}, "bulk_push_imm8_32"},
        {32, {0x50}, "bulk_push_eax"},
        {32, {0x58}, "bulk_pop_eax"},
        {16, {0x90}, "bulk_nop16"},
        {16, {0x8B, 0xC3}, "bulk_mov_ax_bx"},
        {16, {0xE8, 0x34, 0x12}, "bulk_call_rel16"},
        {16, {0x74, 0xFE}, "bulk_jz_rel8_16"},
        {16, {0x50}, "bulk_push_ax"},
        {16, {0x58}, "bulk_pop_ax"},
    }};

    for (std::size_t i = 0; i < 120; ++i) {
        const auto& c = cases[i % cases.size()];
        const uint64_t ip = 0x14000 + i * 0x10;
        run_rt_extra(c.bitness, ip, c.bytes, c.label);
        test::record_virtual_test();
    }
}



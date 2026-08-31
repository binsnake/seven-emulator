#include "seven/flag_liveness.hpp"

#include <iced_x86/code.hpp>
#include <iced_x86/op_kind.hpp>

#include "seven/types.hpp"

namespace seven {

namespace {

// Per-instruction ALU flag sets, hand-verified against seven's handlers since iced_x86's are stubs
// here. `written` may only list a bit GUARANTEED written, never a conditional one; `read` is safe to
// overclaim; anything absent defaults to {read=all, written=0}.
struct FlagsInfo {
  std::uint64_t read = kAluStatusFlagsMask;
  std::uint64_t written = 0;
};

[[nodiscard]] constexpr FlagsInfo flags_info_for_code(iced_x86::Code code) noexcept {
  using iced_x86::Code;
  constexpr std::uint64_t kAll6 = kAluStatusFlagsMask;
  constexpr std::uint64_t kNone = 0;
  constexpr std::uint64_t kCF = kFlagCF;
  constexpr std::uint64_t kZF = kFlagZF;
  constexpr std::uint64_t kAF = kFlagAF;
  constexpr std::uint64_t kCondFlags = kFlagCF | kFlagPF | kFlagZF | kFlagSF | kFlagOF;  // no AF -- no x86 condition code tests it
  constexpr std::uint64_t kIncDecFlags = kFlagOF | kFlagSF | kFlagZF | kFlagAF | kFlagPF;  // no CF -- INC/DEC preserve it
  constexpr std::uint64_t kRotateCarryOverflow = kFlagCF | kFlagOF;
  constexpr std::uint64_t kAsciiAdjustFlags = kFlagCF | kFlagAF | kFlagZF | kFlagSF | kFlagPF;  // no OF
  constexpr std::uint64_t kLahfSahfFlags = kFlagSF | kFlagZF | kFlagAF | kFlagPF | kFlagCF;

  switch (code) {
    // All overwrite every ALU status flag unconditionally, with no incoming-flag dependency.
    case Code::ADD_RM8_R8: case Code::ADD_RM16_R16: case Code::ADD_RM32_R32: case Code::ADD_RM64_R64:
    case Code::ADD_R8_RM8: case Code::ADD_R16_RM16: case Code::ADD_R32_RM32: case Code::ADD_R64_RM64:
    case Code::ADD_RM8_IMM8: case Code::ADD_RM8_IMM8_82: case Code::ADD_RM16_IMM16: case Code::ADD_RM32_IMM32:
    case Code::ADD_RM64_IMM32: case Code::ADD_AL_IMM8: case Code::ADD_AX_IMM16: case Code::ADD_EAX_IMM32:
    case Code::ADD_RAX_IMM32: case Code::ADD_RM16_IMM8: case Code::ADD_RM32_IMM8: case Code::ADD_RM64_IMM8:
    case Code::SUB_RM8_R8: case Code::SUB_RM16_R16: case Code::SUB_RM32_R32: case Code::SUB_RM64_R64:
    case Code::SUB_R8_RM8: case Code::SUB_R16_RM16: case Code::SUB_R32_RM32: case Code::SUB_R64_RM64:
    case Code::SUB_RM8_IMM8: case Code::SUB_RM8_IMM8_82: case Code::SUB_RM16_IMM16: case Code::SUB_RM32_IMM32:
    case Code::SUB_RM64_IMM32: case Code::SUB_AL_IMM8: case Code::SUB_AX_IMM16: case Code::SUB_EAX_IMM32:
    case Code::SUB_RAX_IMM32: case Code::SUB_RM16_IMM8: case Code::SUB_RM32_IMM8: case Code::SUB_RM64_IMM8:
    case Code::CMP_RM8_R8: case Code::CMP_RM16_R16: case Code::CMP_RM32_R32: case Code::CMP_RM64_R64:
    case Code::CMP_R8_RM8: case Code::CMP_R16_RM16: case Code::CMP_R32_RM32: case Code::CMP_R64_RM64:
    case Code::CMP_RM8_IMM8: case Code::CMP_RM8_IMM8_82: case Code::CMP_RM16_IMM16: case Code::CMP_RM32_IMM32:
    case Code::CMP_RM64_IMM32: case Code::CMP_AL_IMM8: case Code::CMP_AX_IMM16: case Code::CMP_EAX_IMM32:
    case Code::CMP_RAX_IMM32: case Code::CMP_RM16_IMM8: case Code::CMP_RM32_IMM8: case Code::CMP_RM64_IMM8:
    case Code::AND_RM8_R8: case Code::AND_RM16_R16: case Code::AND_RM32_R32: case Code::AND_RM64_R64:
    case Code::AND_R8_RM8: case Code::AND_R16_RM16: case Code::AND_R32_RM32: case Code::AND_R64_RM64:
    case Code::AND_RM8_IMM8: case Code::AND_RM8_IMM8_82: case Code::AND_RM16_IMM16: case Code::AND_RM32_IMM32:
    case Code::AND_RM64_IMM32: case Code::AND_AL_IMM8: case Code::AND_AX_IMM16: case Code::AND_EAX_IMM32:
    case Code::AND_RAX_IMM32: case Code::AND_RM16_IMM8: case Code::AND_RM32_IMM8: case Code::AND_RM64_IMM8:
    case Code::OR_RM8_R8: case Code::OR_RM16_R16: case Code::OR_RM32_R32: case Code::OR_RM64_R64:
    case Code::OR_R8_RM8: case Code::OR_R16_RM16: case Code::OR_R32_RM32: case Code::OR_R64_RM64:
    case Code::OR_RM8_IMM8: case Code::OR_RM8_IMM8_82: case Code::OR_RM16_IMM16: case Code::OR_RM32_IMM32:
    case Code::OR_RM64_IMM32: case Code::OR_AL_IMM8: case Code::OR_AX_IMM16: case Code::OR_EAX_IMM32:
    case Code::OR_RAX_IMM32: case Code::OR_RM16_IMM8: case Code::OR_RM32_IMM8: case Code::OR_RM64_IMM8:
    case Code::XOR_RM8_R8: case Code::XOR_R8_RM8: case Code::XOR_RM16_R16: case Code::XOR_RM32_R32:
    case Code::XOR_RM64_R64: case Code::XOR_R16_RM16: case Code::XOR_R32_RM32: case Code::XOR_R64_RM64:
    case Code::XOR_RM8_IMM8: case Code::XOR_RM8_IMM8_82: case Code::XOR_RM16_IMM16: case Code::XOR_RM32_IMM32:
    case Code::XOR_RM64_IMM32: case Code::XOR_AL_IMM8: case Code::XOR_AX_IMM16: case Code::XOR_EAX_IMM32:
    case Code::XOR_RAX_IMM32: case Code::XOR_RM16_IMM8: case Code::XOR_RM32_IMM8: case Code::XOR_RM64_IMM8:
    case Code::TEST_RM8_R8: case Code::TEST_RM16_R16: case Code::TEST_RM32_R32: case Code::TEST_RM64_R64:
    case Code::TEST_RM8_IMM8: case Code::TEST_RM16_IMM16: case Code::TEST_RM32_IMM32: case Code::TEST_RM64_IMM32:
    case Code::TEST_AL_IMM8: case Code::TEST_AX_IMM16: case Code::TEST_EAX_IMM32: case Code::TEST_RAX_IMM32:
    case Code::TEST_RM8_IMM8_F6R1: case Code::TEST_RM16_IMM16_F7R1: case Code::TEST_RM32_IMM32_F7R1: case Code::TEST_RM64_IMM32_F7R1:
    case Code::XADD_RM8_R8: case Code::XADD_RM16_R16: case Code::XADD_RM32_R32: case Code::XADD_RM64_R64:
    case Code::CMPXCHG_RM8_R8: case Code::CMPXCHG_RM16_R16: case Code::CMPXCHG_RM32_R32: case Code::CMPXCHG_RM64_R64:
    case Code::NEG_RM8: case Code::NEG_RM16: case Code::NEG_RM32: case Code::NEG_RM64:
    case Code::MUL_RM8: case Code::MUL_RM16: case Code::MUL_RM32: case Code::MUL_RM64:
    case Code::IMUL_RM8: case Code::IMUL_RM16: case Code::IMUL_RM32: case Code::IMUL_RM64:
    case Code::IMUL_R16_RM16: case Code::IMUL_R16_RM16_IMM16: case Code::IMUL_R16_RM16_IMM8:
    case Code::IMUL_R32_RM32: case Code::IMUL_R32_RM32_IMM32: case Code::IMUL_R32_RM32_IMM8:
    case Code::IMUL_R64_RM64: case Code::IMUL_R64_RM64_IMM32: case Code::IMUL_R64_RM64_IMM8:
    case Code::POPCNT_R16_RM16: case Code::POPCNT_R32_RM32: case Code::POPCNT_R64_RM64:
    case Code::TZCNT_R16_RM16: case Code::TZCNT_R32_RM32: case Code::TZCNT_R64_RM64:
    case Code::LZCNT_R16_RM16: case Code::LZCNT_R32_RM32: case Code::LZCNT_R64_RM64:
    case Code::VEX_BLSI_R32_RM32: case Code::VEX_BLSI_R64_RM64:
    case Code::VEX_BLSR_R32_RM32: case Code::VEX_BLSR_R64_RM64:
    case Code::VEX_BLSMSK_R32_RM32: case Code::VEX_BLSMSK_R64_RM64:
    case Code::VEX_ANDN_R32_R32_RM32: case Code::VEX_ANDN_R64_R64_RM64:
    case Code::VEX_BZHI_R32_RM32_R32: case Code::VEX_BZHI_R64_RM64_R64:
    case Code::VEX_BEXTR_R32_RM32_R32: case Code::VEX_BEXTR_R64_RM64_R64:
    case Code::SHL_RM8_1: case Code::SHL_RM16_1: case Code::SHL_RM32_1: case Code::SHL_RM64_1:
    case Code::SAL_RM8_1: case Code::SAL_RM16_1: case Code::SAL_RM32_1: case Code::SAL_RM64_1:
    case Code::SHR_RM8_1: case Code::SHR_RM16_1: case Code::SHR_RM32_1: case Code::SHR_RM64_1:
    case Code::SAR_RM8_1: case Code::SAR_RM16_1: case Code::SAR_RM32_1: case Code::SAR_RM64_1:
    case Code::POPFQ:
      return {kNone, kAll6};

    // ADC/SBB: written=all6, but both read incoming CF (carry_in/borrow_in) before computing.
    case Code::ADC_RM8_R8: case Code::ADC_RM16_R16: case Code::ADC_RM32_R32: case Code::ADC_RM64_R64:
    case Code::ADC_R8_RM8: case Code::ADC_R16_RM16: case Code::ADC_R32_RM32: case Code::ADC_R64_RM64:
    case Code::ADC_RM8_IMM8: case Code::ADC_RM8_IMM8_82: case Code::ADC_RM16_IMM16: case Code::ADC_RM32_IMM32:
    case Code::ADC_RM64_IMM32: case Code::ADC_AL_IMM8: case Code::ADC_AX_IMM16: case Code::ADC_EAX_IMM32:
    case Code::ADC_RAX_IMM32: case Code::ADC_RM16_IMM8: case Code::ADC_RM32_IMM8: case Code::ADC_RM64_IMM8:
    case Code::SBB_RM8_R8: case Code::SBB_RM16_R16: case Code::SBB_RM32_R32: case Code::SBB_RM64_R64:
    case Code::SBB_R8_RM8: case Code::SBB_R16_RM16: case Code::SBB_R32_RM32: case Code::SBB_R64_RM64:
    case Code::SBB_RM8_IMM8: case Code::SBB_RM8_IMM8_82: case Code::SBB_RM16_IMM16: case Code::SBB_RM32_IMM32:
    case Code::SBB_RM64_IMM32: case Code::SBB_AL_IMM8: case Code::SBB_AX_IMM16: case Code::SBB_EAX_IMM32:
    case Code::SBB_RAX_IMM32: case Code::SBB_RM16_IMM8: case Code::SBB_RM32_IMM8: case Code::SBB_RM64_IMM8:
      return {kCF, kAll6};

    // INC/DEC: everything but CF, which they deliberately preserve.
    case Code::INC_RM8: case Code::INC_RM16: case Code::INC_RM32: case Code::INC_RM64:
    case Code::DEC_RM8: case Code::DEC_RM16: case Code::DEC_RM32: case Code::DEC_RM64:
      return {kNone, kIncDecFlags};

    // No flag write at all. The CL/imm8 shift forms are here because a count masking to zero skips
    // the write entirely, so they must not be modelled as writing.
    case Code::NOT_RM8: case Code::NOT_RM16: case Code::NOT_RM32: case Code::NOT_RM64:
    case Code::SHL_RM8_CL: case Code::SHL_RM16_CL: case Code::SHL_RM32_CL: case Code::SHL_RM64_CL:
    case Code::SHL_RM8_IMM8: case Code::SHL_RM16_IMM8: case Code::SHL_RM32_IMM8: case Code::SHL_RM64_IMM8:
    case Code::SAL_RM8_CL: case Code::SAL_RM16_CL: case Code::SAL_RM32_CL: case Code::SAL_RM64_CL:
    case Code::SAL_RM8_IMM8: case Code::SAL_RM16_IMM8: case Code::SAL_RM32_IMM8: case Code::SAL_RM64_IMM8:
    case Code::SHR_RM8_CL: case Code::SHR_RM16_CL: case Code::SHR_RM32_CL: case Code::SHR_RM64_CL:
    case Code::SHR_RM8_IMM8: case Code::SHR_RM16_IMM8: case Code::SHR_RM32_IMM8: case Code::SHR_RM64_IMM8:
    case Code::SAR_RM8_CL: case Code::SAR_RM16_CL: case Code::SAR_RM32_CL: case Code::SAR_RM64_CL:
    case Code::SAR_RM8_IMM8: case Code::SAR_RM16_IMM8: case Code::SAR_RM32_IMM8: case Code::SAR_RM64_IMM8:
    case Code::ROL_RM8_CL: case Code::ROL_RM16_CL: case Code::ROL_RM32_CL: case Code::ROL_RM64_CL:
    case Code::ROL_RM8_IMM8: case Code::ROL_RM16_IMM8: case Code::ROL_RM32_IMM8: case Code::ROL_RM64_IMM8:
    case Code::ROR_RM8_CL: case Code::ROR_RM16_CL: case Code::ROR_RM32_CL: case Code::ROR_RM64_CL:
    case Code::ROR_RM8_IMM8: case Code::ROR_RM16_IMM8: case Code::ROR_RM32_IMM8: case Code::ROR_RM64_IMM8:
    case Code::SHLD_RM16_R16_IMM8: case Code::SHLD_RM32_R32_IMM8: case Code::SHLD_RM64_R64_IMM8:
    case Code::SHLD_RM16_R16_CL: case Code::SHLD_RM32_R32_CL: case Code::SHLD_RM64_R64_CL:
    case Code::SHRD_RM16_R16_IMM8: case Code::SHRD_RM32_R32_IMM8: case Code::SHRD_RM64_R64_IMM8:
    case Code::SHRD_RM16_R16_CL: case Code::SHRD_RM32_R32_CL: case Code::SHRD_RM64_R64_CL:
    case Code::DIV_RM8: case Code::DIV_RM16: case Code::DIV_RM32: case Code::DIV_RM64:
    case Code::IDIV_RM8: case Code::IDIV_RM16: case Code::IDIV_RM32: case Code::IDIV_RM64:
    case Code::CRC32_R32_RM8: case Code::CRC32_R64_RM8: case Code::CRC32_R32_RM16:
    case Code::CRC32_R32_RM32: case Code::CRC32_R64_RM64:
    case Code::LOOP_REL8_16_CX: case Code::LOOP_REL8_32_CX: case Code::LOOP_REL8_16_ECX:
    case Code::LOOP_REL8_32_ECX: case Code::LOOP_REL8_64_ECX: case Code::LOOP_REL8_16_RCX: case Code::LOOP_REL8_64_RCX:
    case Code::JECXZ_REL8_16: case Code::JECXZ_REL8_32: case Code::JECXZ_REL8_64:
    case Code::JRCXZ_REL8_16: case Code::JRCXZ_REL8_64:
      return {kNone, kNone};

    // ROL/ROR "_1" forms: only CF/OF, never AF/ZF/SF/PF.
    case Code::ROL_RM8_1: case Code::ROL_RM16_1: case Code::ROL_RM32_1: case Code::ROL_RM64_1:
    case Code::ROR_RM8_1: case Code::ROR_RM16_1: case Code::ROR_RM32_1: case Code::ROR_RM64_1:
      return {kNone, kRotateCarryOverflow};

    // RCL/RCR "_1" forms: CF/OF written, CF also read (rotate-through-carry).
    case Code::RCL_RM8_1: case Code::RCL_RM16_1: case Code::RCL_RM32_1: case Code::RCL_RM64_1:
    case Code::RCR_RM8_1: case Code::RCR_RM16_1: case Code::RCR_RM32_1: case Code::RCR_RM64_1:
      return {kCF, kRotateCarryOverflow};

    // RCL/RCR "_CL"/"_IMM8" forms: count can mask to zero (no write), but CF is always read.
    case Code::RCL_RM8_CL: case Code::RCL_RM16_CL: case Code::RCL_RM32_CL: case Code::RCL_RM64_CL:
    case Code::RCL_RM8_IMM8: case Code::RCL_RM16_IMM8: case Code::RCL_RM32_IMM8: case Code::RCL_RM64_IMM8:
    case Code::RCR_RM8_CL: case Code::RCR_RM16_CL: case Code::RCR_RM32_CL: case Code::RCR_RM64_CL:
    case Code::RCR_RM8_IMM8: case Code::RCR_RM16_IMM8: case Code::RCR_RM32_IMM8: case Code::RCR_RM64_IMM8:
      return {kCF, kNone};

    // Jcc / SETcc / CMOVcc: read the 5 flags their condition needs (never AF), write nothing.
    case Code::JO_REL8_64: case Code::JNO_REL8_64: case Code::JB_REL8_64: case Code::JAE_REL8_64:
    case Code::JE_REL8_64: case Code::JNE_REL8_64: case Code::JBE_REL8_64: case Code::JA_REL8_64:
    case Code::JS_REL8_64: case Code::JNS_REL8_64: case Code::JP_REL8_64: case Code::JNP_REL8_64:
    case Code::JL_REL8_64: case Code::JGE_REL8_64: case Code::JLE_REL8_64: case Code::JG_REL8_64:
    case Code::JO_REL32_64: case Code::JNO_REL32_64: case Code::JB_REL32_64: case Code::JAE_REL32_64:
    case Code::JE_REL32_64: case Code::JNE_REL32_64: case Code::JBE_REL32_64: case Code::JA_REL32_64:
    case Code::JS_REL32_64: case Code::JNS_REL32_64: case Code::JP_REL32_64: case Code::JNP_REL32_64:
    case Code::JL_REL32_64: case Code::JGE_REL32_64: case Code::JLE_REL32_64: case Code::JG_REL32_64:
    case Code::JA_REL16: case Code::JA_REL32_32: case Code::JA_REL8_16: case Code::JA_REL8_32:
    case Code::JAE_REL16: case Code::JAE_REL32_32: case Code::JAE_REL8_16: case Code::JAE_REL8_32:
    case Code::JB_REL16: case Code::JB_REL32_32: case Code::JB_REL8_16: case Code::JB_REL8_32:
    case Code::JBE_REL16: case Code::JBE_REL32_32: case Code::JBE_REL8_16: case Code::JBE_REL8_32:
    case Code::JE_REL16: case Code::JE_REL32_32: case Code::JE_REL8_16: case Code::JE_REL8_32:
    case Code::JG_REL16: case Code::JG_REL32_32: case Code::JG_REL8_16: case Code::JG_REL8_32:
    case Code::JGE_REL16: case Code::JGE_REL32_32: case Code::JGE_REL8_16: case Code::JGE_REL8_32:
    case Code::JL_REL16: case Code::JL_REL32_32: case Code::JL_REL8_16: case Code::JL_REL8_32:
    case Code::JLE_REL16: case Code::JLE_REL32_32: case Code::JLE_REL8_16: case Code::JLE_REL8_32:
    case Code::JNE_REL16: case Code::JNE_REL32_32: case Code::JNE_REL8_16: case Code::JNE_REL8_32:
    case Code::JNO_REL16: case Code::JNO_REL32_32: case Code::JNO_REL8_16: case Code::JNO_REL8_32:
    case Code::JNP_REL16: case Code::JNP_REL32_32: case Code::JNP_REL8_16: case Code::JNP_REL8_32:
    case Code::JNS_REL16: case Code::JNS_REL32_32: case Code::JNS_REL8_16: case Code::JNS_REL8_32:
    case Code::JO_REL16: case Code::JO_REL32_32: case Code::JO_REL8_16: case Code::JO_REL8_32:
    case Code::JP_REL16: case Code::JP_REL32_32: case Code::JP_REL8_16: case Code::JP_REL8_32:
    case Code::JS_REL16: case Code::JS_REL32_32: case Code::JS_REL8_16: case Code::JS_REL8_32:
    case Code::CMOVO_R16_RM16: case Code::CMOVO_R32_RM32: case Code::CMOVO_R64_RM64:
    case Code::CMOVNO_R16_RM16: case Code::CMOVNO_R32_RM32: case Code::CMOVNO_R64_RM64:
    case Code::CMOVB_R16_RM16: case Code::CMOVB_R32_RM32: case Code::CMOVB_R64_RM64:
    case Code::CMOVAE_R16_RM16: case Code::CMOVAE_R32_RM32: case Code::CMOVAE_R64_RM64:
    case Code::CMOVE_R16_RM16: case Code::CMOVE_R32_RM32: case Code::CMOVE_R64_RM64:
    case Code::CMOVNE_R16_RM16: case Code::CMOVNE_R32_RM32: case Code::CMOVNE_R64_RM64:
    case Code::CMOVBE_R16_RM16: case Code::CMOVBE_R32_RM32: case Code::CMOVBE_R64_RM64:
    case Code::CMOVA_R16_RM16: case Code::CMOVA_R32_RM32: case Code::CMOVA_R64_RM64:
    case Code::CMOVS_R16_RM16: case Code::CMOVS_R32_RM32: case Code::CMOVS_R64_RM64:
    case Code::CMOVNS_R16_RM16: case Code::CMOVNS_R32_RM32: case Code::CMOVNS_R64_RM64:
    case Code::CMOVP_R16_RM16: case Code::CMOVP_R32_RM32: case Code::CMOVP_R64_RM64:
    case Code::CMOVNP_R16_RM16: case Code::CMOVNP_R32_RM32: case Code::CMOVNP_R64_RM64:
    case Code::CMOVL_R16_RM16: case Code::CMOVL_R32_RM32: case Code::CMOVL_R64_RM64:
    case Code::CMOVGE_R16_RM16: case Code::CMOVGE_R32_RM32: case Code::CMOVGE_R64_RM64:
    case Code::CMOVLE_R16_RM16: case Code::CMOVLE_R32_RM32: case Code::CMOVLE_R64_RM64:
    case Code::CMOVG_R16_RM16: case Code::CMOVG_R32_RM32: case Code::CMOVG_R64_RM64:
    case Code::SETO_RM8: case Code::SETNO_RM8: case Code::SETB_RM8: case Code::SETAE_RM8:
    case Code::SETE_RM8: case Code::SETNE_RM8: case Code::SETBE_RM8: case Code::SETA_RM8:
    case Code::SETS_RM8: case Code::SETNS_RM8: case Code::SETP_RM8: case Code::SETNP_RM8:
    case Code::SETL_RM8: case Code::SETGE_RM8: case Code::SETLE_RM8: case Code::SETG_RM8:
      return {kCondFlags, kNone};

    // LOOPE/LOOPNE additionally read ZF to decide whether to keep looping.
    case Code::LOOPE_REL8_16_CX: case Code::LOOPE_REL8_32_CX: case Code::LOOPE_REL8_16_ECX:
    case Code::LOOPE_REL8_32_ECX: case Code::LOOPE_REL8_64_ECX: case Code::LOOPE_REL8_16_RCX: case Code::LOOPE_REL8_64_RCX:
    case Code::LOOPNE_REL8_16_CX: case Code::LOOPNE_REL8_32_CX: case Code::LOOPNE_REL8_16_ECX:
    case Code::LOOPNE_REL8_32_ECX: case Code::LOOPNE_REL8_64_ECX: case Code::LOOPNE_REL8_16_RCX: case Code::LOOPNE_REL8_64_RCX:
      return {kZF, kNone};

    // BT: only ever writes CF (the tested bit), from the tested value, not from incoming flags.
    case Code::BT_RM16_IMM8: case Code::BT_RM16_R16: case Code::BT_RM32_IMM8:
    case Code::BT_RM32_R32: case Code::BT_RM64_IMM8: case Code::BT_RM64_R64:
      return {kNone, kCF};

    // BSF/BSR: this handler only ever touches ZF.
    case Code::BSF_R16_RM16: case Code::BSF_R32_RM32: case Code::BSF_R64_RM64:
    case Code::BSR_R16_RM16: case Code::BSR_R32_RM32: case Code::BSR_R64_RM64:
    case Code::CMPXCHG8B_M64: case Code::CMPXCHG16B_M128:
      return {kNone, kZF};

    // DAA/DAS: read CF and AF unconditionally at entry, write everything but OF.
    case Code::DAA: case Code::DAS:
      return {kCF | kAF, kAsciiAdjustFlags};

    // AAA/AAS: read AF only, write everything but OF.
    case Code::AAA: case Code::AAS:
      return {kAF, kAsciiAdjustFlags};

    // AAM/AAD: force CF/AF=false and compute ZF/SF/PF, no dependency on incoming flags.
    case Code::AAM_IMM8: case Code::AAD_IMM8:
      return {kNone, kAsciiAdjustFlags};

    // REPE/REPNE CMPS/SCAS forms read ZF (loop-continue test) in addition to the compare write.
    case Code::CMPSB_M8_M8: case Code::CMPSW_M16_M16: case Code::CMPSD_M32_M32: case Code::CMPSQ_M64_M64:
    case Code::SCASB_AL_M8: case Code::SCASW_AX_M16: case Code::SCASD_EAX_M32: case Code::SCASQ_RAX_M64:
      return {kZF, kAll6};

    // PUSHFx: reads (pushes) the whole flags register, writes nothing.
    case Code::PUSHFQ: case Code::PUSHFD: case Code::PUSHFW:
      return {kAll6, kNone};

    // LAHF: reads 5 flags into AH, doesn't touch rflags itself.
    case Code::LAHF:
      return {kLahfSahfFlags, kNone};

    // SAHF: unconditionally overwrites those same 5 flags from AH.
    case Code::SAHF:
      return {kNone, kLahfSahfFlags};

    case Code::CLC: case Code::STC:
      return {kNone, kCF};
    case Code::CMC:
      return {kCF, kCF};

    default:
      return {};
  }
}

}  // namespace

// Whether this instruction can fault, which keeps liveness conservative across it since a fault
// means the covering instruction never runs. Taking the decoded instruction rather than the Code is
// what makes it precise.
bool can_fault(const iced_x86::Instruction& instr) noexcept {
  const auto op_count = instr.op_count();
  for (std::uint32_t i = 0; i < op_count; ++i) {
    // Every memory operand kind, not just OpKind::MEMORY: the implicit rsi/rdi operands have their
    // own distinct kinds, so == MEMORY missed the string instructions entirely. They sit together at
    // the top of the enum with MEMORY last, so one comparison covers all of them.
    if (instr.op_kind(i) >= iced_x86::OpKind::MEMORY_SEG_SI) {
      return true;
    }
  }
  switch (instr.code()) {
    case iced_x86::Code::DIV_RM8: case iced_x86::Code::DIV_RM16:
    case iced_x86::Code::DIV_RM32: case iced_x86::Code::DIV_RM64:
    case iced_x86::Code::IDIV_RM8: case iced_x86::Code::IDIV_RM16:
    case iced_x86::Code::IDIV_RM32: case iced_x86::Code::IDIV_RM64:
    // AAM raises #DE on a zero immediate, making it the third divide-error source. Its flags entry
    // claims an unconditional write, so liveness masked earlier writes as covered while `aam 0`
    // returns before setting any of them.
    case iced_x86::Code::AAM_IMM8:
      return true;  // divide error is a property of the VALUE, independent of operand kind

    // CALL/RET touch the stack implicitly: the operand loop only sees the branch target, never the
    // push or pop, so they need their own list.
    case iced_x86::Code::CALL_REL16: case iced_x86::Code::CALL_REL32_32: case iced_x86::Code::CALL_REL32_64:
    case iced_x86::Code::CALL_RM16: case iced_x86::Code::CALL_RM32: case iced_x86::Code::CALL_RM64:
    case iced_x86::Code::CALL_PTR1616: case iced_x86::Code::CALL_PTR1632:
    case iced_x86::Code::RETNW: case iced_x86::Code::RETND: case iced_x86::Code::RETNQ:
    case iced_x86::Code::RETNW_IMM16: case iced_x86::Code::RETND_IMM16: case iced_x86::Code::RETNQ_IMM16:
    case iced_x86::Code::RETFW: case iced_x86::Code::RETFD: case iced_x86::Code::RETFQ:
    case iced_x86::Code::RETFW_IMM16: case iced_x86::Code::RETFD_IMM16: case iced_x86::Code::RETFQ_IMM16:
      return true;

    // The decoder is inconsistent about MASKMOVDQU's implicit ES:[rDI] destination: the non-VEX form
    // gets a memory kind the loop catches, the VEX form gets a plain REGISTER holding RDI. Both
    // write up to 16 bytes of guest memory.
    case iced_x86::Code::VEX_VMASKMOVDQU_R_DI_XMM_XMM:
      return true;

    // Same implicit-stack-access gap as CALL/RET, just for the rest of the instructions that push
    // or pop through rsp without an explicit memory operand.
    case iced_x86::Code::PUSH_R16: case iced_x86::Code::PUSH_R32: case iced_x86::Code::PUSH_R64:
    case iced_x86::Code::POP_R16: case iced_x86::Code::POP_R32: case iced_x86::Code::POP_R64:
    case iced_x86::Code::PUSH_IMM16: case iced_x86::Code::PUSHD_IMM32: case iced_x86::Code::PUSHQ_IMM32:
    case iced_x86::Code::PUSHD_IMM8: case iced_x86::Code::PUSHQ_IMM8:
    case iced_x86::Code::PUSHAW: case iced_x86::Code::PUSHAD:
    case iced_x86::Code::POPAW: case iced_x86::Code::POPAD:
    case iced_x86::Code::ENTERW_IMM16_IMM8: case iced_x86::Code::ENTERD_IMM16_IMM8: case iced_x86::Code::ENTERQ_IMM16_IMM8:
    case iced_x86::Code::LEAVEW: case iced_x86::Code::LEAVED: case iced_x86::Code::LEAVEQ:
    case iced_x86::Code::XLAT_M8:
    // PUSHF/POPF touch rsp with op_count() == 0, so neither the operand loop nor the PUSH/POP list
    // saw them. Leaving them out also made POPFQ callout-eligible, which let a guest set rflags.TF
    // mid-block.
    case iced_x86::Code::PUSHFW: case iced_x86::Code::PUSHFD: case iced_x86::Code::PUSHFQ:
    case iced_x86::Code::POPFW: case iced_x86::Code::POPFD: case iced_x86::Code::POPFQ:
      return true;

    // Two REGISTER-kind operands, so the loop cannot see them, but both directions fault: #GP at
    // CPL>0, and #UD for a reserved CR or for DR4/DR5 with CR4.DE set.
    case iced_x86::Code::MOV_R32_CR: case iced_x86::Code::MOV_R64_CR:
    case iced_x86::Code::MOV_CR_R32: case iced_x86::Code::MOV_CR_R64:
    case iced_x86::Code::MOV_R32_DR: case iced_x86::Code::MOV_R64_DR:
    case iced_x86::Code::MOV_DR_R32: case iced_x86::Code::MOV_DR_R64:
      return true;

    // Same gap: fixed-register operands with no memory kind, but #GP at CPL>0.
    case iced_x86::Code::CLTS: case iced_x86::Code::SWAPGS:
    case iced_x86::Code::WRMSR: case iced_x86::Code::WRMSRNS: case iced_x86::Code::WRMSRLIST:
    case iced_x86::Code::RDMSR: case iced_x86::Code::RDMSRLIST:
    case iced_x86::Code::XSETBV:
    case iced_x86::Code::CLI: case iced_x86::Code::STI:
    case iced_x86::Code::WRFSBASE_R64: case iced_x86::Code::WRGSBASE_R64:
    // CPL0-only for the same reason. SYSRET is the sharp one: it writes all of rflags, so liveness
    // treats earlier writes as covered by a cover that a #GP means never happens.
    case iced_x86::Code::SYSRETD: case iced_x86::Code::SYSRETQ:
    case iced_x86::Code::SYSEXITD: case iced_x86::Code::SYSEXITQ:
    // Their effects are not modelled, so the CPL check is both all they do and their only fault.
    case iced_x86::Code::INVD: case iced_x86::Code::WBINVD: case iced_x86::Code::HLT:
      return true;
    default:
      return false;
  }
}

// Whether a fault can strike AFTER the flags were written, making that write observable despite
// something later overwriting it. Handlers set flags before the result, so this is only true for a
// memory destination; a fault on a memory SOURCE never reaches the flag computation.
[[nodiscard]] bool faults_after_writing_flags(const iced_x86::Instruction& instr) noexcept {
  return instr.op_count() > 0 && instr.op_kind(0) >= iced_x86::OpKind::MEMORY_SEG_SI;
}

// The other reason a write must stay live: rep CMPS/SCAS test the ZF their own compare wrote. The
// table declares that read, but `read` folds into the live set only after this instruction's mask is
// computed, so it protects the one before it. CMPS is covered above; SCAS writes AL, so it was not.
[[nodiscard]] bool rereads_own_flag_write(iced_x86::Code code) noexcept {
  using Code = iced_x86::Code;
  switch (code) {
    case Code::CMPSB_M8_M8: case Code::CMPSW_M16_M16:
    case Code::CMPSD_M32_M32: case Code::CMPSQ_M64_M64:
    case Code::SCASB_AL_M8: case Code::SCASW_AX_M16:
    case Code::SCASD_EAX_M32: case Code::SCASQ_RAX_M64:
      return true;
    default:
      return false;
  }
}

void compute_flag_liveness(std::span<FlagLivenessInstr> insts) noexcept {
  // Live-out is every flag: there is no cross-block liveness, so whatever runs next might read any.
  std::uint64_t live = kAluStatusFlagsMask;
  for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
    const auto info = flags_info_for_code(it->instr->code());
    const auto written = info.written & kAluStatusFlagsMask;
    // A fault-capable instruction counts as reading every flag, not because it does but because
    // nothing after it is guaranteed to run. That keeps earlier writes from being masked across it.
    const auto read = (info.read | (can_fault(*it->instr) ? kAluStatusFlagsMask : 0)) & kAluStatusFlagsMask;
    // The line above protects earlier writes; this one protects the faulting instruction's own,
    // which the boost missed by only reaching `live` on the next iteration.
    const auto own = (faults_after_writing_flags(*it->instr) ||
                      rereads_own_flag_write(it->instr->code()))
                         ? kAluStatusFlagsMask
                         : 0;
    it->dead_flags_mask = written & ~(live | own);
    live = (live & ~written) | read;
  }
}

}  // namespace seven

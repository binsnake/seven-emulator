#include "seven/handlers_fwd.hpp"
#include "seven/handler_helpers.hpp"

#include <cstdint>

namespace seven::handlers {
namespace {

enum class Cond : std::uint8_t {
  o,
  no,
  b,
  ae,
  e,
  ne,
  be,
  a,
  s,
  ns,
  p,
  np,
  l,
  ge,
  le,
  g,
};

bool eval_cond(const CpuState& state, Cond cond) {
  const auto cf = (state.rflags & kFlagCF) != 0;
  const auto pf = (state.rflags & kFlagPF) != 0;
  const auto zf = (state.rflags & kFlagZF) != 0;
  const auto sf = (state.rflags & kFlagSF) != 0;
  const auto of = (state.rflags & kFlagOF) != 0;
  switch (cond) {
    case Cond::o: return of;
    case Cond::no: return !of;
    case Cond::b: return cf;
    case Cond::ae: return !cf;
    case Cond::e: return zf;
    case Cond::ne: return !zf;
    case Cond::be: return cf || zf;
    case Cond::a: return !cf && !zf;
    case Cond::s: return sf;
    case Cond::ns: return !sf;
    case Cond::p: return pf;
    case Cond::np: return !pf;
    case Cond::l: return sf != of;
    case Cond::ge: return sf == of;
    case Cond::le: return zf || (sf != of);
    case Cond::g: return !zf && (sf == of);
  }
  return false;
}

}  // namespace

ExecutionResult handle_code_CMOVO_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::o)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVNO_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::no)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVB_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::b)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVAE_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ae)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVE_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::e)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVNE_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ne)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVBE_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::be)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVA_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::a)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVS_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::s)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVNS_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ns)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVP_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::p)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVNP_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::np)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVL_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::l)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVGE_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ge)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVLE_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::le)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVG_R16_RM16(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 2, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::g)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 2);
  return {};
}

ExecutionResult handle_code_CMOVO_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::o)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVNO_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::no)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVB_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::b)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVAE_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ae)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVE_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::e)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVNE_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ne)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVBE_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::be)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVA_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::a)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVS_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::s)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVNS_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ns)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVP_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::p)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVNP_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::np)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVL_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::l)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVGE_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ge)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVLE_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::le)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVG_R32_RM32(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 4, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::g)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 4);
  return {};
}

ExecutionResult handle_code_CMOVO_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::o)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVNO_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::no)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVB_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::b)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVAE_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ae)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVE_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::e)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVNE_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ne)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVBE_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::be)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVA_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::a)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVS_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::s)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVNS_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ns)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVP_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::p)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVNP_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::np)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVL_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::l)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVGE_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::ge)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVLE_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::le)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_CMOVG_R64_RM64(ExecutionContext& ctx) {
  bool src_ok = false;
  const auto src = detail::read_operand(ctx, 1, 8, &src_ok);
  if (!src_ok) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  if (!eval_cond(ctx.state, Cond::g)) {
    return {};
  }
  detail::write_register(ctx.state, ctx.instr.op_register(0), src, 8);
  return {};
}

ExecutionResult handle_code_SETO_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::o) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETNO_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::no) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETB_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::b) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETAE_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::ae) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETE_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::e) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETNE_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::ne) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETBE_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::be) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETA_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::a) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETS_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::s) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETNS_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::ns) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETP_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::p) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETNP_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::np) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETL_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::l) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETGE_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::ge) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETLE_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::le) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

ExecutionResult handle_code_SETG_RM8(ExecutionContext& ctx) {
  const std::uint64_t value = eval_cond(ctx.state, Cond::g) ? 1u : 0u;
  if (!detail::write_operand(ctx, 0, value, 1)) {
    return {StopReason::page_fault, 0, ExceptionInfo{StopReason::page_fault, detail::memory_address(ctx), 0}, ctx.instr.code()};
  }
  return {};
}

}  // namespace seven::handlers



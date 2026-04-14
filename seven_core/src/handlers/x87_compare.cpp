#include <cmath>

#include "seven/handler_helpers.hpp"
#include "seven/x87_helpers.hpp"

namespace seven::handlers {

using X87Scalar = ::seven::X87Scalar;

ExecutionResult handle_code_FTST(ExecutionContext& ctx) {
  if (ctx.state.x87_is_empty(0)) {
    x87_set_cmp_flags(ctx, -2);
    return x87_stack_underflow(ctx);
  }
  std::uint16_t exceptions = 0;
  const int rel = x87_cmp(ctx.state.x87_get(0), 0, false, exceptions);
  x87_set_cmp_flags(ctx, rel);
  if (exceptions != 0) {
    return x87_exception(ctx, exceptions);
  }
  return {};
}
ExecutionResult handle_code_FCOM_M32FP(ExecutionContext& ctx) { return x87_compare_mem(ctx, 4, false, false); }
ExecutionResult handle_code_FCOMP_M32FP(ExecutionContext& ctx) { return x87_compare_mem(ctx, 4, true, false); }
ExecutionResult handle_code_FCOM_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, false, false); }
ExecutionResult handle_code_FCOMP_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, false, true); }
ExecutionResult handle_code_FCOMPP(ExecutionContext& ctx) { return x87_compare_regs(ctx, 0, 1, true, true, false); }
ExecutionResult handle_code_FUCOMPP(ExecutionContext& ctx) { return x87_compare_regs(ctx, 0, 1, true, true, false, true); }
ExecutionResult handle_code_FCOMI_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, true, false); }
ExecutionResult handle_code_FUCOMI_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, true, false, true); }
ExecutionResult handle_code_FCOMIP_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, true, true); }
ExecutionResult handle_code_FUCOMIP_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, true, true, true); }
ExecutionResult handle_code_FCOM_M64FP(ExecutionContext& ctx) { return x87_compare_mem(ctx, 8, false, false); }
ExecutionResult handle_code_FCOMP_M64FP(ExecutionContext& ctx) { return x87_compare_mem(ctx, 8, true, false); }
ExecutionResult handle_code_FUCOM_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, false, false, true); }
ExecutionResult handle_code_FUCOMP_ST0_STI(ExecutionContext& ctx) { return x87_compare_st0_sti(ctx, 1, false, true, true); }

ExecutionResult handle_code_FCOM_ST0_STI_DCD0(ExecutionContext& ctx) { return handle_code_FCOM_ST0_STI(ctx); }
ExecutionResult handle_code_FCOMP_ST0_STI_DCD8(ExecutionContext& ctx) { return handle_code_FCOMP_ST0_STI(ctx); }
ExecutionResult handle_code_FCOMP_ST0_STI_DED0(ExecutionContext& ctx) { return handle_code_FCOMP_ST0_STI(ctx); }

ExecutionResult handle_code_FCMOVB_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagCF) != 0); }
ExecutionResult handle_code_FCMOVBE_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagCF) != 0 || (ctx.state.rflags & kFlagZF) != 0); }
ExecutionResult handle_code_FCMOVE_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagZF) != 0); }
ExecutionResult handle_code_FCMOVNB_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagCF) == 0); }
ExecutionResult handle_code_FCMOVNBE_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagCF) == 0 && (ctx.state.rflags & kFlagZF) == 0); }
ExecutionResult handle_code_FCMOVNE_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagZF) == 0); }
ExecutionResult handle_code_FCMOVNU_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagPF) == 0); }
ExecutionResult handle_code_FCMOVU_ST0_STI(ExecutionContext& ctx) { return x87_move_if(ctx, (ctx.state.rflags & kFlagPF) != 0); }

}  // namespace seven::handlers




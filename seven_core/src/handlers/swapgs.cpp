#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {

constexpr std::uint32_t kMsrKernelGsBase = 0xC0000102u;

}

ExecutionResult handle_code_SWAPGS(ExecutionContext& ctx) {
  const auto user_gs = ctx.state.gs_base;
  const auto kernel_gs = detail::read_msr(ctx.state, kMsrKernelGsBase);
  ctx.state.gs_base = kernel_gs;
  detail::write_msr(ctx.state, kMsrKernelGsBase, user_gs);
  return {};
}

}  // namespace seven::handlers


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_STAC(ExecutionContext& ctx) {
  constexpr std::uint64_t kFlagAC = 1ull << 18;
  detail::set_flag(ctx.state.rflags, kFlagAC, true);
  return {};
}

}  // namespace seven::handlers



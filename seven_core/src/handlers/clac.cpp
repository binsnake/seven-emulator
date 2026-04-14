#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CLAC(ExecutionContext& ctx) {
  constexpr std::uint64_t kFlagAC = 1ull << 18;
  detail::set_flag(ctx.state.rflags, kFlagAC, false);
  return {};
}

}  // namespace seven::handlers



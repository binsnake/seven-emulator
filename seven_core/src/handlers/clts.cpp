#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CLTS(ExecutionContext& ctx) {
  constexpr std::uint64_t kTaskSwitched = 1ull << 3;
  ctx.state.cr[0] &= ~kTaskSwitched;
  return {};
}

}  // namespace seven::handlers



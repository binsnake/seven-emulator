#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_CLFLUSHOPT_M8(ExecutionContext& ctx) {
  std::uint8_t ignored = 0;
  if (auto result = detail::read_operand_checked(ctx, 0, 1, ignored); !result.ok()) {
    return result;
  }
  return {};
}

}  // namespace seven::handlers


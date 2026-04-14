#include "seven/handler_helpers.hpp"

namespace seven::handlers {

namespace {
ExecutionResult prefetch(ExecutionContext&) {
  return {};
}
}

ExecutionResult handle_code_PREFETCHNTA_M8(ExecutionContext& ctx) { return prefetch(ctx); }
ExecutionResult handle_code_PREFETCHT0_M8(ExecutionContext& ctx) { return prefetch(ctx); }
ExecutionResult handle_code_PREFETCHT1_M8(ExecutionContext& ctx) { return prefetch(ctx); }
ExecutionResult handle_code_PREFETCHT2_M8(ExecutionContext& ctx) { return prefetch(ctx); }
ExecutionResult handle_code_PREFETCHW_M8(ExecutionContext& ctx) { return prefetch(ctx); }
ExecutionResult handle_code_PREFETCHWT1_M8(ExecutionContext& ctx) { return prefetch(ctx); }

}  // namespace seven::handlers


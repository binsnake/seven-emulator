#include "seven/handlers_fwd.hpp"

namespace seven::handlers {

ExecutionResult handle_code_NOPD(ExecutionContext&) { return {}; }
ExecutionResult handle_code_NOPW(ExecutionContext&) { return {}; }
ExecutionResult handle_code_NOP_RM16(ExecutionContext&) { return {}; }
ExecutionResult handle_code_NOP_RM32(ExecutionContext&) { return {}; }
ExecutionResult handle_code_NOP_RM64(ExecutionContext&) { return {}; }

}  // namespace seven::handlers


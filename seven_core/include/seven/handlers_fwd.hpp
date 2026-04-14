#pragma once

#include "seven/executor.hpp"

namespace seven {
namespace handlers {

#define KUBERA_CODE(code) ExecutionResult handle_code_##code(ExecutionContext& ctx);
#include "seven/handled_codes.def"
#undef KUBERA_CODE

}  // namespace handlers
}  // namespace seven



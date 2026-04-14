#pragma once

#include "seven/executor.hpp"
#include "seven/snapshot.hpp"
#include "seven/system_state.hpp"

namespace seven {

struct StandaloneMachine {
  CpuState state{};
  Memory memory{};
  Executor executor{};
};

}  // namespace seven


#include "seven/handler_helpers.hpp"

namespace seven::handlers {

ExecutionResult handle_code_BSWAP_R16(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto swapped = static_cast<std::uint64_t>(((value & 0x00FFull) << 8ull) | ((value >> 8ull) & 0x00FFull));
  detail::write_register(ctx.state, ctx.instr.op_register(0), swapped, 2);
  return {};
}

ExecutionResult handle_code_BSWAP_R32(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto swapped = ((value & 0x00000000000000FFull) << 24ull) |
                      ((value & 0x000000000000FF00ull) << 8ull) |
                      ((value & 0x0000000000FF0000ull) >> 8ull) |
                      ((value & 0x00000000FF000000ull) >> 24ull);
  detail::write_register(ctx.state, ctx.instr.op_register(0), swapped, 4);
  return {};
}

ExecutionResult handle_code_BSWAP_R64(ExecutionContext& ctx) {
  const auto value = detail::read_register(ctx.state, ctx.instr.op_register(0));
  const auto swapped = ((value & 0x00000000000000FFull) << 56ull) |
                      ((value & 0x000000000000FF00ull) << 40ull) |
                      ((value & 0x0000000000FF0000ull) << 24ull) |
                      ((value & 0x00000000FF000000ull) << 8ull) |
                      ((value & 0x000000FF00000000ull) >> 8ull) |
                      ((value & 0x0000FF0000000000ull) >> 24ull) |
                      ((value & 0x00FF000000000000ull) >> 40ull) |
                      ((value & 0xFF00000000000000ull) >> 56ull);
  detail::write_register(ctx.state, ctx.instr.op_register(0), swapped, 8);
  return {};
}

}  // namespace seven::handlers


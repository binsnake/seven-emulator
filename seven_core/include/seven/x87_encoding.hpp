#pragma once

#include <cstdint>
#include <cstring>

#include "seven/types.hpp"

namespace seven::handlers::x87_encoding {

// extFloat80_t is byte-identical to the x87 80-bit extended memory format on
// little-endian systems: signif (8 bytes) followed by signExp (2 bytes).
// encode/decode are therefore simple field-level memcpy operations.

inline seven::X87Scalar decode_ext80(const std::uint8_t* raw) {
    extFloat80_t v;
    std::memcpy(&v.signif,  raw,     8);
    std::memcpy(&v.signExp, raw + 8, 2);
    return seven::X87Scalar(v);
}

inline void encode_ext80(seven::X87Scalar value, std::uint8_t* raw) {
    std::memcpy(raw,     &value.val.signif,  8);
    std::memcpy(raw + 8, &value.val.signExp, 2);
}

}  // namespace seven::handlers::x87_encoding



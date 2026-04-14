#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "seven/types.hpp"

namespace seven::handlers::x87_encoding {

using X87Scalar = seven::X87Scalar;

inline seven::X87Scalar decode_ext80(const std::uint8_t* raw) {
  std::uint64_t significand = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    significand |= (std::uint64_t(raw[i]) << (8 * i));
  }
  const std::uint16_t se = std::uint16_t(raw[8]) | (std::uint16_t(raw[9]) << 8);
  const bool sign = (se & 0x8000u) != 0;
  const std::uint16_t exp = static_cast<std::uint16_t>(se & 0x7FFFu);

  if (exp == 0 && significand == 0) {
    return sign ? seven::X87Scalar(0) * -1 : seven::X87Scalar(0);
  }

  if (exp == 0x7FFFu) {
    if ((significand & 0x7FFFFFFFFFFFFFFFull) == 0) {
      return sign ? -std::numeric_limits<seven::X87Scalar>::infinity() : std::numeric_limits<seven::X87Scalar>::infinity();
    }
    return std::numeric_limits<X87Scalar>::quiet_NaN();
  }

  const seven::X87Scalar frac = static_cast<seven::X87Scalar>(significand) / static_cast<seven::X87Scalar>(UINT64_C(1) << 63);
  const int unbiased = (exp == 0) ? (1 - 16383) : (int(exp) - 16383);
  seven::X87Scalar value = boost::multiprecision::ldexp(frac, unbiased);
  if (sign) value = -value;
  return value;
}

inline void encode_ext80(seven::X87Scalar value, std::uint8_t* raw) {
  std::memset(raw, 0, 10);
  if (boost::multiprecision::isnan(value)) {
    const std::uint16_t se = 0x7FFFu;
    const std::uint64_t sig = UINT64_C(1) << 62;
    for (std::size_t i = 0; i < 8; ++i) raw[i] = static_cast<std::uint8_t>((sig >> (8 * i)) & 0xFFu);
    raw[8] = static_cast<std::uint8_t>(se & 0xFFu);
    raw[9] = static_cast<std::uint8_t>((se >> 8) & 0xFFu);
    return;
  }
  if (boost::multiprecision::isinf(value)) {
    const std::uint16_t se = static_cast<std::uint16_t>((boost::multiprecision::signbit(value) ? 0x8000u : 0u) | 0x7FFFu);
    const std::uint64_t sig = UINT64_C(1) << 63;
    for (std::size_t i = 0; i < 8; ++i) raw[i] = static_cast<std::uint8_t>((sig >> (8 * i)) & 0xFFu);
    raw[8] = static_cast<std::uint8_t>(se & 0xFFu);
    raw[9] = static_cast<std::uint8_t>((se >> 8) & 0xFFu);
    return;
  }
  if (value == 0) {
    const std::uint16_t se = static_cast<std::uint16_t>(boost::multiprecision::signbit(value) ? 0x8000u : 0u);
    raw[8] = static_cast<std::uint8_t>(se & 0xFFu);
    raw[9] = static_cast<std::uint8_t>((se >> 8) & 0xFFu);
    return;
  }

  const bool sign = boost::multiprecision::signbit(value);
  const seven::X87Scalar av = boost::multiprecision::abs(value);
  int exp2 = 0;
  const seven::X87Scalar frac = boost::multiprecision::frexp(av, &exp2); // av = frac * 2^exp2, frac in [0.5,1)
  const seven::X87Scalar normalized = frac * 2;      // [1,2)
  int exp = exp2 - 1 + 16383;

  if (exp <= 0) {
    const seven::X87Scalar scaled = boost::multiprecision::ldexp(av, 63 + 16382);
    std::uint64_t sig = static_cast<std::uint64_t>(scaled);
    for (std::size_t i = 0; i < 8; ++i) raw[i] = static_cast<std::uint8_t>((sig >> (8 * i)) & 0xFFu);
    const std::uint16_t se = static_cast<std::uint16_t>(sign ? 0x8000u : 0u);
    raw[8] = static_cast<std::uint8_t>(se & 0xFFu);
    raw[9] = static_cast<std::uint8_t>((se >> 8) & 0xFFu);
    return;
  }

  if (exp >= 0x7FFF) {
    const std::uint16_t se = static_cast<std::uint16_t>((sign ? 0x8000u : 0u) | 0x7FFFu);
    const std::uint64_t sig = UINT64_C(1) << 63;
    for (std::size_t i = 0; i < 8; ++i) raw[i] = static_cast<std::uint8_t>((sig >> (8 * i)) & 0xFFu);
    raw[8] = static_cast<std::uint8_t>(se & 0xFFu);
    raw[9] = static_cast<std::uint8_t>((se >> 8) & 0xFFu);
    return;
  }

  std::uint64_t sig = static_cast<std::uint64_t>(boost::multiprecision::ldexp(normalized, 63));
  if (sig == 0xFFFFFFFFFFFFFFFFull) {
    sig = UINT64_C(1) << 63;
    ++exp;
  }

  for (std::size_t i = 0; i < 8; ++i) raw[i] = static_cast<std::uint8_t>((sig >> (8 * i)) & 0xFFu);
  const std::uint16_t se = static_cast<std::uint16_t>((sign ? 0x8000u : 0u) | (exp & 0x7FFFu));
  raw[8] = static_cast<std::uint8_t>(se & 0xFFu);
  raw[9] = static_cast<std::uint8_t>((se >> 8) & 0xFFu);
}

}  // namespace seven::handlers::x87_encoding



#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" {
#include <softfloat.h>
}

namespace seven {

class Float80 {
public:
    extFloat80_t val{};  // zero-initialised → +0.0

    Float80() = default;
    explicit Float80(extFloat80_t v) noexcept : val(v) {}

    Float80(double d) noexcept {
        val = f64_to_extF80(std::bit_cast<float64_t>(d));
    }
    Float80(float f) noexcept : Float80(static_cast<double>(f)) {}
    Float80(int i) noexcept {
        val = i32_to_extF80(static_cast<int32_t>(i));
    }
    Float80(std::int64_t i) noexcept {
        val = i64_to_extF80(i);
    }
    Float80(std::uint64_t u) noexcept {
        val = ui64_to_extF80(u);
    }

    Float80 operator+(Float80 o) const noexcept { return Float80(extF80_add(val, o.val)); }
    Float80 operator-(Float80 o) const noexcept { return Float80(extF80_sub(val, o.val)); }
    Float80 operator*(Float80 o) const noexcept { return Float80(extF80_mul(val, o.val)); }
    Float80 operator/(Float80 o) const noexcept { return Float80(extF80_div(val, o.val)); }

    Float80 operator-() const noexcept {
        extFloat80_t r = val;
        r.signExp ^= 0x8000u;
        return Float80(r);
    }

    Float80& operator+=(Float80 o) noexcept { *this = *this + o; return *this; }
    Float80& operator-=(Float80 o) noexcept { *this = *this - o; return *this; }
    Float80& operator*=(Float80 o) noexcept { *this = *this * o; return *this; }
    Float80& operator/=(Float80 o) noexcept { *this = *this / o; return *this; }

    bool operator==(Float80 o) const noexcept { return extF80_eq(val, o.val); }
    bool operator!=(Float80 o) const noexcept { return !extF80_eq(val, o.val); }
    bool operator< (Float80 o) const noexcept { return extF80_lt(val, o.val); }
    bool operator<=(Float80 o) const noexcept { return extF80_le(val, o.val); }
    bool operator> (Float80 o) const noexcept { return extF80_lt(o.val, val); }
    bool operator>=(Float80 o) const noexcept { return extF80_le(o.val, val); }

    explicit operator double()      const noexcept { return std::bit_cast<double>(extF80_to_f64(val)); }
    explicit operator float()       const noexcept { return static_cast<float>(static_cast<double>(*this)); }
    explicit operator std::int64_t()  const noexcept { return extF80_to_i64(val,  softfloat_round_minMag, false); }
    explicit operator std::uint64_t() const noexcept { return extF80_to_ui64(val, softfloat_round_minMag, false); }
    explicit operator int()         const noexcept { return static_cast<int>(static_cast<std::int64_t>(*this)); }
};

// ── bit-level helpers ────────────────────────────────────────────────────────

inline bool isnan(Float80 x) noexcept {
    return (x.val.signExp & 0x7FFFu) == 0x7FFFu &&
           x.val.signif != 0x8000000000000000ULL;
}

inline bool isinf(Float80 x) noexcept {
    return (x.val.signExp & 0x7FFFu) == 0x7FFFu &&
           x.val.signif == 0x8000000000000000ULL;
}

inline bool signbit(Float80 x) noexcept {
    return (x.val.signExp & 0x8000u) != 0;
}

inline Float80 abs(Float80 x) noexcept {
    extFloat80_t r = x.val;
    r.signExp &= 0x7FFFu;
    return Float80(r);
}

// ── rounding ─────────────────────────────────────────────────────────────────

inline Float80 trunc(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_minMag, false));
}

inline Float80 floor(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_min, false));
}

inline Float80 ceil(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_max, false));
}

// ── remainder ────────────────────────────────────────────────────────────────

inline Float80 remainder(Float80 a, Float80 b) noexcept {
    return Float80(extF80_rem(a.val, b.val));
}

inline Float80 fmod(Float80 a, Float80 b) noexcept {
    if (isnan(a) || isnan(b) || isinf(a) || b == Float80(0)) {
        extFloat80_t nan; nan.signExp = 0x7FFFu; nan.signif = 0xC000000000000000ULL;
        return Float80(nan);
    }
    const Float80 q = trunc(a / b);
    return a - q * b;
}

// ── exponent manipulation (exact, no precision loss) ─────────────────────────

inline Float80 ldexp(Float80 x, int n) noexcept {
    if (isnan(x) || isinf(x) || x == Float80(0)) return x;
    const uint16_t biased = x.val.signExp & 0x7FFFu;
    if (biased == 0) {
        // denormal: round-trip through double is acceptable
        return Float80(std::ldexp(static_cast<double>(x), n));
    }
    const int newexp = static_cast<int>(biased) + n;
    if (newexp >= 0x7FFF) {
        extFloat80_t r;
        r.signExp = static_cast<uint16_t>((x.val.signExp & 0x8000u) | 0x7FFFu);
        r.signif  = 0x8000000000000000ULL;
        return Float80(r);
    }
    if (newexp <= 0) {
        return Float80(std::ldexp(static_cast<double>(x), n));
    }
    extFloat80_t r = x.val;
    r.signExp = static_cast<uint16_t>((x.val.signExp & 0x8000u) | static_cast<uint16_t>(newexp));
    return Float80(r);
}

inline Float80 frexp(Float80 x, int* exp) noexcept {
    if (isnan(x) || isinf(x) || x == Float80(0)) { *exp = 0; return x; }
    const uint16_t biased = x.val.signExp & 0x7FFFu;
    if (biased == 0) {
        double m = std::frexp(static_cast<double>(x), exp);
        return Float80(m);
    }
    // Normal: exponent = biased - 16382, mantissa exponent → 16382 → value in [0.5,1)
    *exp = static_cast<int>(biased) - 16382;
    extFloat80_t r = x.val;
    r.signExp = static_cast<uint16_t>((x.val.signExp & 0x8000u) | 16382u);
    return Float80(r);
}

// ── sqrt (exact via SoftFloat) ────────────────────────────────────────────────

inline Float80 sqrt(Float80 x) noexcept {
    return Float80(extF80_sqrt(x.val));
}

// ── transcendentals via double round-trip ────────────────────────────────────
// x87 transcendentals are not required to be correctly rounded, so double
// precision (53-bit mantissa) is acceptable for FSIN/FCOS/FTAN/FATAN2/FYL2X.

inline Float80 sin(Float80 x)  noexcept { return Float80(std::sin(static_cast<double>(x))); }
inline Float80 cos(Float80 x)  noexcept { return Float80(std::cos(static_cast<double>(x))); }
inline Float80 tan(Float80 x)  noexcept { return Float80(std::tan(static_cast<double>(x))); }

inline Float80 atan2(Float80 y, Float80 x) noexcept {
    return Float80(std::atan2(static_cast<double>(y), static_cast<double>(x)));
}

inline Float80 pow(Float80 base, Float80 exp_) noexcept {
    return Float80(std::pow(static_cast<double>(base), static_cast<double>(exp_)));
}

inline Float80 log2(Float80 x) noexcept { return Float80(std::log2(static_cast<double>(x))); }

}  // namespace seven

// ── std::numeric_limits specialisation ───────────────────────────────────────

namespace std {

template<>
class numeric_limits<seven::Float80> {
public:
    static constexpr bool is_specialized    = true;
    static constexpr bool is_signed         = true;
    static constexpr bool is_integer        = false;
    static constexpr bool is_exact          = false;
    static constexpr bool has_infinity      = true;
    static constexpr bool has_quiet_NaN     = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool is_bounded        = true;
    static constexpr int  radix             = 2;
    static constexpr int  digits            = 64;  // 80-bit explicit integer bit
    static constexpr int  max_exponent      = 16384;
    static constexpr int  min_exponent      = -16381;

    static seven::Float80 infinity() noexcept {
        extFloat80_t r;
        r.signExp = 0x7FFFu;
        r.signif  = 0x8000000000000000ULL;
        return seven::Float80(r);
    }

    static seven::Float80 quiet_NaN() noexcept {
        extFloat80_t r;
        r.signExp = 0x7FFFu;
        r.signif  = 0xC000000000000000ULL;
        return seven::Float80(r);
    }

    static seven::Float80 signaling_NaN() noexcept {
        extFloat80_t r;
        r.signExp = 0x7FFFu;
        r.signif  = 0xA000000000000000ULL;
        return seven::Float80(r);
    }

    // smallest positive normal: biased_exp=1 → actual=-16382
    static seven::Float80 min() noexcept {
        extFloat80_t r;
        r.signExp = 1u;
        r.signif  = 0x8000000000000000ULL;
        return seven::Float80(r);
    }

    // largest finite: biased_exp=0x7FFE, all significand bits set
    static seven::Float80 max() noexcept {
        extFloat80_t r;
        r.signExp = 0x7FFEu;
        r.signif  = 0xFFFFFFFFFFFFFFFFULL;
        return seven::Float80(r);
    }

    static seven::Float80 lowest() noexcept {
        auto v = max();
        v.val.signExp |= 0x8000u;
        return v;
    }

    static seven::Float80 denorm_min() noexcept {
        extFloat80_t r;
        r.signExp = 0u;
        r.signif  = 1u;
        return seven::Float80(r);
    }
};

}  // namespace std

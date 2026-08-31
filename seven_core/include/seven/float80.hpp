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
    extFloat80_t val{};  // zero-initialised -> +0.0

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
    // Straight to f32. Going via double rounds twice, and a value that sits exactly on the f64
    // midpoint can round to even there and then land a full ulp away from what one rounding gives,
    // which FST/FSTP m32 would show.
    explicit operator float()       const noexcept { return std::bit_cast<float>(extF80_to_f32(val)); }
    explicit operator std::int64_t()  const noexcept { return extF80_to_i64(val,  softfloat_round_minMag, false); }
    explicit operator std::uint64_t() const noexcept { return extF80_to_ui64(val, softfloat_round_minMag, false); }
    explicit operator int()         const noexcept { return static_cast<int>(static_cast<std::int64_t>(*this)); }
};

// -- bit-level helpers --------------------------------------------------------

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

// A quiet NaN sets both the integer bit and the top fraction bit. Everything else under the all-ones
// exponent -- signalling NaNs, plus the pseudo-NaN and pseudo-infinity encodings that the explicit
// integer bit makes representable -- raises #IA even on the compares that stay quiet for a QNaN.
inline bool issnan(Float80 x) noexcept {
    if ((x.val.signExp & 0x7FFFu) != 0x7FFFu) return false;
    if (x.val.signif == 0x8000000000000000ULL) return false;  // infinity
    return (x.val.signif & 0xC000000000000000ULL) != 0xC000000000000000ULL;
}

// The 80-bit format carries its integer bit explicitly, so an encoding can contradict its exponent.
// Hardware does not treat those as numbers, which is why FXAM gives them a class of their own.
inline bool isunsupported(Float80 x) noexcept {
    if ((x.val.signExp & 0x7FFFu) == 0u) return false;  // zero, denormal, pseudo-denormal
    return (x.val.signif & 0x8000000000000000ULL) == 0u;
}

// Hardware derives the tag from the register's own bits every time, so anything not plainly
// normalized reads back as SPECIAL. 0 = valid, 1 = zero, 2 = special, 3 = empty.
inline std::uint8_t x87_tag_of(Float80 x) noexcept {
    const std::uint16_t exp = x.val.signExp & 0x7FFFu;
    if (exp == 0x7FFFu) return 2;                        // infinity, NaN, pseudo-NaN, pseudo-infinity
    if (exp == 0) return x.val.signif == 0 ? 1 : 2;      // zero, else (pseudo-)denormal
    return (x.val.signif >> 63) != 0 ? 0 : 2;            // clear integer bit under a live exponent = unnormal
}

// Quieting a signalling NaN sets the top fraction bit and keeps the rest of the payload. Hardware
// hands that back rather than a fresh indefinite, so the guest can still see which NaN it was.
inline Float80 quiet(Float80 x) noexcept {
    extFloat80_t r = x.val;
    r.signif |= 0x4000000000000000ULL;
    return Float80(r);
}

inline Float80 abs(Float80 x) noexcept {
    extFloat80_t r = x.val;
    r.signExp &= 0x7FFFu;
    return Float80(r);
}

// -- rounding -----------------------------------------------------------------

// SoftFloat takes its rounding mode from a mutable global, so honouring FCW.RC means owning that
// global for the call and putting it back on scope exit. The build defines THREAD_LOCAL for the
// softfloat target; without it two guests on separate threads strand each other's mode.
class RoundingGuard {
public:
    explicit RoundingGuard(uint_fast8_t mode) noexcept
        : saved_(softfloat_roundingMode), changed_(mode != softfloat_roundingMode) {
        if (changed_) softfloat_roundingMode = mode;
    }
    ~RoundingGuard() noexcept {
        if (changed_) softfloat_roundingMode = saved_;
    }
    RoundingGuard(const RoundingGuard&) = delete;
    RoundingGuard& operator=(const RoundingGuard&) = delete;

private:
    uint_fast8_t saved_;
    bool changed_;
};

// Raised exceptions come out of a second global that softfloat only ORs into, so a reading means
// nothing unless it was cleared first. Same discipline as RoundingGuard, except raised() must be
// read before the guard leaves scope, which is why most callers go through with_exception_flags.
class ExceptionFlagGuard {
public:
    ExceptionFlagGuard() noexcept : saved_(softfloat_exceptionFlags) { softfloat_exceptionFlags = 0; }
    ~ExceptionFlagGuard() noexcept { softfloat_exceptionFlags = saved_; }
    ExceptionFlagGuard(const ExceptionFlagGuard&) = delete;
    ExceptionFlagGuard& operator=(const ExceptionFlagGuard&) = delete;

    [[nodiscard]] uint_fast8_t raised() const noexcept { return softfloat_exceptionFlags; }

private:
    uint_fast8_t saved_;
};

template <typename T>
struct Flagged {
    T value;
    uint_fast8_t flags;
};

// Runs fn() with the exception flags cleared and hands back both its result and what it raised.
template <typename Fn>
inline auto with_exception_flags(Fn&& fn) {
    ExceptionFlagGuard guard;
    auto value = fn();
    return Flagged<decltype(value)>{value, guard.raised()};
}

// The narrow formats are widened by softfloat rather than by the host, because the host quietens a
// signalling NaN on the way through float/double and there is then nothing left to raise #IA about.
inline Float80 from_f32_bits(std::uint32_t bits) noexcept {
    return Float80(f32_to_extF80(std::bit_cast<float32_t>(bits)));
}

inline Float80 from_f64_bits(std::uint64_t bits) noexcept {
    return Float80(f64_to_extF80(std::bit_cast<float64_t>(bits)));
}

inline Float80 round_near_even(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_near_even, false));
}

inline Float80 trunc(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_minMag, false));
}

inline Float80 floor(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_min, false));
}

inline Float80 ceil(Float80 x) noexcept {
    return Float80(extF80_roundToInt(x.val, softfloat_round_max, false));
}

// -- remainder ----------------------------------------------------------------

inline Float80 remainder(Float80 a, Float80 b) noexcept {
    return Float80(extF80_rem(a.val, b.val));
}

inline Float80 fmod(Float80 a, Float80 b) noexcept {
    if (isnan(a) || isnan(b) || isinf(a) || b == Float80(0)) {
        extFloat80_t nan; nan.signExp = 0x7FFFu; nan.signif = 0xC000000000000000ULL;
        return Float80(nan);
    }
    // IEEE and x87 both give the finite value back unchanged. This used to fall through to the
    // quotient path below, where trunc(a/inf) is zero and zero times infinity is NaN.
    if (isinf(b)) return a;
    // Built on the exact IEEE remainder rather than a - trunc(a/b)*b, whose quotient turns to
    // rounding noise past 64 significand bits. The two differ only in how the quotient rounds, so
    // they are at most one b apart, and fmod is the one whose sign follows the dividend.
    const Float80 r = Float80(extF80_rem(a.val, b.val));
    if (r == Float80(0) || signbit(r) == signbit(a)) return r;
    return signbit(a) ? (r - abs(b)) : (r + abs(b));
}

// -- exponent manipulation (exact, no precision loss) -------------------------

// Entirely in the exponent/significand domain with a 64-bit accumulator. Adding a guest-supplied n
// to the biased exponent as an int overflows, and the std::ldexp fallback covered exactly the
// subnormal cases a double cannot hold.
inline Float80 ldexp(Float80 x, int n) noexcept {
    if (isnan(x) || isinf(x) || x == Float80(0)) return x;

    const std::uint16_t sign = x.val.signExp & 0x8000u;
    std::uint64_t signif = x.val.signif;
    if (signif == 0) {
        extFloat80_t r; r.signExp = sign; r.signif = 0;
        return Float80(r);
    }

    // A biased exponent of 0 means the same scale as 1 with no implicit integer bit, so the two
    // shapes agree once the significand is normalized. Doing that here means the rest of the
    // function only has one case to think about, denormals and pseudo-denormals included.
    std::int64_t exp = static_cast<std::int64_t>(x.val.signExp & 0x7FFFu);
    if (exp == 0) exp = 1;
    while ((signif & 0x8000000000000000ULL) == 0) {
        signif <<= 1;
        --exp;
    }

    exp += static_cast<std::int64_t>(n);

    if (exp >= 0x7FFF) {
        extFloat80_t r;
        r.signExp = static_cast<std::uint16_t>(sign | 0x7FFFu);
        r.signif  = 0x8000000000000000ULL;
        return Float80(r);
    }
    if (exp <= 0) {
        // Below the smallest normal: shift the significand down into the denormal range instead of
        // flushing. Bits shifted out are dropped rather than rounded, which matches the rest of this
        // file ignoring the guest's rounding-control bits.
        const std::int64_t shift = 1 - exp;
        extFloat80_t r;
        r.signExp = sign;
        r.signif  = (shift >= 64) ? 0ULL : (signif >> shift);
        return Float80(r);
    }
    extFloat80_t r;
    r.signExp = static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(exp));
    r.signif  = signif;
    return Float80(r);
}

inline Float80 frexp(Float80 x, int* exp) noexcept {
    if (isnan(x) || isinf(x) || x == Float80(0)) { *exp = 0; return x; }

    const std::uint16_t sign = x.val.signExp & 0x8000u;
    std::uint64_t signif = x.val.signif;
    if (signif == 0) {
        *exp = 0;
        extFloat80_t r; r.signExp = sign; r.signif = 0;
        return Float80(r);
    }

    // Same normalization as ldexp, and for the same reason: the old denormal branch went through a
    // double, which cannot hold a denormal extF80 at all and returned zero with an exponent of zero.
    std::int64_t e = static_cast<std::int64_t>(x.val.signExp & 0x7FFFu);
    if (e == 0) e = 1;
    while ((signif & 0x8000000000000000ULL) == 0) {
        signif <<= 1;
        --e;
    }

    *exp = static_cast<int>(e - 16382);
    extFloat80_t r;
    r.signExp = static_cast<std::uint16_t>(sign | 16382u);
    r.signif  = signif;
    return Float80(r);
}

// -- sqrt (exact via SoftFloat) ------------------------------------------------

inline Float80 sqrt(Float80 x) noexcept {
    return Float80(extF80_sqrt(x.val));
}

// -- transcendentals via double round-trip ------------------------------------
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

// Split off the exponent before narrowing: the mantissa stays in [0.5, 1) so the round trip only
// costs precision, where the whole value turned FYL2X on 2^1100 into infinity rather than 1100.
inline Float80 log2(Float80 x) noexcept {
    if (isnan(x) || isinf(x) || signbit(x) || x == Float80(0)) {
        return Float80(std::log2(static_cast<double>(x)));
    }
    int e = 0;
    const Float80 mantissa = frexp(x, &e);
    return Float80(static_cast<std::int64_t>(e)) + Float80(std::log2(static_cast<double>(mantissa)));
}

}  // namespace seven

// -- std::numeric_limits specialisation ---------------------------------------

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

    // smallest positive normal: biased_exp=1 -> actual=-16382
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

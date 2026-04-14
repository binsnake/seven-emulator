#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace test {

using TestFn = void (*)();

struct TestCase {
    const char* name;
    TestFn fn;
};

std::vector<TestCase>& registry();
std::size_t& virtual_test_count();

inline void record_virtual_test(std::size_t n = 1) {
    virtual_test_count() += n;
}

struct Registrar {
    Registrar(const char* name, TestFn fn) {
        registry().push_back(TestCase{name, fn});
    }
};

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error(message);
}

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        fail(std::string(message));
    }
}

template <typename T, typename U>
inline void require_eq(const T& expected, const U& actual, std::string_view label) {
    if (!(expected == actual)) {
        std::ostringstream oss;
        oss << label << " mismatch";
        fail(oss.str());
    }
}

inline std::string bytes_to_hex(const std::vector<std::uint8_t>& bytes) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(bytes.size() * 3);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        std::uint8_t b = bytes[i];
        out.push_back(kHex[(b >> 4) & 0xF]);
        out.push_back(kHex[b & 0xF]);
        if (i + 1 != bytes.size()) {
            out.push_back(' ');
        }
    }
    return out;
}

} // namespace test

#define ICED_TEST(name)                                                                                                 \
    static void name();                                                                                                 \
    static ::test::Registrar name##_registrar(#name, &name);                                                           \
    static void name()

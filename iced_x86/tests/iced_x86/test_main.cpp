#include "test_utils.hpp"

namespace test {
std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

std::size_t& virtual_test_count() {
    static std::size_t count = 0;
    return count;
}
} // namespace test

int main() {
    int failures = 0;
    test::virtual_test_count() = 0;
    for (const auto& test_case : test::registry()) {
        try {
            test_case.fn();
            std::cout << "[PASS] " << test_case.name << '\n';
        } catch (const std::exception& ex) {
            ++failures;
            std::cout << "[FAIL] " << test_case.name << " -> " << ex.what() << '\n';
        } catch (...) {
            ++failures;
            std::cout << "[FAIL] " << test_case.name << " -> unknown exception\n";
        }
    }

    const std::size_t executed = test::registry().size() + test::virtual_test_count();
    std::cout << "Executed " << executed << " tests, failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}

set(ICED_X86_TEST_SOURCES
    test_main.cpp
    decoder_tests.cpp
    encoder_tests.cpp
    formatter_tests.cpp
)

add_executable(iced_x86_tests ${ICED_X86_TEST_SOURCES})
target_compile_features(iced_x86_tests PRIVATE cxx_std_23)
target_link_libraries(iced_x86_tests PRIVATE iced_x86::iced_x86)

if(MSVC)
    target_compile_options(iced_x86_tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(iced_x86_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME iced_x86_tests COMMAND iced_x86_tests)

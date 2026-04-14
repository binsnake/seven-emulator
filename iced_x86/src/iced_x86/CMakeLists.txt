set(ICED_X86_SOURCES
    src/block_encoder.cpp
    src/decoder.cpp
    src/encoder.cpp
    src/encoder_handlers.cpp
    src/encoder_methods.cpp
    src/encoder_ops.cpp
    src/fastfmt.cpp
    src/gas_formatter.cpp
    src/handlers.cpp
    src/instruction.cpp
    src/instruction_create.cpp
    src/instruction_info.cpp
    src/intel_formatter.cpp
    src/masm_formatter.cpp
    src/memory_size_info.cpp
    src/mvex_info_data.cpp
    src/nasm_formatter.cpp
    src/op_code_info.cpp
    src/register_info.cpp
    src/table_deserializer.cpp
    src/tables.cpp
)

add_library(iced_x86 ${ICEDCPP_LIBRARY_TYPE} ${ICED_X86_SOURCES})
add_library(iced_x86::iced_x86 ALIAS iced_x86)
set_target_properties(iced_x86 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_compile_features(iced_x86 PUBLIC cxx_std_23)
target_compile_definitions(iced_x86 PUBLIC ICED_X86_CONSTEXPR_HANDLERS=0)
target_include_directories(iced_x86
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(iced_x86 PRIVATE /W4 /permissive-)
else()
    target_compile_options(iced_x86 PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ICEDCPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/iced_x86 ${CMAKE_BINARY_DIR}/tests/iced_x86)
endif()

install(TARGETS iced_x86
    EXPORT icedcppTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/iced_x86/
    DESTINATION include/iced_x86
    FILES_MATCHING PATTERN "*.hpp"
)

set(ICEDCPP_EXAMPLE_SOURCES
    decoder_stream.cpp
    decoder_error_handling.cpp
    encoder_roundtrip.cpp
    encoder_from_scratch.cpp
    formatter_styles.cpp
    instruction_info.cpp
    opcode_info.cpp
    block_encoder_relocate.cpp
    code_assembler_labels.cpp
)

foreach(example_src IN LISTS ICEDCPP_EXAMPLE_SOURCES)
    get_filename_component(example_name "${example_src}" NAME_WE)
    set(target_name "icedcpp_example_${example_name}")

    add_executable(${target_name} "${example_src}")
    target_compile_features(${target_name} PRIVATE cxx_std_23)
    target_link_libraries(${target_name} PRIVATE iced_x86::iced_x86)

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

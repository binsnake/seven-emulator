cmake_minimum_required(VERSION 3.24)

project(seven LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

option(SEVEN_ENABLE_LTO "Enable link time optimization" ON)
option(SEVEN_ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors for new targets" OFF)
set(SEVEN_SIMD_PROFILE "AVX512" CACHE STRING "SIMD profile: AVX512, AVX2, or AVX")
set_property(CACHE SEVEN_SIMD_PROFILE PROPERTY STRINGS AVX512 AVX2 AVX)

set(SEVEN_VECTOR_BITS 512)
set(SEVEN_MAX_VECTOR_BYTES 64)
set(SEVEN_ENABLE_AVX 1)
set(SEVEN_ENABLE_AVX2 1)
set(SEVEN_ENABLE_AVX512 1)
if(SEVEN_SIMD_PROFILE STREQUAL "AVX2")
  set(SEVEN_VECTOR_BITS 256)
  set(SEVEN_MAX_VECTOR_BYTES 32)
  set(SEVEN_ENABLE_AVX 1)
  set(SEVEN_ENABLE_AVX2 1)
  set(SEVEN_ENABLE_AVX512 0)
elseif(SEVEN_SIMD_PROFILE STREQUAL "AVX")
  set(SEVEN_VECTOR_BITS 128)
  set(SEVEN_MAX_VECTOR_BYTES 16)
  set(SEVEN_ENABLE_AVX 1)
  set(SEVEN_ENABLE_AVX2 0)
  set(SEVEN_ENABLE_AVX512 0)
elseif(NOT SEVEN_SIMD_PROFILE STREQUAL "AVX512")
  message(FATAL_ERROR "Invalid SEVEN_SIMD_PROFILE='${SEVEN_SIMD_PROFILE}'. Expected AVX512, AVX2, or AVX.")
endif()
message(STATUS "SEVEN SIMD profile: ${SEVEN_SIMD_PROFILE} (${SEVEN_VECTOR_BITS}-bit vectors)")

include(CTest)
include(FetchContent)
include(GoogleTest)

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

set(BOOST_HINT_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../dbemu/vcpkg_installed/x64-windows-static/x64-windows-static/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../dbemu/vcpkg_installed/x64-windows-static/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../dbemu/vcpkg_installed/x64-windows-static/vcpkg/pkgs/boost-multiprecision_x64-windows-static/include"
)

set(BOOST_MULTIPRECISION_HINT_DIRS
  ${BOOST_HINT_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/../kubera/deps/boost_multiprecision/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../minkubera/deps/boost_multiprecision/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Mona/KUBERA/deps/boost_multiprecision/include"
  "$ENV{BOOST_ROOT}/include"
)

set(SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR "")
set(SEVEN_BOOST_FULL_INCLUDE_DIR "")

foreach(_dir IN LISTS BOOST_HINT_DIRS)
  if(EXISTS "${_dir}/boost/config.hpp")
    if(EXISTS "${_dir}/boost/multiprecision/cpp_int.hpp")
      set(SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR "${_dir}")
      set(SEVEN_BOOST_FULL_INCLUDE_DIR "${_dir}")
      break()
    endif()
  endif()
endforeach()

if (NOT SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR)
  find_path(SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR
    NAMES "boost/multiprecision/cpp_int.hpp"
    PATHS ${BOOST_MULTIPRECISION_HINT_DIRS}
  )
endif()

if (NOT SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR)
  message(FATAL_ERROR "Could not locate boost/multiprecision headers. Set BOOST_ROOT to an installation that contains boost/multiprecision.")
endif()

if (NOT SEVEN_BOOST_FULL_INCLUDE_DIR AND
    EXISTS "${SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR}/boost/multiprecision/detail/standalone_config.hpp")
  message(WARNING "Using standalone boost multiprecision headers: ${SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR}. Define BOOST_INCLUDE_DIR for full Boost.Config headers to avoid this.")
endif()

message(STATUS "Using boost multiprecision headers from: ${SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR}")

if(MSVC)
  add_compile_options(/permissive- /EHsc /Zc:__cplusplus /W4 /wd5311 /FS)
  if(SEVEN_ENABLE_WARNINGS_AS_ERRORS)
    add_compile_options(/WX)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(SEVEN_ENABLE_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
  endif()
endif()

if(SEVEN_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT SEVEN_IPO_SUPPORTED OUTPUT SEVEN_IPO_ERROR)
  if(SEVEN_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

set(ICED_X86_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/block_encoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/decoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/encoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/encoder_handlers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/encoder_methods.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/encoder_ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/fastfmt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/gas_formatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/handlers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/instruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/instruction_create.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/instruction_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/intel_formatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/masm_formatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/memory_size_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/mvex_info_data.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/nasm_formatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/op_code_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/register_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/table_deserializer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/src/iced_x86/src/tables.cpp
)

add_library(iced_x86 STATIC ${ICED_X86_SOURCES})
target_include_directories(iced_x86 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/iced_x86/include")
if(MSVC)
  target_compile_options(iced_x86 PRIVATE /WX-)
endif()
target_compile_definitions(
  iced_x86
  PUBLIC
    _CRT_SECURE_NO_WARNINGS
    ICED_X86_CONSTEXPR_HANDLERS=0
)

file(GLOB_RECURSE SEVEN_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/seven_core/src/*.cpp"
)

add_library(seven_core STATIC ${SEVEN_CORE_SOURCES})
target_include_directories(seven_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/seven_core/include")
target_include_directories(seven_core PUBLIC "${SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR}")
target_link_libraries(seven_core PUBLIC iced_x86)
target_compile_definitions(seven_core PUBLIC
  SEVEN_SIMD_PROFILE_${SEVEN_SIMD_PROFILE}=1
  SEVEN_ENABLE_AVX=${SEVEN_ENABLE_AVX}
  SEVEN_ENABLE_AVX2=${SEVEN_ENABLE_AVX2}
  SEVEN_ENABLE_AVX512=${SEVEN_ENABLE_AVX512}
  SEVEN_VECTOR_BITS=${SEVEN_VECTOR_BITS}
  SEVEN_MAX_VECTOR_BYTES=${SEVEN_MAX_VECTOR_BYTES}
)
if(MSVC)
  target_compile_options(seven_core PRIVATE /WX-)
endif()

add_executable(seven_example examples/kubera_example.cpp)
target_link_libraries(seven_example PRIVATE seven_core)
if(MSVC)
  target_compile_options(seven_example PRIVATE /WX-)
endif()

add_executable(seven_tests
  tests/kubera_debug_tests.cpp
  tests/kubera_scalar_tests.cpp
  tests/kubera_simd_tests.cpp
)
target_link_libraries(seven_tests PRIVATE seven_core GTest::gtest_main)
if(MSVC)
  target_compile_options(seven_tests PRIVATE /WX-)
endif()

gtest_discover_tests(seven_tests)

add_library(seven_core_bmi2 STATIC
  seven_core/src/memory.cpp
  seven_core/src/handler_helpers.cpp
  seven_core/src/handlers/bmi.cpp
  tests/kubera_bmi2_executor.cpp
)
target_include_directories(seven_core_bmi2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/seven_core/include")
target_include_directories(seven_core_bmi2 PUBLIC "${SEVEN_BOOST_MULTIPRECISION_INCLUDE_DIR}")
target_link_libraries(seven_core_bmi2 PUBLIC iced_x86)
target_compile_definitions(seven_core_bmi2 PUBLIC
  SEVEN_SIMD_PROFILE_${SEVEN_SIMD_PROFILE}=1
  SEVEN_ENABLE_AVX=${SEVEN_ENABLE_AVX}
  SEVEN_ENABLE_AVX2=${SEVEN_ENABLE_AVX2}
  SEVEN_ENABLE_AVX512=${SEVEN_ENABLE_AVX512}
  SEVEN_VECTOR_BITS=${SEVEN_VECTOR_BITS}
  SEVEN_MAX_VECTOR_BYTES=${SEVEN_MAX_VECTOR_BYTES}
)

add_executable(seven_bmi2_tests tests/kubera_bmi2_tests.cpp)
target_link_libraries(seven_bmi2_tests PRIVATE seven_core_bmi2 GTest::gtest_main)
if(MSVC)
  target_compile_options(seven_bmi2_tests PRIVATE /WX-)
  target_compile_options(seven_core_bmi2 PRIVATE /WX-)
endif()
gtest_discover_tests(seven_bmi2_tests)

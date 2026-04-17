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

FetchContent_Declare(
  softfloat
  GIT_REPOSITORY https://github.com/ucb-bar/berkeley-softfloat-3.git
  GIT_TAG        master
)
FetchContent_GetProperties(softfloat)
if(NOT softfloat_POPULATED)
  FetchContent_Populate(softfloat)
  file(GLOB SOFTFLOAT_SOURCES_ALL
    "${softfloat_SOURCE_DIR}/source/*.c"
    "${softfloat_SOURCE_DIR}/source/8086/*.c"
  )
  # With SOFTFLOAT_FAST_INT64, the *M.c multi-word variants must be excluded.
  list(FILTER SOFTFLOAT_SOURCES_ALL EXCLUDE REGEX "M\\.c$")
  list(FILTER SOFTFLOAT_SOURCES_ALL EXCLUDE REGEX "M_")
  set(SOFTFLOAT_SOURCES ${SOFTFLOAT_SOURCES_ALL})
  # Generate a platform.h appropriate for the current compiler/platform
  set(_sf_platform_dir "${CMAKE_BINARY_DIR}/softfloat_platform")
  file(MAKE_DIRECTORY "${_sf_platform_dir}")
  if(MSVC)
    file(WRITE "${_sf_platform_dir}/platform.h"
      "#define LITTLEENDIAN 1\n#define INLINE __forceinline\n")
  else()
    file(WRITE "${_sf_platform_dir}/platform.h"
      "#define LITTLEENDIAN 1\n#define INLINE static inline\n"
      "#define SOFTFLOAT_BUILTIN_CLZ 1\n")
  endif()

  add_library(softfloat STATIC ${SOFTFLOAT_SOURCES})
  target_include_directories(softfloat PUBLIC
    "${softfloat_SOURCE_DIR}/source/include"
    "${softfloat_SOURCE_DIR}/source/8086"
  )
  target_include_directories(softfloat PRIVATE "${_sf_platform_dir}")
  target_compile_definitions(softfloat PUBLIC SOFTFLOAT_FAST_INT64)
  if(MSVC)
    target_compile_options(softfloat PRIVATE /WX- /wd4146 /wd4244 /wd4267)
  endif()
endif()


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
target_include_directories(seven_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(seven_core PUBLIC iced_x86 softfloat)
target_compile_definitions(seven_core PUBLIC
  SEVEN_SIMD_PROFILE_${SEVEN_SIMD_PROFILE}=1
  SEVEN_ENABLE_AVX=${SEVEN_ENABLE_AVX}
  SEVEN_ENABLE_AVX2=${SEVEN_ENABLE_AVX2}
  SEVEN_ENABLE_AVX512=${SEVEN_ENABLE_AVX512}
  SEVEN_VECTOR_BITS=${SEVEN_VECTOR_BITS}
  SEVEN_MAX_VECTOR_BYTES=${SEVEN_MAX_VECTOR_BYTES}
  SOFTFLOAT_FAST_INT64
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
target_include_directories(seven_core_bmi2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(seven_core_bmi2 PUBLIC iced_x86 softfloat)
target_compile_definitions(seven_core_bmi2 PUBLIC
  SEVEN_SIMD_PROFILE_${SEVEN_SIMD_PROFILE}=1
  SEVEN_ENABLE_AVX=${SEVEN_ENABLE_AVX}
  SEVEN_ENABLE_AVX2=${SEVEN_ENABLE_AVX2}
  SEVEN_ENABLE_AVX512=${SEVEN_ENABLE_AVX512}
  SEVEN_VECTOR_BITS=${SEVEN_VECTOR_BITS}
  SEVEN_MAX_VECTOR_BYTES=${SEVEN_MAX_VECTOR_BYTES}
  SOFTFLOAT_FAST_INT64
)

add_executable(seven_bmi2_tests tests/kubera_bmi2_tests.cpp)
target_link_libraries(seven_bmi2_tests PRIVATE seven_core_bmi2 GTest::gtest_main)
if(MSVC)
  target_compile_options(seven_bmi2_tests PRIVATE /WX-)
  target_compile_options(seven_core_bmi2 PRIVATE /WX-)
endif()
gtest_discover_tests(seven_bmi2_tests)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  *.cpp
  *.hpp
  *.rc
)

list(SORT SRC_FILES)

if(SOGEN_BUILD_STATIC)
  add_library(seven-emulator STATIC ${SRC_FILES})
else()
  add_library(seven-emulator SHARED ${SRC_FILES})
endif()

target_include_directories(seven-emulator INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(seven-emulator PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../../kubera_core/include")
target_compile_features(seven-emulator PRIVATE cxx_std_23)

find_package(Git QUIET)
set(SOGEN_SEVEN_GIT_HASH "unknown")
set(SOGEN_SEVEN_GIT_DIRTY "unknown")
if (GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}/.." rev-parse --short=12 HEAD
    OUTPUT_VARIABLE SOGEN_SEVEN_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}/.." diff --quiet --no-ext-diff
    RESULT_VARIABLE SOGEN_SEVEN_DIFF_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if (SOGEN_SEVEN_DIFF_RESULT EQUAL 0)
    set(SOGEN_SEVEN_GIT_DIRTY "clean")
  else()
    set(SOGEN_SEVEN_GIT_DIRTY "dirty")
  endif()
endif()

target_compile_definitions(seven-emulator PRIVATE
  SOGEN_SEVEN_GIT_HASH="${SOGEN_SEVEN_GIT_HASH}"
  SOGEN_SEVEN_GIT_DIRTY="${SOGEN_SEVEN_GIT_DIRTY}"
  SOGEN_SEVEN_BUILD_STAMP="${CMAKE_SYSTEM_NAME}-${CMAKE_BUILD_TYPE}-${CMAKE_CXX_COMPILER_ID}"
)

target_link_libraries(seven-emulator PUBLIC emulator)
target_link_libraries(seven-emulator PRIVATE emulator-common seven_core)

momo_strip_target(seven-emulator)

cmake_minimum_required(VERSION 3.21)

project(icedcpp VERSION 0.1.0 LANGUAGES CXX)

option(ICEDCPP_BUILD_TESTS "Build iced_x86 tests" ON)
option(ICEDCPP_BUILD_SHARED "Build iced_x86 as a shared library" ON)
option(ICEDCPP_BUILD_DOCS "Build API docs with Doxygen" OFF)
option(ICEDCPP_BUILD_EXAMPLES "Build iced_x86 examples" ON)

if(ICEDCPP_BUILD_SHARED)
    set(ICEDCPP_LIBRARY_TYPE SHARED)
else()
    set(ICEDCPP_LIBRARY_TYPE STATIC)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
endforeach()

if(ICEDCPP_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(src)
add_subdirectory(bin)

if(ICEDCPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ICEDCPP_BUILD_DOCS)
    add_subdirectory(docs)
endif()

install(EXPORT icedcppTargets
    NAMESPACE iced_x86::
    DESTINATION lib/cmake/icedcpp
)

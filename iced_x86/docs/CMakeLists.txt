find_package(Doxygen REQUIRED)

set(DOXYGEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/out")
set(DOXYGEN_HTML_DIR "${DOXYGEN_OUTPUT_DIR}/html")
set(DOXYGEN_XML_DIR "${DOXYGEN_OUTPUT_DIR}/xml")
set(DOXYGEN_WARN_LOG "${CMAKE_CURRENT_BINARY_DIR}/doxygen-warnings.log")
set(DOXYGEN_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md")

if (DOXYGEN_DOT_FOUND)
    set(ICEDCPP_DOXYGEN_HAVE_DOT YES)
else()
    set(ICEDCPP_DOXYGEN_HAVE_DOT NO)
endif()

configure_file("${DOXYGEN_IN}" "${DOXYGEN_OUT}" @ONLY)

add_custom_command(
    OUTPUT "${DOXYGEN_HTML_DIR}/index.html"
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${DOXYGEN_OUT}" "${DOXYGEN_MAINPAGE}"
    COMMENT "Generating HTML API documentation with Doxygen"
    VERBATIM
)

add_custom_target(docs_html DEPENDS "${DOXYGEN_HTML_DIR}/index.html")

add_custom_target(docs_xml
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${DOXYGEN_OUT}" "${DOXYGEN_MAINPAGE}"
    COMMENT "Generating XML API documentation with Doxygen"
    VERBATIM
)

add_custom_target(docs DEPENDS docs_html docs_xml)

add_custom_target(docs_clean
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${DOXYGEN_OUTPUT_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${DOXYGEN_WARN_LOG}"
    COMMENT "Cleaning generated Doxygen documentation"
    VERBATIM
)

install(DIRECTORY "${DOXYGEN_HTML_DIR}/"
    DESTINATION "share/doc/icedcpp/html"
    OPTIONAL
)

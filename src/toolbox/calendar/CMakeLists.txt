add_cython_extension(
    TARGET      calendar_core
    SOURCE      ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyx
    DESTINATION toolbox/calendar
)

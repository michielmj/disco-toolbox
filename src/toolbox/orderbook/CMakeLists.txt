add_pybind11_extension(
    TARGET      orderbook_core
    SOURCE      ${CMAKE_CURRENT_SOURCE_DIR}/_core.cpp
    DESTINATION toolbox/orderbook
)

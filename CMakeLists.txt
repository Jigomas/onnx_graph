cmake_minimum_required(VERSION 3.14)
project(OnnxGraph VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(onnx_graph_lib
    src/graph.cpp
    src/node.cpp
    src/tensor.cpp
)

target_include_directories(onnx_graph_lib PUBLIC include/)

target_compile_options(onnx_graph_lib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O2>
)

add_executable(onnx_graph src/main.cpp)
target_link_libraries(onnx_graph PRIVATE onnx_graph_lib)

enable_testing()
add_subdirectory(tests)
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip)
FetchContent_MakeAvailable(googletest)

add_executable(tests
    test_graph_no_onnx.cpp
)

target_link_libraries(tests
    PRIVATE onnx_graph_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)
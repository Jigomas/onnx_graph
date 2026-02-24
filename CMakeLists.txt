find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS 
    third_party/onnx/onnx.proto3)

add_library(onnx_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)
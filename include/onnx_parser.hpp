#pragma once

#include "graph.hpp"
#include "onnx.pb.h"  // autogen from onnx.proto3 through protoc


class OnnxParser {
    OnnxParser()                                 = default;
    ~OnnxParser()                                = default;
    OnnxParser(const OnnxParser&)                = default;
    OnnxParser& operator=(const OnnxParser&)     = default;
    OnnxParser(OnnxParser&&) noexcept            = default;
    OnnxParser& operator=(OnnxParser&&) noexcept = default;

    Graph Parse(const onnx::ModelProto& model);
    
    

private:
    void ParseNodes(const onnx::GraphProto& proto_graph, Graph& graph);
    void ParseTensors(const onnx::GraphProto& proto_graph, Graph& graph);
    void ParseInitializers(const onnx::GraphProto& proto_graph, Graph& graph);

    Node      ParseNode(const onnx::NodeProto& proto_node);
    Tensor    ParseTensorInfo(const onnx::ValueInfoProto& info);
    AttrValue ParseAttribute(const onnx::AttributeProto& attr);

    DataType  MapDataType(int32_t onnx_type);
};
    
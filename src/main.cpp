#include "../include/graph.hpp"
#include "../include/node.hpp"
#include "../include/tensor.hpp"

#include <iostream>



static std::unique_ptr<Node> MakeNode (
    const std::string&              name,
    const std::string&              op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) 
{
    auto node = std::make_unique<Node>();
    node->name =    name;
    node->op_type = op_type;
    node->inputs =  inputs;
    node->outputs = outputs;

    return node;
}



static Tensor MakeTensor(
    const std::string& name,
    DataType dtype,
    std::vector<int64_t> dims) 
{
    Tensor tensor;
    tensor.name = name;
    tensor.dtype = dtype;
    tensor.shape.dims = std::move(dims);

    return tensor;
}





int main() {
    /*
    input
    │
    Conv 
    │
    conv_out
    │
    Relu
    │
    relu_out
    │
    Gemm
    │
    output 
    */
    Graph graph("graph");

    // Tensors

    graph.AddTensor(MakeTensor("input",
        DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("conv_out",
        DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("relu_out",
        DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("output",
        DataType::FLOAT32, {1, 1000}));

    // Conv: 32 filters, 3 entry channels, core 3x3
    Tensor conv_weights;
    conv_weights.name           = "conv_w";
    conv_weights.dtype          = DataType::FLOAT32;
    conv_weights.shape.dims     = {32, 3, 3, 3};
    conv_weights.is_initialiser = true;  
    graph.AddTensor(conv_weights);

    // Gemm
    Tensor gemm_weights;
    gemm_weights.name           = "gemm_w";
    gemm_weights.dtype          = DataType::FLOAT32;
    gemm_weights.shape.dims     = {1000, 32};
    gemm_weights.is_initialiser = true;
    graph.AddTensor(gemm_weights);


    // Conv: kernel_shape, strides, dilations, pads
    auto conv = MakeNode("conv_0", "Conv", {"input", "conv_w"}, {"conv_out"});
    conv->attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    conv->attributes["strides"]      = std::vector<int64_t>{1, 1};
    conv->attributes["dilations"]    = std::vector<int64_t>{1, 1};
    conv->attributes["pads"]         = std::vector<int64_t>{0, 0, 0, 0};
    conv->attributes["group"]        = int64_t{1};
    graph.AddNode(std::move(conv));

    // Relu:
    graph.AddNode(MakeNode("relu_0", "Relu", {"conv_out"}, {"relu_out"}));

    // Gemm: 
    // Y = alpha * A * B^T + beta * C
    auto gemm = MakeNode("gemm_0", "Gemm", {"relu_out", "gemm_w"}, {"output"});
    gemm->attributes["alpha"]  = float{1.0f};
    gemm->attributes["beta"]   = float{1.0f};
    gemm->attributes["transB"] = int64_t{1};  
    graph.AddNode(std::move(gemm));

    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    graph.DumpGraph();

    std::cout << "\n Топологический обход \n";
    for (const Node* node : graph.TopologicalSort()) {
        std::cout << "  [" << node->op_type << "] " << node->name << '\n';
    }

    std::cout << "\n Атрибуты Conv \n";
    const Node* conv_node = graph.FindNode("conv_0");
    if (conv_node) {
        auto strides = conv_node->GetAttr<std::vector<int64_t>>("strides");
        auto group   = conv_node->GetAttr<int64_t>("group", 1);
        std::cout << "  strides: [" << strides[0] << ", " << strides[1] << "]\n";
        std::cout << "  group:   "  << group << '\n';
    }

    std::cout << "\n Атрибуты Gemm \n";
    const Node* gemm_node = graph.FindNode("gemm_0");
    if (gemm_node) {
        auto alpha  = gemm_node->GetAttr<float>("alpha", 1.0f);
        auto transB = gemm_node->GetAttr<int64_t>("transB", 0);
        std::cout << "  alpha:  " << alpha  << '\n';
        std::cout << "  transB: " << transB << '\n';
    }

    return 0;
}
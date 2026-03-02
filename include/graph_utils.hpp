#pragma once

#include "node.hpp"
#include "tensor.hpp"

#include <vector>
#include <memory>
#include <string>



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
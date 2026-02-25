#include <gtest/gtest.h>

#include "../include/graph.hpp"
#include "../include/node.hpp"
#include "../include/tensor.hpp"



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



TEST(NodeTest, GetAttrString) {
    Node node;
    node.attributes["auto_pad"] = std::string{"SAME_UPPER"};
    EXPECT_EQ(node.GetAttr<std::string>("auto_pad"), "SAME_UPPER");
    EXPECT_EQ(node.GetAttr<std::string>("missing"), "");
}



TEST(GraphTest, FindNonexistentNodeReturnsNullptr) {
    Graph graph("empty");
    EXPECT_EQ(graph.FindNode("ghost"), nullptr);
}



TEST(GraphTest, AddNullNodeThrows) {
    Graph graph("test");
    EXPECT_THROW(graph.AddNode(nullptr), std::invalid_argument);
}



TEST(GraphTest, FindNonexistentTensorReturnsNullopt) {
    Graph graph("empty");
    auto result = graph.FindTensor("ghost");
    EXPECT_FALSE(result.has_value());
}

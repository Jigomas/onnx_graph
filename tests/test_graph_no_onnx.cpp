#include <gtest/gtest.h>

#include "../include/graph.hpp"
#include "../include/node.hpp"
#include "../include/graph_utils.hpp"




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

#pragma once

#include "node.hpp"
#include "tensor.hpp"

#include <vector>
#include <unordered_map>  
#include <memory>
#include <string>
#include <optional>  



class Graph {
public:
    explicit Graph(const std::string& name = "");
    ~Graph() = default;

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&&) noexcept            = default;
    Graph& operator=(Graph&&) noexcept = default;

    Graph Clone() const;
    
    void AddNode (std::unique_ptr<Node> node);
    void AddTensor(const Tensor& tensor);
    void AddTensor(Tensor&& tensor);

    Node* FindNode(const std::string& node_name) const;
    std::optional<Tensor> FindTensor(const std::string& tensor_name) const;

    std::vector<Node*> TopologicalSort() const;

    void DumpGraph() const;

    std::string                             name;
    std::vector<std::unique_ptr<Node>>      nodes;    
    std::unordered_map<std::string, Tensor> tensors;
    std::vector<std::string>                inputs;
    std::vector<std::string>                outputs;
    
    

private:
    void TopologicalSortData(
        Node*                                         node,
        std::unordered_map<std::string, bool>&        visited,
        const std::unordered_map<std::string, Node*>& tensor_producer,
        std::vector<Node*>&                           result
    ) const;
};

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
    std::string                             name;
    std::vector<std::unique_ptr<Node>>      nodes;    
    std::unordered_map<std::string, Tensor> tensors;
    std::vector<std::string>                inputs;
    std::vector<std::string>                outputs;

    explicit Graph(const std::string& name = "");
    ~Graph() = default;

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&&) noexcept            = default;
    Graph& operator=(Graph&&) noexcept = default;

    Graph Clone() const;
    
    void AddNode (std::unique_ptr<Node> node);
    void AddTensor(const Tensor& tensor);

    Node* FindNode(const std::string& name);
    std::optional<Tensor> FindTensor(const std::string& name) const;

    std::vector<Node*> TopologicalSort() const;

    void DumpGraph() const;


    
private:
    void TopologicalSortData(
        const std::string&                     node_name,
        std::unordered_map<std::string, bool>& visited,
        std::vector<Node*>&                    result
    ) const;
};

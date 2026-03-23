#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "graph.hpp"

class MLIRCodegen {
public:
    explicit MLIRCodegen(const Graph& graph);

    std::string GenerateMLIR() const;
    void        WriteToFile(const std::string& path) const;

private:
    const Graph& graph_;

    std::vector<std::string> CollectTensors() const;
    std::string              GetMemRefType(const std::string& name) const;
    int                      GetRank(const std::string& name) const;

    std::string
    GenNode(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;
    std::string
    GenAdd(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;
    std::string
    GenMul(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;
    std::string
    GenRelu(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;
    std::string
    GenMatMul(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;
    std::string
    GenGemm(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;
    std::string
    GenConv(const Node& n, const std::unordered_map<std::string, std::string>& m, int idx) const;

    static std::string IdentityAffineMap(int rank);
    static std::string ParallelIterTypes(int rank);
};

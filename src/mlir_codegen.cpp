#include "../include/mlir_codegen.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

MLIRCodegen::MLIRCodegen(const Graph& graph) : graph_(graph) {}

std::string MLIRCodegen::IdentityAffineMap(int rank) {
    std::string d, c;
    for (int i = 0; i < rank; i++) {
        if (i) {
            d += ", ";
            c += ", ";
        }
        d += "d" + std::to_string(i);
        c += "d" + std::to_string(i);
    }
    return "affine_map<(" + d + ") -> (" + c + ")>";
}

std::string MLIRCodegen::ParallelIterTypes(int rank) {
    std::string r = "[";
    for (int i = 0; i < rank; i++) {
        if (i)
            r += ", ";
        r += "\"parallel\"";
    }
    return r + "]";
}

int MLIRCodegen::GetRank(const std::string& name) const {
    auto it = graph_.tensors.find(name);
    return (it != graph_.tensors.end()) ? (int) it->second.shape.dims.size() : 1;
}

std::string MLIRCodegen::GetMemRefType(const std::string& name) const {
    auto it = graph_.tensors.find(name);
    if (it == graph_.tensors.end())
        return "memref<?xf32>";

    const Tensor& t  = it->second;
    std::string   sc = (t.dtype == DataType::FLOAT64) ? "f64"
                       : (t.dtype == DataType::INT32) ? "i32"
                       : (t.dtype == DataType::INT64) ? "i64"
                                                      : "f32";

    if (t.shape.dims.empty())
        return "memref<" + sc + ">";
    std::string r = "memref<";
    for (size_t i = 0; i < t.shape.dims.size(); i++) {
        if (i)
            r += "x";
        r += (t.shape.dims[i] < 0) ? "?" : std::to_string(t.shape.dims[i]);
    }
    return r + "x" + sc + ">";
}

std::vector<std::string> MLIRCodegen::CollectTensors() const {
    std::vector<std::string> res;
    std::set<std::string>    seen;
    auto                     add = [&](const std::string& n) {
        if (!n.empty() && seen.insert(n).second)
            res.push_back(n);
    };
    for (const auto& n : graph_.inputs)
        add(n);
    for (const auto& [k, t] : graph_.tensors)
        if (t.is_initializer)
            add(k);
    for (const auto& np : graph_.nodes) {
        for (const auto& s : np->inputs)
            add(s);
        for (const auto& s : np->outputs)
            add(s);
    }
    for (const auto& n : graph_.outputs)
        add(n);
    return res;
}

std::string MLIRCodegen::GenAdd(const Node&                                         n,
                                const std::unordered_map<std::string, std::string>& m,
                                int) const {
    const auto &       A = n.inputs[0], B = n.inputs[1], C = n.outputs[0];
    int                rank = std::max(GetRank(A), GetRank(B));
    auto               map = IdentityAffineMap(rank), it = ParallelIterTypes(rank);
    std::ostringstream s;
    s << "    linalg.generic { indexing_maps = [" << map << ", " << map << ", " << map << "],"
      << " iterator_types = " << it << " }\n"
      << "      ins(" << m.at(A) << ", " << m.at(B) << " : " << GetMemRefType(A) << ", "
      << GetMemRefType(B) << ")\n"
      << "      outs(" << m.at(C) << " : " << GetMemRefType(C) << ") {\n"
      << "    ^bb0(%a: f32, %b: f32, %c: f32):\n"
      << "      %r = arith.addf %a, %b : f32\n      linalg.yield %r : f32\n    }\n";
    return s.str();
}

std::string MLIRCodegen::GenMul(const Node&                                         n,
                                const std::unordered_map<std::string, std::string>& m,
                                int) const {
    const auto &       A = n.inputs[0], B = n.inputs[1], C = n.outputs[0];
    int                rank = std::max(GetRank(A), GetRank(B));
    auto               map = IdentityAffineMap(rank), it = ParallelIterTypes(rank);
    std::ostringstream s;
    s << "    linalg.generic { indexing_maps = [" << map << ", " << map << ", " << map << "],"
      << " iterator_types = " << it << " }\n"
      << "      ins(" << m.at(A) << ", " << m.at(B) << " : " << GetMemRefType(A) << ", "
      << GetMemRefType(B) << ")\n"
      << "      outs(" << m.at(C) << " : " << GetMemRefType(C) << ") {\n"
      << "    ^bb0(%a: f32, %b: f32, %c: f32):\n"
      << "      %r = arith.mulf %a, %b : f32\n      linalg.yield %r : f32\n    }\n";
    return s.str();
}

std::string MLIRCodegen::GenRelu(const Node&                                         n,
                                 const std::unordered_map<std::string, std::string>& m,
                                 int) const {
    const auto &       inp = n.inputs[0], out = n.outputs[0];
    int                rank = GetRank(inp);
    auto               map = IdentityAffineMap(rank), it = ParallelIterTypes(rank);
    std::ostringstream s;
    s << "    linalg.generic { indexing_maps = [" << map << ", " << map << "],"
      << " iterator_types = " << it << " }\n"
      << "      ins(" << m.at(inp) << " : " << GetMemRefType(inp) << ")\n"
      << "      outs(" << m.at(out) << " : " << GetMemRefType(out) << ") {\n"
      << "    ^bb0(%in: f32, %out_val: f32):\n"
      << "      %z = arith.constant 0.000000e+00 : f32\n"
      << "      %r = arith.maximumf %in, %z : f32\n      linalg.yield %r : f32\n    }\n";
    return s.str();
}

std::string MLIRCodegen::GenMatMul(const Node&                                         n,
                                   const std::unordered_map<std::string, std::string>& m,
                                   int                                                 idx) const {
    const auto &       A = n.inputs[0], B = n.inputs[1], C = n.outputs[0];
    std::string        zv = "%z" + std::to_string(idx);
    std::ostringstream s;
    s << "    " << zv << " = arith.constant 0.000000e+00 : f32\n"
      << "    linalg.fill ins(" << zv << " : f32) outs(" << m.at(C) << " : " << GetMemRefType(C)
      << ")\n"
      << "    linalg.matmul ins(" << m.at(A) << ", " << m.at(B) << " : " << GetMemRefType(A) << ", "
      << GetMemRefType(B) << ") outs(" << m.at(C) << " : " << GetMemRefType(C) << ")\n";
    return s.str();
}

std::string MLIRCodegen::GenGemm(const Node&                                         n,
                                 const std::unordered_map<std::string, std::string>& m,
                                 int                                                 idx) const {
    const auto &       A = n.inputs[0], B = n.inputs[1], C = n.outputs[0];
    int64_t            transB = n.GetAttr<int64_t>("transB", 0);
    std::string        zv     = "%z" + std::to_string(idx);
    std::string        op     = transB ? "linalg.matmul_transpose_b" : "linalg.matmul";
    std::ostringstream s;
    s << "    " << zv << " = arith.constant 0.000000e+00 : f32\n"
      << "    linalg.fill ins(" << zv << " : f32) outs(" << m.at(C) << " : " << GetMemRefType(C)
      << ")\n"
      << "    " << op << " ins(" << m.at(A) << ", " << m.at(B) << " : " << GetMemRefType(A) << ", "
      << GetMemRefType(B) << ") outs(" << m.at(C) << " : " << GetMemRefType(C) << ")\n";
    return s.str();
}

std::string MLIRCodegen::GenConv(const Node&                                         n,
                                 const std::unordered_map<std::string, std::string>& m,
                                 int                                                 idx) const {
    const auto &X = n.inputs[0], W = n.inputs[1], Y = n.outputs[0];
    auto        strides   = n.GetAttr<std::vector<int64_t>>("strides", {1, 1});
    auto        dilations = n.GetAttr<std::vector<int64_t>>("dilations", {1, 1});
    while (strides.size() < 2)
        strides.push_back(1);
    while (dilations.size() < 2)
        dilations.push_back(1);
    std::string        zv = "%z" + std::to_string(idx);
    std::ostringstream s;
    s << "    " << zv << " = arith.constant 0.000000e+00 : f32\n"
      << "    linalg.fill ins(" << zv << " : f32) outs(" << m.at(Y) << " : " << GetMemRefType(Y)
      << ")\n"
      << "    linalg.conv_2d_nchw_fchw {\n"
      << "      dilations = dense<[" << dilations[0] << ", " << dilations[1]
      << "]> : vector<2xi64>,\n"
      << "      strides   = dense<[" << strides[0] << ", " << strides[1] << "]> : vector<2xi64>\n"
      << "    } ins(" << m.at(X) << ", " << m.at(W) << " : " << GetMemRefType(X) << ", "
      << GetMemRefType(W) << ")" << " outs(" << m.at(Y) << " : " << GetMemRefType(Y) << ")\n";
    return s.str();
}

std::string MLIRCodegen::GenNode(const Node&                                         n,
                                 const std::unordered_map<std::string, std::string>& m,
                                 int                                                 idx) const {
    if (n.op_type == "Add")
        return GenAdd(n, m, idx);
    if (n.op_type == "Mul")
        return GenMul(n, m, idx);
    if (n.op_type == "Relu")
        return GenRelu(n, m, idx);
    if (n.op_type == "MatMul")
        return GenMatMul(n, m, idx);
    if (n.op_type == "Gemm")
        return GenGemm(n, m, idx);
    if (n.op_type == "Conv")
        return GenConv(n, m, idx);
    return "    // [UNSUPPORTED] " + n.op_type + "\n";
}

std::string MLIRCodegen::GenerateMLIR() const {
    auto tensors = CollectTensors();

    std::unordered_map<std::string, std::string> name_map;
    for (size_t i = 0; i < tensors.size(); i++)
        name_map[tensors[i]] = "%arg" + std::to_string(i);

    auto sorted = graph_.TopologicalSort();

    std::ostringstream out;
    out << "// MLIR module: " << graph_.name << "\n";
    out << "// Arguments:\n";
    for (size_t i = 0; i < tensors.size(); i++) {
        const std::string& name = tensors[i];
        auto               it   = graph_.tensors.find(name);
        bool               is_in =
            std::find(graph_.inputs.begin(), graph_.inputs.end(), name) != graph_.inputs.end();
        bool is_out =
            std::find(graph_.outputs.begin(), graph_.outputs.end(), name) != graph_.outputs.end();
        bool is_w = (it != graph_.tensors.end() && it->second.is_initializer);
        out << "//   %arg" << i << " : " << name;
        if (is_in)
            out << " [input]";
        if (is_out)
            out << " [output]";
        if (is_w)
            out << " [weight]";
        out << "\n";
    }
    out << "\nmodule {\n  func.func @model(\n";

    for (size_t i = 0; i < tensors.size(); i++) {
        out << "    %arg" << i << " : " << GetMemRefType(tensors[i]);
        if (i + 1 < tensors.size())
            out << ",";
        out << "\n";
    }

    out << "  ) {\n";
    int idx = 0;
    for (const Node* node : sorted)
        out << GenNode(*node, name_map, idx++);

    out << "    return\n  }\n}\n";
    return out.str();
}

void MLIRCodegen::WriteToFile(const std::string& path) const {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("MLIRCodegen: cannot open: " + path);
    f << GenerateMLIR();
}

#include <iostream>
#include <string>

#include "../include/compiler_driver.hpp"
#include "../include/graph.hpp"
#include "../include/graph_utils.hpp"
#include "../include/node.hpp"
#include "../include/onnx_loader.hpp"
#include "../include/tensor.hpp"
#include "../include/visualizer.hpp"

static void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [model.onnx]\n"
              << "\nOptions:\n"
              << "  --emit-mlir              Generate MLIR file (.mlir)\n"
              << "  --emit-llvmir            Generate LLVM IR file (.ll)\n"
              << "  --emit-asm               Generate assembly file (.s)\n"
              << "  --output <name>          Output file prefix (default: model)\n"
              << "  --target <arch>          Target architecture (default: x86-64)\n"
              << "  --opt-level <0-3>        Optimization level (default: 2)\n"
              << "  --mlir-opt <path>        Path to mlir-opt (default: mlir-opt-18)\n"
              << "  --mlir-translate <path>  Path to mlir-translate (default: mlir-translate-18)\n"
              << "  --llc <path>             Path to llc (default: llc-18)\n"
              << "  --visualize <out.png>    Render graph to PNG\n"
              << "  --help                   Show this message\n";
}

static Graph BuildManualGraph() {
    Graph graph("graph");

    // input(4,8) -> Add -> relu -> Gemm(W^T) -> output(4,4)
    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("bias", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("add_out", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("relu_out", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {4, 4}));

    Tensor w;
    w.name           = "gemm_w";
    w.dtype          = DataType::FLOAT32;
    w.shape.dims     = {4, 8};
    w.is_initializer = true;
    graph.AddTensor(w);

    graph.AddNode(MakeNode("add_0", "Add", {"input", "bias"}, {"add_out"}));
    graph.AddNode(MakeNode("relu_0", "Relu", {"add_out"}, {"relu_out"}));

    auto gemm                  = MakeNode("gemm_0", "Gemm", {"relu_out", "gemm_w"}, {"output"});
    gemm->attributes["transB"] = int64_t{1};
    graph.AddNode(std::move(gemm));

    graph.inputs  = {"input", "bias"};
    graph.outputs = {"output"};

    return graph;
}

int main(int argc, char* argv[]) {
    CompilerOptions opts;
    std::string     onnx_file;
    std::string     png_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--emit-mlir") {
            opts.emit_mlir = true;
        } else if (arg == "--emit-llvmir") {
            opts.emit_mlir   = true;
            opts.emit_llvmir = true;
        } else if (arg == "--emit-asm") {
            opts.emit_mlir   = true;
            opts.emit_llvmir = true;
            opts.emit_asm    = true;
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            opts.target = argv[++i];
        } else if (arg == "--opt-level" && i + 1 < argc) {
            opts.opt_level = std::stoi(argv[++i]);
        } else if (arg == "--mlir-opt" && i + 1 < argc) {
            opts.mlir_opt = argv[++i];
        } else if (arg == "--mlir-translate" && i + 1 < argc) {
            opts.mlir_translate = argv[++i];
        } else if (arg == "--llc" && i + 1 < argc) {
            opts.llc = argv[++i];
        } else if (arg == "--visualize" && i + 1 < argc) {
            png_file = argv[++i];
        } else if (arg[0] != '-') {
            onnx_file = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }

    try {
        Graph graph = onnx_file.empty() ? BuildManualGraph() : OnnxLoader().Load(onnx_file);

        graph.DumpGraph();

        std::cout << "\n--- Topological order ---\n";
        for (const Node* node : graph.TopologicalSort())
            std::cout << "  [" << node->op_type << "] " << node->name << '\n';

        if (!png_file.empty()) {
            Visualizer viz;
            viz.ToDot(graph, opts.output + ".dot");
            viz.Render(opts.output + ".dot", png_file);
            std::cout << "Saved: " << png_file << '\n';
        }

        if (opts.emit_mlir || opts.emit_llvmir || opts.emit_asm) {
            CompilerDriver driver(graph, opts);
            if (!driver.Run()) {
                std::cerr << "Compilation failed.\n";
                return 1;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

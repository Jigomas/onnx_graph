#pragma once

#include <string>

#include "graph.hpp"

struct CompilerOptions {
    bool        emit_mlir      = false;
    bool        emit_llvmir    = false;
    bool        emit_asm       = false;
    std::string output         = "model";
    std::string target         = "x86-64";
    int         opt_level      = 2;
    std::string mlir_opt       = "mlir-opt-18";
    std::string mlir_translate = "mlir-translate-18";
    std::string llc            = "llc-18";
};

class CompilerDriver {
public:
    CompilerDriver(const Graph& graph, const CompilerOptions& opts);

    bool Run();

private:
    bool LowerMLIR();
    bool ToLLVM();
    bool ToAsm();
    bool Exec(const std::string& cmd);

    const Graph&    graph_;
    CompilerOptions opts_;
};

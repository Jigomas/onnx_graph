#include "../include/compiler_driver.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "../include/mlir_codegen.hpp"

CompilerDriver::CompilerDriver(const Graph& graph, const CompilerOptions& opts)
    : graph_(graph), opts_(opts) {}

bool CompilerDriver::Exec(const std::string& cmd) {
    std::cout << "$ " << cmd << '\n';
    return std::system(cmd.c_str()) == 0;
}

bool CompilerDriver::LowerMLIR() {
    std::string mlir_in  = opts_.output + ".mlir";
    std::string mlir_out = opts_.output + "_lowered.mlir";
    std::string passes =
        "--convert-linalg-to-loops "
        "--lower-affine "
        "--convert-scf-to-cf "
        "--convert-arith-to-llvm "
        "--convert-cf-to-llvm "
        "--finalize-memref-to-llvm "
        "--convert-func-to-llvm "
        "--reconcile-unrealized-casts";
    return Exec(opts_.mlir_opt + " " + passes + " " + mlir_in + " -o " + mlir_out);
}

bool CompilerDriver::ToLLVM() {
    std::string mlir_in = opts_.output + "_lowered.mlir";
    std::string ll_out  = opts_.output + ".ll";
    return Exec(opts_.mlir_translate + " --mlir-to-llvmir " + mlir_in + " -o " + ll_out);
}

bool CompilerDriver::ToAsm() {
    std::string ll_in   = opts_.output + ".ll";
    std::string asm_out = opts_.output + ".s";
    return Exec(opts_.llc + " -march=" + opts_.target + " -O" + std::to_string(opts_.opt_level) +
                " " + ll_in + " -o " + asm_out);
}

bool CompilerDriver::Run() {
    std::string mlir_path = opts_.output + ".mlir";
    MLIRCodegen(graph_).WriteToFile(mlir_path);
    std::cout << "Written: " << mlir_path << '\n';

    if (!opts_.emit_mlir && !opts_.emit_llvmir && !opts_.emit_asm)
        return true;

    if (!LowerMLIR())
        return false;

    if (opts_.emit_llvmir || opts_.emit_asm)
        if (!ToLLVM())
            return false;

    if (opts_.emit_asm)
        if (!ToAsm())
            return false;

    return true;
}

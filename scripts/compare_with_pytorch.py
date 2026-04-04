import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("pip install torch --index-url https://download.pytorch.org/whl/cpu")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnxruntime")

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def ort_run(onnx_path, feed):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, feed)


def build_add(path):
    class M(torch.nn.Module):
        def forward(self, a, b): return a + b
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    torch.onnx.export(M(), (a, b), path, input_names=["a", "b"],
                      output_names=["c"], opset_version=13)
    return {"a": a.numpy(), "b": b.numpy()}


def build_mul(path):
    class M(torch.nn.Module):
        def forward(self, a, b): return a * b
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    torch.onnx.export(M(), (a, b), path, input_names=["a", "b"],
                      output_names=["c"], opset_version=13)
    return {"a": a.numpy(), "b": b.numpy()}


def build_relu(path):
    class M(torch.nn.Module):
        def forward(self, x): return torch.relu(x)
    x = torch.randn(1, 16)
    torch.onnx.export(M(), x, path, input_names=["x"],
                      output_names=["y"], opset_version=13)
    return {"x": x.numpy()}


def build_matmul(path):
    class M(torch.nn.Module):
        def forward(self, a, b): return torch.matmul(a, b)
    a, b = torch.randn(4, 8), torch.randn(8, 4)
    torch.onnx.export(M(), (a, b), path, input_names=["a", "b"],
                      output_names=["c"], opset_version=13)
    return {"a": a.numpy(), "b": b.numpy()}


def build_gemm(path):
    class M(torch.nn.Module):
        def forward(self, a, b): return torch.mm(a, b.t())
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    torch.onnx.export(M(), (a, b), path, input_names=["a", "b"],
                      output_names=["c"], opset_version=13)
    return {"a": a.numpy(), "b": b.numpy()}


TESTS = [
    ("Add",    build_add),
    ("Mul",    build_mul),
    ("Relu",   build_relu),
    ("MatMul", build_matmul),
    ("Gemm",   build_gemm),
]

def run_test(name, build_fn, tool, mlir_opt, workdir):
    onnx_path   = os.path.join(workdir, f"{name}.onnx")
    output_pref = os.path.join(workdir, name)

    # 1. Create ONNX model
    feed = build_fn(onnx_path)

    # 2. onnxruntime reference
    ref_outputs = ort_run(onnx_path, feed)
    print(f"  onnxruntime reference:")
    for i, arr in enumerate(ref_outputs):
        print(f"    output[{i}]: shape={arr.shape}  min={arr.min():.4f}"
              f"  max={arr.max():.4f}  mean={arr.mean():.4f}")

    # 3. Our compiler: parse ONNX + emit MLIR
    cmd  = f"{tool} --emit-mlir --output {output_pref} {onnx_path}"
    cmd += f" --mlir-opt {mlir_opt}"
    rc, log = run_cmd(cmd)

    mlir_file = output_pref + ".mlir"
    if rc == 0 and os.path.exists(mlir_file):
        size = os.path.getsize(mlir_file)
        print(f"  {GREEN}[OK]{RESET}  MLIR generated ({size} bytes): {mlir_file}")
    else:
        print(f"  {RED}[FAIL]{RESET} Pipeline error:")
        print("   ", log[:400])
        return False

    # 4. Show first lines of generated MLIR for inspection
    with open(mlir_file) as f:
        lines = f.readlines()
    preview = "".join(lines[:min(20, len(lines))])
    print(f"  --- MLIR preview ---\n{preview}  --------------------")

    return True

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build-dir",  default="build",
                   help="cmake build directory (default: build)")
    p.add_argument("--mlir-opt",   default="mlir-opt-18",
                   help="mlir-opt binary (default: mlir-opt-18)")
    p.add_argument("--test",       default=None,
                   help="run only one test: Add|Mul|Relu|MatMul|Gemm")
    args = p.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.build_dir):
        args.build_dir = os.path.join(repo, args.build_dir)

    tool = os.path.join(args.build_dir, "onnx_graph")
    if not os.path.exists(tool):
        sys.exit(f"onnx_graph not found at {tool}. Build the project first.")

    tests = [(n, f) for n, f in TESTS if args.test is None or n == args.test]

    results = {}
    with tempfile.TemporaryDirectory(prefix="onnx_cmp_") as workdir:
        for name, fn in tests:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            results[name] = run_test(name, fn, tool, args.mlir_opt, workdir)

    print(f"\n{'='*60}")
    passed = sum(v for v in results.values())
    for name, ok in results.items():
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {name}")
    print(f"\n  {passed}/{len(results)} passed")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
# onnx-graph

ONNX tensor compiler — loads a neural network from an .onnx file,
builds a computational graph, and compiles it through MLIR to LLVM IR and assembly.

## What it does

- Parses binary .onnx files via protobuf
- Builds a typed computational graph (nodes = ops, tensors = edges)
- Supports ops: Conv, Relu, Add, Mul, MatMul, Gemm
- Topological sort for correct execution order
- Graphviz visualization
- MLIR codegen (linalg dialect) for all supported ops
- Lowering pipeline: MLIR → LLVM IR → assembly (via mlir-opt / mlir-translate / llc)

## Graph structure

```text
  input
    │
  [Conv]   
    │
 conv_out
    │
  [Relu]
    │
 relu_out
    │
  [Gemm]   
    │
  output
```

## Dependencies

```bash
sudo apt install cmake build-essential protobuf-compiler libprotobuf-dev graphviz
sudo apt install eog
# For MLIR/LLVM codegen (Ubuntu 22.04+):
sudo apt install mlir-18-tools llvm-18
```

## Build

```bash
# Download ONNX schema
mkdir -p third_party/onnx
wget -O third_party/onnx/onnx.proto3 https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3

# Rename for protoc
mv third_party/onnx/onnx.proto3 third_party/onnx/onnx.proto

# Generate protobuf sources and build
mkdir build 
cd build
protoc --cpp_out=. -I ../third_party/onnx ../third_party/onnx/onnx.proto
cmake ..
make -j4
```

## Run

```bash
# Manual graph demo (no .onnx needed)
./onnx_graph

# Load ONNX model
./onnx_graph model.onnx

# Generate MLIR
./onnx_graph --emit-mlir --output out model.onnx
# → out.mlir

# Generate LLVM IR
./onnx_graph --emit-llvmir --output out model.onnx
# → out.mlir, out_lowered.mlir, out.ll

# Generate assembly (x86-64 by default)
./onnx_graph --emit-asm --output out model.onnx
# → out.mlir, out_lowered.mlir, out.ll, out.s

# Specify target and optimization level
./onnx_graph --emit-asm --target aarch64 --opt-level 3 --output out model.onnx

# Custom tool paths
./onnx_graph --emit-asm --mlir-opt mlir-opt --mlir-translate mlir-translate --llc llc --output out

# Visualize graph
./onnx_graph --visualize graph.png model.onnx
```

## Comparison with onnxruntime

The script generates tiny ONNX models (Add, Mul, Relu, MatMul, Gemm) via PyTorch,
runs them with onnxruntime as a reference, then runs our compiler pipeline and
shows the generated MLIR for inspection.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime

python3 scripts/compare_with_pytorch.py
# or for a single op:
python3 scripts/compare_with_pytorch.py --test Add
```

## Tests

```bash
# Run tests
cd build
./tests/tests

# Run specific suite
./tests/tests --gtest_filter=GraphTest.*


#Or with visualizator
mkdir test_viz_output
cd /root/.a_Programming/onnx-graph
chmod +x run_tests_with_viz.sh
./run_tests_with_viz.sh
#Search for them in test_viz_output
```

## Project structure

```text
onnx-graph/
├── include/
│   ├── tensor.hpp        # Tensor, TensorShape, DataType
│   ├── node.hpp          # Node, AttrValue
│   ├── graph.hpp         # Graph
│   ├── graph_utils.hpp   # MakeNode, MakeTensor
│   ├── onnx_loader.hpp   # file I/O
│   ├── onnx_parser.hpp   # protobuf → Graph
│   ├── visualizer.hpp    # Graphviz .dot generator
│   ├── mlir_codegen.hpp  # Graph → MLIR (linalg dialect)
│   └── compiler_driver.hpp # MLIR → LLVM IR → assembly pipeline
├── src/
│   ├── graph.cpp
│   ├── onnx_loader.cpp
│   ├── onnx_parser.cpp
│   ├── visualizer.cpp
│   ├── mlir_codegen.cpp
│   ├── compiler_driver.cpp
│   └── main.cpp
├── tests/
│   ├── tests.cpp
│   └── CMakeLists.txt
├── scripts/
│   └── compare_with_pytorch.py  # PyTorch/onnxruntime vs our compiler
├── third_party/
│   └── onnx/
│       └── onnx.proto
├── test_viz_output/              
├── build/                        
├── .gitignore
├── CMakeLists.txt
├── README.md
└── run_tests_with_viz.sh         
```

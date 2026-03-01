#include "../include/onnx_loader.hpp"
#include "../include/onnx_parser.hpp"

#include "onnx.pb.h"

#include <fstream>
#include <stdexcept>



Graph OnnxLoader::Load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + filepath);

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&file))
        throw std::runtime_error("Failed to parse ONNX model: " + filepath);

    return OnnxParser().Parse(model);
}
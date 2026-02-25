#pragma once 
#include <string>
#include <vector>
#include <cstdint> 



enum class DataType {FLOAT32, INT32, INT64, FLOAT64, UNKNOWN};



class TensorShape {
public:
    std::vector<int64_t> dims;
    bool                 is_dynamic = false; 
};



class Tensor {
public:
    std::string        name;
    DataType           dtype          = DataType::UNKNOWN;
    TensorShape        shape;
    bool               is_initialiser = false;
    std::vector<float> data;

    Tensor()                             = default;
    ~Tensor()                            = default;

    Tensor(const Tensor&)                = default;
    Tensor& operator=(const Tensor&)     = default;

    Tensor(Tensor&&) noexcept            = default;
    Tensor& operator=(Tensor&&) noexcept = default;
};
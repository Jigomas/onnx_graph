#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>



using AttrValue = std::variant 
<
    int64_t,
    float,
    std::string,
    std::vector<int64_t>,
    std::vector<float>
>;



template<typename T>
inline constexpr bool is_attr_type_v =
    std::is_same_v<T, int64_t>              ||
    std::is_same_v<T, float>                ||
    std::is_same_v<T, std::string>          ||
    std::is_same_v<T, std::vector<int64_t>> ||
    std::is_same_v<T, std::vector<float>>;




class Node {
public:
    std::string                      name;
    std::string                      op_type;
    std::vector<std::string>         inputs;
    std::vector<std::string>         outputs;
    std::map<std::string, AttrValue> attributes;
    
    Node()                           = default;
    ~Node()                          = default;

    Node(const Node&)                = default;
    Node& operator=(const Node&)     = default;

    Node(Node&&) noexcept            = default;
    Node& operator=(Node&&) noexcept = default;

    template<typename T>
    T GetAttr(const std::string& key, T default_val = T{}) const {
        static_assert(is_attr_type_v<T>, "T is not an AttrValue type");

        auto it = attributes.find(key);
        if (it == attributes.end()) return default_val;
        if (const auto* val = std::get_if<T>(&it->second))
            return *val;
        return default_val;
    }
};
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



class Node {
public:
    std::string                      name;
    std::string                      op_type;
    std::vector<std::string>         inputs;
    std::map<std::string, AttrValue> attributes;
    
    Node()                           = default;
    ~Node()                          = default;

    Node(const Node&)                = default;
    Node& operator=(const Node&)     = default;

    Node(Node&&) noexcept            = default;
    Node& operator=(Node&&) noexcept = default;

    template<typename T>
    T GetAttr(const std::string& key, T default_val = T{}) const {
        static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, float>);

        auto it = attributes.find(key);
        if (it == attributes.end()) return default_val;
        if (const auto* val = std::get_if<T>(&it->second))
            return *val;
        return default_val;
    }

};
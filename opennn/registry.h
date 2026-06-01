//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I S T R Y   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <string>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <format>

namespace opennn
{

template<typename T>
class Registry
{
public:

    using Creator = std::function<std::unique_ptr<T>()>;

    static Registry& instance()
    {
        static Registry registry;
        return registry;
    }

    void register_component(const std::string& name, Creator creator)
    {
        creators[name] = std::move(creator);
    }

    std::unique_ptr<T> create(const std::string& name) const
    {
        auto it = creators.find(name);

        if (it == creators.end())
            throw std::runtime_error(std::format("Component not found: {}", name));

        return it->second();

    }

    std::vector<std::string> registered_names() const
    {
        std::vector<std::string> names;

        for (const auto& pair : creators)
            names.push_back(pair.first);

        return names;
    }

private:
    std::unordered_map<std::string, Creator> creators;
};

#define REGISTER(BASE, CLASS, NAME) \
namespace { \
    const bool CLASS##_registered = []() { \
        Registry<BASE>::instance().register_component(NAME, []() { \
            return std::make_unique<CLASS>(); \
        }); \
        return true; \
    }(); \
}

void register_classes();

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

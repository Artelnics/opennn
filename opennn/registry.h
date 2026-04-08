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
#include <vector>

namespace opennn
{

using std::string;
using std::unique_ptr;
using std::vector;
using std::function;
using std::unordered_map;
using std::runtime_error;

template<typename T>
class Registry
{
public:

    using Creator = function<unique_ptr<T>()>;

    static Registry& instance()
    {
        static Registry registry;
        return registry;
    }

    void register_component(const string& name, Creator creator)
    {
        creators[name] = std::move(creator);
    }

    unique_ptr<T> create(const string& name) const
    {
        auto it = creators.find(name);

        if(it == creators.end())
            throw runtime_error("Component not found: " + name);

        return it->second();

    }

    vector<string> registered_names() const
    {
        vector<string> names;

        for(const auto& pair : creators)
            names.push_back(pair.first);

        return names;
    }

private:
    unordered_map<string, Creator> creators;
};

#define REGISTER(BASE, CLASS, NAME) \
namespace { \
    const bool CLASS##_registered = []() { \
              Registry<BASE>::instance().register_component(NAME, [](){ \
                          return std::make_unique<CLASS>(); \
                  }); \
              return true; \
      }(); \
}

void reference_dense_layer();
void reference_scaling_layer();
void reference_flatten_layer();
void reference_addition_layer();
void reference_all_layers();

void register_classes();

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

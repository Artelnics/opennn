//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I S T R Y   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef REGISTRY_H
#define REGISTRY_H

#include <string>
#include <functional>
#include <memory>
#include <stdexcept>

using namespace std;

namespace opennn
{

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
                          return make_unique<CLASS>(); \
                  }); \
              return true; \
      }(); \
}

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

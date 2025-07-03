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

//#include "layer.h"

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

/*
class ForwardRegistry
{
public:
    using Creator = function<unique_ptr<LayerForwardPropagation>(size_t, Layer*)>;

    static ForwardRegistry& instance()
    {
        static ForwardRegistry reg;
        return reg;
    }

    void register_creator(const string& layer_type, Creator creator)
    {
        creators[layer_type] = std::move(creator);
    }

    unique_ptr<LayerForwardPropagation> create(const string& name, size_t samples, Layer* layer)
    {
        auto it = creators.find(name);
        if(it == creators.end())
            throw runtime_error("No forward propagation registered for: " + name);
        return it->second(samples, layer);
    }

    vector<string> registered_names() const
    {
        vector<string> names;

        for(const auto& kv : creators)
            names.push_back(kv.first);

        return names;
    }

private:
    unordered_map<string, Creator> creators;
};


#define REGISTER_FORWARD_PROPAGATION(LAYER_NAME, FORWARD_CLASS) \
namespace { \
    const bool FORWARD_CLASS##_registered = []() { \
              ForwardRegistry::instance().register_creator(LAYER_NAME, \
                                    [](size_t samples, Layer* layer) { \
                                            return make_unique<FORWARD_CLASS>(samples, layer); \
                                    }); \
              return true; \
      }(); \
}


class BackRegistry
{
public:
    using Creator = function<unique_ptr<LayerBackPropagation>(size_t, Layer*)>;

    static BackRegistry& instance()
    {
        static BackRegistry registry;
        return registry;
    }

    void register_creator(const string& layer_type, Creator creator)
    {
        creators[layer_type] = std::move(creator);
    }

    unique_ptr<LayerBackPropagation> create(const string& layer_type, size_t samples, Layer* layer) const
    {
        auto it = creators.find(layer_type);

        if (it == creators.end())
            throw runtime_error("No backpropagation registered for: " + layer_type);

        return it->second(samples, layer);
    }

    vector<string> registered_names() const
    {
        vector<string> names;

        for (const auto& kv : creators)
            names.push_back(kv.first);

        return names;
    }

private:

    unordered_map<string, Creator> creators;
};


#define REGISTER_BACK_PROPAGATION(LAYER_NAME, BACK_CLASS) \
namespace { \
    const bool BACK_CLASS##_registered = []() { \
              BackRegistry::instance().register_creator(LAYER_NAME, \
                                    [](size_t samples, Layer* layer) { \
                                            return std::make_unique<BACK_CLASS>(samples, layer); \
                                    }); \
              return true; \
      }(); \
}

#ifdef OPENNN_CUDA

class ForwardCudaRegistry
{
public:
    using Creator = function<unique_ptr<LayerForwardPropagationCuda>(size_t, Layer*)>;

    static ForwardCudaRegistry& instance()
    {
        static ForwardCudaRegistry reg;
        return reg;
    }

    void register_creator(const string& layer_type, Creator creator)
    {
        creators[layer_type] = move(creator);
    }

    unique_ptr<LayerForwardPropagationCuda> create(const string& layer_type, size_t samples, Layer* layer) const
    {
        auto it = creators.find(layer_type);
        if (it == creators.end())
            throw runtime_error("No CUDA forward propagation registered for: " + layer_type);
        return it->second(samples, layer);
    }

    vector<string> registered_names() const
    {
        vector<string> names;
        for (const auto& kv : creators)
            names.push_back(kv.first);
        return names;
    }

private:

    unordered_map<string, Creator> creators;
};


#define REGISTER_FORWARD_CUDA(LAYER_NAME, FORWARD_CLASS) \
namespace { \
    const bool FORWARD_CLASS##_cuda_registered = []() { \
              ForwardCudaRegistry::instance().register_creator(LAYER_NAME, \
                                    [](size_t samples, Layer* layer) { \
                                            return std::make_unique<FORWARD_CLASS>(samples, layer); \
                                    }); \
              return true; \
      }(); \
}

class BackCudaRegistry
{
public:

    using Creator = function<unique_ptr<LayerBackPropagationCuda>(size_t, Layer*)>;

    static BackCudaRegistry& instance()
    {
        static BackCudaRegistry reg;
        return reg;
    }

    void register_creator(const string& layer_type, Creator creator)
    {
        creators[layer_type] = std::move(creator);
    }

    unique_ptr<LayerBackPropagationCuda> create(const string& layer_type, size_t samples, Layer* layer) const
    {
        auto it = creators.find(layer_type);
        if (it == creators.end())
            throw runtime_error("No CUDA backpropagation registered for: " + layer_type);
        return it->second(samples, layer);
    }

    vector<string> registered_names() const
    {
        vector<string> names;
        for (const auto& kv : creators)
            names.push_back(kv.first);
        return names;
    }

private:
    unordered_map<string, Creator> creators;
};


#define REGISTER_BACK_CUDA(LAYER_NAME, BACK_CLASS) \
namespace { \
    const bool BACK_CLASS##_cuda_registered = []() { \
              BackCudaRegistry::instance().register_creator(LAYER_NAME, \
                                    [](size_t samples, Layer* layer) { \
                                            return make_unique<BACK_CLASS>(samples, layer); \
                                    }); \
              return true; \
      }(); \
}

#endif
*/
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

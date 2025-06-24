//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRAININGSTRATEGY_H
#define TRAININGSTRATEGY_H

#include "mean_squared_error.h"
#include "normalized_squared_error.h"
#include "minkowski_error.h"
#include "cross_entropy_error.h"
#include "cross_entropy_error_3d.h"
#include "weighted_squared_error.h"

#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"
#include "stochastic_gradient_descent.h"
#include "adaptive_moment_estimation.h"

namespace opennn
{

//class NeuralNetwork;
class LossIndex;
class OptimizationAlgorithm;

struct TrainingResults;

class TrainingStrategy
{

public:

    TrainingStrategy(NeuralNetwork* = nullptr, Dataset* = nullptr);

    enum class LossMethod
    {
        MEAN_SQUARED_ERROR,
        NORMALIZED_SQUARED_ERROR,
        MINKOWSKI_ERROR,
        WEIGHTED_SQUARED_ERROR,
        CROSS_ENTROPY_ERROR_2D,
        CROSS_ENTROPY_ERROR_3D
    };

    enum class OptimizationMethod
    {
        QUASI_NEWTON_METHOD,
        LEVENBERG_MARQUARDT_ALGORITHM,
        STOCHASTIC_GRADIENT_DESCENT,
        ADAPTIVE_MOMENT_ESTIMATION
    };

    Dataset* get_data_set();

    NeuralNetwork* get_neural_network() const;

    LossIndex* get_loss_index();
    OptimizationAlgorithm* get_optimization_algorithm();

    bool has_neural_network() const;
    bool has_data_set() const;

    MeanSquaredError* get_mean_squared_error();
    NormalizedSquaredError* get_normalized_squared_error();
    MinkowskiError* get_Minkowski_error();
    CrossEntropyError2d* get_cross_entropy_error();
    WeightedSquaredError* get_weighted_squared_error();

    QuasiNewtonMethod* get_quasi_Newton_method();
    LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm();
    StochasticGradientDescent* get_stochastic_gradient_descent();
    AdaptiveMomentEstimation* get_adaptive_moment_estimation();

    const LossMethod& get_loss_method() const;
    const OptimizationMethod& get_optimization_method() const;

    string write_loss_method() const;
    string write_optimization_method() const;

    string write_optimization_method_text() const;
    string write_loss_method_text() const;

    const bool& get_display() const;

    // Set

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);
    void set_default();

    void set_threads_number(const int&);

    void set_data_set(Dataset*);
    void set_neural_network(NeuralNetwork*);

    void set_loss_index(LossIndex*);

    void set_loss_method(const LossMethod&);
    void set_optimization_method(const OptimizationMethod&);

    void set_loss_method(const string&);
    void set_optimization_method(const string&);

    void set_display(const bool&);

    void set_loss_goal(const type&);
    void set_maximum_selection_failures(const Index&);
    void set_maximum_epochs_number(const int&);
    void set_display_period(const int&);

    void set_maximum_time(const type&);

    // Training

    TrainingResults perform_training();
#ifdef OPENNN_CUDA
    TrainingResults perform_training_cuda();
#endif

    // Check

    void fix_forecasting();

    // Serialization

    void print() const;

    void from_XML(const XMLDocument&);
    void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    Dataset* dataset = nullptr;

    NeuralNetwork* neural_network = nullptr;

    // Loss index

    MeanSquaredError mean_squared_error;

    NormalizedSquaredError normalized_squared_error;

    MinkowskiError Minkowski_error;

    CrossEntropyError2d cross_entropy_error_2d;

    CrossEntropyError3d cross_entropy_error_3d;

    WeightedSquaredError weighted_squared_error;

    LossMethod loss_method;

    // Optimization algorithm

    QuasiNewtonMethod quasi_Newton_method;

    LevenbergMarquardtAlgorithm Levenberg_Marquardt_algorithm;

    StochasticGradientDescent stochastic_gradient_descent;

    AdaptiveMomentEstimation adaptive_moment_estimation;

    OptimizationMethod optimization_method;

    bool display = true;

};

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

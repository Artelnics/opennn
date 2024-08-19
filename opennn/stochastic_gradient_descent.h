//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STOCHASTICGRADIENTDESCENT_H
#define STOCHASTICGRADIENTDESCENT_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"

#include "loss_index.h"
#include "optimization_algorithm.h"

namespace opennn
{

struct StochasticGradientDescentData;

class StochasticGradientDescent : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit StochasticGradientDescent(); 

   explicit StochasticGradientDescent(LossIndex*);

   //Training operators

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   // Stopping criteria

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;

   // Set methods

   void set_loss_index(LossIndex*) override;

   void set_default() final;

   void set_batch_samples_number(const Index& new_batch_samples_number)
   {
       batch_samples_number = new_batch_samples_number;
   }

   // Get methods

   Index get_batch_samples_number() const;

   //Training operators

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);

   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);

   // Training methods

   void update_parameters(BackPropagation& , StochasticGradientDescentData&) const;

   TrainingResults perform_training() final;

   string write_optimization_algorithm_type() const final;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const final;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

private:

   // Training operators

   type initial_learning_rate;

   type initial_decay;

   type momentum = type(0);

   bool nesterov = false;

   Index batch_samples_number = 1000;

   // Stopping criteria

   type training_loss_goal = type(0);

   Index maximum_selection_failures = numeric_limits<Index>::max();

   Index maximum_epochs_number = 10000;

   type maximum_time = type(3600);

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/stochastic_gradient_descent_cuda.h"
#endif

};


struct StochasticGradientDescentData : public OptimizationAlgorithmData
{
    explicit StochasticGradientDescentData()
    {
    }

    explicit StochasticGradientDescentData(StochasticGradientDescent* new_stochastic_gradient_descent)
    {
        set(new_stochastic_gradient_descent);
    }

    virtual ~StochasticGradientDescentData() {}

    void set(StochasticGradientDescent* new_stochastic_gradient_descent);

    StochasticGradientDescent* stochastic_gradient_descent = nullptr;

    Index iteration = 0;

    Tensor<type, 1> parameters_increment;
    Tensor<type, 1> last_parameters_increment;
};

}

#endif

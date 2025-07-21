//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STOCHASTICGRADIENTDESCENT_H
#define STOCHASTICGRADIENTDESCENT_H

#include "optimization_algorithm.h"

namespace opennn
{

struct BackPropagation;
struct StochasticGradientDescentData;

#ifdef OPENNN_CUDA
struct SGDOptimizationDataCuda;
#endif


class StochasticGradientDescent : public OptimizationAlgorithm
{

public:

   StochasticGradientDescent(const LossIndex* = nullptr);

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   const type& get_loss_goal() const;

   void set_default();

   void set_batch_size(const Index&);

   Index get_samples_number() const;

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);

   void set_maximum_epochs_number(const Index&);

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);

   void update_parameters(BackPropagation& , StochasticGradientDescentData&) const;

   TrainingResults train() override;

   string get_name() const override;

   Tensor<string, 2> to_string_matrix() const override;

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

private:

   type initial_learning_rate;

   type initial_decay;

   type momentum = type(0);

   bool nesterov = false;

   Index batch_size = 1000;

   type training_loss_goal = type(0);

   Index maximum_selection_failures = numeric_limits<Index>::max();

#ifdef OPENNN_CUDA

public:

    TrainingResults perform_training_cuda() override;

    void update_parameters_cuda(BackPropagationCuda&, SGDOptimizationDataCuda&) const;

#endif

};


struct StochasticGradientDescentData : public OptimizationAlgorithmData
{
    StochasticGradientDescentData(StochasticGradientDescent* = nullptr);

    void set(StochasticGradientDescent* = nullptr);

    StochasticGradientDescent* stochastic_gradient_descent = nullptr;

    Index iteration = 0;

    vector<vector<Tensor<type, 1>>> parameters_increment;
    vector<vector<Tensor<type, 1>>> last_parameters_increment;
};


#ifdef OPENNN_CUDA

struct SGDOptimizationDataCuda : public OptimizationAlgorithmData
{
    SGDOptimizationDataCuda(StochasticGradientDescent* = nullptr);

    ~SGDOptimizationDataCuda() { free(); }

    void set(StochasticGradientDescent* = nullptr);

    void free();

    void print() const;

    StochasticGradientDescent* stochastic_gradient_descent = nullptr;

    Index iteration = 0;

    vector<vector<float*>> velocity;
};

#endif

}

#endif

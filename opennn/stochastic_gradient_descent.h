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

struct StochasticGradientDescentData;

#ifdef OPENNN_CUDA
struct SGDOptimizationDataCuda;
#endif


class StochasticGradientDescent : public OptimizationAlgorithm
{

public:

   StochasticGradientDescent(LossIndex* = nullptr);

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;

   void set_default();

   void set_batch_samples_number(const Index&);

   Index get_samples_number() const;

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);

   void set_maximum_epochs_number(const Index&);


   void set_loss_goal(const type&);
   void set_maximum_time(const type&);

   void update_parameters(BackPropagation& , StochasticGradientDescentData&) const;

   TrainingResults perform_training() override;

   string write_optimization_algorithm_type() const override;

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

   Index maximum_epochs_number = 10000;

   type maximum_time = type(3600000);

#ifdef OPENNN_CUDA

public:

    TrainingResults perform_training_cuda();

protected:

    void update_parameteres_cuda(LossIndex::BackPropagationCuda& back_propagation_cuda,
        SGDOptimizationDataCuda& optimization_data_cuda);

    bool display = true;
    Index display_period = 1;
    Index maximum_epochs_number = 1000;
    Index batch_samples_number = 64;
    float initial_learning_rate = 0.01f;
    float training_loss_goal = 0.01f;
    float maximum_time = 3600.0f;
    Index maximum_selection_failures = 10;
    Index save_period = 10;
    //std::string neural_network_file_name = "network.nn";

    LossIndex* loss_index = nullptr;

#endif

};


struct StochasticGradientDescentData : public OptimizationAlgorithmData
{
    StochasticGradientDescentData(StochasticGradientDescent* = nullptr);

    void set(StochasticGradientDescent* = nullptr);

    StochasticGradientDescent* stochastic_gradient_descent = nullptr;

    Index iteration = 0;

    Tensor<type, 1> parameters_increment;
    Tensor<type, 1> last_parameters_increment;
};

#ifdef OPENNN_CUDA

struct SGDOptimizationDataCuda : public OptimizationAlgorithmData
{
    explicit SGDOptimizationDataCuda(OptimizationAlgorithm* new_optimization_algorithm);
    virtual ~SGDOptimizationDataCuda();

    void free();

    OptimizationAlgorithm* optimization_algorithm = nullptr;
    Index iteration = 0;
};

#endif

}

#endif

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H



#include <string>
#include <iostream>



#include "loss_index.h"

#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"
#include "config.h"

namespace opennn
{

struct GradientDescentData;

class GradientDescent : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit GradientDescent(); 

   explicit GradientDescent(LossIndex*);

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm();

   // Stopping criteria   

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;

   const Index& get_maximum_selection_failures() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Set

   void set_loss_index(LossIndex*) final;

   void set_default() final;

   // Stopping criteria

   void set_maximum_epochs_number(const Index&);

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);

   void set_maximum_selection_failures(const Index&);

   void set_maximum_time(const type&);

   // Training

   void calculate_training_direction(const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagation&,
           GradientDescentData&) const;

   TrainingResults perform_training() final;

   string write_optimization_algorithm_type() const final;

   // Serialization

   Tensor<string, 2> to_string_matrix() const final;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;

private:

   // TRAINING OPERATORS

   LearningRateAlgorithm learning_rate_algorithm;

   const type first_learning_rate = type(0.01);

   // Stopping criteria 

   type minimum_loss_decrease;

   type training_loss_goal;

   Index maximum_selection_failures;

   Index maximum_epochs_number;

   type maximum_time;

};


struct GradientDescentData : public OptimizationAlgorithmData
{

    explicit GradientDescentData()
    {
    }


    explicit GradientDescentData(GradientDescent* new_gradient_descent)
    {
        set(new_gradient_descent);
    }


    virtual ~GradientDescentData() {}

    void set(GradientDescent* new_gradient_descent);


    virtual void print() const
    {
        cout << "Training direction:" << endl;
        cout << training_direction << endl;

        cout << "Learning rate:" << endl;
        cout << learning_rate << endl;
    }

    GradientDescent* gradient_descent = nullptr;

    // Optimization algorithm data

    Index epoch = 0;

    Tensor<type, 0> training_slope;

    type learning_rate = type(0);
    type old_learning_rate = type(0);
};

}

#endif

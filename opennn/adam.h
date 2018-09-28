/********************************************/
/*                                          */
/*   OpenNN: Open Neural Networks Library   */
/*   www.opennn.net                         */
/*                                          */
/*   ADAM   C L A S S   H E A D E R         */
/*                                          */
/*   Russell Standish                       */
/*   High Performance Coders                */
/*   hpcoder@hpcoders.com.au                */
/*                                          */
/********************************************/

#ifndef __ADAM_H__
#define __ADAM_H__

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

#include "loss_index.h"

#include "training_algorithm.h"
#include "training_rate_algorithm.h"


namespace OpenNN
{

  /// This concrete class represents the gradient descent training algorithm for
  /// a loss functional of a neural network.

  class Adam : public TrainingAlgorithm
  {

  public:

    Adam() {}

    /// ownership not passed
    explicit Adam(LossIndex* li): TrainingAlgorithm(li) {}
    explicit Adam(const tinyxml2::XMLDocument&); 
  
    ///
    /// This structure contains the training results for the gradient descent. 
    ///

    struct AdamResults : public TrainingAlgorithm::TrainingAlgorithmResults
    {
      AdamResults(Adam& a): adam(a) {}
      Adam& adam;

      // Training history

      /// History of the neural network parameters over the training iterations.
      Vector< Vector<double> > parameters_history;

      /// History of the parameters norm over the training iterations.
      Vector<double> parameters_norm_history;

      /// History of the loss function loss over the training iterations.
      Vector<double> loss_history;

      /// History of the selection loss over the training iterations.
      Vector<double> selection_loss_history;

      /// History of the loss function gradient over the training iterations.
      Vector< Vector<double> > gradient_history;

      /// History of the gradient norm over the training iterations.
      Vector<double> gradient_norm_history;

      /// History of the random search training direction over the training iterations.
      Vector< Vector<double> >  training_direction_history;

      /// History of the elapsed time over the training iterations.
      Vector<double> elapsed_time_history;

      // Final values

      /// Final neural network parameters vector. 
      Vector<double> final_parameters;

      /// Final neural network parameters norm. 
      double final_parameters_norm;

      /// Final loss function evaluation.
      double final_loss;

      /// Final selection loss.
      double final_selection_loss;

      /// Final loss function gradient. 
      Vector<double> final_gradient;

      /// Final gradient norm. 
      double final_gradient_norm;

      /// Final gradient descent training direction. 
      Vector<double> final_training_direction;

      /// Final gradient descent training rate. 
      double final_training_rate;

      /// Elapsed time of the training process. 
      double elapsed_time;

      /// Maximum number of training iterations.
      size_t iterations_number;

      /// Stopping criterion
      std::string stopping_criterion;

      /// Resizes the training history variables which are to be reserved by the training algorithm.
      /// @param new_size Size of training history variables. 
      void resize_training_history(const size_t&);

      std::string to_string(void) const override;

      //Matrix<std::string> write_final_results(const size_t& precision = 3) const;
    };

    /// Adam parameters - see https://sefiks.com/2018/06/23/the-insiders-guide-to-adam-optimization-algorithm-for-deep-learning/
    double learningRate=1e-3;
    double beta1=0.9, beta2=0.999;
    // TRAINING HISTORY
    /// True if the parameters history matrix is to be reserved, false otherwise.
    bool reserve_parameters_history=false;
    /// True if the parameters norm history vector is to be reserved, false otherwise.
    bool reserve_parameters_norm_history=false;
    /// True if the loss history vector is to be reserved, false otherwise.
    bool reserve_loss_history=true;
    /// True if the gradient history matrix is to be reserved, false otherwise.
    bool reserve_gradient_history=false;
    /// True if the gradient norm history vector is to be reserved, false otherwise.
    bool reserve_gradient_norm_history=false;
    /// True if the training direction history matrix is to be reserved, false otherwise.
    bool reserve_training_direction_history=false;
    /// True if the training rate history vector is to be reserved, false otherwise.
    bool reserve_training_rate_history=false;
    /// True if the elapsed time history vector is to be reserved, false otherwise.
    bool reserve_elapsed_time_history=false;
    /// True if the selection loss history vector is to be reserved, false otherwise.
    bool reserve_selection_loss_history=false;

    // STOPPING CRITERIA

    /// Maximum number of iterations to perform_training. It is used as a stopping criterion.
    size_t maximum_iterations_number=1000;
    /// Norm of the parameters increment vector at which training stops.
    double minimum_parameters_increment_norm = 0.0;
    /// Minimum loss improvement between two successive iterations. It is used as a stopping criterion.
    double minimum_loss_increase = 0.0;
    /// Goal value for the loss. It is used as a stopping criterion.
    double loss_goal = -1.0e99;
    /// Goal value for the norm of the objective function gradient. It is used as a stopping criterion.
    double gradient_norm_goal = 0.0;
    /// Maximum number of iterations at which the selection loss increases.
    /// This is an early stopping method for improving selection.
    size_t maximum_selection_loss_decreases = 1000000;
    /// Maximum training time. It is used as a stopping criterion.
    double maximum_time=1000;
    /// True if the final model will be the neural network with the minimum selection error, false otherwise.
    bool return_minimum_selection_error_neural_network= false;

    AdamResults* perform_training() override;
  
  private:
    Vector<double> vdw, sdw;

    //
    //   // Utilities
    //
    //   void set_display_period(const size_t&);
    //
    //   // Training methods
    //
    //   Vector<double> calculate_training_direction(const Vector<double>&) const;
    //
    //  /// ownership passed - use delete to destroy
    //   GradientDescentResults* perform_training(void);
    //
    //   std::string write_training_algorithm_type(void) const;
    //
    //   // Serialization methods
    //
    //   Matrix<std::string> to_string_matrix(void) const;
    //
    //  /// ownership passed - use delete to destroy
    //   tinyxml2::XMLDocument* to_XML(void) const;
    //   void from_XML(const tinyxml2::XMLDocument&);
    //
    //   void write_XML(tinyxml2::XMLPrinter&) const;
    //   // void read_XML(   );
    //
  };

}

#endif

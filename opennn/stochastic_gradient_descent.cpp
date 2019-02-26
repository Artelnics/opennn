/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S                                          */
/*                                                                                                              */
/*   Carlos Barranquero                                                                                         */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   carlosbarranquero@artelnics.com                                                                            */
/*                                                                                                              */
/****************************************************************************************************************/

// Open NN includes

#include "stochastic_gradient_descent.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>

void initCUDA();

int mallocCUDA(double** A_d, int nBytes);
int memcpyCUDA(double* A_d, const double* A_h, int nBytes);
int getHostVector(const double* A_d, double* A_h, int nBytes);
void freeCUDA(double* A_d);

void randomizeVector(double* A_d, int n);

void calculateOutputsCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
                          double* output_data_h, const size_t output_rows, const size_t output_columns,
                          const std::vector<std::string> layers_activations);

void calculateFirstOrderForwardPropagationCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                               const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                               const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                               std::vector<double*> layers_activations_data, std::vector<double*> layers_activation_derivatives_data,
                                               const std::vector<size_t> activations_rows_numbers, const std::vector<size_t> activations_columns_numbers,
                                               const std::vector<std::string> layers_activations);

void calculateFirstOrderLossCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                double* input_data_h, const size_t input_rows, const size_t input_columns,
                                double* target_data_h, const size_t target_rows, const size_t target_columns,
                                std::vector<double*> error_gradient_data,
                                const std::vector<std::string> layers_activations, const std::string loss_method,
                                const std::vector<double> loss_parameters = vector<double>());

void updateParametersCUDA(std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* gradient_h, const size_t parameters_number);

#endif

namespace OpenNN
{

/// Default constructor.
/// It creates a stochastic gradient descent optimization algorithm not associated to any loss index object.
/// It also initializes the class members to their default values.

StochasticGradientDescent::StochasticGradientDescent()
:OptimizationAlgorithm()
{
   set_default();
}


/// Loss index constructor.
/// It creates a stochastic gradient descent optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

StochasticGradientDescent::StochasticGradientDescent(LossIndex* new_loss_index_pointer)
: OptimizationAlgorithm(new_loss_index_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also loads the class members from a XML document.
/// @param document TinyXML document with the members of a gradient descent object.

StochasticGradientDescent::StochasticGradientDescent(const tinyxml2::XMLDocument& document)
: OptimizationAlgorithm(document)
{
   set_default();

   from_XML(document);
}


// DESTRUCTOR

/// Destructor.

StochasticGradientDescent::~StochasticGradientDescent()
{
}

// METHODS

/// Returns the initial learning rate.
///
const double& StochasticGradientDescent::get_initial_learning_rate() const
{
   return(initial_learning_rate);
}


/// Returns the initial decay.

const double& StochasticGradientDescent::get_initial_decay() const
{
   return(initial_decay);
}


/// Returns the momentum.

const double& StochasticGradientDescent::get_momentum() const
{
   return(momentum);
}


///Returns true if nesterov is active, and false otherwise.

const bool& StochasticGradientDescent::get_nesterov() const
{
   return(nesterov);
}


/// Returns the minimum value for the norm of the parameters vector at wich a warning message is
/// written to the screen.

const double& StochasticGradientDescent::get_warning_parameters_norm() const
{
   return(warning_parameters_norm);
}


/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written
/// to the screen.

const double& StochasticGradientDescent::get_warning_gradient_norm() const
{
   return(warning_gradient_norm);
}


/// Returns the value for the norm of the parameters vector at wich an error message is
/// written to the screen and the program exits.

const double& StochasticGradientDescent::get_error_parameters_norm() const
{
   return(error_parameters_norm);
}


/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits.

const double& StochasticGradientDescent::get_error_gradient_norm() const
{
   return(error_gradient_norm);
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training.

const double& StochasticGradientDescent::get_minimum_parameters_increment_norm() const
{
   return(minimum_parameters_increment_norm);
}


/// Returns the minimum loss improvement during training.

const double& StochasticGradientDescent::get_minimum_loss_increase() const
{
   return(minimum_loss_decrease);
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a multilayer perceptron

const double& StochasticGradientDescent::get_loss_goal() const
{
   return(loss_goal);
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a multilayer perceptron

const double& StochasticGradientDescent::get_gradient_norm_goal() const
{
   return(gradient_norm_goal);
}


/// Returns the maximum number of selection failures during the training process.

const size_t& StochasticGradientDescent::get_maximum_selection_failures() const
{
   return(maximum_selection_failures);
}


/// Returns the maximum training time.

const double& StochasticGradientDescent::get_maximum_time() const
{
   return(maximum_time);
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& StochasticGradientDescent::get_return_minimum_selection_error_neural_network() const
{
    return(return_minimum_selection_error_neural_network);
}


/// Returns true if the selection error decrease stopping criteria has to be taken in account, false otherwise.

const bool& StochasticGradientDescent::get_apply_early_stopping() const
{
    return(apply_early_stopping);
}


/// Returns true if the parameters history matrix is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_parameters_history() const
{
   return(reserve_parameters_history);
}


/// Returns true if the parameters norm history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_parameters_norm_history() const
{
   return(reserve_parameters_norm_history);
}


/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_error_history() const
{
   return(reserve_error_history);
}


/// Returns true if the gradient history vector of vectors is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_gradient_history() const
{
   return(reserve_gradient_history);
}


/// Returns true if the gradient norm history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_gradient_norm_history() const
{
   return(reserve_gradient_norm_history);
}

/// Returns true if the learning rate history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_learning_rate_history() const
{
   return(reserve_learning_rate_history);
}


/// Returns true if the elapsed time history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_elapsed_time_history() const
{
   return(reserve_elapsed_time_history);
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_selection_error_history() const
{
   return(reserve_selection_error_history);
}


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void StochasticGradientDescent::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;
}


void StochasticGradientDescent::set_default()
{
   // TRAINING OPERATORS

   initial_learning_rate = 0.01;
   initial_decay = 0.0e-6;
   momentum = 0.0;
   nesterov = false;

   // TRAINING PARAMETERS

   warning_parameters_norm = 1.0e6;
   warning_gradient_norm = 1.0e6;

   error_parameters_norm = 1.0e9;
   error_gradient_norm = 1.0e9;

   // STOPPING CRITERIA

   minimum_parameters_increment_norm = 0.0;
   minimum_loss_decrease = 0.0;
   loss_goal = -numeric_limits<double>::max();
   gradient_norm_goal = 0.0;
   maximum_selection_failures = 1000000;
   maximum_time = 1000.0;
   maximum_epochs_number =1000;
   return_minimum_selection_error_neural_network = false;
   apply_early_stopping = true;

   // TRAINING HISTORY

   reserve_parameters_history = false;
   reserve_parameters_norm_history = false;
   reserve_error_history = true;
   reserve_gradient_history = false;
   reserve_gradient_norm_history = false;
   reserve_selection_error_history = false;
   reserve_learning_rate_history = false;
   reserve_elapsed_time_history = false;

   // UTILITIES

   display = true;
   display_period = 5;
}


/// Set the initial value for the learning rate. If dacay is not active learning rate will be constant
/// otherwise learning rate will decay over each update.
/// @param new_initial_learning_rate initial learning rate value.

void StochasticGradientDescent::set_initial_learning_rate(const double& new_learning_rate)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_learning_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_initial_learning_rate(const double&) method.\n"
             << "initial_learning_rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set learning rate

   initial_learning_rate = new_learning_rate;
}


/// Set the initial value for the decay.
/// @param new_initial_learning_rate initial value for the decay.

void StochasticGradientDescent::set_initial_decay(const double& new_dacay)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_dacay < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_initial_decay(const double&) method.\n"
             << "new_dacay must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set  initial decay

   initial_decay = new_dacay;
}


/// Set a new value for momentum, this parameter accelerates SGD in the relevant direction
/// and dampens oscillations.
/// @param new_momentum initial value for the mometum.

void StochasticGradientDescent::set_momentum(const double& new_momentum)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_momentum < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_momentum(const double&) method.\n"
             << "new_momentum must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set momentum

   momentum = new_momentum;
}


/// Set nesterov, boolean. Whether to apply Nesterov momentum.
/// @param new_momentum initial value for the mometum.

void StochasticGradientDescent::set_nesterov(const bool& new_nesterov_momentum)
{
   nesterov = new_nesterov_momentum;
}


/// Makes the training history of all variables to reseved or not in memory:
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> Loss.
/// <li> Gradient.
/// <li> Gradient norm.
/// <li> Selection loss.
/// <li> Learning rate.
/// <li> Elapsed_time.
/// </ul>
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
   // Multilayer perceptron

   reserve_parameters_history = new_reserve_all_training_history;
   reserve_parameters_norm_history = new_reserve_all_training_history;

   // Loss index

   reserve_error_history = new_reserve_all_training_history;
   reserve_gradient_history = new_reserve_all_training_history;
   reserve_gradient_norm_history = new_reserve_all_training_history;
   reserve_selection_error_history = new_reserve_all_training_history;

   // Optimization algorithm

   reserve_learning_rate_history = new_reserve_all_training_history;
   reserve_elapsed_time_history = new_reserve_all_training_history;
}


/// Sets a new value for the parameters vector norm at which a warning message is written to the
/// screen.
/// @param new_warning_parameters_norm Warning norm of parameters vector value.

void StochasticGradientDescent::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_warning_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_warning_parameters_norm(const double&) method.\n"
             << "Warning parameters norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set warning parameters norm

   warning_parameters_norm = new_warning_parameters_norm;
}


/// Sets a new value for the gradient vector norm at which
/// a warning message is written to the screen.
/// @param new_warning_gradient_norm Warning norm of gradient vector value.

void StochasticGradientDescent::set_warning_gradient_norm(const double& new_warning_gradient_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_warning_gradient_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_warning_gradient_norm(const double&) method.\n"
             << "Warning gradient norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set warning gradient norm

   warning_gradient_norm = new_warning_gradient_norm;
}


/// Sets a new value for the parameters vector norm at which an error message is written to the
/// screen and the program exits.
/// @param new_error_parameters_norm Error norm of parameters vector value.

void StochasticGradientDescent::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_error_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_error_parameters_norm(const double&) method.\n"
             << "Error parameters norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set error parameters norm

   error_parameters_norm = new_error_parameters_norm;
}


/// Sets a new value for the gradient vector norm at which an error message is written to the screen
/// and the program exits.
/// @param new_error_gradient_norm Error norm of gradient vector value.

void StochasticGradientDescent::set_error_gradient_norm(const double& new_error_gradient_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_error_gradient_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_error_gradient_norm(const double&) method.\n"
             << "Error gradient norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set error gradient norm

   error_gradient_norm = new_error_gradient_norm;
}

/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void StochasticGradientDescent:: set_maximum_epochs_number(const size_t& new_maximum_epochs_number)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_maximum_epochs_number < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_maximum_epochs_number(const double&) method.\n"
             << "Maximum epochs number must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set maximum_epochs number

   maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new value for the minimum parameters increment norm stopping criterion.
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training.

void StochasticGradientDescent::set_minimum_parameters_increment_norm(const double& new_minimum_parameters_increment_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_minimum_parameters_increment_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void new_minimum_parameters_increment_norm(const double&) method.\n"
             << "Minimum parameters increment norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set error training rate

   minimum_parameters_increment_norm = new_minimum_parameters_increment_norm;
}


/// Sets a new minimum loss improvement during training.
/// @param new_minimum_loss_increase Minimum improvement in the loss between two iterations.

void StochasticGradientDescent::set_minimum_loss_increase(const double& new_minimum_loss_increase)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_minimum_loss_increase < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_minimum_loss_increase(const double&) method.\n"
             << "Minimum loss improvement must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set minimum loss improvement

   minimum_loss_decrease = new_minimum_loss_increase;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a multilayer perceptron
/// @param new_loss_goal Goal value for the loss.

void StochasticGradientDescent::set_loss_goal(const double& new_loss_goal)
{
   loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a multilayer perceptron
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void StochasticGradientDescent::set_gradient_norm_goal(const double& new_gradient_norm_goal)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_gradient_norm_goal < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_gradient_norm_goal(const double&) method.\n"
             << "Gradient norm goal must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set gradient norm goal

   gradient_norm_goal = new_gradient_norm_goal;
}


/// Sets a new maximum number of selection failures.
/// @param new_maximum_selection_failures Maximum number of iterations in which the selection evalutation decreases.

void StochasticGradientDescent::set_maximum_selection_error_increases(const size_t& new_maximum_selection_failures)
{
   maximum_selection_failures = new_maximum_selection_failures;
}


/// Sets a maximum number of iterations for training.
/// @param new_maximum_iterations_number Maximum number of iterations for training.

//void StochasticGradientDescent::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
//{
//   maximum_iterations_number = new_maximum_iterations_number;
//}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void StochasticGradientDescent::set_maximum_time(const double& new_maximum_time)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_maximum_time < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_maximum_time(const double&) method.\n"
             << "Maximum time must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set maximum time

   maximum_time = new_maximum_time;
}


/// Makes the minimum selection error neural network of all the iterations to be returned or not.
/// @param new_return_minimum_selection_error_neural_network True if the final model will be the neural network with the minimum selection error, false otherwise.

void StochasticGradientDescent::set_return_minimum_selection_error_neural_network(const bool& new_return_minimum_selection_error_neural_network)
{
   return_minimum_selection_error_neural_network = new_return_minimum_selection_error_neural_network;
}


/// Makes the selection error decrease stopping criteria has to be taken in account or not.
/// @param new_apply_early_stopping True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

void StochasticGradientDescent::set_apply_early_stopping(const bool& new_apply_early_stopping)
{
    apply_early_stopping = new_apply_early_stopping;
}


/// Makes the parameters history vector of vectors to be reseved or not in memory.
/// @param new_reserve_parameters_history True if the parameters history vector of vectors is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_parameters_history(const bool& new_reserve_parameters_history)
{
   reserve_parameters_history = new_reserve_parameters_history;
}


/// Makes the parameters norm history vector to be reseved or not in memory.
/// @param new_reserve_parameters_norm_history True if the parameters norm history vector is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_parameters_norm_history(const bool& new_reserve_parameters_norm_history)
{
   reserve_parameters_norm_history = new_reserve_parameters_norm_history;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_error_history True if the error history vector is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_error_history(const bool& new_reserve_error_history)
{
   reserve_error_history = new_reserve_error_history;
}


/// Makes the gradient history vector of vectors to be reseved or not in memory.
/// @param new_reserve_gradient_history True if the gradient history matrix is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_gradient_history(const bool& new_reserve_gradient_history)
{
   reserve_gradient_history = new_reserve_gradient_history;
}


/// Makes the gradient norm history vector to be reseved or not in memory.
/// @param new_reserve_gradient_norm_history True if the gradient norm history matrix is to be reserved, false
/// otherwise.

void StochasticGradientDescent::set_reserve_gradient_norm_history(const bool& new_reserve_gradient_norm_history)
{
   reserve_gradient_norm_history = new_reserve_gradient_norm_history;
}


/// Makes the training rate history vector to be reseved or not in memory.
/// @param new_reserve_learning_rate_history True if the training rate history vector is to be reserved, false
/// otherwise.

void StochasticGradientDescent::set_reserve_learning_rate_history(const bool& new_reserve_learning_rate_history)
{
   reserve_learning_rate_history = new_reserve_learning_rate_history;
}


/// Makes the elapsed time over the iterations to be reseved or not in memory. This is a vector.
/// @param new_reserve_elapsed_time_history True if the elapsed time history vector is to be reserved, false
/// otherwise.

void StochasticGradientDescent::set_reserve_elapsed_time_history(const bool& new_reserve_elapsed_time_history)
{
   reserve_elapsed_time_history = new_reserve_elapsed_time_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector.
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
   reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Sets a new number of iterations between the training showing progress.
/// @param new_display_period
/// Number of iterations between the training showing progress.

void StochasticGradientDescent::set_display_period(const size_t& new_display_period)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_display_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
             << "void set_display_period(const double&) method.\n"
             << "First training rate must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   display_period = new_display_period;
}



string StochasticGradientDescent::StochasticGradientDescentResults::object_to_string() const
{
   ostringstream buffer;

   // Parameters history

   if(!parameters_history.empty())
   {
      if(!parameters_history[0].empty())
      {
          buffer << "% Parameters history:\n"
                 << parameters_history << "\n";
      }
   }

   // Parameters norm history

   if(!parameters_norm_history.empty())
   {
       buffer << "% Parameters norm history:\n"
              << parameters_norm_history << "\n";
   }

   // Loss history

   if(!loss_history.empty())
   {
       buffer << "% Loss history:\n"
              << loss_history << "\n";
   }

   // Selection loss history

   if(!selection_error_history.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_error_history << "\n";
   }

   // Gradient history

   if(!gradient_history.empty())
   {
      if(!gradient_history[0].empty())
      {
          buffer << "% Gradient history:\n"
                 << gradient_history << "\n";
      }
   }

   // Gradient norm history

   if(!gradient_norm_history.empty())
   {
       buffer << "% Gradient norm history:\n"
              << gradient_norm_history << "\n";
   }

   // Training rate history

   if(!learning_rate_history.empty())
   {
       buffer << "% Training rate history:\n"
              << learning_rate_history << "\n";
   }

   // Elapsed time history

   if(!elapsed_time_history.empty())
   {
       buffer << "% Elapsed time history:\n"
              << elapsed_time_history << "\n";
   }

   // Stopping criterion

   if(!stopping_criterion.empty())
   {
       buffer << "% Stopping criterion:\n"
              << stopping_criterion << "\n";
   }

   return(buffer.str());
}


Matrix<string> StochasticGradientDescent::StochasticGradientDescentResults::write_final_results(const int& precision) const
{
   ostringstream buffer;

   Vector<string> names;
   Vector<string> values;

   // Final parameters norm

   names.push_back("Final parameters norm");

   buffer.str("");
   buffer << setprecision(precision) << final_parameters_norm;

   values.push_back(buffer.str());

   // Final loss

   names.push_back("Final training error");

   buffer.str("");
   buffer << setprecision(precision) << final_loss;

   values.push_back(buffer.str());

   names.push_back("Final selection error");

   buffer.str("");
   buffer << setprecision(precision) << final_selection_error;

   values.push_back(buffer.str());

   // Final gradient norm

   names.push_back("Final gradient norm");

   buffer.str("");
   buffer << setprecision(precision) << final_gradient_norm;

   values.push_back(buffer.str());

   // Iterations number

   names.push_back("Iterations number");

   buffer.str("");
   buffer << iterations_number;

   values.push_back(buffer.str());

   // Elapsed time

   names.push_back("Elapsed time");

   buffer.str("");
   buffer << write_elapsed_time(elapsed_time);

   values.push_back(buffer.str());

   // Stopping criteria

   names.push_back("Stopping criterion");

   values.push_back(write_stopping_condition());

   const size_t rows_number = names.size();
   const size_t columns_number = 2;

   Matrix<string> final_results(rows_number, columns_number);

   final_results.set_column(0, names, "name");
   final_results.set_column(1, values, "value");

   return(final_results);
}


/// Resizes the training history variables which are to be reserved by the optimization algorithm.
/// @param new_size Size of training history variables.

void StochasticGradientDescent::StochasticGradientDescentResults::resize_training_history(const size_t& new_size)
{
    // Control sentence(if debug)

    if(stochastic_gradient_descent_pointer->get_reserve_parameters_history())
    {
        parameters_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_parameters_norm_history())
    {
        parameters_norm_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_error_history())
    {
        loss_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_selection_error_history())
    {
        selection_error_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_gradient_history())
    {
        gradient_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_gradient_norm_history())
    {
        gradient_norm_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_learning_rate_history())
    {
        learning_rate_history.resize(new_size);
    }

    if(stochastic_gradient_descent_pointer->get_reserve_elapsed_time_history())
    {
        elapsed_time_history.resize(new_size);
    }
}


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

StochasticGradientDescent::StochasticGradientDescentResults* StochasticGradientDescent::perform_training()
{
    StochasticGradientDescentResults* results_pointer = new StochasticGradientDescentResults(this);

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Start training

   if(display) cout << "Training with stochastic gradient descent...\n";

   // Data set stuff

   DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   const Instances& instances = data_set_pointer->get_instances();

   const size_t selection_instances_number = instances.get_selection_instances_number();

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> parameters(parameters_number);
   Vector<double> parameters_increment(parameters_number);
   Vector<double> last_increment(parameters_number,0.0);

   double parameters_norm = 0.0;

   // Loss index stuff

   LossIndex::FirstOrderLoss first_order_loss(parameters_number);

   double training_error = 0.0;
   double old_training_error = 0.0;

   double selection_error = 0.0;
   double old_selection_error = 0.0;

   double loss = 0.0;

   double gradient_norm = 0.0;

   // Optimization algorithm stuff

   double learning_rate = initial_learning_rate;

   size_t selection_failures = 0;

   Vector<double> nesterov_increment(parameters_number);

   Vector<double> minimum_selection_error_parameters(parameters_number);
   double minimum_selection_error = numeric_limits<double>::max();

   bool stop_training = false;

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time = 0.0;

   results_pointer->resize_training_history(maximum_epochs_number + 1);

   size_t current_iteration = 0;
   size_t learning_rate_iteration = 1;

   // Main loop

   for(size_t epoch = 0; epoch <= epochs_number; epoch++)
   {
       const Vector< Vector<size_t> > training_batches = instances.get_training_batches(training_batch_size);

       const size_t batches_number = training_batches.size();

       parameters = neural_network_pointer->get_parameters();

       parameters_norm = parameters.calculate_L2_norm();

       if(display && parameters_norm >= warning_parameters_norm) cout << "OpenNN Warning: Parameters norm is " << parameters_norm << ".\n";

       loss = 0.0;

       for(size_t iteration = 0; iteration < batches_number; iteration++)
       {
           //Loss

           first_order_loss = loss_index_pointer->calculate_batch_first_order_loss(training_batches[iteration]);

           loss += first_order_loss.loss;

           // Gradient

            gradient_norm = first_order_loss.gradient.calculate_L2_norm();

            if(display && gradient_norm >= warning_gradient_norm) cout << "OpenNN Warning: Gradient norm is " << gradient_norm << ".\n";

            initial_decay > 0.0 ? learning_rate =  initial_learning_rate * (1.0 / (1.0 + learning_rate_iteration*initial_decay)) : initial_learning_rate ;

            parameters = neural_network_pointer->get_parameters();

            parameters_increment = first_order_loss.gradient*(-learning_rate);

            if(momentum > 0.0 && !nesterov) {

            parameters_increment += last_increment*momentum;

            last_increment = parameters_increment;

            neural_network_pointer->set_parameters(parameters + parameters_increment);

            }

            else if(momentum > 0.0 && nesterov ){

            parameters_increment += last_increment*momentum;

            last_increment = parameters_increment;

            nesterov_increment = parameters_increment*momentum - first_order_loss.gradient*(learning_rate) ;

            neural_network_pointer->set_parameters(parameters + nesterov_increment);

            }
            else{

            neural_network_pointer->set_parameters(parameters + parameters_increment);

            }

            learning_rate_iteration++;
       }

       // Loss

       training_error = loss/static_cast<double>(batches_number);

       if(selection_instances_number > 0) selection_error = loss_index_pointer->calculate_selection_error();

       if(epoch == 0)
       {
          minimum_selection_error = selection_error;
          minimum_selection_error_parameters = neural_network_pointer->get_parameters();
       }
       else if(epoch != 0 && selection_error > old_selection_error)
       {
          selection_failures++;
       }
       else if(selection_error <= minimum_selection_error)
       {
          minimum_selection_error = selection_error;
          minimum_selection_error_parameters = neural_network_pointer->get_parameters();
       }

       // Elapsed time

       time(&current_time);
       elapsed_time = difftime(current_time, beginning_time);

       // Training history neural network

       if(reserve_parameters_history) results_pointer->parameters_history[epoch] = parameters;

       if(reserve_parameters_norm_history) results_pointer->parameters_norm_history[epoch] = parameters_norm;

       // Training history loss index

       if(reserve_gradient_norm_history) results_pointer->gradient_norm_history[epoch] = gradient_norm;

       if(reserve_error_history) results_pointer->loss_history[epoch] = training_error;

       if(reserve_selection_error_history) results_pointer->selection_error_history[epoch] = selection_error;

       // Training history optimization algorithm

       if(reserve_learning_rate_history) results_pointer->learning_rate_history[epoch] = learning_rate;

       if(reserve_elapsed_time_history) results_pointer->elapsed_time_history[epoch] = elapsed_time;

       // Stopping Criteria

        if(selection_failures >= maximum_selection_failures && apply_early_stopping)
        {
           if(display)
           {
              cout << "Epoch " << epoch << ", iteration " << epoch << ": Maximum selection failures reached.\n"
                   << "Selection failures: " << selection_failures << endl;
           }

           stop_training = true;

           results_pointer->stopping_condition = MaximumSelectionErrorIncreases;
        }

        else if(epoch == maximum_epochs_number)
        {
           if(display)
           {
              cout << "Epoch " << epoch << ": Maximum number of iterations reached.\n";
           }

           stop_training = true;

           results_pointer->stopping_condition = MaximumIterationsNumber;
        }

        else if(elapsed_time >= maximum_time)
        {
           if(display)
           {
              cout << "Epoch " << epoch << ": Maximum training time reached.\n";
           }

           stop_training = true;

           results_pointer->stopping_condition = MaximumTime;
        }

        if(epoch != 0 && epoch % save_period == 0)
        {
              neural_network_pointer->save(neural_network_file_name);
        }

        if(stop_training)
        {
           if(display)
           {
              cout << "Parameters norm: " << parameters_norm << "\n"
                        << "Training loss: " << training_error << "\n"
                        << "Batch size: " << training_batch_size << "\n"
                        << "Gradient norm: " << gradient_norm << "\n"
                        << loss_index_pointer->write_information()
                        << "Learning rate: " << learning_rate << "\n"
                        << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n"
                        << "Selection error: " << selection_error << endl;
           }

           results_pointer->resize_training_history(1 + epoch);

           results_pointer->final_parameters = parameters;

           results_pointer->final_parameters_norm = parameters_norm;

           results_pointer->final_loss = training_error;

           results_pointer->final_selection_error = selection_error;

           results_pointer->final_gradient_norm = gradient_norm;

           results_pointer->elapsed_time = elapsed_time;

           results_pointer->iterations_number = epoch;

           break;
        }
        else if(display && epoch % display_period == 0)
        {
           cout << "Epoch " << epoch << ";\n"
                << "Parameters norm: " << parameters_norm << "\n"
                << "Training loss: " << training_error << "\n"
                << "Batch size: " << training_batch_size << "\n"
                << "Gradient norm: " << gradient_norm << "\n"
                << loss_index_pointer->write_information()
                << "Learning rate: " << learning_rate<< "\n"
                << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n"
                << "Selection error: " << selection_error << endl;
        }

          // Update stuff

          old_training_error = training_error;
          old_selection_error = selection_error;

          current_iteration++;

       if(stop_training) break;
   }

   if(return_minimum_selection_error_neural_network)
   {
       parameters = minimum_selection_error_parameters;
       parameters_norm = parameters.calculate_L2_norm();

       neural_network_pointer->set_parameters(parameters);

       selection_error = minimum_selection_error;
   }

   results_pointer->final_parameters = parameters;
   results_pointer->final_parameters_norm = parameters_norm;

   results_pointer->final_loss = training_error;
   results_pointer->final_selection_error = selection_error;

   results_pointer->final_gradient_norm = gradient_norm;

   results_pointer->elapsed_time = elapsed_time;

   return(results_pointer);
}


StochasticGradientDescent::StochasticGradientDescentResults* StochasticGradientDescent::perform_training_cuda()
{
    // Check CUDA

    StochasticGradientDescentResults* results_pointer = new StochasticGradientDescentResults(this);

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif
#ifdef __OPENNN_CUDA__
   // Start training

   if(display) cout << "Training with stochastic gradient descent with CUDA...\n";

   // Data set stuff

   DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   const Instances& instances = data_set_pointer->get_instances();

   const size_t selection_instances_number = instances.get_selection_instances_number();

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> parameters = neural_network_pointer->get_parameters();
   Vector<double> parameters_increment(parameters_number, 0.0);
   Vector<double> last_increment(parameters_number,0.0);

   double parameters_norm = 0.0;

   MultilayerPerceptron::Pointers multilayer_perceptron_pointers_device = multilayer_perceptron_pointer->host_to_device();

   // Loss index stuff

   LossIndex::FirstOrderLoss first_order_loss(parameters_number);

   double training_error = 0.0;
   double old_training_error = 0.0;

   double selection_error = 0.0;
   double old_selection_error = 0.0;

   double loss = 0.0;

   double gradient_norm = 0.0;

   // CUDA stuff

   omp_set_num_threads(2);
   const size_t num_threads = static_cast<size_t>(omp_get_max_threads());

   Vector< Vector<double*> > last_data_device(num_threads);

   for(size_t i = 0; i < num_threads; i++)
   {
       last_data_device[i].set(2);
   }

   // Optimization algorithm stuff

   double learning_rate = initial_learning_rate;

   size_t selection_failures = 0;

   Vector<double> nesterov_increment(parameters_number);

   Vector<double> minimum_selection_error_parameters(parameters_number);
   double minimum_selection_error = numeric_limits<double>::max();

   bool stop_training = false;

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time = 0.0;

   results_pointer->resize_training_history(maximum_epochs_number + 1);

   size_t current_iteration = 0;
   size_t learning_rate_iteration = 1;

   // Main loop

   const Vector< Vector<size_t> > training_batches = instances.get_training_batches(training_batch_size);

   const size_t batches_number = training_batches.size();

   for(size_t epoch = 0; epoch <= epochs_number; epoch++)
   {
       loss = 0.0;

       last_data_device[0] = data_set_pointer->host_to_device(training_batches[0]);

       for(size_t iteration = 0; iteration < batches_number; iteration++)
       {
           current_iteration++;

#pragma omp parallel
           {
               const size_t thread_num = static_cast<size_t>(omp_get_thread_num());

               if(num_threads >= 2)
               {
                   if(thread_num == 0)
                   {
                       first_order_loss = loss_index_pointer->calculate_batch_first_order_loss_cuda(training_batches[iteration],
                                                                                                    multilayer_perceptron_pointers_device, last_data_device[0]);

                       freeCUDA(last_data_device[0][0]);
                       freeCUDA(last_data_device[0][1]);
                   }
                   else if(iteration == 0 && iteration+thread_num < batches_number)
                   {
                       last_data_device[thread_num] = data_set_pointer->host_to_device(training_batches[iteration+thread_num]);
                   }
                   else if(thread_num == num_threads-1 && iteration+thread_num < batches_number)
                   {
                       last_data_device[thread_num] = data_set_pointer->host_to_device(training_batches[iteration+thread_num]);
                   }
               }
               else
               {
                   first_order_loss = loss_index_pointer->calculate_batch_first_order_loss_cuda(training_batches[iteration],
                                                                                                multilayer_perceptron_pointers_device, last_data_device[0]);

                   freeCUDA(last_data_device[0][0]);
                   freeCUDA(last_data_device[0][1]);

                   if(iteration != batches_number-1) last_data_device[0] = data_set_pointer->host_to_device(training_batches[iteration+1]);
               }
           }

           for(size_t i = 0; i < num_threads-1; i++)
           {
               last_data_device[i][0] = last_data_device[i+1][0];
               last_data_device[i][1] = last_data_device[i+1][1];
           }

           //Loss

           loss += first_order_loss.loss;

           // Gradient

           learning_rate =  initial_learning_rate * (1.0 / (1.0 + learning_rate_iteration*initial_decay));

           parameters_increment =  first_order_loss.gradient*(-learning_rate);

           if(momentum > 0.0 && !nesterov)
           {
               parameters_increment += last_increment*momentum;

               last_increment = parameters_increment;

               multilayer_perceptron_pointers_device.update_parameters(parameters_increment);
               parameters = parameters +  parameters_increment;
           }
           else if(momentum > 0.0 && nesterov)
           {
               parameters_increment += last_increment*momentum;

               last_increment = parameters_increment;

               nesterov_increment = parameters_increment*momentum - first_order_loss.gradient*(learning_rate) ;

               multilayer_perceptron_pointers_device.update_parameters(nesterov_increment);
               parameters = parameters +  nesterov_increment;
           }
           else
           {
               multilayer_perceptron_pointers_device.update_parameters(parameters_increment);
               parameters = parameters +  parameters_increment;
           }

           learning_rate_iteration++;
       }

       for(size_t i = 0; i < num_threads; i++)
       {
           freeCUDA(last_data_device[i][0]);
           freeCUDA(last_data_device[i][1]);
       }

       // Loss

       training_error = loss/static_cast<double>(batches_number);

       if(selection_instances_number > 0) selection_error = loss_index_pointer->calculate_batch_error_cuda(data_set_pointer->get_instances().get_selection_indices(),
                                                                                                           multilayer_perceptron_pointers_device);
       if(epoch == 0)
       {
          minimum_selection_error = selection_error;
          minimum_selection_error_parameters = parameters;
       }
       else if(epoch != 0 && selection_error > old_selection_error)
       {
          selection_failures++;
       }
       else if(selection_error <= minimum_selection_error)
       {
          minimum_selection_error = selection_error;
          minimum_selection_error_parameters = parameters;
       }

       // Elapsed time

       time(&current_time);
       elapsed_time = difftime(current_time, beginning_time);

       // Training history neural network

       if(reserve_parameters_history) results_pointer->parameters_history[epoch] = parameters;

       if(reserve_parameters_norm_history) results_pointer->parameters_norm_history[epoch] = parameters.calculate_L2_norm();

       // Training history loss index

       if(reserve_error_history) results_pointer->loss_history[epoch] = training_error;

       if(reserve_gradient_history) results_pointer->gradient_history[epoch] = first_order_loss.gradient;

       if(reserve_gradient_norm_history) results_pointer->gradient_norm_history[epoch] = first_order_loss.gradient.calculate_L2_norm();

       if(reserve_selection_error_history) results_pointer->selection_error_history[epoch] = selection_error;

       // Training history optimization algorithm

       if(reserve_elapsed_time_history) results_pointer->elapsed_time_history[epoch] = elapsed_time;

       // Stopping Criteria

        if(selection_failures >= maximum_selection_failures && apply_early_stopping)
        {
           if(display)
           {
              cout << "Epoch " << epoch << ", iteration " << epoch << ": Maximum selection failures reached.\n"
                   << "Selection failures: " << selection_failures << endl;
           }

           stop_training = true;

           results_pointer->stopping_condition = MaximumSelectionErrorIncreases;
        }

        else if(epoch == maximum_epochs_number)
        {
           if(display)
           {
              cout << "Epoch " << epoch << ": Maximum number of iterations reached.\n";
           }

           stop_training = true;

           results_pointer->stopping_condition = MaximumIterationsNumber;
        }

        else if(elapsed_time >= maximum_time)
        {
           if(display)
           {
              cout << "Epoch " << epoch << ": Maximum training time reached.\n";
           }

           stop_training = true;

           results_pointer->stopping_condition = MaximumTime;
        }

        if(epoch != 0 && epoch % save_period == 0)
        {
              neural_network_pointer->save(neural_network_file_name);
        }

        if(stop_training)
        {
           if(display)
           {
              cout << "Training loss: " << training_error << "\n"
                        << "Batch size: " << training_batch_size << "\n"
                        << loss_index_pointer->write_information()
                        << "Learning rate: " << learning_rate << "\n"
                        << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n"
                        << "Selection error: " << selection_error << endl;
           }

           results_pointer->resize_training_history(1+epoch);

           results_pointer->final_parameters = parameters;

           results_pointer->final_parameters_norm = parameters.calculate_L2_norm();

           results_pointer->final_loss = training_error;

           results_pointer->final_selection_error = selection_error;

           results_pointer->final_gradient_norm = first_order_loss.gradient.calculate_L2_norm();

           results_pointer->elapsed_time = elapsed_time;

           results_pointer->iterations_number = epoch;

           break;
        }
        else if(display && epoch % display_period == 0)
        {
           cout << "Epoch " << epoch << ";\n"
                << "Training loss: " << training_error << "\n"
                << "Batch size: " << training_batch_size << "\n"
                << loss_index_pointer->write_information()
                << "Learning rate: " << learning_rate<< "\n"
                << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n"
                << "Selection error: " << selection_error << endl;
        }

          // Update stuff

          old_training_error = training_error;
          old_selection_error = selection_error;

          current_iteration++;

       if(stop_training) break;
   }

   if(return_minimum_selection_error_neural_network)
   {
       parameters = minimum_selection_error_parameters;
       parameters_norm = parameters.calculate_L2_norm();

       neural_network_pointer->set_parameters(parameters);

       selection_error = minimum_selection_error;
   }

   results_pointer->final_parameters = parameters;
   results_pointer->final_parameters_norm = parameters_norm;

   results_pointer->final_loss = training_error;
   results_pointer->final_selection_error = selection_error;

   results_pointer->final_gradient_norm = first_order_loss.gradient.calculate_L2_norm();

   results_pointer->elapsed_time = elapsed_time;

#endif
   return(results_pointer);
}


void StochasticGradientDescent::perform_training_void()
{
    StochasticGradientDescentResults* results = perform_training();

    delete results;
}


string StochasticGradientDescent::write_optimization_algorithm_type() const
{
   return("GRADIENT_DESCENT");
}


/// Writes as matrix of strings the most representative atributes.

Matrix<string> StochasticGradientDescent::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

   // Minimum parameters increment norm

   labels.push_back("Minimum parameters increment norm");

   buffer.str("");
   buffer << minimum_parameters_increment_norm;

   values.push_back(buffer.str());

   // Minimum loss decrease

   labels.push_back("Minimum loss decrease");

   buffer.str("");
   buffer << minimum_loss_decrease;

   values.push_back(buffer.str());

   // Loss goal

   labels.push_back(" Loss goal");

   buffer.str("");
   buffer << loss_goal;

   values.push_back(buffer.str());

   // Gradient norm goal

   labels.push_back("Gradient norm goal");

   buffer.str("");
   buffer << gradient_norm_goal;

   values.push_back(buffer.str());

   // Maximum selection error decreases

   labels.push_back("Maximum selection error increases");

   buffer.str("");
   buffer << maximum_selection_failures;

   values.push_back(buffer.str());

   // Maximum iterations number

   labels.push_back("Maximum iterations number");

   buffer.str("");
   buffer << maximum_epochs_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Reserve parameters norm history

   labels.push_back("Reserve parameters norm history");

   buffer.str("");

   if(reserve_parameters_norm_history)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Reserve error history

   labels.push_back("Reserve error history");

   buffer.str("");

   if(reserve_error_history)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Reserve selection error history

   labels.push_back("Reserve selection error history");

   buffer.str("");

   if(reserve_selection_error_history)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Reserve gradient norm history

   labels.push_back("Reserve gradient norm history");

   buffer.str("");

   if(reserve_gradient_norm_history)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return(string_matrix);
}


/// Serializes the training parameters, the stopping criteria and other user stuff
/// concerning the gradient descent object.

tinyxml2::XMLDocument* StochasticGradientDescent::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Optimization algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("StochasticGradientDescent");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Return minimum selection error neural network

   element = document->NewElement("ReturnMinimumSelectionErrorNN");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << return_minimum_selection_error_neural_network;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Apply early stopping

   element = document->NewElement("ApplyEarlyStopping");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << apply_early_stopping;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Warning parameters norm

   element = document->NewElement("WarningParametersNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << warning_parameters_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Warning gradient norm

   element = document->NewElement("WarningGradientNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << warning_gradient_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Error parameters norm

   element = document->NewElement("ErrorParametersNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << error_parameters_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Error gradient norm

   element = document->NewElement("ErrorGradientNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << error_gradient_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Minimum parameters increment norm

   element = document->NewElement("MinimumParametersIncrementNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_parameters_increment_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Minimum loss decrease

   element = document->NewElement("MinimumLossDecrease");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_loss_decrease;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Loss goal

   element = document->NewElement("LossGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << loss_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Gradient norm goal

   element = document->NewElement("GradientNormGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << gradient_norm_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum selection error decreases

   element = document->NewElement("MaximumSelectionErrorIncreases");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_failures;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum iterations number

   element = document->NewElement("MaximumIterationsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_epochs_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum time

   element = document->NewElement("MaximumTime");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_time;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve parameters norm history

   element = document->NewElement("ReserveParametersNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_parameters_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve parameters history

   element = document->NewElement("ReserveParametersHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_parameters_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve loss history

   element = document->NewElement("ReserveErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_error_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve selection error history

   element = document->NewElement("ReserveSelectionErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_error_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve gradient history

   element = document->NewElement("ReserveGradientHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_gradient_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve gradient norm history

   element = document->NewElement("ReserveGradientNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_gradient_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

    //Reserve training rate history

   element = document->NewElement("ReserveLearningRateHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_learning_rate_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

    //Reserve elapsed time history

   element = document->NewElement("ReserveElapsedTimeHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_elapsed_time_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

    //Reserve selection error history

   element = document->NewElement("ReserveSelectionErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_error_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Display period

   element = document->NewElement("DisplayPeriod");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << display_period;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Save period

   element = document->NewElement("SavePeriod");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << save_period;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Neural network file name

   element = document->NewElement("NeuralNetworkFileName");
   root_element->LinkEndChild(element);

   text = document->NewText(neural_network_file_name.c_str());
   element->LinkEndChild(text);

   // Display warnings

   element = document->NewElement("Display");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << display;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   return(document);
}


/// Serializes the gradient descent object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void StochasticGradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("StochasticGradientDescent");

    // Return minimum selection error neural network


    file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

    buffer.str("");
    buffer << return_minimum_selection_error_neural_network;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    // Apply early stopping

    file_stream.OpenElement("ApplyEarlyStopping");

    buffer.str("");
    buffer << apply_early_stopping;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum parameters increment norm

    file_stream.OpenElement("MinimumParametersIncrementNorm");

    buffer.str("");
    buffer << minimum_parameters_increment_norm;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum loss decrease

    file_stream.OpenElement("MinimumLossDecrease");

    buffer.str("");
    buffer << minimum_loss_decrease;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Loss goal

    file_stream.OpenElement("LossGoal");

    buffer.str("");
    buffer << loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Gradient norm goal

    file_stream.OpenElement("GradientNormGoal");

    buffer.str("");
    buffer << gradient_norm_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection error decreases

    file_stream.OpenElement("MaximumSelectionErrorIncreases");

    buffer.str("");
    buffer << maximum_selection_failures;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations number

    file_stream.OpenElement("MaximumIterationsNumber");

    buffer.str("");
    buffer << maximum_epochs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve parameters norm history

    file_stream.OpenElement("ReserveParametersNormHistory");

    buffer.str("");
    buffer << reserve_parameters_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve error history

    file_stream.OpenElement("ReserveErrorHistory");

    buffer.str("");
    buffer << reserve_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error history

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve gradient norm history

    file_stream.OpenElement("ReserveGradientNormHistory");

    buffer.str("");
    buffer << reserve_gradient_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();
}


void StochasticGradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("StochasticGradientDescent");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Gradient descent element is nullptr.\n";

        throw logic_error(buffer.str());
    }

   // Warning parameters norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningParametersNorm");

       if(element)
       {
          const double new_warning_parameters_norm = atof(element->GetText());

          try
          {
             set_warning_parameters_norm(new_warning_parameters_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Warning gradient norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningGradientNorm");

       if(element)
       {
          const double new_warning_gradient_norm = atof(element->GetText());

          try
          {
             set_warning_gradient_norm(new_warning_gradient_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Error parameters norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorParametersNorm");

       if(element)
       {
          const double new_error_parameters_norm = atof(element->GetText());

          try
          {
              set_error_parameters_norm(new_error_parameters_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Error gradient norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorGradientNorm");

       if(element)
       {
          const double new_error_gradient_norm = atof(element->GetText());

          try
          {
             set_error_gradient_norm(new_error_gradient_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

    // Return minimum selection error neural network

    const tinyxml2::XMLElement* return_minimum_selection_error_neural_network_element = root_element->FirstChildElement("ReturnMinimumSelectionErrorNN");

    if(return_minimum_selection_error_neural_network_element)
    {
        string new_return_minimum_selection_error_neural_network = return_minimum_selection_error_neural_network_element->GetText();

        try
        {
           set_return_minimum_selection_error_neural_network(new_return_minimum_selection_error_neural_network != "0");
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
    }

    // Apply early stopping

    const tinyxml2::XMLElement* apply_early_stopping_element = root_element->FirstChildElement("ApplyEarlyStopping");

    if(apply_early_stopping_element)
    {
        string new_apply_early_stopping = apply_early_stopping_element->GetText();

        try
        {
            set_apply_early_stopping(new_apply_early_stopping != "0");
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

   // Minimum parameters increment norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumParametersIncrementNorm");

       if(element)
       {
          const double new_minimum_parameters_increment_norm = atof(element->GetText());

          try
          {
             set_minimum_parameters_increment_norm(new_minimum_parameters_increment_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Minimum loss decrease
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumLossDecrease");

       if(element)
       {
          const double new_minimum_loss_increase = atof(element->GetText());

          try
          {
             set_minimum_loss_increase(new_minimum_loss_increase);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Loss goal
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossGoal");

       if(element)
       {
          const double new_loss_goal = atof(element->GetText());

          try
          {
             set_loss_goal(new_loss_goal);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Gradient norm goal
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("GradientNormGoal");

       if(element)
       {
          const double new_gradient_norm_goal = atof(element->GetText());

          try
          {
             set_gradient_norm_goal(new_gradient_norm_goal);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum selection error decreases
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionErrorIncreases");

       if(element)
       {
          const size_t new_maximum_selection_failures = static_cast<size_t>(atoi(element->GetText()));

          try
          {
             set_maximum_selection_error_increases(new_maximum_selection_failures);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum iterations number
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumIterationsNumber");

       if(element)
       {
          const size_t new_maximum_iterations_number = static_cast<size_t>(atoi(element->GetText()));

          try
          {
             set_maximum_epochs_number(new_maximum_iterations_number);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum time
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

       if(element)
       {
          const double new_maximum_time = atof(element->GetText());

          try
          {
             set_maximum_time(new_maximum_time);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve parameters history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersHistory");

       if(element)
       {
          const string new_reserve_parameters_history = element->GetText();

          try
          {
             set_reserve_parameters_history(new_reserve_parameters_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve parameters norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersNormHistory");

       if(element)
       {
          const string new_reserve_parameters_norm_history = element->GetText();

          try
          {
             set_reserve_parameters_norm_history(new_reserve_parameters_norm_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveErrorHistory");

       if(element)
       {
          const string new_reserve_error_history = element->GetText();

          try
          {
             set_reserve_error_history(new_reserve_error_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

    // Reserve selection error history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
           const string new_reserve_selection_error_history = element->GetText();

           try
           {
              set_reserve_selection_error_history(new_reserve_selection_error_history != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

   // Reserve gradient history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGradientHistory");

       if(element)
       {
          const string new_reserve_gradient_history = element->GetText();

          try
          {
             set_reserve_gradient_history(new_reserve_gradient_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve gradient norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGradientNormHistory");

       if(element)
       {
          const string new_reserve_gradient_norm_history = element->GetText();

          try
          {
             set_reserve_gradient_norm_history(new_reserve_gradient_norm_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve training rate history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveLearningRateHistory");

       if(element)
       {
          const string new_reserve_learning_rate_history = element->GetText();

          try
          {
             set_reserve_learning_rate_history(new_reserve_learning_rate_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve elapsed time history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveElapsedTimeHistory");

       if(element)
       {
          const string new_reserve_elapsed_time_history = element->GetText();

          try
          {
             set_reserve_elapsed_time_history(new_reserve_elapsed_time_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve selection error history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

       if(element)
       {
          const string new_reserve_selection_error_history = element->GetText();

          try
          {
             set_reserve_selection_error_history(new_reserve_selection_error_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Display period
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("DisplayPeriod");

       if(element)
       {
          const size_t new_display_period = static_cast<size_t>(atoi(element->GetText()));

          try
          {
             set_display_period(new_display_period);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

    // Save period
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SavePeriod");

        if(element)
        {
           const size_t new_save_period = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              set_save_period(new_save_period);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Neural network file name
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("NeuralNetworkFileName");

        if(element)
        {
           const string new_neural_network_file_name = element->GetText();

           try
           {
              set_neural_network_file_name(new_neural_network_file_name);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

   // Display
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

       if(element)
       {
          const string new_display = element->GetText();

          try
          {
             set_display(new_display != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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


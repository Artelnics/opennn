/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R A D I E N T   D E S C E N T   C L A S S                                                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Open NN includes

#include "gradient_descent.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a gradient descent training algorithm not associated to any loss functional object.
/// It also initializes the class members to their default values.

GradientDescent::GradientDescent(void) 
 : TrainingAlgorithm()
{
   set_default();
}


// PERFORMANCE FUNCTIONAL CONSTRUCTOR 

/// Loss index constructor. 
/// It creates a gradient descent training algorithm associated to a loss functional.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss functional object.

GradientDescent::GradientDescent(LossIndex* new_loss_index_pointer)
: TrainingAlgorithm(new_loss_index_pointer)
{
   training_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a gradient descent training algorithm not associated to any loss functional object.
/// It also loads the class members from a XML document.
/// @param document TinyXML document with the members of a gradient descent object.

GradientDescent::GradientDescent(const tinyxml2::XMLDocument& document) : TrainingAlgorithm(document)
{
   set_default();

   from_XML(document);
}


// DESTRUCTOR

/// Destructor.

GradientDescent::~GradientDescent(void)
{
}


// METHODS

// const TrainingRateAlgorithm& get_training_rate_algorithm(void) const method

/// Returns a constant reference to the training rate algorithm object inside the gradient descent object. 

const TrainingRateAlgorithm& GradientDescent::get_training_rate_algorithm(void) const
{
   return(training_rate_algorithm);
}


// TrainingRateAlgorithm* get_training_rate_algorithm_pointer(void) method

/// Returns a pointer to the training rate algorithm object inside the gradient descent object. 

TrainingRateAlgorithm* GradientDescent::get_training_rate_algorithm_pointer(void)
{
   return(&training_rate_algorithm);
}


// const double& get_warning_parameters_norm(void) const method

/// Returns the minimum value for the norm of the parameters vector at wich a warning message is 
/// written to the screen. 

const double& GradientDescent::get_warning_parameters_norm(void) const
{
   return(warning_parameters_norm);       
}


// const double& get_warning_gradient_norm(void) const method

/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written
/// to the screen. 

const double& GradientDescent::get_warning_gradient_norm(void) const
{
   return(warning_gradient_norm);       
}


// const double& get_warning_training_rate(void) const method

/// Returns the training rate value at wich a warning message is written to the screen during line 
/// minimization.

const double& GradientDescent::get_warning_training_rate(void) const
{
   return(warning_training_rate);
}


// const double& get_error_parameters_norm(void) const method

/// Returns the value for the norm of the parameters vector at wich an error message is 
/// written to the screen and the program exits. 

const double& GradientDescent::get_error_parameters_norm(void) const
{
   return(error_parameters_norm);
}


// const double& get_error_gradient_norm(void) const method

/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits. 

const double& GradientDescent::get_error_gradient_norm(void) const
{
   return(error_gradient_norm);
}


// const double& get_error_training_rate(void) const method

/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.

const double& GradientDescent::get_error_training_rate(void) const
{
   return(error_training_rate);
}


// const double& get_minimum_parameters_increment_norm(void) const method

/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training. 

const double& GradientDescent::get_minimum_parameters_increment_norm(void) const
{
   return(minimum_parameters_increment_norm);
}


// const double& get_minimum_loss_increase(void) const method

/// Returns the minimum loss improvement during training.  

const double& GradientDescent::get_minimum_loss_increase(void) const
{
   return(minimum_loss_increase);
}


// const double& get_loss_goal(void) const method

/// Returns the goal value for the loss. 
/// This is used as a stopping criterion when training a multilayer perceptron

const double& GradientDescent::get_loss_goal(void) const
{
   return(loss_goal);
}


// const double& get_gradient_norm_goal(void) const method

/// Returns the goal value for the norm of the objective function gradient.
/// This is used as a stopping criterion when training a multilayer perceptron

const double& GradientDescent::get_gradient_norm_goal(void) const
{
   return(gradient_norm_goal);
}


// const size_t& get_maximum_selection_loss_decreases(void) const method

/// Returns the maximum number of selection failures during the training process. 

const size_t& GradientDescent::get_maximum_selection_loss_decreases(void) const
{
   return(maximum_selection_loss_decreases);
}


// const size_t& get_maximum_iterations_number(void) const method

/// Returns the maximum number of iterations for training.

const size_t& GradientDescent::get_maximum_iterations_number(void) const
{
   return(maximum_iterations_number);
}


// const double& get_maximum_time(void) const method

/// Returns the maximum training time.  

const double& GradientDescent::get_maximum_time(void) const
{
   return(maximum_time);
}

// const bool& get_return_minimum_selection_error_neural_network(void) const method

/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& GradientDescent::get_return_minimum_selection_error_neural_network(void) const
{
    return(return_minimum_selection_error_neural_network);
}

// const bool& get_reserve_parameters_history(void) const method

/// Returns true if the parameters history matrix is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_parameters_history(void) const
{
   return(reserve_parameters_history);     
}


// const bool& get_reserve_parameters_norm_history(void) const method 

/// Returns true if the parameters norm history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_parameters_norm_history(void) const
{
   return(reserve_parameters_norm_history);     
}


// const bool& get_reserve_loss_history(void) const method

/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_loss_history(void) const
{
   return(reserve_loss_history);     
}


// const bool& get_reserve_gradient_history(void) const method

/// Returns true if the gradient history vector of vectors is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_gradient_history(void) const
{
   return(reserve_gradient_history);     
}


// const bool& get_reserve_gradient_norm_history(void) const method

/// Returns true if the gradient norm history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_gradient_norm_history(void) const
{
   return(reserve_gradient_norm_history);     
}


// const bool& get_reserve_training_direction_history(void) const method

/// Returns true if the training direction history matrix is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_training_direction_history(void) const
{
   return(reserve_training_direction_history);     
}


// const bool& get_reserve_training_rate_history(void) const method

/// Returns true if the training rate history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_training_rate_history(void) const
{
   return(reserve_training_rate_history);     
}


// const bool& get_reserve_elapsed_time_history(void) const method

/// Returns true if the elapsed time history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_elapsed_time_history(void) const
{
   return(reserve_elapsed_time_history);     
}


// const bool& get_reserve_selection_loss_history(void) const method

/// Returns true if the selection loss history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_selection_loss_history(void) const
{
   return(reserve_selection_loss_history);
}


// void set_training_rate_algorithm(const TrainingRateAlgorithm&) method

/// Sets a new training rate algorithm object into the gradient descent object.
/// @param new_training_rate_algorithm Object of the class TrainingRateAlgorithm

void GradientDescent::set_training_rate_algorithm(const TrainingRateAlgorithm& new_training_rate_algorithm)
{
   training_rate_algorithm = new_training_rate_algorithm;
}


// void set_loss_index_pointer(LossIndex*) method

/// Sets a pointer to a loss functional object to be associated to the gradient descent object.
/// It also sets that loss functional to the training rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss functional object.

void GradientDescent::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   training_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


// void set_default(void) method

void GradientDescent::set_default(void)
{
   // TRAINING PARAMETERS

   warning_parameters_norm = 1.0e6;
   warning_gradient_norm = 1.0e6;   
   warning_training_rate = 1.0e6;

   error_parameters_norm = 1.0e9;
   error_gradient_norm = 1.0e9;
   error_training_rate = 1.0e9;

   // STOPPING CRITERIA

   minimum_parameters_increment_norm = 0.0;

   minimum_loss_increase = 0.0;
   loss_goal = -1.0e99;
   gradient_norm_goal = 0.0;
   maximum_selection_loss_decreases = 1000000;

   maximum_iterations_number = 1000;
   maximum_time = 1000.0;

   return_minimum_selection_error_neural_network = false;

   // TRAINING HISTORY

   reserve_parameters_history = false;
   reserve_parameters_norm_history = false;

   reserve_loss_history = true;
   reserve_gradient_history = false;
   reserve_gradient_norm_history = false;
   reserve_selection_loss_history = false;

   reserve_training_direction_history = false;
   reserve_training_rate_history = false;
   reserve_elapsed_time_history = false;

   // UTILITIES

   display = true;
   display_period = 5;
}


// void set_reserve_all_training_history(bool) method

/// Makes the training history of all variables to reseved or not in memory:
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> Performance.
/// <li> Gradient. 
/// <li> Gradient norm. 
/// <li> Selection loss.
/// <li> Training direction.
/// <li> Training direction norm. 
/// <li> Training rate.
/// </ul>
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved, false otherwise.

void GradientDescent::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
   // Multilayer perceptron

   reserve_parameters_history = new_reserve_all_training_history;
   reserve_parameters_norm_history = new_reserve_all_training_history;
   
   // Loss index

   reserve_loss_history = new_reserve_all_training_history;
   reserve_gradient_history = new_reserve_all_training_history;
   reserve_gradient_norm_history = new_reserve_all_training_history;

   reserve_selection_loss_history = new_reserve_all_training_history;

   // Training algorithm

   reserve_training_direction_history = new_reserve_all_training_history;
   reserve_training_rate_history = new_reserve_all_training_history;

   reserve_elapsed_time_history = new_reserve_all_training_history;
}


// void set_warning_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which a warning message is written to the 
/// screen. 
/// @param new_warning_parameters_norm Warning norm of parameters vector value. 

void GradientDescent::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_parameters_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_warning_parameters_norm(const double&) method.\n"
             << "Warning parameters norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set warning parameters norm

   warning_parameters_norm = new_warning_parameters_norm;     
}


// void set_warning_gradient_norm(const double&) method

/// Sets a new value for the gradient vector norm at which 
/// a warning message is written to the screen. 
/// @param new_warning_gradient_norm Warning norm of gradient vector value. 

void GradientDescent::set_warning_gradient_norm(const double& new_warning_gradient_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_gradient_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_warning_gradient_norm(const double&) method.\n"
             << "Warning gradient norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set warning gradient norm

   warning_gradient_norm = new_warning_gradient_norm;     
}


// void set_warning_training_rate(const double&) method

/// Sets a new training rate value at wich a warning message is written to the screen during line 
/// minimization.
/// @param new_warning_training_rate Warning training rate value.

void GradientDescent::set_warning_training_rate(const double& new_warning_training_rate)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_training_rate < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_warning_training_rate(const double&) method.\n"
             << "Warning training rate must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   warning_training_rate = new_warning_training_rate;
}


// void set_error_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which an error message is written to the 
/// screen and the program exits. 
/// @param new_error_parameters_norm Error norm of parameters vector value. 

void GradientDescent::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_parameters_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_error_parameters_norm(const double&) method.\n"
             << "Error parameters norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set error parameters norm

   error_parameters_norm = new_error_parameters_norm;
}


// void set_error_gradient_norm(const double&) method

/// Sets a new value for the gradient vector norm at which an error message is written to the screen 
/// and the program exits. 
/// @param new_error_gradient_norm Error norm of gradient vector value. 

void GradientDescent::set_error_gradient_norm(const double& new_error_gradient_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_gradient_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_error_gradient_norm(const double&) method.\n"
             << "Error gradient norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set error gradient norm

   error_gradient_norm = new_error_gradient_norm;
}


// void set_error_training_rate(const double&) method

/// Sets a new training rate value at wich a the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.
/// @param new_error_training_rate Error training rate value.

void GradientDescent::set_error_training_rate(const double& new_error_training_rate)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_training_rate < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_error_training_rate(const double&) method.\n"
             << "Error training rate must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set error training rate

   error_training_rate = new_error_training_rate;
}


// void set_minimum_parameters_increment_norm(const double&) method

/// Sets a new value for the minimum parameters increment norm stopping criterion. 
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training. 

void GradientDescent::set_minimum_parameters_increment_norm(const double& new_minimum_parameters_increment_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_parameters_increment_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void new_minimum_parameters_increment_norm(const double&) method.\n"
             << "Minimum parameters increment norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set error training rate

   minimum_parameters_increment_norm = new_minimum_parameters_increment_norm;
}


// void set_minimum_loss_increase(const double&) method

/// Sets a new minimum loss improvement during training.  
/// @param new_minimum_loss_increase Minimum improvement in the loss between two iterations.

void GradientDescent::set_minimum_loss_increase(const double& new_minimum_loss_increase)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_loss_increase < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_minimum_loss_increase(const double&) method.\n"
             << "Minimum loss improvement must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set minimum loss improvement

   minimum_loss_increase = new_minimum_loss_increase;
}


// void set_loss_goal(const double&) method

/// Sets a new goal value for the loss. 
/// This is used as a stopping criterion when training a multilayer perceptron
/// @param new_loss_goal Goal value for the loss.

void GradientDescent::set_loss_goal(const double& new_loss_goal)
{
   loss_goal = new_loss_goal;
}


// void set_gradient_norm_goal(const double&) method

/// Sets a new the goal value for the norm of the objective function gradient. 
/// This is used as a stopping criterion when training a multilayer perceptron
/// @param new_gradient_norm_goal Goal value for the norm of the objective function gradient.

void GradientDescent::set_gradient_norm_goal(const double& new_gradient_norm_goal)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_gradient_norm_goal < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_gradient_norm_goal(const double&) method.\n"
             << "Gradient norm goal must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set gradient norm goal

   gradient_norm_goal = new_gradient_norm_goal;
}


// void set_maximum_selection_loss_decreases(const size_t&) method

/// Sets a new maximum number of selection failures. 
/// @param new_maximum_selection_loss_decreases Maximum number of iterations in which the selection evalutation decreases.

void GradientDescent::set_maximum_selection_loss_decreases(const size_t& new_maximum_selection_loss_decreases)
{
   maximum_selection_loss_decreases = new_maximum_selection_loss_decreases;
}


// void set_maximum_iterations_number(size_t) method

/// Sets a maximum number of iterations for training.
/// @param new_maximum_iterations_number Maximum number of iterations for training.

void GradientDescent::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
   maximum_iterations_number = new_maximum_iterations_number;
}


// void set_maximum_time(const double&) method

/// Sets a new maximum training time.  
/// @param new_maximum_time Maximum training time.

void GradientDescent::set_maximum_time(const double& new_maximum_time)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_time < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_maximum_time(const double&) method.\n"
             << "Maximum time must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }
   
   #endif

   // Set maximum time

   maximum_time = new_maximum_time;
}

// void set_return_minimum_selection_error_neural_network(const bool&) method

/// Makes the minimum selection error neural network of all the iterations to be returned or not.
/// @param new_return_minimum_selection_error_neural_network True if the final model will be the neural network with the minimum selection error, false otherwise.

void GradientDescent::set_return_minimum_selection_error_neural_network(const bool& new_return_minimum_selection_error_neural_network)
{
   return_minimum_selection_error_neural_network = new_return_minimum_selection_error_neural_network;
}

// void set_reserve_parameters_history(bool) method

/// Makes the parameters history vector of vectors to be reseved or not in memory.
/// @param new_reserve_parameters_history True if the parameters history vector of vectors is to be reserved, false otherwise.

void GradientDescent::set_reserve_parameters_history(const bool& new_reserve_parameters_history)
{
   reserve_parameters_history = new_reserve_parameters_history;     
}


// void set_reserve_parameters_norm_history(bool) method

/// Makes the parameters norm history vector to be reseved or not in memory.
/// @param new_reserve_parameters_norm_history True if the parameters norm history vector is to be reserved, false otherwise.

void GradientDescent::set_reserve_parameters_norm_history(const bool& new_reserve_parameters_norm_history)
{
   reserve_parameters_norm_history = new_reserve_parameters_norm_history;     
}


// void set_reserve_loss_history(bool) method

/// Makes the loss history vector to be reseved or not in memory.
/// @param new_reserve_loss_history True if the loss history vector is to be reserved, false otherwise.

void GradientDescent::set_reserve_loss_history(const bool& new_reserve_loss_history)
{
   reserve_loss_history = new_reserve_loss_history;     
}


// void set_reserve_gradient_history(bool) method

/// Makes the gradient history vector of vectors to be reseved or not in memory.
/// @param new_reserve_gradient_history True if the gradient history matrix is to be reserved, false otherwise.

void GradientDescent::set_reserve_gradient_history(const bool& new_reserve_gradient_history)
{
   reserve_gradient_history = new_reserve_gradient_history;    
}


// void set_reserve_gradient_norm_history(bool) method

/// Makes the gradient norm history vector to be reseved or not in memory.
/// @param new_reserve_gradient_norm_history True if the gradient norm history matrix is to be reserved, false 
/// otherwise.

void GradientDescent::set_reserve_gradient_norm_history(const bool& new_reserve_gradient_norm_history)
{
   reserve_gradient_norm_history = new_reserve_gradient_norm_history;     
}


// void set_reserve_training_direction_history(bool) method

/// Makes the training direction history vector of vectors to be reseved or not in memory.
/// @param new_reserve_training_direction_history True if the training direction history matrix is to be reserved, 
/// false otherwise.

void GradientDescent::set_reserve_training_direction_history(const bool& new_reserve_training_direction_history)
{
   reserve_training_direction_history = new_reserve_training_direction_history;          
}


// void set_reserve_training_rate_history(bool) method

/// Makes the training rate history vector to be reseved or not in memory.
/// @param new_reserve_training_rate_history True if the training rate history vector is to be reserved, false 
/// otherwise.

void GradientDescent::set_reserve_training_rate_history(const bool& new_reserve_training_rate_history)
{
   reserve_training_rate_history = new_reserve_training_rate_history;          
}


// void set_reserve_elapsed_time_history(bool) method

/// Makes the elapsed time over the iterations to be reseved or not in memory. This is a vector.
/// @param new_reserve_elapsed_time_history True if the elapsed time history vector is to be reserved, false 
/// otherwise.

void GradientDescent::set_reserve_elapsed_time_history(const bool& new_reserve_elapsed_time_history)
{
   reserve_elapsed_time_history = new_reserve_elapsed_time_history;     
}


// void set_reserve_selection_loss_history(bool) method

/// Makes the selection loss history to be reserved or not in memory.
/// This is a vector. 
/// @param new_reserve_selection_loss_history True if the selection loss history is to be reserved, false otherwise.

void GradientDescent::set_reserve_selection_loss_history(const bool& new_reserve_selection_loss_history)  
{
   reserve_selection_loss_history = new_reserve_selection_loss_history;
}


// void set_display_period(size_t) method

/// Sets a new number of iterations between the training showing progress.
/// @param new_display_period
/// Number of iterations between the training showing progress.

void GradientDescent::set_display_period(const size_t& new_display_period)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 
     
   if(new_display_period <= 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_display_period(const double&) method.\n"
             << "First training rate must be greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   display_period = new_display_period;
}


// Vector<double> calculate_training_direction(const Vector<double>&) const method

/// Returns the gradient descent training direction,
/// which is the negative of the normalized gradient.
/// @param gradient Performance function gradient.

Vector<double> GradientDescent::calculate_training_direction(const Vector<double>& gradient) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: GradientDescent class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&) const method.\n"
              << "Loss index pointer is NULL.\n";

       throw std::logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: GradientDescent class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&) const method.\n"
              << "Size of gradient (" << gradient_size << ") is not equal to number of parameters (" << parameters_number << ").\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   return(gradient.calculate_normalized()*(-1.0));
}


// std::string GradientDescentResults::to_string(void) const method

std::string GradientDescent::GradientDescentResults::to_string(void) const
{
   std::ostringstream buffer;

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

   // Performance history   

   if(!loss_history.empty())
   {
       buffer << "% Performance history:\n"
              << loss_history << "\n"; 
   }

   // Selection loss history

   if(!selection_loss_history.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_loss_history << "\n"; 
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

   // Training direction history

   if(!training_direction_history.empty())
   {
      if(!training_direction_history[0].empty())
      {
         buffer << "% Training direction history:\n"
                << training_direction_history << "\n"; 
	  }
   }

   // Training rate history   

   if(!training_rate_history.empty())
   {
       buffer << "% Training rate history:\n"
              << training_rate_history << "\n"; 
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


// Matrix<std::string> write_final_results(const size_t& precision) const method

Matrix<std::string> GradientDescent::GradientDescentResults::write_final_results(const size_t& precision) const
{
   std::ostringstream buffer;

   Vector<std::string> names;
   Vector<std::string> values;

   // Final parameters norm

   names.push_back("Final parameters norm");

   buffer.str("");
   buffer << std::setprecision(precision) << final_parameters_norm;

   values.push_back(buffer.str());

   // Final loss

   names.push_back("Final loss");

   buffer.str("");
   buffer << std::setprecision(precision) << final_loss;

   values.push_back(buffer.str());

   // Final selection loss

   const LossIndex* loss_index_pointer = gradient_descent_pointer->get_loss_index_pointer();

   if(loss_index_pointer->has_selection())
   {
       names.push_back("Final selection loss");

       buffer.str("");
       buffer << std::setprecision(precision) << final_selection_loss;

       values.push_back(buffer.str());
    }

   // Final gradient norm

   names.push_back("Final gradient norm");

   buffer.str("");
   buffer << std::setprecision(precision) << final_gradient_norm;

   values.push_back(buffer.str());

   // Final training rate

//   names.push_back("Final training rate");

//   buffer.str("");
//   buffer << std::setprecision(precision) << final_training_rate;

//   values.push_back(buffer.str());

   // Iterations number

   names.push_back("Iterations number");

   buffer.str("");
   buffer << iterations_number;

   values.push_back(buffer.str());

   // Elapsed time

   names.push_back("Elapsed time");

   buffer.str("");
   buffer << elapsed_time;

   values.push_back(buffer.str());

   // Stopping criteria

   names.push_back("Stopping criterion");

   values.push_back(write_stopping_condition());

   const size_t rows_number = names.size();
   const size_t columns_number = 2;

   Matrix<std::string> final_results(rows_number, columns_number);

   final_results.set_column(0, names);
   final_results.set_column(1, values);

   return(final_results);
}


// void GradientDescentResults::resize_training_history(const size_t&) method

/// Resizes the training history variables which are to be reserved by the training algorithm.
/// @param new_size Size of training history variables. 

void GradientDescent::GradientDescentResults::resize_training_history(const size_t& new_size)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(gradient_descent_pointer == NULL)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: GradientDescentResults structure.\n"
              << "void resize_training_history(const size_t&) method.\n"
              << "Gradient descent pointer is NULL.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    if(gradient_descent_pointer->get_reserve_parameters_history())
    {
        parameters_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_parameters_norm_history())
    {
        parameters_norm_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_loss_history())
    {
        loss_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_selection_loss_history())
    {
        selection_loss_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_gradient_history())
    {
        gradient_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_gradient_norm_history())
    {
        gradient_norm_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_training_direction_history())
    {
        training_direction_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_training_rate_history())
    {
        training_rate_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_elapsed_time_history())
    {
        elapsed_time_history.resize(new_size);
    }
}


// GradientDescentResults* perform_training(void) method

/// Trains a neural network with an associated loss functional,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

GradientDescent::GradientDescentResults* GradientDescent::perform_training(void)
{
    GradientDescentResults* results_pointer = new GradientDescentResults(this);

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Start training 

   if(display)
   {
      std::cout << "Training with gradient descent...\n";
   }

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Vector<double> parameters(parameters_number);
   double parameters_norm;

   Vector<double> parameters_increment(parameters_number);
   double parameters_increment_norm;

   // Loss index stuff

   double selection_loss = 0.0; 
   double old_selection_loss = 0.0;
      
   double loss = 0.0;
   double old_loss = 0.0;
   double loss_increase = 0.0;

   Vector<double> gradient(parameters_number);
   double gradient_norm;

   LossIndex::FirstOrderloss first_order_loss;

   // Training algorithm stuff 

   size_t selection_failures = 0;

   Vector<double> training_direction(parameters_number);

   const double first_training_rate = 0.01;

   double initial_training_rate = 0.0;
   double training_rate = 0.0;
   double old_training_rate = 0.0;

   Vector<double> directional_point(2, 0.0);

   Vector<double> minimum_selection_error_parameters(parameters_number);
   double minimum_selection_error;

   bool stop_training = false;

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time;

   results_pointer->resize_training_history(maximum_iterations_number+1);
   
   // Main loop

   for(size_t iteration = 0; iteration <= maximum_iterations_number; iteration++)
   {
      // Neural network stuff

      parameters = neural_network_pointer->arrange_parameters();

      parameters_norm = parameters.calculate_norm();

      if(display && parameters_norm >= warning_parameters_norm)
      {
         std::cout << "OpenNN Warning: Parameters norm is " << parameters_norm << ".\n";          
      }

      // Loss index stuff

      if(iteration == 0)
      {      
         loss = loss_index_pointer->calculate_loss();
         loss_increase = 1.0e99;
      }
      else
      {
         loss = directional_point[1];
         loss_increase = old_loss - loss; 
      }

      gradient = loss_index_pointer->calculate_gradient();

      gradient_norm = gradient.calculate_norm();

      if(display && gradient_norm >= warning_gradient_norm)
      {
         std::cout << "OpenNN Warning: Gradient norm is " << gradient_norm << ".\n";          
      }

      selection_loss = loss_index_pointer->calculate_selection_loss();

      if(iteration == 0)
      {
          minimum_selection_error = selection_loss;

          minimum_selection_error_parameters = neural_network_pointer->arrange_parameters();
      }
      else if(iteration != 0 && selection_loss > old_selection_loss)
      {
         selection_failures++;
      }
      else if(selection_loss < minimum_selection_error)
      {
          minimum_selection_error = selection_loss;

          minimum_selection_error_parameters = neural_network_pointer->arrange_parameters();
      }

      // Training algorithm 

      training_direction = calculate_training_direction(gradient);
                        
      if(iteration == 0)
      {
         initial_training_rate = first_training_rate;//training_rate_algorithm.get_first_training_rate();
      }
      else
      {
         initial_training_rate = old_training_rate;
      }    

      directional_point = training_rate_algorithm.calculate_directional_point(loss, training_direction, initial_training_rate);

      training_rate = directional_point[0];

      parameters_increment = training_direction*training_rate;
      parameters_increment_norm = parameters_increment.calculate_norm();
      
      // Elapsed time

      time(&current_time);
      elapsed_time = difftime(current_time, beginning_time);

      // Training history neural network

      if(reserve_parameters_history)
      {
         results_pointer->parameters_history[iteration] = parameters;
      }

      if(reserve_parameters_norm_history)
      {
         results_pointer->parameters_norm_history[iteration] = parameters_norm;
      }

      // Training history loss functional

      if(reserve_loss_history)
      {
         results_pointer->loss_history[iteration] = loss;
      }

      if(reserve_gradient_history)
      {
         results_pointer->gradient_history[iteration] = gradient;
      }

      if(reserve_gradient_norm_history)
      {
         results_pointer->gradient_norm_history[iteration] = gradient_norm;
      }

      if(reserve_selection_loss_history)
      {
         results_pointer->selection_loss_history[iteration] = selection_loss;
      }

      // Training history training algorithm

      if(reserve_training_direction_history)
      {
         results_pointer->training_direction_history[iteration] = training_direction;
      }

      if(reserve_training_rate_history)
      {
         results_pointer->training_rate_history[iteration] = training_rate;
      }

      if(reserve_elapsed_time_history)
      {
         results_pointer->elapsed_time_history[iteration] = elapsed_time;
      }

      // Stopping Criteria

      if(parameters_increment_norm <= minimum_parameters_increment_norm)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Minimum parameters increment norm reached.\n";
            std::cout << "Parameters increment norm: " << parameters_increment_norm << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MinimumParametersIncrementNorm;
      }

      else if(iteration != 0 && loss_increase <= minimum_loss_increase)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Minimum loss increase reached.\n"
                      << "Performance increase: " << loss_increase << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MinimumPerformanceIncrease;
      }

      else if(loss <= loss_goal)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Performance goal reached.\n";
         }

         stop_training = true;

         results_pointer->stopping_condition = PerformanceGoal;
      }

      else if(selection_failures >= maximum_selection_loss_decreases)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Maximum selection failures reached.\n"
                      << "Selection failures: " << selection_failures << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MaximumSelectionPerformanceDecreases;
      }

      else if(gradient_norm <= gradient_norm_goal)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Gradient norm goal reached.\n";
         }

         stop_training = true;

         results_pointer->stopping_condition = GradientNormGoal;
      }

      else if(iteration == maximum_iterations_number)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Maximum number of iterations reached.\n";
         }

         stop_training = true;

         results_pointer->stopping_condition = MaximumIterationsNumber;
      }

      else if(elapsed_time >= maximum_time)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Maximum training time reached.\n";
         }

         stop_training = true;

         results_pointer->stopping_condition = MaximumTime;
      }

      if(iteration != 0 && iteration % save_period == 0)
      {
            neural_network_pointer->save(neural_network_file_name);
      }

      if(stop_training)
      {
         if(display)
		 {
            std::cout << "Parameters norm: " << parameters_norm << "\n"
                      << "Training loss: " << loss << "\n"
                      << "Gradient norm: " << gradient_norm << "\n"
                      << loss_index_pointer->write_information()
                      << "Training rate: " << training_rate << "\n"
                      << "Elapsed time: " << elapsed_time << std::endl; 

            if(selection_loss != 0)
            {
               std::cout << "Selection loss: " << selection_loss << std::endl;
            }
		 }
    
         results_pointer->resize_training_history(1+iteration);

         results_pointer->final_parameters = parameters;

         results_pointer->final_parameters_norm = parameters_norm;

         results_pointer->final_loss = loss;

         results_pointer->final_selection_loss = selection_loss;

         results_pointer->final_gradient = gradient;

         results_pointer->final_gradient_norm = gradient_norm;

         results_pointer->final_training_direction = training_direction;

         results_pointer->final_training_rate = training_rate;

         results_pointer->elapsed_time = elapsed_time;

         results_pointer->iterations_number = iteration;

         break;
      }
      else if(display && iteration % display_period == 0)
      {
         std::cout << "Iteration " << iteration << ";\n"
                   << "Parameters norm: " << parameters_norm << "\n"
                   << "Training loss: " << loss << "\n"
                   << "Gradient norm: " << gradient_norm << "\n"
                   << loss_index_pointer->write_information()
                   << "Training rate: " << training_rate << "\n"
                   << "Elapsed time: " << elapsed_time << std::endl;

         if(selection_loss != 0)
         {
            std::cout << "Selection loss: " << selection_loss << std::endl;
         }
      }

      // Set new parameters

      parameters += parameters_increment;

      neural_network_pointer->set_parameters(parameters);

      // Update stuff

      old_loss = loss;
      old_selection_loss = selection_loss;
   
      old_training_rate = training_rate;
   } 

   if(return_minimum_selection_error_neural_network)
   {
       parameters = minimum_selection_error_parameters;
       parameters_norm = parameters.calculate_norm();

       neural_network_pointer->set_parameters(parameters);

       loss = loss_index_pointer->calculate_loss();
       selection_loss = minimum_selection_error;
   }

   results_pointer->final_parameters = parameters;
   results_pointer->final_parameters_norm = parameters_norm;

   results_pointer->final_loss = loss;
   results_pointer->final_selection_loss = selection_loss;

   results_pointer->final_gradient = gradient;
   results_pointer->final_gradient_norm = gradient_norm;

   results_pointer->elapsed_time = elapsed_time;

   return(results_pointer);
}


// std::string write_training_algorithm_type(void) const method

std::string GradientDescent::write_training_algorithm_type(void) const
{
   return("GRADIENT_DESCENT");
}


// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> GradientDescent::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

   // Training rate method

   labels.push_back("Training rate method");

   const std::string training_rate_method = training_rate_algorithm.write_training_rate_method();

   values.push_back(training_rate_method);

   // Training rate tolerance

   labels.push_back("Training rate tolerance");

   buffer.str("");
   buffer << training_rate_algorithm.get_training_rate_tolerance();

   values.push_back(buffer.str());

   // Minimum parameters increment norm

   labels.push_back("Minimum parameters increment norm");

   buffer.str("");
   buffer << minimum_parameters_increment_norm;

   values.push_back(buffer.str());

   // Minimum loss increase

   labels.push_back("Minimum loss increase");

   buffer.str("");
   buffer << minimum_loss_increase;

   values.push_back(buffer.str());

   // Performance goal

   labels.push_back("Performance goal");

   buffer.str("");
   buffer << loss_goal;

   values.push_back(buffer.str());

   // Gradient norm goal

   labels.push_back("Gradient norm goal");

   buffer.str("");
   buffer << gradient_norm_goal;

   values.push_back(buffer.str());

   // Maximum selection loss decreases

   labels.push_back("Maximum selection loss increases");

   buffer.str("");
   buffer << maximum_selection_loss_decreases;

   values.push_back(buffer.str());

   // Maximum iterations number

   labels.push_back("Maximum iterations number");

   buffer.str("");
   buffer << maximum_iterations_number;

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

   // Reserve loss history

   labels.push_back("Reserve loss history");

   buffer.str("");

   if(reserve_loss_history)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Reserve selection loss history

   labels.push_back("Reserve selection loss history");

   buffer.str("");

   if(reserve_selection_loss_history)
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

   // Reserve training direction norm history

//   labels.push_back("");

//   buffer.str("");
//   buffer << reserve_training_direction_norm_history;

   // Reserve training rate history

//   labels.push_back("");

//   buffer.str("");
//   buffer << reserve_training_rate_history;

//   values.push_back(buffer.str());

   // Reserve elapsed time history

//   labels.push_back("Reserve elapsed time history");

//   buffer.str("");
//   buffer << reserve_elapsed_time_history;

//   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the training parameters, the stopping criteria and other user stuff 
/// concerning the gradient descent object.

tinyxml2::XMLDocument* GradientDescent::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Training algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("GradientDescent");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Training rate algorithm
   {
      tinyxml2::XMLElement* element = document->NewElement("TrainingRateAlgorithm");
      root_element->LinkEndChild(element);

      const tinyxml2::XMLDocument* training_rate_algorithm_document = training_rate_algorithm.to_XML();

      const tinyxml2::XMLElement* training_rate_algorithm_element = training_rate_algorithm_document->FirstChildElement("TrainingRateAlgorithm");

      DeepClone(element, training_rate_algorithm_element, document, NULL);

      delete training_rate_algorithm_document;
   }


   // Return minimum selection error neural network

   element = document->NewElement("ReturnMinimumSelectionErrorNN");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << return_minimum_selection_error_neural_network;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Warning parameters norm

//   element = document->NewElement("WarningParametersNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_parameters_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Warning gradient norm 

//   element = document->NewElement("WarningGradientNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_gradient_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Warning training rate 

//   element = document->NewElement("WarningTrainingRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_training_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Error parameters norm

//   element = document->NewElement("ErrorParametersNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_parameters_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Error gradient norm 

//   element = document->NewElement("ErrorGradientNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_gradient_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Error training rate

//   element = document->NewElement("ErrorTrainingRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_training_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Minimum parameters increment norm

   element = document->NewElement("MinimumParametersIncrementNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_parameters_increment_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Minimum loss increase 

   element = document->NewElement("MinimumPerformanceIncrease");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_loss_increase;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Performance goal 

   element = document->NewElement("PerformanceGoal");
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

   // Maximum selection loss decreases

   element = document->NewElement("MaximumSelectionLossDecreases");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_loss_decreases;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum iterations number

   element = document->NewElement("MaximumIterationsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_iterations_number;

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

//   element = document->NewElement("ReserveParametersHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_parameters_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Reserve loss history 

   element = document->NewElement("ReservePerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve selection loss history

   element = document->NewElement("ReserveSelectionLossHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve gradient history 

//   element = document->NewElement("ReserveGradientHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_gradient_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Reserve gradient norm history 

   element = document->NewElement("ReserveGradientNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_gradient_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve training direction history 

//   element = document->NewElement("ReserveTrainingDirectionHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_training_direction_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Reserve training rate history 

//   element = document->NewElement("ReserveTrainingRateHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_training_rate_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Reserve elapsed time history 

//   element = document->NewElement("ReserveElapsedTimeHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_elapsed_time_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Reserve selection loss history

//   element = document->NewElement("ReserveSelectionLossHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_selection_loss_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Display period

//   element = document->NewElement("DisplayPeriod");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display_period;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Save period

//   element = document->NewElement("SavePeriod");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << save_period;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Neural network file name

//   element = document->NewElement("NeuralNetworkFileName");
//   root_element->LinkEndChild(element);

//   text = document->NewText(neural_network_file_name.c_str());
//   element->LinkEndChild(text);

   // Display warnings 

//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the gradient descent object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("GradientDescent");

    // Training rate algorithm

    training_rate_algorithm.write_XML(file_stream);

    // Return minimum selection error neural network

    file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

    buffer.str("");
    buffer << return_minimum_selection_error_neural_network;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum parameters increment norm

    file_stream.OpenElement("MinimumParametersIncrementNorm");

    buffer.str("");
    buffer << minimum_parameters_increment_norm;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum loss increase

    file_stream.OpenElement("MinimumPerformanceIncrease");

    buffer.str("");
    buffer << minimum_loss_increase;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Performance goal

    file_stream.OpenElement("PerformanceGoal");

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

    // Maximum selection loss decreases

    file_stream.OpenElement("MaximumSelectionLossDecreases");

    buffer.str("");
    buffer << maximum_selection_loss_decreases;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations number

    file_stream.OpenElement("MaximumIterationsNumber");

    buffer.str("");
    buffer << maximum_iterations_number;

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

    // Reserve loss history

    file_stream.OpenElement("ReservePerformanceHistory");

    buffer.str("");
    buffer << reserve_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection loss history

    file_stream.OpenElement("ReserveSelectionLossHistory");

    buffer.str("");
    buffer << reserve_selection_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve gradient norm history

    file_stream.OpenElement("ReserveGradientNormHistory");

    buffer.str("");
    buffer << reserve_gradient_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

void GradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GradientDescent");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Gradient descent element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Training rate algorithm
    {
       const tinyxml2::XMLElement* training_rate_algorithm_element = root_element->FirstChildElement("TrainingRateAlgorithm");

       if(training_rate_algorithm_element)
       {
           tinyxml2::XMLDocument training_rate_algorithm_document;

           tinyxml2::XMLElement* element_clone = training_rate_algorithm_document.NewElement("TrainingRateAlgorithm");
           training_rate_algorithm_document.InsertFirstChild(element_clone);

           DeepClone(element_clone, training_rate_algorithm_element, &training_rate_algorithm_document, NULL);

           training_rate_algorithm.from_XML(training_rate_algorithm_document);
       }
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Warning training rate 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningTrainingRate");

       if(element)
       {
          const double new_warning_training_rate = atof(element->GetText());

          try
          {
             set_warning_training_rate(new_warning_training_rate);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Error training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorTrainingRate");

       if(element)
       {
          const double new_error_training_rate = atof(element->GetText());

          try
          {
             set_error_training_rate(new_error_training_rate);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

    // Return minimum selection error neural network

    const tinyxml2::XMLElement* return_minimum_selection_error_neural_network_element = root_element->FirstChildElement("ReturnMinimumSelectionErrorNN");

    if(return_minimum_selection_error_neural_network_element)
    {
        std::string new_return_minimum_selection_error_neural_network = return_minimum_selection_error_neural_network_element->GetText();

        try
        {
           set_return_minimum_selection_error_neural_network(new_return_minimum_selection_error_neural_network != "0");
        }
        catch(const std::logic_error& e)
        {
           std::cout << e.what() << std::endl;
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Minimum loss increase 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumPerformanceIncrease");

       if(element)
       {
          const double new_minimum_loss_increase = atof(element->GetText());

          try
          {
             set_minimum_loss_increase(new_minimum_loss_increase);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Performance goal 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("PerformanceGoal");

       if(element)
       {
          const double new_loss_goal = atof(element->GetText());

          try
          {
             set_loss_goal(new_loss_goal);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Maximum selection loss decreases
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionLossDecreases");

       if(element)
       {
          const size_t new_maximum_selection_loss_decreases = atoi(element->GetText());

          try
          {
             set_maximum_selection_loss_decreases(new_maximum_selection_loss_decreases);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Maximum iterations number
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumIterationsNumber");

       if(element)
       {
          const size_t new_maximum_iterations_number = atoi(element->GetText());

          try
          {
             set_maximum_iterations_number(new_maximum_iterations_number);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
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
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve parameters history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersHistory");

       if(element)
       {
          const std::string new_reserve_parameters_history = element->GetText();

          try
          {
             set_reserve_parameters_history(new_reserve_parameters_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve parameters norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersNormHistory");

       if(element)
       {
          const std::string new_reserve_parameters_norm_history = element->GetText();

          try
          {
             set_reserve_parameters_norm_history(new_reserve_parameters_norm_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve loss history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

       if(element)
       {
          const std::string new_reserve_loss_history = element->GetText();

          try
          {
             set_reserve_loss_history(new_reserve_loss_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

    // Reserve selection loss history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

        if(element)
        {
           const std::string new_reserve_selection_loss_history = element->GetText();

           try
           {
              set_reserve_selection_loss_history(new_reserve_selection_loss_history != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

   // Reserve gradient history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGradientHistory");

       if(element)
       {
          const std::string new_reserve_gradient_history = element->GetText();

          try
          {
             set_reserve_gradient_history(new_reserve_gradient_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve gradient norm history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGradientNormHistory");

       if(element)
       {
          const std::string new_reserve_gradient_norm_history = element->GetText();

          try
          {
             set_reserve_gradient_norm_history(new_reserve_gradient_norm_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve training direction history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingDirectionHistory");

       if(element)
       {
          const std::string new_reserve_training_direction_history = element->GetText();

          try
          {
             set_reserve_training_direction_history(new_reserve_training_direction_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve training rate history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingRateHistory");

       if(element)
       {
          const std::string new_reserve_training_rate_history = element->GetText();

          try
          {
             set_reserve_training_rate_history(new_reserve_training_rate_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve elapsed time history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveElapsedTimeHistory");

       if(element)
       {
          const std::string new_reserve_elapsed_time_history = element->GetText();

          try
          {
             set_reserve_elapsed_time_history(new_reserve_elapsed_time_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve selection loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

       if(element)
       {
          const std::string new_reserve_selection_loss_history = element->GetText();

          try
          {
             set_reserve_selection_loss_history(new_reserve_selection_loss_history != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Display period
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("DisplayPeriod");

       if(element)
       {
          const size_t new_display_period = atoi(element->GetText());

          try
          {
             set_display_period(new_display_period);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

    // Save period
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SavePeriod");

        if(element)
        {
           const size_t new_save_period = atoi(element->GetText());

           try
           {
              set_save_period(new_save_period);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Neural network file name
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("NeuralNetworkFileName");

        if(element)
        {
           const std::string new_neural_network_file_name = element->GetText();

           try
           {
              set_neural_network_file_name(new_neural_network_file_name);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

   // Display
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

       if(element)
       {
          const std::string new_display = element->GetText();

          try
          {
             set_display(new_display != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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


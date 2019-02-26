/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R A D I E N T   D E S C E N T   C L A S S                                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Open NN includes

#include "gradient_descent.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also initializes the class members to their default values.

GradientDescent::GradientDescent() 
 : OptimizationAlgorithm()
{
   set_default();
}


/// Loss index constructor. 
/// It creates a gradient descent optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

GradientDescent::GradientDescent(LossIndex* new_loss_index_pointer)
: OptimizationAlgorithm(new_loss_index_pointer)
{
   learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also loads the class members from a XML document.
/// @param document TinyXML document with the members of a gradient descent object.

GradientDescent::GradientDescent(const tinyxml2::XMLDocument& document) : OptimizationAlgorithm(document)
{
   set_default();

   from_XML(document);
}


// DESTRUCTOR

/// Destructor.

GradientDescent::~GradientDescent()
{
}


// METHODS


/// Returns a constant reference to the learning rate algorithm object inside the gradient descent object. 

const LearningRateAlgorithm& GradientDescent::get_learning_rate_algorithm() const
{
   return(learning_rate_algorithm);
}


/// Returns a pointer to the learning rate algorithm object inside the gradient descent object. 

LearningRateAlgorithm* GradientDescent::get_learning_rate_algorithm_pointer()
{
   return(&learning_rate_algorithm);
}


/// Returns the minimum value for the norm of the parameters vector at wich a warning message is 
/// written to the screen. 

const double& GradientDescent::get_warning_parameters_norm() const
{
   return(warning_parameters_norm);       
}


/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written
/// to the screen. 

const double& GradientDescent::get_warning_gradient_norm() const
{
   return(warning_gradient_norm);       
}


/// Returns the training rate value at wich a warning message is written to the screen during line 
/// minimization.

const double& GradientDescent::get_warning_training_rate() const
{
   return(warning_training_rate);
}


/// Returns the value for the norm of the parameters vector at wich an error message is 
/// written to the screen and the program exits. 

const double& GradientDescent::get_error_parameters_norm() const
{
   return(error_parameters_norm);
}


/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits. 

const double& GradientDescent::get_error_gradient_norm() const
{
   return(error_gradient_norm);
}


/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.

const double& GradientDescent::get_error_training_rate() const
{
   return(error_training_rate);
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training. 

const double& GradientDescent::get_minimum_parameters_increment_norm() const
{
   return(minimum_parameters_increment_norm);
}


/// Returns the minimum loss improvement during training.  

const double& GradientDescent::get_minimum_loss_increase() const
{
   return(minimum_loss_decrease);
}


/// Returns the goal value for the loss. 
/// This is used as a stopping criterion when training a multilayer perceptron

const double& GradientDescent::get_loss_goal() const
{
   return(loss_goal);
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a multilayer perceptron

const double& GradientDescent::get_gradient_norm_goal() const
{
   return(gradient_norm_goal);
}


/// Returns the maximum number of selection failures during the training process. 

const size_t& GradientDescent::get_maximum_selection_error_decreases() const
{
   return(maximum_selection_error_decreases);
}


/// Returns the maximum number of iterations for training.

const size_t& GradientDescent::get_maximum_epochs_number() const
{
   return(maximum_epochs_number);
}


/// Returns the maximum training time.  

const double& GradientDescent::get_maximum_time() const
{
   return(maximum_time);
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& GradientDescent::get_return_minimum_selection_error_neural_network() const
{
    return(return_minimum_selection_error_neural_network);
}


/// Returns true if the selection error decrease stopping criteria has to be taken in account, false otherwise.

const bool& GradientDescent::get_apply_early_stopping() const
{
    return(apply_early_stopping);
}


/// Returns true if the parameters history matrix is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_parameters_history() const
{
   return(reserve_parameters_history);     
}


/// Returns true if the parameters norm history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_parameters_norm_history() const
{
   return(reserve_parameters_norm_history);     
}


/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_error_history() const
{
   return(reserve_error_history);
}


/// Returns true if the gradient history vector of vectors is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_gradient_history() const
{
   return(reserve_gradient_history);     
}


/// Returns true if the gradient norm history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_gradient_norm_history() const
{
   return(reserve_gradient_norm_history);     
}


/// Returns true if the training direction history matrix is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_training_direction_history() const
{
   return(reserve_training_direction_history);     
}


/// Returns true if the training rate history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_training_rate_history() const
{
   return(reserve_training_rate_history);     
}


/// Returns true if the elapsed time history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_elapsed_time_history() const
{
   return(reserve_elapsed_time_history);     
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_selection_error_history() const
{
   return(reserve_selection_error_history);
}


/// Sets a new learning rate algorithm object into the gradient descent object.
/// @param new_learning_rate_algorithm Object of the class LearningRateAlgorithm

//void GradientDescent::set_learning_rate_algorithm(const LearningRateAlgorithm& new_learning_rate_algorithm)
//{
//   learning_rate_algorithm = new_learning_rate_algorithm;
//}


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void GradientDescent::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


void GradientDescent::set_default()
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

   minimum_loss_decrease = 0.0;
   loss_goal = -numeric_limits<double>::max();
   gradient_norm_goal = 0.0;
   maximum_selection_error_decreases = 1000000;

   maximum_epochs_number = 1000;
   maximum_time = 1000.0;

   return_minimum_selection_error_neural_network = false;
   apply_early_stopping = true;

   // TRAINING HISTORY

   reserve_parameters_history = false;
   reserve_parameters_norm_history = false;

   reserve_error_history = true;
   reserve_gradient_history = false;
   reserve_gradient_norm_history = false;
   reserve_selection_error_history = false;

   reserve_training_direction_history = false;
   reserve_training_rate_history = false;
   reserve_elapsed_time_history = false;

   // UTILITIES

   display = true;
   display_period = 5;
}


/// Makes the training history of all variables to reseved or not in memory:
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> Loss.
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

   reserve_error_history = new_reserve_all_training_history;
   reserve_gradient_history = new_reserve_all_training_history;
   reserve_gradient_norm_history = new_reserve_all_training_history;

   reserve_selection_error_history = new_reserve_all_training_history;

   // Optimization algorithm

   reserve_training_direction_history = new_reserve_all_training_history;
   reserve_training_rate_history = new_reserve_all_training_history;

   reserve_elapsed_time_history = new_reserve_all_training_history;
}


/// Sets a new value for the parameters vector norm at which a warning message is written to the 
/// screen. 
/// @param new_warning_parameters_norm Warning norm of parameters vector value. 

void GradientDescent::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
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

void GradientDescent::set_warning_gradient_norm(const double& new_warning_gradient_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_gradient_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_warning_gradient_norm(const double&) method.\n"
             << "Warning gradient norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set warning gradient norm

   warning_gradient_norm = new_warning_gradient_norm;     
}


/// Sets a new training rate value at wich a warning message is written to the screen during line 
/// minimization.
/// @param new_warning_training_rate Warning training rate value.

void GradientDescent::set_warning_training_rate(const double& new_warning_training_rate)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_training_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_warning_training_rate(const double&) method.\n"
             << "Warning training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   warning_training_rate = new_warning_training_rate;
}


/// Sets a new value for the parameters vector norm at which an error message is written to the 
/// screen and the program exits. 
/// @param new_error_parameters_norm Error norm of parameters vector value. 

void GradientDescent::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
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

void GradientDescent::set_error_gradient_norm(const double& new_error_gradient_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_gradient_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_error_gradient_norm(const double&) method.\n"
             << "Error gradient norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set error gradient norm

   error_gradient_norm = new_error_gradient_norm;
}


/// Sets a new training rate value at wich a the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.
/// @param new_error_training_rate Error training rate value.

void GradientDescent::set_error_training_rate(const double& new_error_training_rate)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_training_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_error_training_rate(const double&) method.\n"
             << "Error training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set error training rate

   error_training_rate = new_error_training_rate;
}


/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void GradientDescent::set_maximum_epochs_number(const size_t& new_maximum_epochs_number)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_maximum_epochs_number < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
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

void GradientDescent::set_minimum_parameters_increment_norm(const double& new_minimum_parameters_increment_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_parameters_increment_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
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

void GradientDescent::set_minimum_loss_decrease(const double& new_minimum_loss_increase)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_loss_increase < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_minimum_loss_decrease(const double&) method.\n"
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

void GradientDescent::set_loss_goal(const double& new_loss_goal)
{
   loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient. 
/// This is used as a stopping criterion when training a multilayer perceptron
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void GradientDescent::set_gradient_norm_goal(const double& new_gradient_norm_goal)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_gradient_norm_goal < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_gradient_norm_goal(const double&) method.\n"
             << "Gradient norm goal must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set gradient norm goal

   gradient_norm_goal = new_gradient_norm_goal;
}


/// Sets a new maximum number of selection failures. 
/// @param new_maximum_selection_error_decreases Maximum number of iterations in which the selection evalutation decreases.

void GradientDescent::set_maximum_selection_error_increases(const size_t& new_maximum_selection_error_decreases)
{
   maximum_selection_error_decreases = new_maximum_selection_error_decreases;
}


/// Sets a new maximum training time.  
/// @param new_maximum_time Maximum training time.

void GradientDescent::set_maximum_time(const double& new_maximum_time)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_time < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
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

void GradientDescent::set_return_minimum_selection_error_neural_network(const bool& new_return_minimum_selection_error_neural_network)
{
   return_minimum_selection_error_neural_network = new_return_minimum_selection_error_neural_network;
}


/// Makes the selection error decrease stopping criteria has to be taken in account or not.
/// @param new_apply_early_stopping True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

void GradientDescent::set_apply_early_stopping(const bool& new_apply_early_stopping)
{
    apply_early_stopping = new_apply_early_stopping;
}


/// Makes the parameters history vector of vectors to be reseved or not in memory.
/// @param new_reserve_parameters_history True if the parameters history vector of vectors is to be reserved, false otherwise.

void GradientDescent::set_reserve_parameters_history(const bool& new_reserve_parameters_history)
{
   reserve_parameters_history = new_reserve_parameters_history;     
}


/// Makes the parameters norm history vector to be reseved or not in memory.
/// @param new_reserve_parameters_norm_history True if the parameters norm history vector is to be reserved, false otherwise.

void GradientDescent::set_reserve_parameters_norm_history(const bool& new_reserve_parameters_norm_history)
{
   reserve_parameters_norm_history = new_reserve_parameters_norm_history;     
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_error_history True if the loss history vector is to be reserved, false otherwise.

void GradientDescent::set_reserve_error_history(const bool& new_reserve_error_history)
{
   reserve_error_history = new_reserve_error_history;
}


/// Makes the gradient history vector of vectors to be reseved or not in memory.
/// @param new_reserve_gradient_history True if the gradient history matrix is to be reserved, false otherwise.

void GradientDescent::set_reserve_gradient_history(const bool& new_reserve_gradient_history)
{
   reserve_gradient_history = new_reserve_gradient_history;    
}


/// Makes the gradient norm history vector to be reseved or not in memory.
/// @param new_reserve_gradient_norm_history True if the gradient norm history matrix is to be reserved, false 
/// otherwise.

void GradientDescent::set_reserve_gradient_norm_history(const bool& new_reserve_gradient_norm_history)
{
   reserve_gradient_norm_history = new_reserve_gradient_norm_history;     
}


/// Makes the training direction history vector of vectors to be reseved or not in memory.
/// @param new_reserve_training_direction_history True if the training direction history matrix is to be reserved, 
/// false otherwise.

void GradientDescent::set_reserve_training_direction_history(const bool& new_reserve_training_direction_history)
{
   reserve_training_direction_history = new_reserve_training_direction_history;          
}


/// Makes the training rate history vector to be reseved or not in memory.
/// @param new_reserve_training_rate_history True if the training rate history vector is to be reserved, false 
/// otherwise.

void GradientDescent::set_reserve_training_rate_history(const bool& new_reserve_training_rate_history)
{
   reserve_training_rate_history = new_reserve_training_rate_history;          
}


/// Makes the elapsed time over the iterations to be reseved or not in memory. This is a vector.
/// @param new_reserve_elapsed_time_history True if the elapsed time history vector is to be reserved, false 
/// otherwise.

void GradientDescent::set_reserve_elapsed_time_history(const bool& new_reserve_elapsed_time_history)
{
   reserve_elapsed_time_history = new_reserve_elapsed_time_history;     
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector. 
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void GradientDescent::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)  
{
   reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Sets a new number of iterations between the training showing progress.
/// @param new_display_period
/// Number of iterations between the training showing progress.

void GradientDescent::set_display_period(const size_t& new_display_period)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 
     
   if(new_display_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GradientDescent class.\n"
             << "void set_display_period(const double&) method.\n"
             << "First training rate must be greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   display_period = new_display_period;
}


/// Returns the gradient descent training direction,
/// which is the negative of the normalized gradient.
/// @param gradient Performance function gradient.

Vector<double> GradientDescent::calculate_training_direction(const Vector<double>& gradient) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: GradientDescent class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&) const method.\n"
              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: GradientDescent class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

   return(gradient.calculate_normalized()*(-1.0));
}


string GradientDescent::GradientDescentResults::object_to_string() const
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


Matrix<string> GradientDescent::GradientDescentResults::write_final_results(const int& precision) const
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

   // Final selection error

/*   const LossIndex* loss_index_pointer = gradient_descent_pointer->get_loss_index_pointer();

   if(loss_index_pointer->has_selection())
   {
       names.push_back("Final selection error");

       buffer.str("");
       buffer << setprecision(precision) << final_selection_error;

       values.push_back(buffer.str());
    }
*/
   // Final gradient norm

   names.push_back("Final gradient norm");

   buffer.str("");
   buffer << setprecision(precision) << final_gradient_norm;

   values.push_back(buffer.str());

   // Final training rate

//   names.push_back("Final training rate");

//   buffer.str("");
//   buffer << setprecision(precision) << final_training_rate;

//   values.push_back(buffer.str());

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

void GradientDescent::GradientDescentResults::resize_training_history(const size_t& new_size)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(gradient_descent_pointer == nullptr)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: GradientDescentResults structure.\n"
              << "void resize_training_history(const size_t&) method.\n"
              << "Gradient descent pointer is nullptr.\n";

       throw logic_error(buffer.str());
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

    if(gradient_descent_pointer->get_reserve_error_history())
    {
        loss_history.resize(new_size);
    }

    if(gradient_descent_pointer->get_reserve_selection_error_history())
    {
        selection_error_history.resize(new_size);
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


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

GradientDescent::GradientDescentResults* GradientDescent::perform_training()
{
    GradientDescentResults* results_pointer = new GradientDescentResults(this);

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Start training 

   if(display) cout << "Training with gradient descent...\n";

   // Data set stuff

   DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();
   const Vector<size_t> selection_indices = instances.get_selection_indices();

   const size_t selection_instances_number = instances.get_selection_instances_number();

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> parameters(parameters_number);
   double parameters_norm = 0.0;

   Vector<double> parameters_increment(parameters_number);
   double parameters_increment_norm = 0.0;

   // Loss index stuff

   double selection_error = 0.0;
   double old_selection_error = 0.0;
      
   double training_loss = 0.0;
   double old_training_loss = 0.0;
   double training_loss_decrease = -numeric_limits<double>::max();

   Vector<double> gradient(parameters_number);
   double gradient_norm = 0.0;

   // Optimization algorithm stuff 

   size_t selection_failures = 0;

   Vector<double> training_direction(parameters_number);

   const double first_training_rate = 0.01;

   double initial_training_rate = 0.0;
   double training_rate = 0.0;
   double old_training_rate = 0.0;

   Vector<double> directional_point(2, 0.0);

   Vector<double> minimum_selection_error_parameters(parameters_number);
   double minimum_selection_error = numeric_limits<double>::max();

   bool stop_training = false;

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time = 0.0;

   results_pointer->resize_training_history(maximum_epochs_number+1);

   // Main loop

   for(size_t epoch = 0; epoch <= maximum_epochs_number; epoch++)
   {
      // Neural network stuff

      parameters = neural_network_pointer->get_parameters();

      parameters_norm = parameters.calculate_L2_norm();

      if(display && parameters_norm >= warning_parameters_norm) cout << "OpenNN Warning: Parameters norm is " << parameters_norm << ".\n";

      // Loss index stuff

      if(epoch == 0)
      {
         training_loss = loss_index_pointer->calculate_training_loss();
      }
      else
      {
         training_loss = directional_point[1];
         training_loss_decrease = training_loss - old_training_loss;
      }

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

      gradient = loss_index_pointer->calculate_training_loss_gradient();

      if(gradient == 0.0) throw logic_error("Gradient is zero");

      gradient_norm = gradient.calculate_L2_norm();

      if(display && gradient_norm >= warning_gradient_norm)
      {
          cout << "OpenNN Warning: Gradient norm is " << gradient_norm << ".\n";
      }

      // Optimization algorithm

      training_direction = calculate_training_direction(gradient);

      if(training_direction == 0.0) throw logic_error("Training direction is zero");

      const double training_slope = (gradient/gradient_norm).dot(training_direction);

      // Check for a descent direction

      if(training_slope >= 0.0) throw logic_error("Training slope is equal or greater than zero");

      if(epoch == 0)
      {
         initial_training_rate = first_training_rate;
      }
      else
      {
         initial_training_rate = old_training_rate;
      }

      directional_point = learning_rate_algorithm.calculate_directional_point(training_loss, training_direction, initial_training_rate);

      training_rate = directional_point[0];

      if(training_rate == 0.0)
      {
          cout << "Training rate is zero" << endl;

          training_rate = 1.0e-2;
          //throw logic_error("Training rate is zero");
      }

      parameters_increment = training_direction*training_rate;
      parameters_increment_norm = parameters_increment.calculate_L2_norm();

      // Elapsed time

      time(&current_time);
      elapsed_time = difftime(current_time, beginning_time);

      // Training history neural network

      if(reserve_parameters_history)
      {
         results_pointer->parameters_history[epoch] = parameters;
      }

      if(reserve_parameters_norm_history)
      {
         results_pointer->parameters_norm_history[epoch] = parameters_norm;
      }

      // Training history loss index

      if(reserve_error_history)
      {
         results_pointer->loss_history[epoch] = training_loss;
      }

      if(reserve_gradient_history)
      {
         results_pointer->gradient_history[epoch] = gradient;
      }

      if(reserve_gradient_norm_history)
      {
         results_pointer->gradient_norm_history[epoch] = gradient_norm;
      }

      if(reserve_selection_error_history)
      {
         results_pointer->selection_error_history[epoch] = selection_error;
      }

      // Training history optimization algorithm

      if(reserve_training_direction_history)
      {
         results_pointer->training_direction_history[epoch] = training_direction;
      }

      if(reserve_training_rate_history)
      {
         results_pointer->training_rate_history[epoch] = training_rate;
      }

      if(reserve_elapsed_time_history)
      {
         results_pointer->elapsed_time_history[epoch] = elapsed_time;
      }

      // Stopping Criteria

      if(parameters_increment_norm <= minimum_parameters_increment_norm)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Minimum parameters increment norm reached.\n";
            cout << "Parameters increment norm: " << parameters_increment_norm << endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MinimumParametersIncrementNorm;
      }

//      else if(epoch != 0 && training_loss_decrease <= minimum_loss_decrease)
//      {
//         if(display)
//         {
//            cout << "Epoch " << epoch << ": Minimum loss decrease (" << minimum_loss_decrease << ") reached.\n"
//                 << "Loss increase: " << training_loss_decrease << endl;
//         }

//         stop_training = true;

//         results_pointer->stopping_condition = MinimumLossDecrease;
//      }

      /*else*/ if(training_loss <= loss_goal)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Loss goal reached.\n";
         }

         stop_training = true;

         results_pointer->stopping_condition = LossGoal;
      }

      else if(selection_failures >= maximum_selection_error_decreases && apply_early_stopping)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Maximum selection failures reached.\n"
                 << "Selection failures: " << selection_failures << endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MaximumSelectionErrorIncreases;
      }

      else if(gradient_norm <= gradient_norm_goal)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Gradient norm goal reached.\n";
         }

         stop_training = true;

         results_pointer->stopping_condition = GradientNormGoal;
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
                      << "Training loss: " << training_loss << "\n"
                      << "Gradient norm: " << gradient_norm << "\n"
                      << loss_index_pointer->write_information()
                      << "Training rate: " << training_rate << "\n"
                      << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            if(!selection_indices.empty() != 0)
            {
               cout << "Selection error: " << selection_error << endl;
            }
         }

         results_pointer->resize_training_history(1+epoch);

         results_pointer->final_parameters = parameters;

         results_pointer->final_parameters_norm = parameters_norm;

         results_pointer->final_loss = training_loss;

         results_pointer->final_selection_error = selection_error;

         results_pointer->final_gradient = gradient;

         results_pointer->final_gradient_norm = gradient_norm;

         results_pointer->final_training_direction = training_direction;

         results_pointer->final_training_rate = training_rate;

         results_pointer->elapsed_time = elapsed_time;

         results_pointer->iterations_number = epoch;

         break;
      }
      else if(display && epoch % display_period == 0)
      {
         cout << "Epoch " << epoch << ";\n"
              << "Parameters norm: " << parameters_norm << "\n"
              << "Training loss: " << training_loss << "\n"
              << "Gradient norm: " << gradient_norm << "\n"
              << loss_index_pointer->write_information()
              << "Training rate: " << training_rate << "\n"
              << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

         if(!selection_indices.empty() != 0)
         {
            cout << "Selection error: " << selection_error << endl;
         }
      }

      // Set new parameters

      parameters += parameters_increment;

      neural_network_pointer->set_parameters(parameters);

      // Update stuff

      old_training_loss = training_loss;
      old_selection_error = selection_error;

      old_training_rate = training_rate;

      if(stop_training) {break;}
   }

   if(return_minimum_selection_error_neural_network)
   {
       parameters = minimum_selection_error_parameters;
       parameters_norm = parameters.calculate_L2_norm();

       neural_network_pointer->set_parameters(parameters);

       //training_loss = loss_index_pointer->calculate_loss(training_indices);
       selection_error = minimum_selection_error;
   }

   results_pointer->final_parameters = parameters;
   results_pointer->final_parameters_norm = parameters_norm;

   results_pointer->final_loss = training_loss;
   results_pointer->final_selection_error = selection_error;

   results_pointer->final_gradient = gradient;
   results_pointer->final_gradient_norm = gradient_norm;

   results_pointer->elapsed_time = elapsed_time;

   return(results_pointer);
}


void GradientDescent::perform_training_void()
{
    GradientDescentResults* results = perform_training();

    delete results;
}

string GradientDescent::write_optimization_algorithm_type() const
{
   return("GRADIENT_DESCENT");
}


/// Writes as matrix of strings the most representative atributes.

Matrix<string> GradientDescent::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

   // Training rate method

   labels.push_back("Training rate method");

   const string training_rate_method = learning_rate_algorithm.write_training_rate_method();

   values.push_back(training_rate_method);

   // Loss tolerance

   labels.push_back("Loss tolerance");

   buffer.str("");
   buffer << learning_rate_algorithm.get_loss_tolerance();

   values.push_back(buffer.str());

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

   labels.push_back("Loss goal");

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
   buffer << maximum_selection_error_decreases;

   values.push_back(buffer.str());

   // Maximum iterations number

   labels.push_back("Maximum epochs number");

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

   // Reserve loss history

   labels.push_back("Reserve loss history");

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

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return(string_matrix);
}


/// Serializes the training parameters, the stopping criteria and other user stuff 
/// concerning the gradient descent object.

tinyxml2::XMLDocument* GradientDescent::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Optimization algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("GradientDescent");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Training rate algorithm
   {
      const tinyxml2::XMLDocument* learning_rate_algorithm_document = learning_rate_algorithm.to_XML();

      const tinyxml2::XMLElement* inputs_element = learning_rate_algorithm_document->FirstChildElement("Inputs");

      tinyxml2::XMLNode* node = inputs_element->DeepClone(document);

      root_element->InsertEndChild(node);

      delete learning_rate_algorithm_document;
   }


   //   // Return minimum selection error neural network

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

//   element = document->NewElement("WarningLearningRate");
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

//   element = document->NewElement("ErrorLearningRate");
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
   buffer << maximum_selection_error_decreases;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum iterations number

   element = document->NewElement("MaximumEpochsNumber");
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

//   element = document->NewElement("ReserveParametersHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_parameters_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

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

//   element = document->NewElement("ReserveLearningRateHistory");
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

   // Reserve selection error history

//   element = document->NewElement("ReserveSelectionErrorHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_selection_error_history;

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


/// Serializes the gradient descent object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("GradientDescent");

    // Training rate algorithm

    learning_rate_algorithm.write_XML(file_stream);

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
    buffer << maximum_selection_error_decreases;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations number

    file_stream.OpenElement("MaximumEpochsNumber");

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

    // Reserve loss history

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


    //file_stream.CloseElement();
}


void GradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GradientDescent");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Gradient descent element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Training rate algorithm
    {
       const tinyxml2::XMLElement* learning_rate_algorithm_element = root_element->FirstChildElement("LearningRateAlgorithm");

       if(learning_rate_algorithm_element)
       {
           tinyxml2::XMLDocument learning_rate_algorithm_document;
           tinyxml2::XMLNode* element_clone;

           element_clone = learning_rate_algorithm_element->DeepClone(&learning_rate_algorithm_document);

           learning_rate_algorithm_document.InsertFirstChild(element_clone);

           learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
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

   // Warning training rate 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningLearningRate");

       if(element)
       {
          const double new_warning_training_rate = atof(element->GetText());

          try
          {
             set_warning_training_rate(new_warning_training_rate);
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

   // Error training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorLearningRate");

       if(element)
       {
          const double new_error_training_rate = atof(element->GetText());

          try
          {
             set_error_training_rate(new_error_training_rate);
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
             set_minimum_loss_decrease(new_minimum_loss_increase);
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
          const size_t new_maximum_selection_error_decreases = static_cast<size_t>(atoi(element->GetText()));

          try
          {
             set_maximum_selection_error_increases(new_maximum_selection_error_decreases);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum iterations number
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumEpochsNumber");

       if(element)
       {
          const size_t new_maximum_epochs_number = static_cast<size_t>(atoi(element->GetText()));

          try
          {
             set_maximum_epochs_number(new_maximum_epochs_number);
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

   // Reserve training direction history 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingDirectionHistory");

       if(element)
       {
          const string new_reserve_training_direction_history = element->GetText();

          try
          {
             set_reserve_training_direction_history(new_reserve_training_direction_history != "0");
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
          const string new_reserve_training_rate_history = element->GetText();

          try
          {
             set_reserve_training_rate_history(new_reserve_training_rate_history != "0");
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


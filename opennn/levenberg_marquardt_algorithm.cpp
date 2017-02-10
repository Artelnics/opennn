/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "levenberg_marquardt_algorithm.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a Levenberg-Marquardt training algorithm object not associated to any loss functional object. 
/// It also initializes the class members to their default values.

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(void)
 : TrainingAlgorithm()
{
   set_default();
}


// PERFORMANCE FUNCTIONAL CONSTRUCTOR

/// Loss index constructor. 
/// It creates a Levenberg-Marquardt training algorithm object associated associated with a given loss functional object. 
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to an external loss functional object. 

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(LossIndex* new_loss_index_pointer)
 : TrainingAlgorithm(new_loss_index_pointer)
{
    if(!new_loss_index_pointer->is_sum_squared_terms())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
               << "explicit LevenbergMarquardtAlgorithm(LossIndex*) constructor." << std::endl
               << "Loss index cannot be expressed as a series of sum squared terms." << std::endl;

        throw std::logic_error(buffer.str());

    }

   set_default();
}


// XML CONSTRUCTOR

/// XML Constructor.
/// Creates a Levenberg-Marquardt algorithm object, and loads its members from a XML document. 
/// @param document TinyXML document containing the Levenberg-Marquardt algorithm data.

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(const tinyxml2::XMLDocument& document)
 : TrainingAlgorithm(document)
{
   set_default();

   from_XML(document);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any object. 

LevenbergMarquardtAlgorithm::~LevenbergMarquardtAlgorithm(void)
{
}


// const double& get_warning_parameters_norm(void) const method

/// Returns the minimum value for the norm of the parameters vector at wich a warning message is 
/// written to the screen. 

const double& LevenbergMarquardtAlgorithm::get_warning_parameters_norm(void) const
{
   return(warning_parameters_norm);       
}


// const double& get_warning_gradient_norm(void) const method

/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written
/// to the screen. 

const double& LevenbergMarquardtAlgorithm::get_warning_gradient_norm(void) const
{
   return(warning_gradient_norm);       
}


// const double& get_error_parameters_norm(void) const method

/// Returns the value for the norm of the parameters vector at wich an error message is 
/// written to the screen and the program exits. 

const double& LevenbergMarquardtAlgorithm::get_error_parameters_norm(void) const
{
   return(error_parameters_norm);
}


// const double& get_error_gradient_norm(void) const method

/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits. 

const double& LevenbergMarquardtAlgorithm::get_error_gradient_norm(void) const
{
   return(error_gradient_norm);
}


// const double& get_minimum_parameters_increment_norm(void) const method

/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training. 

const double& LevenbergMarquardtAlgorithm::get_minimum_parameters_increment_norm(void) const
{
   return(minimum_parameters_increment_norm);
}


// const double& get_minimum_loss_increase(void) const method

/// Returns the minimum loss improvement during training.  

const double& LevenbergMarquardtAlgorithm::get_minimum_loss_increase(void) const
{
   return(minimum_loss_increase);
}


// const double& get_loss_goal(void) const method

/// Returns the goal value for the loss. 
/// This is used as a stopping criterion when training a neural network.

const double& LevenbergMarquardtAlgorithm::get_loss_goal(void) const
{
   return(loss_goal);
}


// const double& get_gradient_norm_goal(void) const method

/// Returns the goal value for the norm of the loss function gradient.
/// This is used as a stopping criterion when training a neural network.

const double& LevenbergMarquardtAlgorithm::get_gradient_norm_goal(void) const
{
   return(gradient_norm_goal);
}


// const size_t& get_maximum_selection_loss_decreases(void) const method

/// Returns the maximum number of selection failures during the training process. 

const size_t& LevenbergMarquardtAlgorithm::get_maximum_selection_loss_decreases(void) const
{
   return(maximum_selection_loss_decreases);
}


// const size_t& get_maximum_iterations_number(void) const method

/// Returns the maximum number of iterations for training.

const size_t& LevenbergMarquardtAlgorithm::get_maximum_iterations_number(void) const
{
   return(maximum_iterations_number);
}


// const double& get_maximum_time(void) const method

/// Returns the maximum training time.  

const double& LevenbergMarquardtAlgorithm::get_maximum_time(void) const
{
   return(maximum_time);
}

// const bool& get_return_minimum_selection_error_neural_network(void) const method

/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_return_minimum_selection_error_neural_network(void) const
{
    return(return_minimum_selection_error_neural_network);
}

// const bool& get_reserve_parameters_history(void) const method

/// Returns true if the parameters history matrix is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_parameters_history(void) const
{
   return(reserve_parameters_history);     
}


// const bool& get_reserve_parameters_norm_history(void) const method 

/// Returns true if the parameters norm history vector is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_parameters_norm_history(void) const
{
   return(reserve_parameters_norm_history);     
}


// const bool& get_reserve_loss_history(void) const method

/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_loss_history(void) const
{
   return(reserve_loss_history);
}


// const bool& get_reserve_gradient_history(void) const method

/// Returns true if the gradient history vector of vectors is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_gradient_history(void) const
{
   return(reserve_gradient_history);     
}


// const bool& get_reserve_gradient_norm_history(void) const method

/// Returns true if the gradient norm history vector is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_gradient_norm_history(void) const
{
   return(reserve_gradient_norm_history);     
}


// const bool& get_reserve_Hessian_approximation_history(void) const method

/// Returns true if the history of the Hessian approximations is to be reserved,
/// and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_Hessian_approximation_history(void) const
{
   return(reserve_Hessian_approximation_history);
}


// const bool& get_reserve_elapsed_time_history(void) const method

/// Returns true if the elapsed time history vector is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_elapsed_time_history(void) const
{
   return(reserve_elapsed_time_history);     
}


// const bool& get_reserve_selection_loss_history(void) const method

/// Returns true if the selection loss history vector is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_selection_loss_history(void) const
{
   return(reserve_selection_loss_history);
}


// const double& get_damping_parameter(void) const method

/// Returns the damping parameter for the Hessian approximation. 

const double& LevenbergMarquardtAlgorithm::get_damping_parameter(void) const
{
   return(damping_parameter);
}


// const double& get_damping_parameter_factor(void) const method

/// Returns the damping parameter factor (beta in the User's Guide) for the Hessian approximation. 

const double& LevenbergMarquardtAlgorithm::get_damping_parameter_factor(void) const
{
   return(damping_parameter_factor);
}


// const double& get_minimum_damping_parameter(void) const method

/// Returns the minimum damping parameter allowed in the algorithm. 

const double& LevenbergMarquardtAlgorithm::get_minimum_damping_parameter(void) const
{
   return(minimum_damping_parameter);
}


// const double& get_maximum_damping_parameter(void) const method

/// Returns the maximum damping parameter allowed in the algorithm. 

const double& LevenbergMarquardtAlgorithm::get_maximum_damping_parameter(void) const
{
   return(maximum_damping_parameter);
}


// const bool& get_reserve_damping_parameter_history(void) const method

/// Returns true if the damping parameter history vector is to be reserved, and false otherwise.

const bool& LevenbergMarquardtAlgorithm::get_reserve_damping_parameter_history(void) const
{
   return(reserve_damping_parameter_history);
}


// const Vector<double> const get_damping_parameter_history(void) const method

/// Returns a vector containing the damping parameter history over the training iterations.

const Vector<double>& LevenbergMarquardtAlgorithm::get_damping_parameter_history(void) const
{
   return(damping_parameter_history);
}


// void set_default(void) method

/// Sets the following default values for the Levenberg-Marquardt algorithm:
/// Training parameters:
/// <ul>
/// <li> Levenberg-Marquardt parameter: 0.001.
/// </ul>
/// Stopping criteria:
/// <ul> 
/// <li> Performance goal: 1.0e-6.
/// <li> Gradient norm goal: 1.0e-6.
/// <li> Maximum training time: 1000 seconds.
/// <li> Maximum number of iterations: 1000.
/// </ul> 
/// User stuff: 
/// <ul>
/// <li> Iterations between showing progress: 10.
/// </ul>

void LevenbergMarquardtAlgorithm::set_default(void)
{
   // TRAINING PARAMETERS

   warning_parameters_norm = 1.0e6;
   warning_gradient_norm = 1.0e6;   

   error_parameters_norm = 1.0e9;
   error_gradient_norm = 1.0e9;

   // STOPPING CRITERIA

   minimum_parameters_increment_norm = 1.0e-3;

   minimum_loss_increase = 1.0e-9;
   loss_goal = 1.0e-3;
   gradient_norm_goal = 1.0e-3;
   maximum_selection_loss_decreases = 1000;

   maximum_iterations_number = 1000;
   maximum_time = 1000.0;

   return_minimum_selection_error_neural_network = false;

   // TRAINING HISTORY

   reserve_parameters_history = false;
   reserve_parameters_norm_history = false;

   reserve_loss_history = false;
   reserve_gradient_history = false;
   reserve_gradient_norm_history = false;
   reserve_selection_loss_history = false;

   reserve_Hessian_approximation_history = false;

   reserve_elapsed_time_history = false;

   // UTILITIES

   display = true;
   display_period = 5;

   // Training parameters

   damping_parameter = 1.0e-3;

   damping_parameter_factor = 10.0;

   minimum_damping_parameter = 1.0e-6;
   maximum_damping_parameter = 1.0e6;

   reserve_damping_parameter_history = false;
}


// void set_damping_parameter(const double&) method

/// Sets a new damping parameter (lambda in the User's Guide) for the Hessian approximation. 
/// @param new_damping_parameter Damping parameter value. 

void LevenbergMarquardtAlgorithm::set_damping_parameter(const double& new_damping_parameter)
{
   if(new_damping_parameter <= minimum_damping_parameter)
   {
      damping_parameter = minimum_damping_parameter;
   }
   else if(new_damping_parameter >= maximum_damping_parameter)
   {
      damping_parameter = maximum_damping_parameter;
   }
   else
   {
      damping_parameter = new_damping_parameter;
   }
}


// void set_damping_parameter_factor(const double&) method

/// Sets a new damping parameter factor (beta in the User's Guide) for the Hessian approximation. 
/// @param new_damping_parameter_factor Damping parameter factor value. 

void LevenbergMarquardtAlgorithm::set_damping_parameter_factor(const double& new_damping_parameter_factor)
{
   #ifdef __OPENNN_DEBUG__ 

   if(new_damping_parameter_factor <= 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "void set_damping_parameter_factor(const double&) method." << std::endl
             << "Damping parameter factor must be greater than zero." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   #endif

   damping_parameter_factor = new_damping_parameter_factor;
}


// void set_minimum_damping_parameter(const double&) method

/// Sets a new minimum damping parameter allowed in the algorithm. 
/// @param new_minimum_damping_parameter Minimum damping parameter value. 

void LevenbergMarquardtAlgorithm::set_minimum_damping_parameter(const double& new_minimum_damping_parameter)
{
   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_damping_parameter <= 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "void set_minimum_damping_parameter(const double&) method." << std::endl
             << "Minimum damping parameter must be greater than zero." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   #endif

   minimum_damping_parameter = new_minimum_damping_parameter;
}


// void set_maximum_damping_parameter(const double&) method

/// Sets a new maximum damping parameter allowed in the algorithm. 
/// @param new_maximum_damping_parameter Maximum damping parameter value. 

void LevenbergMarquardtAlgorithm::set_maximum_damping_parameter(const double& new_maximum_damping_parameter)
{
   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_damping_parameter <= 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "void set_maximum_damping_parameter(const double&) method." << std::endl
             << "Maximum damping parameter must be greater than zero." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   #endif

   maximum_damping_parameter = new_maximum_damping_parameter;
}


// void set_reserve_damping_parameter_history(bool) method

/// Makes the damping parameter history vector to be reseved or not in memory.
/// @param new_reserve_damping_parameter_history True if the damping parameter history vector is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_damping_parameter_history(const bool& new_reserve_damping_parameter_history)
{
   reserve_damping_parameter_history = new_reserve_damping_parameter_history;
}


// void set_warning_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which a warning message is written to the 
/// screen. 
/// @param new_warning_parameters_norm Warning norm of parameters vector value. 

void LevenbergMarquardtAlgorithm::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_parameters_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
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

void LevenbergMarquardtAlgorithm::set_warning_gradient_norm(const double& new_warning_gradient_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_gradient_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_warning_gradient_norm(const double&) method.\n"
             << "Warning gradient norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set warning gradient norm

   warning_gradient_norm = new_warning_gradient_norm;     
}


// void set_error_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which an error message is written to the 
/// screen and the program exits. 
/// @param new_error_parameters_norm Error norm of parameters vector value. 

void LevenbergMarquardtAlgorithm::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_parameters_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
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

void LevenbergMarquardtAlgorithm::set_error_gradient_norm(const double& new_error_gradient_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_gradient_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_error_gradient_norm(const double&) method.\n"
             << "Error gradient norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set error gradient norm

   error_gradient_norm = new_error_gradient_norm;
}


// void set_minimum_parameters_increment_norm(const double&) method

/// Sets a new value for the minimum parameters increment norm stopping criterion. 
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training. 

void LevenbergMarquardtAlgorithm::set_minimum_parameters_increment_norm(const double& new_minimum_parameters_increment_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_parameters_increment_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
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

void LevenbergMarquardtAlgorithm::set_minimum_loss_increase(const double& new_minimum_loss_increase)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_loss_increase < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
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
/// This is used as a stopping criterion when training a neural network.
/// @param new_loss_goal Goal value for the loss.

void LevenbergMarquardtAlgorithm::set_loss_goal(const double& new_loss_goal)
{
   loss_goal = new_loss_goal;
}


// void set_gradient_norm_goal(const double&) method

/// Sets a new the goal value for the norm of the loss function gradient.
/// This is used as a stopping criterion when training a neural network.
/// @param new_gradient_norm_goal Goal value for the norm of the loss function gradient.

void LevenbergMarquardtAlgorithm::set_gradient_norm_goal(const double& new_gradient_norm_goal)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_gradient_norm_goal < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
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

void LevenbergMarquardtAlgorithm::set_maximum_selection_loss_decreases(const size_t& new_maximum_selection_loss_decreases)
{
   maximum_selection_loss_decreases = new_maximum_selection_loss_decreases;
}


// void set_maximum_iterations_number(size_t) method

/// Sets a maximum number of iterations for training.
/// @param new_maximum_iterations_number Maximum number of iterations for training.

void LevenbergMarquardtAlgorithm::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
   maximum_iterations_number = new_maximum_iterations_number;
}


// void set_maximum_time(const double&) method

/// Sets a new maximum training time.  
/// @param new_maximum_time Maximum training time.

void LevenbergMarquardtAlgorithm::set_maximum_time(const double& new_maximum_time)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_time < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
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

void LevenbergMarquardtAlgorithm::set_return_minimum_selection_error_neural_network(const bool& new_return_minimum_selection_error_neural_network)
{
   return_minimum_selection_error_neural_network = new_return_minimum_selection_error_neural_network;
}

// void set_reserve_parameters_history(bool) method

/// Makes the parameters history vector of vectors to be reseved or not in memory.
/// @param new_reserve_parameters_history True if the parameters history vector of vectors is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_parameters_history(const bool& new_reserve_parameters_history)
{
   reserve_parameters_history = new_reserve_parameters_history;     
}


// void set_reserve_parameters_norm_history(bool) method

/// Makes the parameters norm history vector to be reseved or not in memory.
/// @param new_reserve_parameters_norm_history True if the parameters norm history vector is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_parameters_norm_history(const bool& new_reserve_parameters_norm_history)
{
   reserve_parameters_norm_history = new_reserve_parameters_norm_history;     
}


// void set_reserve_loss_history(bool) method

/// Makes the loss history vector to be reseved or not in memory.
/// @param new_reserve_loss_history True if the loss history vector is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_loss_history(const bool& new_reserve_loss_history)
{
   reserve_loss_history = new_reserve_loss_history;
}


// void set_reserve_gradient_history(bool) method

/// Makes the gradient history vector of vectors to be reseved or not in memory.
/// @param new_reserve_gradient_history True if the gradient history matrix is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_gradient_history(const bool& new_reserve_gradient_history)
{
   reserve_gradient_history = new_reserve_gradient_history;    
}


// void set_reserve_gradient_norm_history(bool) method

/// Makes the gradient norm history vector to be reseved or not in memory.
/// @param new_reserve_gradient_norm_history True if the gradient norm history matrix is to be reserved, false 
/// otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_gradient_norm_history(const bool& new_reserve_gradient_norm_history)
{
   reserve_gradient_norm_history = new_reserve_gradient_norm_history;     
}


// void set_reserve_Hessian_approximation_history(bool) method

/// Sets the history of the Hessian approximation to be reserved or not in memory.
/// This is a vector of matrices. 
/// @param new_reserve_Hessian_approximation_history True if the Hessian approximation history is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_Hessian_approximation_history(const bool& new_reserve_Hessian_approximation_history)
{
   reserve_Hessian_approximation_history = new_reserve_Hessian_approximation_history;
}


// void set_reserve_elapsed_time_history(bool) method

/// Makes the elapsed time over the iterations to be reseved or not in memory. This is a vector.
/// @param new_reserve_elapsed_time_history True if the elapsed time history vector is to be reserved, false 
/// otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_elapsed_time_history(const bool& new_reserve_elapsed_time_history)
{
   reserve_elapsed_time_history = new_reserve_elapsed_time_history;     
}


// void set_reserve_selection_loss_history(bool) method

/// Makes the selection loss history to be reserved or not in memory.
/// This is a vector. 
/// @param new_reserve_selection_loss_history True if the selection loss history is to be reserved, false otherwise.

void LevenbergMarquardtAlgorithm::set_reserve_selection_loss_history(const bool& new_reserve_selection_loss_history)
{
   reserve_selection_loss_history = new_reserve_selection_loss_history;
}


// void set_display_period(size_t) method

/// Sets a new number of iterations between the training showing progress. 
/// @param new_display_period
/// Number of iterations between the training showing progress. 

void LevenbergMarquardtAlgorithm::set_display_period(const size_t& new_display_period)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 
     
   if(new_display_period <= 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_display_period(const double&) method.\n"
             << "First training rate must be greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   display_period = new_display_period;
}


// void check(void) const method

/// Checks that the Levenberg-Marquard object is ok for training.  
/// In particular, it checks that:
/// <ul>
/// <li> The loss functional pointer associated to the training algorithm is not NULL,
/// <li> The neural network associated to that loss functional is neither NULL.
/// <li> The data set associated to that loss functional is neither NULL.
/// </ul>
/// If that checkings are not hold, an exception is thrown. 

void LevenbergMarquardtAlgorithm::check(void) const
{
   std::ostringstream buffer;

   if(!loss_index_pointer)
   {
      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class.\n"
             << "void check(void) const method.\n"
             << "Pointer to loss functional is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "void check(void) const method.\n"
             << "The loss funcional has no data set." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

}


// double calculate_loss(const Vector<double>&) const method

/// Evaluates the loss function from the evaluation of the terms function.
/// @param terms Vector of error terms.

double LevenbergMarquardtAlgorithm::calculate_loss(const Vector<double>& terms) const
{           
    return((terms*terms).calculate_sum());
}


// Vector<double> calculate_gradient(const Vector<double>&, const Matrix<double>&) const method

/// Returns the exact gradient vector of the loss function as a function of the terms vector and the terms Jacobian matrix.
/// @param terms Vector with the error terms values.
/// @param terms_Jacobian Jacobian matrix of the error terms function.

Vector<double> LevenbergMarquardtAlgorithm
::calculate_gradient(const Vector<double>& terms, const Matrix<double>& terms_Jacobian) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   std::ostringstream buffer;

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   const size_t columns_number = terms_Jacobian.get_columns_number();

   if(columns_number != parameters_number)
   {
      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "Vector<double> calculate_gradient(const Vector<double>&, const Matrix<double>&) const method." << std::endl
             << "Number of columns in terms Jacobian must be equal to number of parameters." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   #endif

   return(terms_Jacobian.calculate_transpose().dot(terms)*2.0);
}


// Matrix<double> calculate_Hessian_approximation(const Matrix<double>&) const method

/// Returns an approximation of the Hessian matrix of the loss function
/// as a function of the error terms Jacobian.
/// @param terms_Jacobian Jacobian matrix of the terms function.

Matrix<double> LevenbergMarquardtAlgorithm::calculate_Hessian_approximation(const Matrix<double>& terms_Jacobian) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   std::ostringstream buffer;

   const size_t columns_number = terms_Jacobian.get_columns_number();

   if(columns_number != parameters_number)
   {
      buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << std::endl
             << "Matrix<double> calculate_Hessian_approximation(const Matrix<double>&) const method." << std::endl
             << "Number of columns in terms Jacobian must be equal to number of parameters." << std::endl;

      throw std::logic_error(buffer.str());
   }

   #endif

   return((terms_Jacobian.calculate_transpose().dot(terms_Jacobian)*2.0).sum_diagonal(damping_parameter));
}


// void resize_training_history(const size_t&) method

/// Resizes all the training history variables. 
/// @param new_size Size of training history variables. 

void LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults::resize_training_history(const size_t& new_size)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(Levenberg_Marquardt_algorithm_pointer == NULL)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LevenbergMarquardtAlgorithmResults structure.\n"
              << "void resize_training_history(const size_t&) method.\n"
              << "Levenberg-Marquardt algorithm pointer is NULL.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_parameters_history())
    {
        parameters_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_parameters_norm_history())
    {
        parameters_norm_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_loss_history())
    {
        loss_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_selection_loss_history())
    {
        selection_loss_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_gradient_history())
    {
        gradient_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_gradient_norm_history())
    {
        gradient_norm_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_Hessian_approximation_history())
    {
        Hessian_approximation_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_damping_parameter_history())
    {
        damping_parameter_history.resize(new_size);
    }

    if(Levenberg_Marquardt_algorithm_pointer->get_reserve_elapsed_time_history())
    {
        elapsed_time_history.resize(new_size);
    }
}


// std::string to_string(void) const method

/// Returns a string representation of the current Levenberg-Marquardt algorithm results structure. 

std::string LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults::to_string(void) const
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

   // Hessian approximation history

   if(!Hessian_approximation_history.empty())
   {
      if(!Hessian_approximation_history[0].empty())
      {
          buffer << "% Hessian approximation history:\n"
                 << Hessian_approximation_history << "\n";
	  }
   }

   // Damping parameter history

   if(!damping_parameter_history.empty())
   {
       buffer << "% Damping parameter history:\n"
              << damping_parameter_history << "\n"; 
   }

   // Elapsed time history

   if(!elapsed_time_history.empty())
   {
       buffer << "% Elapsed time history:\n"
              << elapsed_time_history << "\n"; 
   }

   return(buffer.str());
}


// Matrix<std::string> write_final_results(const size_t& precision) const method

Matrix<std::string> LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults::write_final_results(const size_t& precision) const
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

   const LossIndex* loss_index_pointer = Levenberg_Marquardt_algorithm_pointer->get_loss_index_pointer();

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


// LevenbergMarquardtAlgorithmResults* perform_training(void) method

/// Trains a neural network with an associated loss functional according to the Levenberg-Marquardt algorithm.
/// Training occurs according to the training parameters.

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults* LevenbergMarquardtAlgorithm::perform_training(void)
{
    std::ostringstream buffer;

   // Control sentence

   #ifdef __OPENNN_DEBUG__ 
  
   check();

   #endif

   // Start training

   if(display)
   {
      std::cout << "Training with Levenberg-Marquardt algorithm...\n";
   }

   LevenbergMarquardtAlgorithmResults* results_pointer = new LevenbergMarquardtAlgorithmResults(this);

   results_pointer->resize_training_history(1+maximum_iterations_number);

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Vector<double> parameters = neural_network_pointer->arrange_parameters();

   double parameters_norm;

   // Data set stuff

   const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   // Loss index stuff

   double loss = 0.0;
   double old_loss = 0.0;
   double loss_increase = 0.0;

   double selection_loss = 0.0;
   double old_selection_loss = 0.0;

   size_t selection_failures = 0;

   Vector<double> terms(training_instances_number);
   Matrix<double> terms_Jacobian(training_instances_number, parameters_number);

   Vector<double> gradient(parameters_number);

   double gradient_norm;

   Matrix<double> JacobianT_dot_Jacobian(parameters_number, parameters_number);

   Matrix<double> Hessian_approximation(parameters_number, parameters_number);

   // Training strategy stuff

   Vector<double> parameters_increment(parameters_number);
   double parameters_increment_norm;

   Vector<double> minimum_selection_error_parameters(parameters_number);
   double minimum_selection_error;

   bool stop_training = false;

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time;

   // Main loop

   for(size_t iteration = 0; iteration <= maximum_iterations_number; iteration++)
   {
      // Neural network

      parameters_norm = parameters.calculate_norm();

      if(display && parameters_norm >= warning_parameters_norm)
      {
         std::cout << "OpenNN Warning: Parameters norm is " << parameters_norm << "." << std::endl;          
      }

      // Loss index 

      terms = loss_index_pointer->calculate_terms();

      loss = calculate_loss(terms);//*error_terms).calculate_sum()/2.0;

      terms_Jacobian = loss_index_pointer->calculate_terms_Jacobian();

      gradient = calculate_gradient(terms, terms_Jacobian);

      gradient_norm = gradient.calculate_norm();

      JacobianT_dot_Jacobian = terms_Jacobian.calculate_transpose().dot(terms_Jacobian);

      if(display && gradient_norm >= warning_gradient_norm)
      {
         std::cout << "OpenNN Warning: Gradient norm is " << gradient_norm << "." << std::endl;
      }

      do
      {
         Hessian_approximation = (JacobianT_dot_Jacobian.sum_diagonal(damping_parameter));

         parameters_increment = perform_Householder_QR_decomposition(Hessian_approximation, gradient*(-1.0));

         const double new_loss = loss_index_pointer->calculate_loss(parameters+parameters_increment);

         if(new_loss <= loss) // succesfull step
         {
             set_damping_parameter(damping_parameter/damping_parameter_factor);

             parameters += parameters_increment;

             loss = new_loss;

            break;
         }
         else
         {
             set_damping_parameter(damping_parameter*damping_parameter_factor);
         }
      }while(damping_parameter < maximum_damping_parameter);

      parameters_increment_norm = parameters_increment.calculate_norm();

      if(iteration == 0)
      {
         loss_increase = 0.0;
      }
      else
      {
         loss_increase = old_loss - loss;
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

      if(reserve_selection_loss_history)
      {
         results_pointer->selection_loss_history[iteration] = selection_loss;
      }

      if(reserve_gradient_history)
      {
         results_pointer->gradient_history[iteration] = gradient;
      }

      if(reserve_gradient_norm_history)
      {
         results_pointer->gradient_norm_history[iteration] = gradient_norm;
      }

      if(reserve_Hessian_approximation_history)
      {
         results_pointer->Hessian_approximation_history[iteration] = Hessian_approximation; // as computed by linear algebraic equations object
      }

      // Training history training algorithm

      if(reserve_damping_parameter_history)
      {
         results_pointer->damping_parameter_history[iteration] = damping_parameter;
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
            std::cout << "Iteration " << iteration << ": Minimum parameters increment norm reached.\n"
			          << "Parameters increment norm: " << parameters_increment_norm << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MinimumParametersIncrementNorm;
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

      else if(iteration != 0 && loss_increase <= minimum_loss_increase)
      {
         if(display)
         {
             std::cout << "Iteration " << iteration << ": Minimum loss increase (" << minimum_loss_increase << ") reached.\n"
                      << "Performance increase: " << loss_increase << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MinimumPerformanceIncrease;
      }

      else if(gradient_norm <= gradient_norm_goal)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Gradient norm goal reached." << std::endl;  
         }

         stop_training = true;

         results_pointer->stopping_condition = GradientNormGoal;
      }

      else if(selection_failures >= maximum_selection_loss_decreases)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Maximum selection loss increases reached.\n"
                      << "Selection loss increases: "<< selection_failures << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MaximumSelectionPerformanceDecreases;
      }

      else if(iteration == maximum_iterations_number)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Maximum number of iterations reached." << std::endl;
         }

         stop_training = true;

         results_pointer->stopping_condition = MaximumIterationsNumber;
      }

      else if(elapsed_time >= maximum_time)
      {
         if(display)
         {
            std::cout << "Iteration " << iteration << ": Maximum training time reached." << std::endl;
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
                       << "Damping parameter: " << damping_parameter << "\n"
                       << "Elapsed time: " << elapsed_time << std::endl;

             if(selection_loss != 0)
             {
                std::cout << "Selection loss: " << selection_loss << std::endl;
             }
          }

          neural_network_pointer->set_parameters(parameters);

          results_pointer->resize_training_history(1+iteration);

         results_pointer->final_parameters = parameters;
         results_pointer->final_parameters_norm = parameters_norm;

         results_pointer->final_loss = loss;
         results_pointer->final_selection_loss = selection_loss;

         results_pointer->final_gradient = gradient;
         results_pointer->final_gradient_norm = gradient_norm;
   
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
                   << "Damping parameter: " << damping_parameter << "\n"
                   << "Elapsed time: " << elapsed_time << std::endl; 

         if(selection_loss != 0)
         {
            std::cout << "Selection loss: " << selection_loss << std::endl;
         }

      }

      // Update stuff

      old_loss = loss;
      old_selection_loss = selection_loss;

      // Set new parameters

      neural_network_pointer->set_parameters(parameters);
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


// void set_reserve_all_training_history(const bool&) method

void LevenbergMarquardtAlgorithm::set_reserve_all_training_history(const bool&)
{
   reserve_parameters_history = true;
   reserve_parameters_norm_history = true;

   reserve_loss_history = true;
   reserve_selection_loss_history = true;

   reserve_gradient_history = true;
   reserve_gradient_norm_history = true;
   reserve_Hessian_approximation_history = true;

   reserve_damping_parameter_history = true;
   reserve_elapsed_time_history = true;
}


// std::string write_training_algorithm_type(void) const method

std::string LevenbergMarquardtAlgorithm::write_training_algorithm_type(void) const
{
   return("LEVENBERG_MARQUARDT_ALGORITHM");
}


// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> LevenbergMarquardtAlgorithm::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

    // Damping parameter factor

    labels.push_back("Damping parameter factor");

    buffer.str("");
    buffer << damping_parameter_factor;

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

tinyxml2::XMLDocument* LevenbergMarquardtAlgorithm::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Training algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("LevenbergMarquardtAlgorithm");
   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Damping parameter

//   element = document->NewElement("DampingParameter");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << damping_parameter;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Minimum damping parameter.

//   element = document->NewElement("MinimumDampingParameter");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << minimum_damping_parameter;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Maximum damping parameter.

//   element = document->NewElement("MaximumDampingParameter");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << maximum_damping_parameter;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Damping parameter factor.

   element = document->NewElement("DampingParameterFactor");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << damping_parameter_factor;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

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

   // Reserve parameters history 

//   element = document->NewElement("ReserveParametersHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_parameters_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   // Reserve parameters norm history 

   element = document->NewElement("ReserveParametersNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_parameters_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

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

   // Reserve Hessian approximation history

//   element = document->NewElement("ReserveHessianApproximationHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_Hessian_approximation_history;

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
//   {
//       element = document->NewElement("SavePeriod");
//       root_element->LinkEndChild(element);

//       buffer.str("");
//       buffer << save_period;

//       text = document->NewText(buffer.str().c_str());
//       element->LinkEndChild(text);
//   }

   // Neural network file name
//   {
//       element = document->NewElement("NeuralNetworkFileName");
//       root_element->LinkEndChild(element);

//       text = document->NewText(neural_network_file_name.c_str());
//       element->LinkEndChild(text);
//   }

   // Display

//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the Levenberg Marquardt algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LevenbergMarquardtAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("LevenbergMarquardtAlgorithm");

    // Damping paramterer factor.

    file_stream.OpenElement("DampingParameterFactor");

    buffer.str("");
    buffer << damping_parameter_factor;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

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

/// Loads a Levenberg-Marquardt method object from a XML document.
/// Please mind about the format, wich is specified in the OpenNN manual. 
/// @param document TinyXML document containint the object data.

void LevenbergMarquardtAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("LevenbergMarquardtAlgorithm");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Levenberg-Marquardt algorithm element is NULL.\n";

        throw std::logic_error(buffer.str());
    }


    // Damping parameter

    const tinyxml2::XMLElement* damping_parameter_element = root_element->FirstChildElement("DampingParameter");

    if(damping_parameter_element)
    {
       const double new_damping_parameter = atof(damping_parameter_element->GetText());

       try
       {
          set_damping_parameter(new_damping_parameter);
       }
       catch(const std::logic_error& e)
       {
          std::cout << e.what() << std::endl;
       }
    }

    // Minimum damping parameter

    const tinyxml2::XMLElement* minimum_damping_parameter_element = root_element->FirstChildElement("MinimumDampingParameter");

    if(minimum_damping_parameter_element)
    {
       const double new_minimum_damping_parameter = atof(minimum_damping_parameter_element->GetText());

       try
       {
          set_minimum_damping_parameter(new_minimum_damping_parameter);
       }
       catch(const std::logic_error& e)
       {
          std::cout << e.what() << std::endl;
       }
    }

    // Maximum damping parameter

    const tinyxml2::XMLElement* maximum_damping_parameter_element = root_element->FirstChildElement("MaximumDampingParameter");

    if(maximum_damping_parameter_element)
    {
       const double new_maximum_damping_parameter = atof(maximum_damping_parameter_element->GetText());

       try
       {
          set_maximum_damping_parameter(new_maximum_damping_parameter);
       }
       catch(const std::logic_error& e)
       {
          std::cout << e.what() << std::endl;
       }
    }

    // Damping parameter factor

    const tinyxml2::XMLElement* damping_parameter_factor_element = root_element->FirstChildElement("DampingParameterFactor");

    if(damping_parameter_factor_element)
    {
       const double new_damping_parameter_factor = atof(damping_parameter_factor_element->GetText());

       try
       {
          set_damping_parameter_factor(new_damping_parameter_factor);
       }
       catch(const std::logic_error& e)
       {
          std::cout << e.what() << std::endl;
       }
    }

   // Warning parameters norm

   const tinyxml2::XMLElement* warning_parameters_norm_element = root_element->FirstChildElement("WarningParametersNorm");

   if(warning_parameters_norm_element)
   {
      const double new_warning_parameters_norm = atof(warning_parameters_norm_element->GetText());

      try
      {
         set_warning_parameters_norm(new_warning_parameters_norm);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Warning gradient norm 

   const tinyxml2::XMLElement* warning_gradient_norm_element = root_element->FirstChildElement("WarningGradientNorm");

   if(warning_gradient_norm_element)
   {
      const double new_warning_gradient_norm = atof(warning_gradient_norm_element->GetText());

      try
      {
         set_warning_gradient_norm(new_warning_gradient_norm);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Error parameters norm

   const tinyxml2::XMLElement* error_parameters_norm_element = root_element->FirstChildElement("ErrorParametersNorm");

   if(error_parameters_norm_element)
   {
      const double new_error_parameters_norm = atof(error_parameters_norm_element->GetText());

      try
      {
          set_error_parameters_norm(new_error_parameters_norm);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Error gradient norm 

   const tinyxml2::XMLElement* error_gradient_norm_element = root_element->FirstChildElement("ErrorGradientNorm");

   if(error_gradient_norm_element)
   {
      const double new_error_gradient_norm = atof(error_gradient_norm_element->GetText());

      try
      {
         set_error_gradient_norm(new_error_gradient_norm);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
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

   const tinyxml2::XMLElement* minimum_parameters_increment_norm_element = root_element->FirstChildElement("MinimumParametersIncrementNorm");

   if(minimum_parameters_increment_norm_element)
   {
      const double new_minimum_parameters_increment_norm = atof(minimum_parameters_increment_norm_element->GetText());

      try
      {
         set_minimum_parameters_increment_norm(new_minimum_parameters_increment_norm);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Minimum loss increase 

   const tinyxml2::XMLElement* minimum_loss_increase_element = root_element->FirstChildElement("MinimumPerformanceIncrease");

   if(minimum_loss_increase_element)
   {
      const double new_minimum_loss_increase = atof(minimum_loss_increase_element->GetText());

      try
      {
         set_minimum_loss_increase(new_minimum_loss_increase);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Performance goal 

   const tinyxml2::XMLElement* loss_goal_element = root_element->FirstChildElement("PerformanceGoal");

   if(loss_goal_element)
   {
      const double new_loss_goal = atof(loss_goal_element->GetText());

      try
      {
         set_loss_goal(new_loss_goal);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Gradient norm goal 

   const tinyxml2::XMLElement* gradient_norm_goal_element = root_element->FirstChildElement("GradientNormGoal");

   if(gradient_norm_goal_element)
   {
      const double new_gradient_norm_goal = atof(gradient_norm_goal_element->GetText());

      try
      {
         set_gradient_norm_goal(new_gradient_norm_goal);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Maximum selection loss decreases

   const tinyxml2::XMLElement* maximum_selection_loss_decreases_element = root_element->FirstChildElement("MaximumSelectionLossDecreases");

   if(maximum_selection_loss_decreases_element)
   {
      const size_t new_maximum_selection_loss_decreases = atoi(maximum_selection_loss_decreases_element->GetText());

      try
      {
         set_maximum_selection_loss_decreases(new_maximum_selection_loss_decreases);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Maximum iterations number 

   const tinyxml2::XMLElement* maximum_iterations_number_element = root_element->FirstChildElement("MaximumIterationsNumber");

   if(maximum_iterations_number_element)
   {
      const size_t new_maximum_iterations_number = atoi(maximum_iterations_number_element->GetText());

      try
      {
         set_maximum_iterations_number(new_maximum_iterations_number);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Maximum time 

   const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

   if(maximum_time_element)
   {
      const double new_maximum_time = atof(maximum_time_element->GetText());

      try
      {
         set_maximum_time(new_maximum_time);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Reserve parameters history 

   const tinyxml2::XMLElement* reserve_parameters_history_element = root_element->FirstChildElement("ReserveParametersHistory");

   if(reserve_parameters_history_element)
   {
      std::string new_reserve_parameters_history = reserve_parameters_history_element->GetText(); 

      try
      {
         set_reserve_parameters_history(new_reserve_parameters_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Reserve parameters norm history 

   const tinyxml2::XMLElement* reserve_parameters_norm_history_element = root_element->FirstChildElement("ReserveParametersNormHistory");

   if(reserve_parameters_norm_history_element)
   {
      const std::string new_reserve_parameters_norm_history = reserve_parameters_norm_history_element->GetText();

      try
      {
         set_reserve_parameters_norm_history(new_reserve_parameters_norm_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Reserve loss history

   const tinyxml2::XMLElement* reserve_loss_history_element = root_element->FirstChildElement("ReservePerformanceHistory");

   if(reserve_loss_history_element)
   {
      const std::string new_reserve_loss_history = reserve_loss_history_element->GetText();

      try
      {
         set_reserve_loss_history(new_reserve_loss_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Reserve selection loss history

   const tinyxml2::XMLElement* reserve_selection_loss_history_element = root_element->FirstChildElement("ReserveSelectionLossHistory");

   if(reserve_selection_loss_history_element)
   {
      const std::string new_reserve_selection_loss_history = reserve_selection_loss_history_element->GetText();

      try
      {
         set_reserve_selection_loss_history(new_reserve_selection_loss_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;
      }
   }

   // Reserve gradient history 

   const tinyxml2::XMLElement* reserve_gradient_history_element = root_element->FirstChildElement("ReserveGradientHistory");

   if(reserve_gradient_history_element)
   {
      std::string new_reserve_gradient_history = reserve_gradient_history_element->GetText(); 

      try
      {
         set_reserve_gradient_history(new_reserve_gradient_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Reserve gradient norm history 

   const tinyxml2::XMLElement* reserve_gradient_norm_history_element = root_element->FirstChildElement("ReserveGradientNormHistory");

   if(reserve_gradient_norm_history_element)
   {
      const std::string new_reserve_gradient_norm_history = reserve_gradient_norm_history_element->GetText();

      try
      {
         set_reserve_gradient_norm_history(new_reserve_gradient_norm_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Reserve elapsed time history 

   const tinyxml2::XMLElement* reserve_elapsed_time_history_element = root_element->FirstChildElement("ReserveElapsedTimeHistory");

   if(reserve_elapsed_time_history_element)
   {
      const std::string new_reserve_elapsed_time_history = reserve_elapsed_time_history_element->GetText();

      try
      {
         set_reserve_elapsed_time_history(new_reserve_elapsed_time_history != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }

   // Display period

   const tinyxml2::XMLElement* display_period_element = root_element->FirstChildElement("DisplayPeriod");

   if(display_period_element)
   {
      const size_t new_display_period = atoi(display_period_element->GetText());

      try
      {
         set_display_period(new_display_period);
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
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

   const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

   if(display_element)
   {
      const std::string new_display = display_element->GetText();

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


// Vector<double> perform_Householder_QR_decomposition(Matrix<double>&, Matrix<double>&) const method

/// Uses Eigen to solve the system of equations by means of the Householder QR decomposition.

Vector<double> LevenbergMarquardtAlgorithm::perform_Householder_QR_decomposition(const Matrix<double>& A, const Vector<double>& b) const
{
    const size_t n = A.get_rows_number();

    Vector<double> x(n);

    const Eigen::Map<Eigen::MatrixXd> A_eigen((double*)A.data(), n, n);
    const Eigen::Map<Eigen::VectorXd> b_eigen((double*)b.data(), n);
    Eigen::Map<Eigen::VectorXd> x_eigen(x.data(), n);

    x_eigen = A_eigen.colPivHouseholderQr().solve(b_eigen);

    return(x);
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

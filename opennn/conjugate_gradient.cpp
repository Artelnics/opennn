//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "conjugate_gradient.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a conjugate gradient optimization algorithm object not associated to any loss index object.
/// It also initializes the class members to their default values.

ConjugateGradient::ConjugateGradient() : OptimizationAlgorithm()
{
   set_default();
}


/// Destructor.
/// It creates a conjugate gradient optimization algorithm associated to a loss index object.
/// It also initializes the rest of class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

ConjugateGradient::ConjugateGradient(LossIndex* new_loss_index_pointer)
: OptimizationAlgorithm(new_loss_index_pointer)
{
   learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);   

   set_default();
}


/// XML constructor.
/// It creates a conjugate gradient optimization algorithm not associated to any loss index object.
/// It also loads the class members from a XML document.
/// @param conjugate_gradient_document TinyXML document with the members of a conjugate gradient object.

ConjugateGradient::ConjugateGradient(const tinyxml2::XMLDocument& conjugate_gradient_document)
 : OptimizationAlgorithm(conjugate_gradient_document)
{
   set_default();

   from_XML(conjugate_gradient_document);
}


/// Destructor.

ConjugateGradient::~ConjugateGradient()
{
}


/// Returns a constant reference to the learning rate algorithm object inside the conjugate gradient method object. 

const LearningRateAlgorithm& ConjugateGradient::get_learning_rate_algorithm() const
{
   return(learning_rate_algorithm);
}


/// Returns a pointer to the learning rate algorithm object inside the conjugate gradient method object. 

LearningRateAlgorithm* ConjugateGradient::get_learning_rate_algorithm_pointer()
{
   return &learning_rate_algorithm;
}


/// Returns the conjugate gradient training direction method used for training.

const ConjugateGradient::TrainingDirectionMethod& ConjugateGradient::get_training_direction_method() const
{
   return(training_direction_method);
}


/// Returns a string with the name of the training direction.  

string ConjugateGradient::write_training_direction_method() const
{
   switch(training_direction_method)
   {
      case PR:
      {
         return "PR";
      }

      case FR:
      {
         return "FR";
      }
   }

   return string();
}


/// Returns the minimum value for the norm of the parameters vector at wich a warning message is written to the screen. 

const double& ConjugateGradient::get_warning_parameters_norm() const
{
   return(warning_parameters_norm);       
}


/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written to the screen. 

const double& ConjugateGradient::get_warning_gradient_norm() const
{
   return(warning_gradient_norm);       
}


/// Returns the training rate value at wich a warning message is written to the screen during line minimization.

const double& ConjugateGradient::get_warning_learning_rate() const
{
   return(warning_learning_rate);
}


/// Returns the value for the norm of the parameters vector at wich an error message is written to the screen and the program exits. 

const double& ConjugateGradient::get_error_parameters_norm() const
{
   return(error_parameters_norm);
}


/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits. 

const double& ConjugateGradient::get_error_gradient_norm() const
{
   return(error_gradient_norm);
}


/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.

const double& ConjugateGradient::get_error_learning_rate() const
{
   return(error_learning_rate);
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training. 

const double& ConjugateGradient::get_minimum_parameters_increment_norm() const
{
   return(minimum_parameters_increment_norm);
}


/// Returns the minimum loss improvement during training.  

const double& ConjugateGradient::get_minimum_loss_increase() const
{
   return(minimum_loss_decrease);
}


/// Returns the goal value for the loss. 
/// This is used as a stopping criterion when training a neural network

const double& ConjugateGradient::get_loss_goal() const
{
   return(loss_goal);
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network

const double& ConjugateGradient::get_gradient_norm_goal() const
{
   return(gradient_norm_goal);
}


/// Returns the maximum number of selection failures during the training process. 

const size_t& ConjugateGradient::get_maximum_selection_error_increases() const
{
   return(maximum_selection_error_decreases);
}


/// Returns the maximum number of epochs for training.

const size_t& ConjugateGradient::get_maximum_epochs_number() const
{
   return(maximum_epochs_number);
}


/// Returns the maximum training time.  

const double& ConjugateGradient::get_maximum_time() const
{
   return(maximum_time);
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& ConjugateGradient::get_return_minimum_selection_error_neural_network() const
{
    return(return_minimum_selection_error_neural_network);
}


/// Returns true if the selection error decrease stopping criteria has to be taken in account, false otherwise.

const bool& ConjugateGradient::get_apply_early_stopping() const
{
    return(apply_early_stopping);
}


/// Returns true if the error history vector is to be reserved, and false otherwise.

const bool& ConjugateGradient::get_reserve_training_error_history() const
{
   return(reserve_training_error_history);
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& ConjugateGradient::get_reserve_selection_error_history() const
{
   return(reserve_selection_error_history);
}


/// Sets a pointer to a loss index object to be associated to the conjugate gradient object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void ConjugateGradient::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


/// Sets a new training direction method to be used for training. 
/// @param new_training_direction_method Conjugate gradient training direction method.

void ConjugateGradient::set_training_direction_method
(const ConjugateGradient::TrainingDirectionMethod& new_training_direction_method)
{
   training_direction_method = new_training_direction_method;   
}


/// Sets a new conjugate gradient training direction from a string representation. 
/// Possible values are:
/// <ul>
/// <li> "PR"
/// <li> "FR"
/// </ul>
/// @param new_training_direction_method_name String with the name of the training direction method.

void ConjugateGradient::set_training_direction_method(const string& new_training_direction_method_name)
{
   if(new_training_direction_method_name == "PR")
   {
      training_direction_method = PR;
   }
   else if(new_training_direction_method_name == "FR")
   {
      training_direction_method = FR;
   }
   else
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_training_direction_method(const string&) method.\n"
			 << "Unknown training direction method: " << new_training_direction_method_name << ".\n";
   
      throw logic_error(buffer.str());
   }
}


/// Makes the training history of all variables to reseved or not in memory when training.
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> loss.
/// <li> Gradient. 
/// <li> Gradient norm. 
/// <li> selection error.
/// <li> Training direction.
/// <li> Training direction norm. 
/// <li> Training rate.
/// </ul>
///
/// @param new_reserve_all_training_history True if all training history variables are to be reserved, 
/// false otherwise.

void ConjugateGradient::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{

   // Loss index

   reserve_training_error_history = new_reserve_all_training_history;
   reserve_selection_error_history = new_reserve_all_training_history;

}


/// Sets the default values into a conjugate gradient object.
/// Training operators:
/// <ul>
/// <li> Training direction method = Polak-Ribiere;
/// <li> Training rate method = Brent;
/// </ul>
/// Training parameters:
/// <ul>
/// <li> First training rate: 1.0.
/// <li> Bracketing factor: 2.0.
/// <li> Training rate tolerance: 1.0e-3.
/// </ul>
/// Stopping criteria:
/// <ul> 
/// <li> Loss goal: -numeric_limits<double>::max().
/// <li> Gradient norm goal: 0.0.
/// <li> Maximum training time: 1.0e6.
/// <li> Maximum number of iterations: 100. 
/// </ul> 
/// User stuff:
/// <ul>
/// <li> Warning training rate: 1.0e6.
/// <li> Error training rate: 1.0e12.
/// <li> Display: true.
/// <li> Display period: 10.
/// <li> Save period: 0.
/// </ul>
/// Reserve:
/// <ul>
/// <li> Reserve training direction history: false.
/// <li> Reserve training direction norm history: false.
/// <li> Reserve training rate history: false.
/// </ul>
///

void ConjugateGradient::set_default()
{
   // TRAINING PARAMETERS

   warning_parameters_norm = 1.0e6;
   warning_gradient_norm = 1.0e6;   
   warning_learning_rate = 1.0e6;

   error_parameters_norm = 1.0e9;
   error_gradient_norm = 1.0e9;
   error_learning_rate = 1.0e9;

   // Stopping criteria

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

   reserve_training_error_history = true;
   reserve_selection_error_history = false;

   // UTILITIES

   display = true;
   display_period = 5;

   training_direction_method = PR;
}


/// Sets a new value for the parameters vector norm at which a warning message is written to the 
/// screen. 
/// @param new_warning_parameters_norm Warning norm of parameters vector value. 

void ConjugateGradient::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
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

void ConjugateGradient::set_warning_gradient_norm(const double& new_warning_gradient_norm)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_gradient_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_warning_gradient_norm(const double&) method.\n"
             << "Warning gradient norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set warning gradient norm

   warning_gradient_norm = new_warning_gradient_norm;     
}


/// Sets a new training rate value at which a warning message is written to the screen during line
/// minimization.
/// @param new_warning_learning_rate Warning training rate value.

void ConjugateGradient::set_warning_learning_rate(const double& new_warning_learning_rate)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_learning_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_warning_learning_rate(const double&) method.\n"
             << "Warning training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   warning_learning_rate = new_warning_learning_rate;
}


/// Sets a new value for the parameters vector norm at which an error message is written to the 
/// screen and the program exits. 
/// @param new_error_parameters_norm Error norm of parameters vector value. 

void ConjugateGradient::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
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

void ConjugateGradient::set_error_gradient_norm(const double& new_error_gradient_norm)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_gradient_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
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
/// @param new_error_learning_rate Error training rate value.

void ConjugateGradient::set_error_learning_rate(const double& new_error_learning_rate)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_learning_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_error_learning_rate(const double&) method.\n"
             << "Error training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set error training rate

   error_learning_rate = new_error_learning_rate;
}


/// Sets a new value for the minimum parameters increment norm stopping criterion. 
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training. 

void ConjugateGradient::set_minimum_parameters_increment_norm(const double& new_minimum_parameters_increment_norm)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_parameters_increment_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
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

void ConjugateGradient::set_minimum_loss_decrease(const double& new_minimum_loss_increase)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_minimum_loss_increase < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_minimum_loss_decrease(const double&) method.\n"
             << "Minimum loss improvement must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set minimum loss improvement

   minimum_loss_decrease = new_minimum_loss_increase;
}


/// Sets a new goal value for the loss. 
/// This is used as a stopping criterion when training a neural network
/// @param new_loss_goal Goal value for the loss.

void ConjugateGradient::set_loss_goal(const double& new_loss_goal)
{
   loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient. 
/// This is used as a stopping criterion when training a neural network
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void ConjugateGradient::set_gradient_norm_goal(const double& new_gradient_norm_goal)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_gradient_norm_goal < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
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

void ConjugateGradient::set_maximum_selection_error_increases(const size_t& new_maximum_selection_error_increases)
{
   maximum_selection_error_decreases = new_maximum_selection_error_increases;
}


/// Sets a maximum number of epochs for training.
/// @param new_maximum_iterations_number Maximum number of iterations for training.

void ConjugateGradient::set_maximum_epochs_number(const size_t& new_maximum_epochs_number)
{
   maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new maximum training time.  
/// @param new_maximum_time Maximum training time.

void ConjugateGradient::set_maximum_time(const double& new_maximum_time)
{
   

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_time < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
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

void ConjugateGradient::set_return_minimum_selection_error_neural_network(const bool& new_return_minimum_selection_error_neural_network)
{
   return_minimum_selection_error_neural_network = new_return_minimum_selection_error_neural_network;
}


/// Makes the selection error decrease stopping criteria has to be taken in account or not.
/// @param new_apply_early_stopping True if the selection error decrease stopping criteria has to be taken in account, false otherwise.

void ConjugateGradient::set_apply_early_stopping(const bool& new_apply_early_stopping)
{
    apply_early_stopping = new_apply_early_stopping;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_training_error_history True if the loss history vector is to be reserved, false otherwise.

void ConjugateGradient::set_reserve_training_error_history(const bool& new_reserve_training_error_history)
{
   reserve_training_error_history = new_reserve_training_error_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector. 
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void ConjugateGradient::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
   reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Sets a new number of iterations between the training showing progress. 
/// @param new_display_period
/// Number of iterations between the training showing progress. 

void ConjugateGradient::set_display_period(const size_t& new_display_period)
{
   

   #ifdef __OPENNN_DEBUG__ 
     
   if(new_display_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_display_period(const double&) method.\n"
             << "Display period must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   display_period = new_display_period;
}


/// Sets a new number of iterations between the training saving progress.
/// @param new_save_period
/// Number of iterations between the training saving progress.

void ConjugateGradient::set_save_period(const size_t& new_save_period)
{
   #ifdef __OPENNN_DEBUG__

   if(new_save_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_save_period(const double&) method.\n"
             << "Save period must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   save_period = new_save_period;
}


/// Returns the Fletcher-Reeves parameter used to calculate the training direction.
/// @param old_gradient Previous error function gradient.
/// @param gradient: Current error function gradient.

double ConjugateGradient::calculate_FR_parameter(const Vector<double>& old_gradient, const Vector<double>& gradient) const
{
    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "double calculate_FR_parameter(const Vector<double>&, const Vector<double>&) const method.\n"

              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    const size_t old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "double calculate_FR_parameter(const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "double calculate_FR_parameter(const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

   double FR_parameter = 0.0;

   const double numerator = dot(gradient, gradient);
   const double denominator = dot(old_gradient, old_gradient);

   // Prevent a possible division by 0

   if(denominator == 0.0)
   {
      FR_parameter = 0.0;
   }
   else
   {
      FR_parameter = numerator/denominator;
   }

   // Bound the Fletcher-Reeves parameter between 0 and 1

   if(FR_parameter < 0.0)
      FR_parameter = 0.0;

   if(FR_parameter > 1.0)
      FR_parameter = 1.0;

   return(FR_parameter);
}


/// Returns the Polak-Ribiere parameter used to calculate the training direction.
/// @param old_gradient Previous error function gradient.
/// @param gradient Current error function gradient.

double ConjugateGradient::calculate_PR_parameter(const Vector<double>& old_gradient, const Vector<double>& gradient) const
{
    

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "double calculate_PR_parameter(const Vector<double>&, const Vector<double>&) const method.\n"

              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    const size_t old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "double calculate_PR_parameter(const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "double calculate_PR_parameter(const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

   double PR_parameter = 0.0;

   const double numerator = dot(gradient-old_gradient, gradient);
   const double denominator = dot(old_gradient, old_gradient);

   // Prevent a possible division by 0

   if(denominator == 0.0)
   {
      PR_parameter = 0.0;
   }
   else
   {
      PR_parameter = numerator/denominator;
   }

   // Bound the Polak-Ribiere parameter between 0 and 1

   if(PR_parameter < 0.0)
   {
      PR_parameter = 0.0;
   }

   if(PR_parameter > 1.0)
   {
      PR_parameter = 1.0;
   }

   return(PR_parameter);
}


/// Returns the training direction using the Polak-Ribiere update.
/// @param old_gradient Previous error function gradient.
/// @param gradient Current error function gradient.
/// @param old_training_direction Previous training direction vector.

Vector<double> ConjugateGradient::calculate_PR_training_direction
(const Vector<double>& old_gradient, const Vector<double>& gradient, const Vector<double>& old_training_direction) const
{
    

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_PR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    const size_t old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_PR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_PR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t old_training_direction_size = old_training_direction.size();

    if(old_training_direction_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_PR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old training direction(" << old_training_direction_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

   const double PR_parameter = calculate_PR_parameter(old_gradient, gradient);

   const Vector<double> gradient_descent_term = calculate_gradient_descent_training_direction(gradient);
   const Vector<double> conjugate_direction_term = old_training_direction*PR_parameter;

   const Vector<double> PR_training_direction = gradient_descent_term + conjugate_direction_term;

   const double PR_training_direction_norm = l2_norm(PR_training_direction);

   return(PR_training_direction/PR_training_direction_norm);
}


/// Returns the training direction using the Fletcher-Reeves update.
/// @param old_gradient Previous error function gradient.
/// @param gradient Current error function gradient.
/// @param old_training_direction Previous training direction vector.

Vector<double> ConjugateGradient::calculate_FR_training_direction
(const Vector<double>& old_gradient, const Vector<double>& gradient, const Vector<double>& old_training_direction) const
{
    

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_FR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    const size_t old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_FR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_FR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t old_training_direction_size = old_training_direction.size();

    if(old_training_direction_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_FR_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old training direction(" << old_training_direction_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

   const double FR_parameter = calculate_FR_parameter(old_gradient, gradient);

   const Vector<double> gradient_descent_term = calculate_gradient_descent_training_direction(gradient);
   const Vector<double> conjugate_direction_term = old_training_direction*FR_parameter;

   const Vector<double> FR_training_direction = gradient_descent_term + conjugate_direction_term;

   const double FR_training_direction_norm = l2_norm(FR_training_direction);

   return(FR_training_direction/FR_training_direction_norm);
}


/// Returns the conjugate gradient training direction, which has been previously normalized.
/// @param old_gradient Gradient vector in the previous iteration. 
/// @param gradient Current gradient vector. 
/// @param old_training_direction Training direction in the previous iteration. 

Vector<double> ConjugateGradient::calculate_training_direction
(const Vector<double>& old_gradient, const Vector<double>& gradient, const Vector<double>& old_training_direction) const
{
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const size_t old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    const size_t old_training_direction_size = old_training_direction.size();

    if(old_training_direction_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_training_direction(const Vector<double>&, const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of old training direction(" << old_training_direction_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

   switch(training_direction_method)
   {
      case FR:
      {
         return(calculate_FR_training_direction(old_gradient, gradient, old_training_direction));
      }    

      case PR:
      {
         return(calculate_PR_training_direction(old_gradient, gradient, old_training_direction));
      }            
   }

   // Never reach here

   const Vector<double> training_direction(parameters_number, 0.0);

   return(training_direction);

}


/// Returns the gradient descent training direction, which is the negative of the normalized gradient. 
/// @param gradient Gradient vector.

Vector<double> ConjugateGradient::calculate_gradient_descent_training_direction(const Vector<double>& gradient) const
{
    

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_gradient_descent_training_direction(const Vector<double>&) const method.\n"
              << "Loss index pointer is nullptr.\n";

       throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    const size_t gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
       buffer << "OpenNN Exception: ConjugateGradient class.\n"
              << "Vector<double> calculate_gradient_descent_training_direction(const Vector<double>&) const method.\n"
              << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    return normalized(gradient)*(-1.0);
}


/// Trains a neural network with an associated loss index according to the conjugate gradient algorithm.
/// Training occurs according to the training operators, training parameters and stopping criteria.

OptimizationAlgorithm::Results ConjugateGradient::perform_training()
{
   

   check();

   // Start training

   if(display) cout << "Training with conjugate gradient...\n";

   Results results; //= new ConjugateGradientResults(this);
   results.resize_training_history(maximum_epochs_number+1);

   // Elapsed time

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time = 0.0;

   // Data set stuff

   DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   const size_t selection_instances_number = data_set_pointer->get_selection_instances_number();

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> parameters = neural_network_pointer->get_parameters();
   double parameters_norm = 0.0;

   // Loss index stuff

   double training_loss = 0.0;
   double old_training_loss = 0.0;
   double training_loss_decrease = 0.0;
      
   Vector<double> gradient(parameters_number);
   double gradient_norm = 0.0;

   double selection_error = 0.0;
   double old_selection_error = 0.0;

   string information;

   // Optimization algorithm stuff 

   Vector<double> parameters_increment(parameters_number);
   double parameters_increment_norm;

   Vector<double> old_gradient(parameters_number);
   Vector<double> training_direction(parameters_number);
   Vector<double> old_training_direction(parameters_number);

   double training_slope;

   double initial_learning_rate = 0.0;
   double learning_rate = 0.0;
   double old_learning_rate = 0.0;

   pair<double,double> directional_point(2, 0.0);

   Vector<double> minimum_selection_error_parameters(parameters_number);
   double minimum_selection_error = 0.0;

   bool stop_training = false;

   size_t selection_failures = 0;
   
   double first_learning_rate = 0.01;

   // Main loop    
   
   for(size_t epoch = 0; epoch <= maximum_epochs_number; epoch++)
   {
      // Neural network

      parameters = neural_network_pointer->get_parameters();

      parameters_norm = l2_norm(parameters);

      if(parameters_norm >= error_parameters_norm)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: ConjugateGradient class.\n"
                << "ConjugateGradientResults* perform_training() method.\n"
                << "Parameters norm is greater than error parameters norm.\n";
 
         throw logic_error(buffer.str());
      }
      else if(display && parameters_norm >= warning_parameters_norm)
      {
         cout << "OpenNN Warning: Parameters norm is " << parameters_norm << ".\n";          
      }

      // Loss index stuff

      if(epoch == 0)
      {      
         training_loss = loss_index_pointer->calculate_training_loss();
         training_loss_decrease = 0.0;
      }
      else
      {
         training_loss = directional_point.second;
         training_loss_decrease = training_loss - old_training_loss;
      }

      gradient = loss_index_pointer->calculate_training_loss_gradient();

      gradient_norm = l2_norm(gradient);

      if(display && gradient_norm >= warning_gradient_norm)
      {
         cout << "OpenNN Warning: Gradient norm is " << gradient_norm << ".\n";          
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

      // Optimization algorithm 

      if(epoch == 0 || epoch % parameters_number == 0)
      {
         // Gradient descent training direction

         training_direction = calculate_gradient_descent_training_direction(gradient);
      }
      else
      {
         // Conjugate gradient training direction

         training_direction = calculate_training_direction(old_gradient, gradient, old_training_direction);
      }

      // Calculate loss training_slope

      training_slope = dot(gradient/gradient_norm, training_direction);

      // Check for a descent direction 

      if(training_slope >= 0.0)
      {
         // Reset training direction

         training_direction = calculate_gradient_descent_training_direction(gradient);

         cout << "Epoch " << epoch << ": Gradient descent training direction" << endl;
      }

      // Get initial training rate

      if(epoch == 0)
      {
         initial_learning_rate = first_learning_rate;
      }
      else
      {
         initial_learning_rate = old_learning_rate;
      }

      directional_point = learning_rate_algorithm.calculate_directional_point(training_loss, training_direction, initial_learning_rate);

      learning_rate = directional_point.first;

      if(epoch != 0
      && abs(learning_rate) < numeric_limits<double>::min())
      {
         // Reset training direction

         training_direction = calculate_gradient_descent_training_direction(gradient);         

         directional_point = learning_rate_algorithm.calculate_directional_point(training_loss, training_direction, first_learning_rate);

         learning_rate = directional_point.first;
      }

      parameters_increment = training_direction*learning_rate;
      parameters_increment_norm = l2_norm(parameters_increment);
      
      // Elapsed time

      time(&current_time);
      elapsed_time = difftime(current_time, beginning_time);

      // Training error history index

      if(reserve_training_error_history)
      {
         results.training_error_history[epoch] = training_loss;
      }

      if(reserve_selection_error_history)
      {
         results.selection_error_history[epoch] = selection_error;
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

         results.stopping_condition = MinimumParametersIncrementNorm;
      }

      else if(epoch != 0 && training_loss_decrease >= minimum_loss_decrease)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Minimum loss decrease (" << minimum_loss_decrease << ") reached.\n";
            cout << "Loss decrease: " << training_loss_decrease << endl;
         }

         stop_training = true;

         results.stopping_condition = MinimumLossDecrease;
      }

      else if(training_loss <= loss_goal)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Loss goal reached.\n";
         }

         stop_training = true;

         results.stopping_condition = LossGoal;
      }

      else if(gradient_norm <= gradient_norm_goal)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Gradient norm goal reached.\n";
         }

         stop_training = true;

         results.stopping_condition = GradientNormGoal;
      }

      else if(selection_failures > maximum_selection_error_decreases && apply_early_stopping)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Maximum selection failures reached.\n"
                      << "Selection failures: " << selection_failures << endl;
         }

         stop_training = true;

         results.stopping_condition = MaximumSelectionErrorIncreases;
      }

      else if(epoch == maximum_epochs_number)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Maximum number of iterations reached.\n";
         }

         stop_training = true;

         results.stopping_condition = MaximumEpochsNumber;
      }

      else if(elapsed_time >= maximum_time)
      {
         if(display)
         {
            cout << "Epoch " << epoch << ": Maximum training time reached.\n";
         }

         stop_training = true;

         results.stopping_condition = MaximumTime;
      }

      if(epoch != 0 && epoch % save_period == 0)
      {
            neural_network_pointer->save(neural_network_file_name);
      }

      if(stop_training)
      {
          if(display)
          {
             information = loss_index_pointer->write_information();

             cout << "Parameters norm: " << parameters_norm << "\n"
                  << "Training loss: " << training_loss << "\n"
                  << "Gradient norm: " << gradient_norm << "\n"
                  << information
                  << "Training rate: " << learning_rate << "\n"
                  << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

             if(selection_instances_number > 0)
             {
                cout << "Selection error: " << selection_error << endl;
             }
          }

         results.resize_training_history(1+epoch);

         results.final_parameters = parameters;
         results.final_parameters_norm = parameters_norm;

         results.final_training_error = training_loss;
         results.final_selection_error = selection_error;

         results.final_gradient_norm = gradient_norm;

         results.elapsed_time = elapsed_time;

         results.epochs_number = epoch;
  
         break;
      }

      else if(display && epoch % display_period == 0)
      {
         information = loss_index_pointer->write_information();

         cout << "Epoch " << epoch << ";\n"
                   << "Parameters norm: " << parameters_norm << "\n"
                   << "Training loss: " << training_loss << "\n"
                   << "Gradient norm: " << gradient_norm << "\n"
                   << information 
                   << "Training rate: " << learning_rate << "\n"
                   << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

         if(selection_instances_number > 0)
         {
            cout << "Selection error: " << selection_error << endl;
         }
      }

      // Set new parameters

      parameters += parameters_increment;

      neural_network_pointer->set_parameters(parameters);

      // Update stuff

      old_training_loss = training_loss;
      old_gradient = gradient;
      old_selection_error = selection_error;

      old_training_direction = training_direction;   
      old_learning_rate = learning_rate;
   } 

   if(return_minimum_selection_error_neural_network)
   {
       parameters = minimum_selection_error_parameters;
       parameters_norm = l2_norm(parameters);

       neural_network_pointer->set_parameters(parameters);

       training_loss = loss_index_pointer->calculate_training_loss();
       selection_error = minimum_selection_error;
   }

   results.final_parameters = parameters;
   results.final_parameters_norm = parameters_norm;

   results.final_training_error = training_loss;
   results.final_selection_error = selection_error;

   results.final_gradient_norm = gradient_norm;

   results.elapsed_time = elapsed_time;

   return results;

}


/// Perform the training.

void ConjugateGradient::perform_training_void()
{
    perform_training();
}

/// Write a string with best algorithm type for the model.

string ConjugateGradient::write_optimization_algorithm_type() const
{
   return "CONJUGATE_GRADIENT";
}


/// Writes as matrix of strings the most representative atributes.

Matrix<string> ConjugateGradient::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

    // Training direction method

    labels.push_back("Training direction method");

    const string training_direction_method_string = write_training_direction_method();

    values.push_back(training_direction_method_string);

   // Training rate method

   labels.push_back("Training rate method");

   const string learning_rate_method = learning_rate_algorithm.write_learning_rate_method();

   values.push_back(learning_rate_method);

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

   labels.push_back("Maximum iterations number");

   buffer.str("");
   buffer << maximum_epochs_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Reserve training error history

   labels.push_back("Reserve training error history");

   buffer.str("");

   if(reserve_training_error_history)
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

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return string_matrix;
}


/// Serializes the conjugate gradient object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element.

tinyxml2::XMLDocument* ConjugateGradient::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Conjugate gradient

   tinyxml2::XMLElement* root_element = document->NewElement("ConjugateGradient");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;


   // Training direction method
   {
      element = document->NewElement("TrainingDirectionMethod");
      root_element->LinkEndChild(element);

      text = document->NewText(write_training_direction_method().c_str());
      element->LinkEndChild(text);
   }

   // Training rate algorithm
   {
      const tinyxml2::XMLDocument* learning_rate_algorithm_document = learning_rate_algorithm.to_XML();

      const tinyxml2::XMLElement* learning_rate_algorithm_element = learning_rate_algorithm_document->FirstChildElement("LearningRateAlgorithm");

      tinyxml2::XMLNode* node = learning_rate_algorithm_element->DeepClone(document);

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
//   {
//      element = document->NewElement("WarningParametersNorm");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << warning_parameters_norm;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   // Warning gradient norm
//   {
//      element = document->NewElement("WarningGradientNorm");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << warning_gradient_norm;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   // Warning training rate
//   {
//      element = document->NewElement("WarningLearningRate");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << warning_learning_rate;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   // Error parameters norm
//   {
//      element = document->NewElement("ErrorParametersNorm");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << error_parameters_norm;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   // Error gradient norm
//   {
//      element = document->NewElement("ErrorGradientNorm");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << error_gradient_norm;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   // Error training rate
//   {
//      element = document->NewElement("ErrorLearningRate");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << error_learning_rate;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   // Minimum parameters increment norm
   {
      element = document->NewElement("MinimumParametersIncrementNorm");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << minimum_parameters_increment_norm;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Minimum loss decrease
   {
      element = document->NewElement("MinimumLossDecrease");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << minimum_loss_decrease;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Loss goal
   {
      element = document->NewElement("LossGoal");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << loss_goal;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Gradient norm goal
   {
      element = document->NewElement("GradientNormGoal");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << gradient_norm_goal;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Maximum selection error decreases
   {
      element = document->NewElement("MaximumSelectionErrorIncreases");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << maximum_selection_error_decreases;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Maximum iterations number
   {
      element = document->NewElement("MaximumEpochsNumber");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << maximum_epochs_number;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Maximum time
   {
      element = document->NewElement("MaximumTime");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << maximum_time;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Reserve training error history
   {
      element = document->NewElement("ReserveTrainingErrorHistory");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << reserve_training_error_history;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Reserve selection error history
   {
      element = document->NewElement("ReserveSelectionErrorHistory");
      root_element->LinkEndChild(element);

      buffer.str("");
      buffer << reserve_selection_error_history;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Display period
//   {
//      element = document->NewElement("DisplayPeriod");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display_period;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

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
//   {
//      element = document->NewElement("Display");
//      root_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return document;
}


/// Serializes the conjugate gradient object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this element.

void ConjugateGradient::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
   ostringstream buffer;

   //file_stream.OpenElement("ConjugateGradient");

   // Training direction method

   {
       file_stream.OpenElement("TrainingDirectionMethod");

       file_stream.PushText(write_training_direction_method().c_str());

       file_stream.CloseElement();
   }

   // Training rate algorithm

   learning_rate_algorithm.write_XML(file_stream);

//   // Return minimum selection error neural network

   {
       file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

       buffer.str("");
       buffer << return_minimum_selection_error_neural_network;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Apply early stopping

   {
       file_stream.OpenElement("ApplyEarlyStopping");

       buffer.str("");
       buffer << apply_early_stopping;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Minimum parameters increment norm

   {
       file_stream.OpenElement("MinimumParametersIncrementNorm");

       buffer.str("");
       buffer << minimum_parameters_increment_norm;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Minimum loss decrease

   {
       file_stream.OpenElement("MinimumLossDecrease");

       buffer.str("");
       buffer << minimum_loss_decrease;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Loss goal

   {
       file_stream.OpenElement("LossGoal");

       buffer.str("");
       buffer << loss_goal;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Gradient norm goal

   {
       file_stream.OpenElement("GradientNormGoal");

       buffer.str("");
       buffer << gradient_norm_goal;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Maximum selection error decreases

   {
       file_stream.OpenElement("MaximumSelectionErrorIncreases");

       buffer.str("");
       buffer << maximum_selection_error_decreases;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Maximum iterations number

   {
       file_stream.OpenElement("MaximumEpochsNumber");

       buffer.str("");
       buffer << maximum_epochs_number;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Maximum time

   {
       file_stream.OpenElement("MaximumTime");

       buffer.str("");
       buffer << maximum_time;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Reserve training error history

   {
       file_stream.OpenElement("ReserveTrainingErrorHistory");

       buffer.str("");
       buffer << reserve_training_error_history;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Reserve selection error history

   {
       file_stream.OpenElement("ReserveSelectionErrorHistory");

       buffer.str("");
       buffer << reserve_selection_error_history;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   //file_stream.CloseElement();
}


/// Deserializes the conjugate gradient object from a XML document of the TinyXML library.
/// @param document TinyXML document containing the member data.

void ConjugateGradient::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("ConjugateGradient");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Conjugate gradient element is nullptr.\n";

        throw logic_error(buffer.str());
    }

  // Training direction method
  {
     const tinyxml2::XMLElement* training_direction_method_element = root_element->FirstChildElement("TrainingDirectionMethod");

     if(training_direction_method_element)
     {
        const string new_training_direction_method = training_direction_method_element->GetText();

        try
        {
           set_training_direction_method(new_training_direction_method);
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }

  // Learning rate algorithm
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
     const tinyxml2::XMLElement* warning_parameters_norm_element = root_element->FirstChildElement("WarningParametersNorm");

     if(warning_parameters_norm_element)
     {
        const double new_warning_parameters_norm = atof(warning_parameters_norm_element->GetText());

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
     const tinyxml2::XMLElement* warning_gradient_norm_element = root_element->FirstChildElement("WarningGradientNorm");

     if(warning_gradient_norm_element)
     {
        const double new_warning_gradient_norm = atof(warning_gradient_norm_element->GetText());

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
     const tinyxml2::XMLElement* warning_learning_rate_element = root_element->FirstChildElement("WarningLearningRate");

     if(warning_learning_rate_element)
     {
        const double new_warning_learning_rate = atof(warning_learning_rate_element->GetText());

        try
        {
           set_warning_learning_rate(new_warning_learning_rate);
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }

  // Error parameters norm
  {
     const tinyxml2::XMLElement* error_parameters_norm_element = root_element->FirstChildElement("ErrorParametersNorm");

     if(error_parameters_norm_element)
     {
        const double new_error_parameters_norm = atof(error_parameters_norm_element->GetText());

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
     const tinyxml2::XMLElement* error_gradient_norm_element = root_element->FirstChildElement("ErrorGradientNorm");

     if(error_gradient_norm_element)
     {
        const double new_error_gradient_norm = atof(error_gradient_norm_element->GetText());

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
     const tinyxml2::XMLElement* error_learning_rate_element = root_element->FirstChildElement("ErrorLearningRate");

     if(error_learning_rate_element)
     {
        const double new_error_learning_rate = atof(error_learning_rate_element->GetText());

        try
        {
           set_error_learning_rate(new_error_learning_rate);
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
     const tinyxml2::XMLElement* minimum_parameters_increment_norm_element = root_element->FirstChildElement("MinimumParametersIncrementNorm");

     if(minimum_parameters_increment_norm_element)
     {
        const double new_minimum_parameters_increment_norm = atof(minimum_parameters_increment_norm_element->GetText());

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
     const tinyxml2::XMLElement* minimum_loss_increase_element = root_element->FirstChildElement("MinimumLossDecrease");

     if(minimum_loss_increase_element)
     {
        const double new_minimum_loss_increase = atof(minimum_loss_increase_element->GetText());

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
     const tinyxml2::XMLElement* loss_goal_element = root_element->FirstChildElement("LossGoal");

     if(loss_goal_element)
     {
        const double new_loss_goal = atof(loss_goal_element->GetText());

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
     const tinyxml2::XMLElement* gradient_norm_goal_element = root_element->FirstChildElement("GradientNormGoal");

     if(gradient_norm_goal_element)
     {
        const double new_gradient_norm_goal = atof(gradient_norm_goal_element->GetText());

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
     const tinyxml2::XMLElement* maximum_selection_error_decreases_element = root_element->FirstChildElement("MaximumSelectionErrorIncreases");

     if(maximum_selection_error_decreases_element)
     {
        const size_t new_maximum_selection_error_decreases = static_cast<size_t>(atoi(maximum_selection_error_decreases_element->GetText()));

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
     const tinyxml2::XMLElement* maximum_iterations_number_element = root_element->FirstChildElement("MaximumEpochsNumber");

     if(maximum_iterations_number_element)
     {
         const size_t new_maximum_iterations_number = static_cast<size_t>(atoi(maximum_iterations_number_element->GetText()));

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
     const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

     if(maximum_time_element)
     {
        const double new_maximum_time = atof(maximum_time_element->GetText());

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

  // Reserve training error history
  {
     const tinyxml2::XMLElement* reserve_training_error_history_element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

     if(reserve_training_error_history_element)
     {
        const string new_reserve_training_error_history = reserve_training_error_history_element->GetText();

        try
        {
           set_reserve_training_error_history(new_reserve_training_error_history != "0");
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }

    // Reserve selection error history
    {
       const tinyxml2::XMLElement* reserve_selection_error_history_element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

       if(reserve_selection_error_history_element)
       {
          const string new_reserve_selection_error_history = reserve_selection_error_history_element->GetText();

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


  // Reserve selection error history
  {
     const tinyxml2::XMLElement* reserve_selection_error_history_element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

     if(reserve_selection_error_history_element)
     {
        const string new_reserve_selection_error_history = reserve_selection_error_history_element->GetText();

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
     const tinyxml2::XMLElement* display_period_element = root_element->FirstChildElement("DisplayPeriod");

     if(display_period_element)
     {
        const size_t new_display_period = static_cast<size_t>(atoi(display_period_element->GetText()));

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
     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

     if(display_element)
     {
        const string new_display = display_element->GetText();

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
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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

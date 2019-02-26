/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   S T R A T E G Y   C L A S S                                                              */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "training_strategy.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a training strategy object not associated to any loss index object.
/// It also constructs the main optimization algorithm object.

TrainingStrategy::TrainingStrategy()
{
    data_set_pointer = nullptr;

    neural_network_pointer = nullptr;

    set_loss_method(NORMALIZED_SQUARED_ERROR);

    set_training_method(QUASI_NEWTON_METHOD);

    set_default();
}


TrainingStrategy::TrainingStrategy(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;

    neural_network_pointer = new_neural_network_pointer;

    set_loss_method(NORMALIZED_SQUARED_ERROR);

    set_training_method(QUASI_NEWTON_METHOD);

    set_default();
}


/// XML constructor. 
/// It creates a training strategy object not associated to any loss index object.
/// It also loads the members of this object from a XML document. 
/// @param document Document of the TinyXML library.

TrainingStrategy::TrainingStrategy(const tinyxml2::XMLDocument& document)
{
    set_training_method(QUASI_NEWTON_METHOD);

   set_default();

   from_XML(document);
}


/// File constructor.
/// It creates a training strategy object associated to a loss index object.
/// It also loads the members of this object from a XML file.
/// @param file_name Name of training strategy XML file.

TrainingStrategy::TrainingStrategy(const string& file_name)
{
   set_training_method(QUASI_NEWTON_METHOD);

   set_default();

   load(file_name);
}


// DESTRUCTOR 

/// Destructor.
/// This destructor deletes the loss index and optimization algorithm objects.

TrainingStrategy::~TrainingStrategy()
{
    // Delete loss index objects


    // Delete optimization algorithm objects

    delete gradient_descent_pointer;
    delete conjugate_gradient_pointer;
    delete quasi_Newton_method_pointer;
    delete Levenberg_Marquardt_algorithm_pointer;
    delete stochastic_gradient_descent_pointer;
}


// METHODS

NeuralNetwork* TrainingStrategy::get_neural_network_pointer() const
{
    return neural_network_pointer;
}

LossIndex* TrainingStrategy::get_loss_index_pointer() const
{
    if(sum_squared_error_pointer != nullptr) return sum_squared_error_pointer;
    else if(mean_squared_error_pointer != nullptr) return mean_squared_error_pointer;
    else if(normalized_squared_error_pointer != nullptr) return normalized_squared_error_pointer;
    else if(Minkowski_error_pointer != nullptr) return Minkowski_error_pointer;
    else if(cross_entropy_error_pointer != nullptr) return cross_entropy_error_pointer;
    else if(weighted_squared_error_pointer != nullptr) return weighted_squared_error_pointer;
    else return nullptr;
}


bool TrainingStrategy::has_loss_index() const
{
    return true;
}


/// Initializes the loss index and optimization algorithm at random.
/// @todo

void TrainingStrategy::initialize_random()
{
    // Optimization algorithm

    switch(rand()%2)
    {
      case 0:
      break;

      case 1:
      break;

      default:
         ostringstream buffer;


         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void initialize_random() method.\n"
                << "Unknown optimization algorithm.\n";

         throw logic_error(buffer.str());      
   }
}


/// Returns a pointer to the gradient descent main algorithm.
/// It also throws an exception if that pointer is nullptr.

GradientDescent* TrainingStrategy::get_gradient_descent_pointer() const
{
    if(!gradient_descent_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "GradientDescent* get_gradient_descent_pointer() const method.\n"
               << "Gradient descent pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    return(gradient_descent_pointer);
}


// ConjugateGradient* get_conjugate_gradient_pointer() const method

/// Returns a pointer to the conjugate gradient main algorithm.
/// It also throws an exception if that pointer is nullptr.

ConjugateGradient* TrainingStrategy::get_conjugate_gradient_pointer() const
{
    if(!conjugate_gradient_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "ConjugateGradient* get_conjugate_gradient_pointer() const method.\n"
               << "Conjugate gradient pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    return(conjugate_gradient_pointer);
}


// QuasiNewtonMethod* get_quasi_Newton_method_pointer() const method

/// Returns a pointer to the Newton method main algorithm.
/// It also throws an exception if that pointer is nullptr.

QuasiNewtonMethod* TrainingStrategy::get_quasi_Newton_method_pointer() const
{
    if(!quasi_Newton_method_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "QuasiNetwtonMethod* get_quasi_Newton_method_pointer() const method.\n"
               << "Quasi-Newton method pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    return(quasi_Newton_method_pointer);
}


// LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer() const method

/// Returns a pointer to the Levenberg-Marquardt main algorithm.
/// It also throws an exception if that pointer is nullptr.

LevenbergMarquardtAlgorithm* TrainingStrategy::get_Levenberg_Marquardt_algorithm_pointer() const
{
    if(!Levenberg_Marquardt_algorithm_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer() const method.\n"
               << "Levenberg-Marquardt algorithm pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    return(Levenberg_Marquardt_algorithm_pointer);
}


/// Returns a pointer to the stochastic gradient descent main algorithm.
/// It also throws an exception if that pointer is nullptr.

StochasticGradientDescent* TrainingStrategy::get_stochastic_gradient_descent_pointer() const
{
    if(!stochastic_gradient_descent_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "StochasticGradientDescent* get_stochastic_gradient_descent_pointer() const method.\n"
               << "stochastic gradient descent pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    return(stochastic_gradient_descent_pointer);
}

/// Returns a pointer to the adaptive moment estimation main algorithm.
/// It also throws an exception if that pointer is nullptr.

AdaptiveMomentEstimation* TrainingStrategy::get_adaptive_moment_estimation_pointer() const
{
    if(!adaptive_moment_estimation_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "AdaptiveMomentEstimation* get_adaptive_moment_estimation_pointer() const method.\n"
               << "adaptive_moment_estimation_pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    return(adaptive_moment_estimation_pointer);
}


/// Returns a pointer to the sum squared error which is used as error.
/// If that object does not exists, an exception is thrown.

SumSquaredError* TrainingStrategy::get_sum_squared_error_pointer() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(!sum_squared_error_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "SumSquaredError* get_sum_squared_error_pointer() const method.\n"
              << "Pointer to sum squared error error is nullptr.\n";

       throw logic_error(buffer.str());
     }

     #endif

    return(sum_squared_error_pointer);
}


/// Returns a pointer to the mean squared error which is used as error.
/// If that object does not exists, an exception is thrown.

MeanSquaredError* TrainingStrategy::get_mean_squared_error_pointer() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(!mean_squared_error_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "MeanSquaredError* get_mean_squared_error_pointer() const method.\n"
              << "Pointer to mean squared error error is nullptr.\n";

       throw logic_error(buffer.str());
     }

     #endif

    return(mean_squared_error_pointer);
}


/// Returns a pointer to the normalized squared error which is used as error.
/// If that object does not exists, an exception is thrown.

NormalizedSquaredError* TrainingStrategy::get_normalized_squared_error_pointer() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(!normalized_squared_error_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "NormalizedSquaredError* get_normalized_squared_error_pointer() const method.\n"
              << "Pointer to normalized squared error error is nullptr.\n";

       throw logic_error(buffer.str());
     }

     #endif

    return(normalized_squared_error_pointer);
}


// MinkowskiError* get_Minkowski_error_pointer() const method

/// Returns a pointer to the Minkowski error which is used as error.
/// If that object does not exists, an exception is thrown.

MinkowskiError* TrainingStrategy::get_Minkowski_error_pointer() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(!Minkowski_error_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "MinkowskiError* get_Minkowski_error_pointer() const method.\n"
              << "Pointer to Minkowski error error is nullptr.\n";

       throw logic_error(buffer.str());
     }

     #endif

    return(Minkowski_error_pointer);
}


// CrossEntropyError* get_cross_entropy_error_pointer() const method

/// Returns a pointer to the cross entropy error which is used as error.
/// If that object does not exists, an exception is thrown.

CrossEntropyError* TrainingStrategy::get_cross_entropy_error_pointer() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(!cross_entropy_error_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "CrossEntropyError* get_cross_entropy_error_pointer() const method.\n"
              << "Pointer to cross entropy error error is nullptr.\n";

       throw logic_error(buffer.str());
     }

     #endif

    return(cross_entropy_error_pointer);
}


// WeightedSquaredError* get_weighted_squared_error_pointer() const method

/// Returns a pointer to the weighted squared error which is used as error.
/// If that object does not exists, an exception is thrown.

WeightedSquaredError* TrainingStrategy::get_weighted_squared_error_pointer() const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(!weighted_squared_error_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "WeightedSquaredError* get_weighted_squared_error_pointer() const method.\n"
              << "Pointer to weighted squared error error is nullptr.\n";

       throw logic_error(buffer.str());
     }

     #endif

    return(weighted_squared_error_pointer);
}


const TrainingStrategy::LossMethod& TrainingStrategy::get_loss_method() const
{
   return(loss_method);
}


/// Returns the type of the main optimization algorithm composing this training strategy object.

const TrainingStrategy::TrainingMethod& TrainingStrategy::get_training_method() const
{
   return(training_method);
}


string TrainingStrategy::write_loss_method() const
{
    switch(loss_method)
    {
       case SUM_SQUARED_ERROR:
       {
          return "SUM_SQUARED_ERROR";
       }

       case MEAN_SQUARED_ERROR:
       {
        return "MEAN_SQUARED_ERROR";
       }

       case NORMALIZED_SQUARED_ERROR:
       {
        return "NORMALIZED_SQUARED_ERROR";
       }

       case MINKOWSKI_ERROR:
       {
        return "MINKOWSKI_ERROR";
       }

         case WEIGHTED_SQUARED_ERROR:
         {
        return "WEIGHTED_SQUARED_ERROR";
         }

         case CROSS_ENTROPY_ERROR:
         {
        return "CROSS_ENTROPY_ERROR";
         }
    }

    return string();
}


/// Returns a string with the type of the main optimization algorithm composing this training strategy object.

string TrainingStrategy::write_training_method() const
{
   if(training_method == GRADIENT_DESCENT)
   {
      return("GRADIENT_DESCENT");
   }
   else if(training_method == CONJUGATE_GRADIENT)
   {
      return("CONJUGATE_GRADIENT");
   }
   else if(training_method == QUASI_NEWTON_METHOD)
   {
      return("QUASI_NEWTON_METHOD");
   }
   else if(training_method == LEVENBERG_MARQUARDT_ALGORITHM)
   {
      return("LEVENBERG_MARQUARDT_ALGORITHM");
   }
   else if(training_method == STOCHASTIC_GRADIENT_DESCENT)
   {
      return("STOCHASTIC_GRADIENT_DESCENT");
   }
   else if(training_method == ADAPTIVE_MOMENT_ESTIMATION)
   {
      return("ADAPTIVE_MOMENT_ESTIMATION");
   }
   else
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "string write_training_method() const method.\n"
             << "Unknown main type.\n";
 
	  throw logic_error(buffer.str());
   } 

}



// string TrainingStrategy::write_training_method_text() const

/// Returns a string with the main type in text format.

string TrainingStrategy::write_training_method_text() const
{
   if(training_method == GRADIENT_DESCENT)
   {
      return("gradient descent");
   }
   else if(training_method == CONJUGATE_GRADIENT)
   {
      return("conjugate gradient");
   }
   else if(training_method == QUASI_NEWTON_METHOD)
   {
      return("quasi-Newton method");
   }
   else if(training_method == LEVENBERG_MARQUARDT_ALGORITHM)
   {
      return("Levenberg-Marquardt algorithm");
   }
   else if(training_method == STOCHASTIC_GRADIENT_DESCENT)
   {
      return("stochastic gradient descent");
   }
   else if(training_method == ADAPTIVE_MOMENT_ESTIMATION)
   {
      return("adaptive moment estimation");
   }
   else
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "string write_training_method_text() const method.\n"
             << "Unknown main type.\n";

      throw logic_error(buffer.str());
   }
}


/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& TrainingStrategy::get_display() const
{
   return(display);
}


/// Sets the loss index pointer to nullptr.
/// It also destructs the loss index and the optimization algorithm.
/// Finally, it sets the rest of members to their default values. 

void TrainingStrategy::set()
{
   set_training_method(QUASI_NEWTON_METHOD);

   set_default();
}

void TrainingStrategy::set_loss_method(const string& new_loss_method)
{
    if(new_loss_method == "SUM_SQUARED_ERROR")
    {
        set_loss_method(SUM_SQUARED_ERROR);
    }
    else if(new_loss_method == "MEAN_SQUARED_ERROR")
    {
        set_loss_method(MEAN_SQUARED_ERROR);
    }
    else if(new_loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        set_loss_method(NORMALIZED_SQUARED_ERROR);
    }
    else if(new_loss_method == "MINKOWSKI_ERROR")
    {
        set_loss_method(MINKOWSKI_ERROR);
    }
    else if(new_loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        set_loss_method(WEIGHTED_SQUARED_ERROR);
    }
    else if(new_loss_method == "CROSS_ENTROPY_ERROR")
    {
        set_loss_method(CROSS_ENTROPY_ERROR);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "void set_loss_method(const string&) method.\n"
               << "Unknown loss method: " << new_loss_method << ".\n";

        throw logic_error(buffer.str());
    }
}

void TrainingStrategy::set_loss_method(const LossMethod& new_loss_method)
{
   loss_method = new_loss_method;

   switch(loss_method)
   {
      case SUM_SQUARED_ERROR:
      {
         sum_squared_error_pointer = new SumSquaredError(neural_network_pointer, data_set_pointer);
      }
      break;

      case MEAN_SQUARED_ERROR:
      {
          mean_squared_error_pointer = new MeanSquaredError(neural_network_pointer, data_set_pointer);
      }
      break;

      case NORMALIZED_SQUARED_ERROR:
      {
         normalized_squared_error_pointer = new NormalizedSquaredError(neural_network_pointer, data_set_pointer);
      }
      break;

      case MINKOWSKI_ERROR:
      {
         Minkowski_error_pointer = new MinkowskiError(neural_network_pointer, data_set_pointer);
      }
      break;

      case WEIGHTED_SQUARED_ERROR:
      {
        weighted_squared_error_pointer = new WeightedSquaredError(neural_network_pointer, data_set_pointer);
      }
      break;

      case CROSS_ENTROPY_ERROR:
      {
        cross_entropy_error_pointer = new CrossEntropyError(neural_network_pointer, data_set_pointer);
      }
      break;
   }
}


/// Sets a new type of main optimization algorithm.
/// @param new_training_method Type of main optimization algorithm.

void TrainingStrategy::set_training_method(const TrainingMethod& new_training_method)
{
   training_method = new_training_method;

   LossIndex* loss_index_pointer = get_loss_index_pointer();

   switch(training_method)
   {
      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer = new GradientDescent(loss_index_pointer);
      }
      break;

      case CONJUGATE_GRADIENT:
      {
         conjugate_gradient_pointer = new ConjugateGradient(loss_index_pointer);
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
         quasi_Newton_method_pointer = new QuasiNewtonMethod(loss_index_pointer);
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
         Levenberg_Marquardt_algorithm_pointer = new LevenbergMarquardtAlgorithm(loss_index_pointer);
      }
      break;

      case STOCHASTIC_GRADIENT_DESCENT:
      {
         stochastic_gradient_descent_pointer= new StochasticGradientDescent(loss_index_pointer);
      }
      break;

      case ADAPTIVE_MOMENT_ESTIMATION:
      {
         adaptive_moment_estimation_pointer= new AdaptiveMomentEstimation(loss_index_pointer);
      }
      break;
   }
}


/// Sets a new main optimization algorithm from a string containing the type.
/// @param new_training_method String with the type of main optimization algorithm.

void TrainingStrategy::set_training_method(const string& new_training_method)
{
   if(new_training_method == "GRADIENT_DESCENT")
   {
      set_training_method(GRADIENT_DESCENT);
   }
   else if(new_training_method == "CONJUGATE_GRADIENT")
   {
      set_training_method(CONJUGATE_GRADIENT);
   }
   else if(new_training_method == "QUASI_NEWTON_METHOD")
   {
      set_training_method(QUASI_NEWTON_METHOD);
   }
   else if(new_training_method == "LEVENBERG_MARQUARDT_ALGORITHM")
   {
      set_training_method(LEVENBERG_MARQUARDT_ALGORITHM);
   }
   else if(new_training_method == "STOCHASTIC_GRADIENT_DESCENT")
   {
      set_training_method(STOCHASTIC_GRADIENT_DESCENT);
   }
   else if(new_training_method == "ADAPTIVE_MOMENT_ESTIMATION")
   {
      set_training_method(ADAPTIVE_MOMENT_ESTIMATION);
   }
   else
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "void set_training_method(const string&) method.\n"
             << "Unknown main type: " << new_training_method << ".\n";

      throw logic_error(buffer.str());
   }   
}


/// Sets a pointer to a loss index object to be associated to the training strategy.
/// @param new_loss_index_pointer Pointer to a loss index object.

void TrainingStrategy::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   // Main

   switch(training_method)
   {
      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer->set_loss_index_pointer(new_loss_index_pointer);
      }
      break;

      case CONJUGATE_GRADIENT:
      {
         conjugate_gradient_pointer->set_loss_index_pointer(new_loss_index_pointer);
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
         quasi_Newton_method_pointer->set_loss_index_pointer(new_loss_index_pointer);
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
         Levenberg_Marquardt_algorithm_pointer->set_loss_index_pointer(new_loss_index_pointer);
      }
      break;

      case STOCHASTIC_GRADIENT_DESCENT:
      {
         stochastic_gradient_descent_pointer->set_loss_index_pointer(new_loss_index_pointer);
      }
      break;

      case ADAPTIVE_MOMENT_ESTIMATION:
      {
         adaptive_moment_estimation_pointer->set_loss_index_pointer(new_loss_index_pointer);
      }
      break;
   }

}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TrainingStrategy::set_display(const bool& new_display)
{
   display = new_display;

   switch(training_method)
   {
      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer->set_display(display);
      }
      break;

      case CONJUGATE_GRADIENT:
      {
           conjugate_gradient_pointer->set_display(display);
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           quasi_Newton_method_pointer->set_display(display);
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           Levenberg_Marquardt_algorithm_pointer->set_display(display);
      }
      break;

      case STOCHASTIC_GRADIENT_DESCENT:
      {
           stochastic_gradient_descent_pointer->set_display(display);
      }
      break;

      case ADAPTIVE_MOMENT_ESTIMATION:
      {
           adaptive_moment_estimation_pointer->set_display(display);
      }
      break;
   }
}


/// Sets the members of the training strategy object to their default values:
/// <ul>
/// <li> Display: true.
/// </ul> 

void TrainingStrategy::set_default()
{
   display = true;
}


/// This method deletes the optimization algorithm object which composes this training strategy object.

void TrainingStrategy::destruct_optimization_algorithm()
{
    delete gradient_descent_pointer;
    delete conjugate_gradient_pointer;
    delete quasi_Newton_method_pointer;
    delete Levenberg_Marquardt_algorithm_pointer;
    delete stochastic_gradient_descent_pointer;
    delete adaptive_moment_estimation_pointer;

    gradient_descent_pointer = nullptr;
    conjugate_gradient_pointer = nullptr;
    quasi_Newton_method_pointer = nullptr;
    Levenberg_Marquardt_algorithm_pointer = nullptr;
    stochastic_gradient_descent_pointer = nullptr;
    adaptive_moment_estimation_pointer = nullptr;
}


/// @todo

void TrainingStrategy::initialize_layers_autoencoding()
{
/*
    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Instances* instances_pointer = data_set_pointer->get_instances_pointer();

    const size_t training_instances_number = instances_pointer->get_training_instances_number();

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Vector<size_t> architecture = multilayer_perceptron_pointer->get_architecture();

    // Autoencoding

    Matrix<double> inputs;

    DataSet data_set;

    NeuralNetwork neural_network;

    size_t inputs_number;
    size_t layer_size;

    Vector<double> parameters;

    LossIndex loss_index(&neural_network, &data_set);

    QuasiNewtonMethod quasi_Newton_method(&loss_index);
    quasi_Newton_method.set_loss_goal(1.0e-3);
    quasi_Newton_method.set_gradient_norm_goal(1.0e-3);

    quasi_Newton_method.set_display_period(1000);

    cout << "Layers number: " << architecture.size() - 1 << endl;

    for(size_t i = 1; i < architecture.size()-1; i++)
    {
        cout << "Layer: " << i-1 << endl;
        cout << "Size: " << architecture[i-1] << endl;

        // Neural network

        inputs_number = architecture[i-1];
        layer_size = architecture[i];

        neural_network.set(inputs_number, layer_size, inputs_number);

        // Data set

        inputs.set(training_instances_number, inputs_number);
        inputs.randomize_normal();

        data_set.set(training_instances_number, inputs_number, inputs_number);
        data_set.set_data(inputs.assemble_columns(inputs));

        Vector<Variables::Use> inputs(inputs_number, Variables::Input);
        Vector<Variables::Use> targets(inputs_number, Variables::Target);

        data_set.get_variables_pointer()->set_uses(inputs.assemble(targets));

        data_set.get_instances_pointer()->set_training();

        // Training strategy

        quasi_Newton_method.perform_training();

        // Set parameters

        parameters = neural_network.get_multilayer_perceptron_pointer()->get_layer(0).get_parameters();

        multilayer_perceptron_pointer->set_layer_parameters(i-1, parameters);
    }
*/
}


/// This is the most important method of this class.
/// It optimizes the loss index of a neural network.
/// This method also returns a structure with the results from training.

TrainingStrategy::Results TrainingStrategy::perform_training() const
{
   #ifdef __OPENNN_DEBUG__ 

//    check_loss_index();

//    check_optimization_algorithms();

   #endif

   Results training_strategy_results;

   // Main

   switch(training_method)
   {
      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer->set_display(display);

         training_strategy_results.gradient_descent_results_pointer
         = gradient_descent_pointer->perform_training();

      }
      break;

      case CONJUGATE_GRADIENT:
      {
           conjugate_gradient_pointer->set_display(display);

           training_strategy_results.conjugate_gradient_results_pointer
           = conjugate_gradient_pointer->perform_training();
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           quasi_Newton_method_pointer->set_display(display);

           training_strategy_results.quasi_Newton_method_results_pointer
           = quasi_Newton_method_pointer->perform_training();
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           Levenberg_Marquardt_algorithm_pointer->set_display(display);

           training_strategy_results.Levenberg_Marquardt_algorithm_results_pointer
           = Levenberg_Marquardt_algorithm_pointer->perform_training();
      }
      break;

      case STOCHASTIC_GRADIENT_DESCENT:
      {
           stochastic_gradient_descent_pointer->set_display(display);

           if(stochastic_gradient_descent_pointer->check_cuda())
           {
               training_strategy_results.stochastic_gradient_descent_results_pointer
               = stochastic_gradient_descent_pointer->perform_training_cuda();
           }
           else
           {
               training_strategy_results.stochastic_gradient_descent_results_pointer
               = stochastic_gradient_descent_pointer->perform_training();
           }

      }
      break;

      case ADAPTIVE_MOMENT_ESTIMATION:
      {
           adaptive_moment_estimation_pointer->set_display(display);

           training_strategy_results.adaptive_moment_estimation_results_pointer
           = adaptive_moment_estimation_pointer->perform_training();
      }
      break;
   }

   return training_strategy_results;
}


void TrainingStrategy::perform_training_void() const
{
#ifdef __OPENNN_DEBUG__

//    check_loss_index();

//    check_optimization_algorithms();

#endif

switch(training_method)
{
   case GRADIENT_DESCENT:
   {
      gradient_descent_pointer->set_display(display);

      gradient_descent_pointer->perform_training_void();

//      training_strategy_results.gradient_descent_results_pointer
//      = gradient_descent_pointer->perform_training();

   }
   break;

   case CONJUGATE_GRADIENT:
   {
        conjugate_gradient_pointer->set_display(display);

        conjugate_gradient_pointer->perform_training_void();

//        training_strategy_results.conjugate_gradient_results_pointer
//        = conjugate_gradient_pointer->perform_training();
   }
   break;

   case QUASI_NEWTON_METHOD:
   {
        quasi_Newton_method_pointer->set_display(display);

        quasi_Newton_method_pointer->perform_training_void();
   }
   break;

   case LEVENBERG_MARQUARDT_ALGORITHM:
   {
        Levenberg_Marquardt_algorithm_pointer->set_display(display);

        Levenberg_Marquardt_algorithm_pointer->perform_training_void();

//        training_strategy_results.Levenberg_Marquardt_algorithm_results_pointer
//        = Levenberg_Marquardt_algorithm_pointer->perform_training();
   }
   break;

   case STOCHASTIC_GRADIENT_DESCENT:
   {
        stochastic_gradient_descent_pointer->set_display(display);

        stochastic_gradient_descent_pointer->perform_training_void();

//        training_strategy_results.stochastic_gradient_descent_results_pointer
//        = stochastic_gradient_descent_pointer->perform_training();
   }
   break;


   case ADAPTIVE_MOMENT_ESTIMATION:
   {
        adaptive_moment_estimation_pointer->set_display(display);

        adaptive_moment_estimation_pointer->perform_training_void();

//        training_strategy_results.stochastic_gradient_descent_results_pointer
//        = stochastic_gradient_descent_pointer->perform_training();
}
break;
}
}


/// Returns a string representation of the training strategy.

string TrainingStrategy::object_to_string() const
{
   ostringstream buffer;

   buffer << "Training strategy\n";

   // Main

   buffer << "Loss method: " << write_loss_method() << "\n";

   buffer << "Training method: " << write_training_method() << "\n";

   switch(training_method)
   {
      case GRADIENT_DESCENT:

         buffer << gradient_descent_pointer->object_to_string();

      break;

      case CONJUGATE_GRADIENT:

           buffer << conjugate_gradient_pointer->object_to_string();

      break;

      case QUASI_NEWTON_METHOD:

           buffer << quasi_Newton_method_pointer->object_to_string();

      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:

           buffer << Levenberg_Marquardt_algorithm_pointer->object_to_string();

      break;

      case STOCHASTIC_GRADIENT_DESCENT:

           buffer << stochastic_gradient_descent_pointer->object_to_string();

      break;


      case ADAPTIVE_MOMENT_ESTIMATION:

           buffer << adaptive_moment_estimation_pointer->object_to_string();

      break;


      default:

         ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "string object_to_string() const method.\n"
                << "Unknown main type.\n";

         throw logic_error(buffer.str());      
   }

   return(buffer.str());
}


// void print() const method

/// Prints to the screen the string representation of the training strategy object.

void TrainingStrategy::print() const
{
   cout << object_to_string();
}


// tinyxml2::XMLDocument* to_XML() const method

/// Returns a default string representation in XML-type format of the optimization algorithm object.
/// This containts the training operators, the training parameters, stopping criteria and other stuff.

tinyxml2::XMLDocument* TrainingStrategy::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Training strategy

   tinyxml2::XMLElement* training_strategy_element = document->NewElement("TrainingStrategy");

   document->InsertFirstChild(training_strategy_element);

    // Main

   switch(training_method)
   {
      case GRADIENT_DESCENT:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "GRADIENT_DESCENT");

           const tinyxml2::XMLDocument* gradient_descent_document = gradient_descent_pointer->to_XML();

           const tinyxml2::XMLElement* gradient_descent_element = gradient_descent_document->FirstChildElement("GradientDescent");

           for( const tinyxml2::XMLNode* nodeFor=gradient_descent_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
               tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
               main_element->InsertEndChild( copy );
           }

           delete gradient_descent_document;
      }
      break;

      case CONJUGATE_GRADIENT:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "CONJUGATE_GRADIENT");

           const tinyxml2::XMLDocument* conjugate_gradient_document = conjugate_gradient_pointer->to_XML();

           const tinyxml2::XMLElement* conjugate_gradient_element = conjugate_gradient_document->FirstChildElement("ConjugateGradient");

           for( const tinyxml2::XMLNode* nodeFor=conjugate_gradient_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
               tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
               main_element->InsertEndChild( copy );
           }

           delete conjugate_gradient_document;
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "QUASI_NEWTON_METHOD");

           const tinyxml2::XMLDocument* quasi_Newton_method_document = quasi_Newton_method_pointer->to_XML();

           const tinyxml2::XMLElement* quasi_Newton_method_element = quasi_Newton_method_document->FirstChildElement("QuasiNewtonMethod");

           for( const tinyxml2::XMLNode* nodeFor=quasi_Newton_method_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
               tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
               main_element->InsertEndChild( copy );
           }

           delete quasi_Newton_method_document;
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "LEVENBERG_MARQUARDT_ALGORITHM");

           const tinyxml2::XMLDocument* Levenberg_Marquardt_algorithm_document = Levenberg_Marquardt_algorithm_pointer->to_XML();

           const tinyxml2::XMLElement* Levenberg_Marquardt_algorithm_element = Levenberg_Marquardt_algorithm_document->FirstChildElement("LevenbergMarquardtAlgorithm");

           for( const tinyxml2::XMLNode* nodeFor=Levenberg_Marquardt_algorithm_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
               tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
               main_element->InsertEndChild( copy );
           }

           delete Levenberg_Marquardt_algorithm_document;
      }
      break;

      case STOCHASTIC_GRADIENT_DESCENT:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "STOCHASTIC_GRADIENT_DESCENT");

           const tinyxml2::XMLDocument* stochastic_gradient_descent_document = stochastic_gradient_descent_pointer->to_XML();

           const tinyxml2::XMLElement* stochastic_gradient_descent_element = stochastic_gradient_descent_document->FirstChildElement("StochasticGradientDescent");

           for( const tinyxml2::XMLNode* nodeFor = stochastic_gradient_descent_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
               tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
               main_element->InsertEndChild( copy );
           }

           delete stochastic_gradient_descent_document;
      }
      break;

      case ADAPTIVE_MOMENT_ESTIMATION:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "ADAPTIVE_MOMENT_ESTIMATION");

           const tinyxml2::XMLDocument* adaptive_moment_estimation_document = adaptive_moment_estimation_pointer->to_XML();

           const tinyxml2::XMLElement* adaptive_moment_estimation_element = adaptive_moment_estimation_document->FirstChildElement("AdaptiveMomentEstimation");

           for( const tinyxml2::XMLNode* nodeFor = adaptive_moment_estimation_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
               tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
               main_element->InsertEndChild( copy );
           }

           delete adaptive_moment_estimation_document;
      }
      break;

      default:
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "tinyxml2::XMLDocument* to_XML() const method.\n"
                << "Unknown main type.\n";

         throw logic_error(buffer.str());
      }
   }

   // Display
//   {
//      element = document->NewElement("Display");
//      training_strategy_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the training strategy object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TrainingStrategy::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("TrainingStrategy");

    // Loss index

    switch(loss_method)
    {
        case SUM_SQUARED_ERROR:
        {
            file_stream.OpenElement("LossIndex");

            sum_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            file_stream.OpenElement("LossIndex");

            mean_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            file_stream.OpenElement("LossIndex");

            normalized_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
        break;

        case MINKOWSKI_ERROR:
        {
            file_stream.OpenElement("LossIndex");

            Minkowski_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            file_stream.OpenElement("LossIndex");

            cross_entropy_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            file_stream.OpenElement("LossIndex");

            weighted_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    switch(training_method)
    {
       case GRADIENT_DESCENT:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "GRADIENT_DESCENT");

            gradient_descent_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case CONJUGATE_GRADIENT:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "CONJUGATE_GRADIENT");

            conjugate_gradient_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case QUASI_NEWTON_METHOD:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "QUASI_NEWTON_METHOD");

            quasi_Newton_method_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case LEVENBERG_MARQUARDT_ALGORITHM:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "LEVENBERG_MARQUARDT_ALGORITHM");

            Levenberg_Marquardt_algorithm_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case STOCHASTIC_GRADIENT_DESCENT:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "STOCHASTIC_GRADIENT_DESCENT");

            stochastic_gradient_descent_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;


       case ADAPTIVE_MOMENT_ESTIMATION:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "ADAPTIVE_MOMENT_ESTIMATION");

            stochastic_gradient_descent_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       default:
       {
          ostringstream buffer;

          file_stream.CloseElement();

          buffer << "OpenNN Exception: TrainingStrategy class.\n"
                 << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
                 << "Unknown main type.\n";

          throw logic_error(buffer.str());
       }
    }

    file_stream.CloseElement();
}


/// Loads the members of this training strategy object from a XML document.
/// @param document XML document of the TinyXML library.

void TrainingStrategy::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TrainingStrategy");

   if(!root_element)
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: TrainingStrategy class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Training strategy element is nullptr.\n";

       throw logic_error(buffer.str());
   }

   // Loss index
   {
       sum_squared_error_pointer = nullptr;
       mean_squared_error_pointer = nullptr;
       normalized_squared_error_pointer = nullptr;
       Minkowski_error_pointer = nullptr;
       cross_entropy_error_pointer = nullptr;
       weighted_squared_error_pointer = nullptr;

       const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossIndex");

       if(element)
       {           
          const tinyxml2::XMLElement* error_element = element->FirstChildElement("Error");

          const string new_loss_method = error_element->Attribute("Type");

          set_loss_method(new_loss_method);

          switch(loss_method)
          {
              case SUM_SQUARED_ERROR:
              {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* sum_squared_error_element = new_document.NewElement("SumSquaredError");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      sum_squared_error_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(sum_squared_error_element);

                  sum_squared_error_pointer->from_XML(new_document);
              }
              break;

              case MEAN_SQUARED_ERROR:
              {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* mean_squared_error_element = new_document.NewElement("MeanSquaredError");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      mean_squared_error_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(mean_squared_error_element);

                  mean_squared_error_pointer->from_XML(new_document);
              }
              break;

              case NORMALIZED_SQUARED_ERROR:
              {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* normalized_squared_error_element = new_document.NewElement("NormalizedSquaredError");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      normalized_squared_error_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(normalized_squared_error_element);

                  normalized_squared_error_pointer->from_XML(new_document);
              }
              break;

              case MINKOWSKI_ERROR:
              {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* Minkowski_error_element = new_document.NewElement("MinkowskiError");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      Minkowski_error_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(Minkowski_error_element);

                  Minkowski_error_pointer->from_XML(new_document);
              }
              break;

              case CROSS_ENTROPY_ERROR:
              {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* cross_entropy_error_element = new_document.NewElement("CrossEntropyError");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      cross_entropy_error_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(cross_entropy_error_element);

                  cross_entropy_error_pointer->from_XML(new_document);
              }
              break;

              case WEIGHTED_SQUARED_ERROR:
              {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* weighted_squared_error_element = new_document.NewElement("WeightedSquaredError");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      weighted_squared_error_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(weighted_squared_error_element);

                  weighted_squared_error_pointer->from_XML(new_document);
              }
          }
       }

   }

   // Main
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Main");

       if(element)
       {
          const string new_training_method = element->Attribute("Type");

          set_training_method(new_training_method);

          switch(training_method)
          {
             case GRADIENT_DESCENT:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* gradient_descent_element = new_document.NewElement("GradientDescent");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      gradient_descent_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(gradient_descent_element);

                  gradient_descent_pointer->from_XML(new_document);
             }
             break;

             case CONJUGATE_GRADIENT:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* conjugate_gradient_element = new_document.NewElement("ConjugateGradient");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      conjugate_gradient_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(conjugate_gradient_element);

                  conjugate_gradient_pointer->from_XML(new_document);
             }
             break;

             case QUASI_NEWTON_METHOD:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* quasi_newton_method_element = new_document.NewElement("QuasiNewtonMethod");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      quasi_newton_method_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(quasi_newton_method_element);

                  quasi_Newton_method_pointer->from_XML(new_document);
             }
             break;

             case LEVENBERG_MARQUARDT_ALGORITHM:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* levenberg_marquardt_algorithm_element = new_document.NewElement("LevenbergMarquardtAlgorithm");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      levenberg_marquardt_algorithm_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(levenberg_marquardt_algorithm_element);

                  Levenberg_Marquardt_algorithm_pointer->from_XML(new_document);
             }
             break;

             case STOCHASTIC_GRADIENT_DESCENT:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* stochastic_gradient_descent_element = new_document.NewElement("StochasticGradientDescent");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      stochastic_gradient_descent_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(stochastic_gradient_descent_element);

                  stochastic_gradient_descent_pointer->from_XML(new_document);
             }
             break;

             case ADAPTIVE_MOMENT_ESTIMATION:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* adaptive_moment_estimation_element = new_document.NewElement("AdaptiveMomentEstimation");

                  for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                      tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                      adaptive_moment_estimation_element->InsertEndChild( copy );
                  }

                  new_document.InsertEndChild(adaptive_moment_estimation_element);

                  adaptive_moment_estimation_pointer->from_XML(new_document);
             }
             break;

             default:
             {
                ostringstream buffer;

                buffer << "OpenNN Exception: TrainingStrategy class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown main type.\n";

                throw logic_error(buffer.str());
             }
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


// void save(const string&) const method

/// Saves to a XML-type file the members of the optimization algorithm object.
/// @param file_name Name of optimization algorithm XML-type file. 

void TrainingStrategy::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();   

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads a gradient descent object from a XML-type file.
/// Please mind about the file format, wich is specified in the User's Guide. 
/// @param file_name Name of optimization algorithm XML-type file. 

void TrainingStrategy::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;
   
   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
}


// Results constructor

TrainingStrategy::Results::Results()
{
    gradient_descent_results_pointer = nullptr;

    conjugate_gradient_results_pointer = nullptr;

    quasi_Newton_method_results_pointer = nullptr;

    Levenberg_Marquardt_algorithm_results_pointer = nullptr;

    stochastic_gradient_descent_results_pointer = nullptr;

    adaptive_moment_estimation_results_pointer = nullptr;
}


// Results destructor

TrainingStrategy::Results::~Results()
{

//    delete gradient_descent_results_pointer;

//    delete conjugate_gradient_results_pointer;

//    delete quasi_Newton_method_results_pointer;

//    delete Levenberg_Marquardt_algorithm_results_pointer;

//    delete stochastic_gradient_descent_results_pointer;

//    delete adaptive_moment_estimation_results_pointer;
}


// void Results::save(const string&) const method

/// Saves the results structure to a data file.
/// @param file_name Name of training strategy results data file. 

void TrainingStrategy::Results::save(const string& file_name) const
{
   ofstream file(file_name.c_str());

   if(gradient_descent_results_pointer)
   {
      file << gradient_descent_results_pointer->object_to_string();
   }

   if(conjugate_gradient_results_pointer)
   {
      file << conjugate_gradient_results_pointer->object_to_string();
   }

   if(quasi_Newton_method_results_pointer)
   {
      file << quasi_Newton_method_results_pointer->object_to_string();
   }

   if(Levenberg_Marquardt_algorithm_results_pointer)
   {
      file << Levenberg_Marquardt_algorithm_results_pointer->object_to_string();
   }

   if(stochastic_gradient_descent_results_pointer)
   {
      file << stochastic_gradient_descent_results_pointer->object_to_string();
   }

   if(adaptive_moment_estimation_results_pointer)
   {
      file << adaptive_moment_estimation_results_pointer->object_to_string();
   }

   file.close();
}

#ifdef __OPENNN_MPI__

void TrainingStrategy::set_MPI(LossIndex* new_loss_index, const TrainingStrategy* training_strategy)
{

    set_loss_index_pointer(new_loss_index);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int original_training_method;

    int training_rate_method;
    double loss_tolerance;

    int training_direction_method;

    int inverse_hessian_method;

    double damping_parameter_factor;

    int return_minimum_selection_error_model;
    double minimum_parameters_increment_norm;
    double minimum_loss_decrease;
    double loss_goal;
    double gradient_norm_goal;
    int maximum_selection_error_increases;
    int maximum_iterations_number;
    int maximum_time;
    int reserve_parameters_norm_history;
    int reserve_training_loss_history;
    int reserve_selection_error_history;
    int reserve_gradient_norm_history;

    if(rank == 0)
    {
        // Variables to send initialization

        original_training_method = (int)training_strategy->get_training_method();

        switch(original_training_method)
        {
            case(int)TrainingStrategy::GRADIENT_DESCENT:

                training_rate_method = (int)training_strategy->get_gradient_descent_pointer()->get_learning_rate_algorithm_pointer()->get_training_rate_method();
                loss_tolerance = training_strategy->get_gradient_descent_pointer()->get_learning_rate_algorithm_pointer()->get_loss_tolerance();

                return_minimum_selection_error_model = training_strategy->get_gradient_descent_pointer()->get_return_minimum_selection_error_neural_network();
                minimum_parameters_increment_norm = training_strategy->get_gradient_descent_pointer()->get_minimum_parameters_increment_norm();
                minimum_loss_decrease = training_strategy->get_gradient_descent_pointer()->get_minimum_loss_increase();
                loss_goal = training_strategy->get_gradient_descent_pointer()->get_loss_goal();
                gradient_norm_goal = training_strategy->get_gradient_descent_pointer()->get_gradient_norm_goal();
                maximum_selection_error_increases = (int)training_strategy->get_gradient_descent_pointer()->get_maximum_selection_error_decreases();
                maximum_iterations_number = (int)training_strategy->get_gradient_descent_pointer()->get_maximum_iterations_number();
                maximum_time = (int)training_strategy->get_gradient_descent_pointer()->get_maximum_time();
                reserve_parameters_norm_history = training_strategy->get_gradient_descent_pointer()->get_reserve_parameters_norm_history();
                reserve_training_loss_history = training_strategy->get_gradient_descent_pointer()->get_reserve_error_history();
                reserve_selection_error_history = training_strategy->get_gradient_descent_pointer()->get_reserve_selection_error_history();
                reserve_gradient_norm_history = training_strategy->get_gradient_descent_pointer()->get_reserve_gradient_norm_history();

                break;

            case(int)TrainingStrategy::CONJUGATE_GRADIENT:

                training_direction_method = (int)training_strategy->get_conjugate_gradient_pointer()->get_training_direction_method();

                training_rate_method = (int)training_strategy->get_conjugate_gradient_pointer()->get_learning_rate_algorithm_pointer()->get_training_rate_method();
                loss_tolerance = training_strategy->get_conjugate_gradient_pointer()->get_learning_rate_algorithm_pointer()->get_loss_tolerance();

                return_minimum_selection_error_model = training_strategy->get_conjugate_gradient_pointer()->get_return_minimum_selection_error_neural_network();
                minimum_parameters_increment_norm = training_strategy->get_conjugate_gradient_pointer()->get_minimum_parameters_increment_norm();
                minimum_loss_decrease = training_strategy->get_conjugate_gradient_pointer()->get_minimum_loss_increase();
                loss_goal = training_strategy->get_conjugate_gradient_pointer()->get_loss_goal();
                gradient_norm_goal = training_strategy->get_conjugate_gradient_pointer()->get_gradient_norm_goal();
                maximum_selection_error_increases = (int)training_strategy->get_conjugate_gradient_pointer()->get_maximum_selection_error_decreases();
                maximum_iterations_number = (int)training_strategy->get_conjugate_gradient_pointer()->get_maximum_iterations_number();
                maximum_time = (int)training_strategy->get_conjugate_gradient_pointer()->get_maximum_time();
                reserve_parameters_norm_history = training_strategy->get_conjugate_gradient_pointer()->get_reserve_parameters_norm_history();
                reserve_training_loss_history = training_strategy->get_conjugate_gradient_pointer()->get_reserve_error_history();
                reserve_selection_error_history = training_strategy->get_conjugate_gradient_pointer()->get_reserve_selection_error_history();
                reserve_gradient_norm_history = training_strategy->get_conjugate_gradient_pointer()->get_reserve_gradient_norm_history();

                break;

            case(int)TrainingStrategy::QUASI_NEWTON_METHOD:

                inverse_hessian_method = (int)training_strategy->get_quasi_Newton_method_pointer()->get_inverse_Hessian_approximation_method();

                training_rate_method = (int)training_strategy->get_quasi_Newton_method_pointer()->get_learning_rate_algorithm_pointer()->get_training_rate_method();
                loss_tolerance = training_strategy->get_quasi_Newton_method_pointer()->get_learning_rate_algorithm_pointer()->get_loss_tolerance();

                return_minimum_selection_error_model = training_strategy->get_quasi_Newton_method_pointer()->get_return_minimum_selection_error_neural_network();
                minimum_parameters_increment_norm = training_strategy->get_quasi_Newton_method_pointer()->get_minimum_parameters_increment_norm();
                minimum_loss_decrease = training_strategy->get_quasi_Newton_method_pointer()->get_minimum_loss_increase();
                loss_goal = training_strategy->get_quasi_Newton_method_pointer()->get_loss_goal();
                gradient_norm_goal = training_strategy->get_quasi_Newton_method_pointer()->get_gradient_norm_goal();
                maximum_selection_error_increases = (int)training_strategy->get_quasi_Newton_method_pointer()->get_maximum_selection_error_decreases();
                maximum_iterations_number = (int)training_strategy->get_quasi_Newton_method_pointer()->get_maximum_iterations_number();
                maximum_time = (int)training_strategy->get_quasi_Newton_method_pointer()->get_maximum_time();
                reserve_parameters_norm_history = training_strategy->get_quasi_Newton_method_pointer()->get_reserve_parameters_norm_history();
                reserve_training_loss_history = training_strategy->get_quasi_Newton_method_pointer()->get_reserve_error_history();
                reserve_selection_error_history = training_strategy->get_quasi_Newton_method_pointer()->get_reserve_selection_error_history();
                reserve_gradient_norm_history = training_strategy->get_quasi_Newton_method_pointer()->get_reserve_gradient_norm_history();

                break;

            case(int)TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:

                damping_parameter_factor = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_damping_parameter_factor();

                return_minimum_selection_error_model = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_return_minimum_selection_error_neural_network();
                minimum_parameters_increment_norm = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_minimum_parameters_increment_norm();
                minimum_loss_decrease = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_minimum_loss_increase();
                loss_goal = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_loss_goal();
                gradient_norm_goal = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_gradient_norm_goal();
                maximum_selection_error_increases = (int)training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_maximum_selection_error_decreases();
                maximum_iterations_number = (int)training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_maximum_iterations_number();
                maximum_time = (int)training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_maximum_time();
                reserve_parameters_norm_history = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_reserve_parameters_norm_history();
                reserve_training_loss_history = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_reserve_error_history();
                reserve_selection_error_history = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_reserve_selection_error_history();
                reserve_gradient_norm_history = training_strategy->get_Levenberg_Marquardt_algorithm_pointer()->get_reserve_gradient_norm_history();

                break;

           case(int)TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:

               training_rate_method = (int)training_strategy->get_stochastic_gradient_descent_pointer()->get_learning_rate_algorithm_pointer()->get_training_rate_method();
               loss_tolerance = training_strategy->get_stochastic_gradient_descent_pointer()->get_learning_rate_algorithm_pointer()->get_loss_tolerance();

               return_minimum_selection_error_model = training_strategy->get_stochastic_gradient_descent_pointer()->get_return_minimum_selection_error_neural_network();
               minimum_parameters_increment_norm = training_strategy->get_stochastic_gradient_descent_pointer()->get_minimum_parameters_increment_norm();
               minimum_loss_decrease = training_strategy->get_stochastic_gradient_descent_pointer()->get_minimum_loss_increase();
               loss_goal = training_strategy->get_stochastic_gradient_descent_pointer()->get_loss_goal();
               gradient_norm_goal = training_strategy->get_stochastic_gradient_descent_pointer()->get_gradient_norm_goal();
               maximum_selection_error_increases = (int)training_strategy->get_stochastic_gradient_descent_pointer()->get_maximum_selection_error_decreases();
               maximum_iterations_number = (int)training_strategy->get_stochastic_gradient_descent_pointer()->get_maximum_iterations_number();
               maximum_time = (int)training_strategy->get_stochastic_gradient_descent_pointer()->get_maximum_time();
               reserve_parameters_norm_history = training_strategy->get_stochastic_gradient_descent_pointer()->get_reserve_parameters_norm_history();
               reserve_training_loss_history = training_strategy->get_stochastic_gradient_descent_pointer()->get_reserve_error_history();
               reserve_selection_error_history = training_strategy->get_stochastic_gradient_descent_pointer()->get_reserve_selection_error_history();
               reserve_gradient_norm_history = training_strategy->get_stochastic_gradient_descent_pointer()->get_reserve_gradient_norm_history();

               break;

           case(int)TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION:

               training_rate_method = (int)training_strategy->get_adaptive_moment_estimation_pointer()->get_learning_rate_algorithm_pointer()->get_training_rate_method();
               loss_tolerance = training_strategy->get_adaptive_moment_estimation_pointer()->get_learning_rate_algorithm_pointer()->get_loss_tolerance();

               return_minimum_selection_error_model = training_strategy->get_adaptive_moment_estimation_pointer()->get_return_minimum_selection_error_neural_network();
               minimum_parameters_increment_norm = training_strategy->get_adaptive_moment_estimation_pointer()->get_minimum_parameters_increment_norm();
               minimum_loss_decrease = training_strategy->get_adaptive_moment_estimation_pointer()->get_minimum_loss_increase();
               loss_goal = training_strategy->get_adaptive_moment_estimation_pointer()->get_loss_goal();
               gradient_norm_goal = training_strategy->get_adaptive_moment_estimation_pointer()->get_gradient_norm_goal();
               maximum_selection_error_increases = (int)training_strategy->get_adaptive_moment_estimation_pointer()->get_maximum_selection_error_decreases();
               maximum_iterations_number = (int)training_strategy->get_adaptive_moment_estimation_pointer()->get_maximum_iterations_number();
               maximum_time = (int)training_strategy->get_adaptive_moment_estimation_pointer()->get_maximum_time();
               reserve_parameters_norm_history = training_strategy->get_adaptive_moment_estimation_pointer()->get_reserve_parameters_norm_history();
               reserve_training_loss_history = training_strategy->get_adaptive_moment_estimation_pointer()->get_reserve_error_history();
               reserve_selection_error_history = training_strategy->get_adaptive_moment_estimation_pointer()->get_reserve_selection_error_history();
               reserve_gradient_norm_history = training_strategy->get_adaptive_moment_estimation_pointer()->get_reserve_gradient_norm_history();

               break;


            default:
                break;
        }
    }

    // Send variables

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank > 0)
    {
        MPI_Recv(&original_training_method, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request req[12];

        switch(original_training_method)
        {
            case(int)TrainingStrategy::GRADIENT_DESCENT:

                MPI_Irecv(&training_rate_method, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&loss_tolerance, 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Waitall(2, req, MPI_STATUS_IGNORE);

                break;

            case(int)TrainingStrategy::CONJUGATE_GRADIENT:

                MPI_Irecv(&training_rate_method, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&loss_tolerance, 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Irecv(&training_direction_method, 1, MPI_INT, rank-1, 3, MPI_COMM_WORLD, &req[2]);

                MPI_Waitall(3, req, MPI_STATUS_IGNORE);

                break;

            case(int)TrainingStrategy::QUASI_NEWTON_METHOD:

                MPI_Irecv(&training_rate_method, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&loss_tolerance, 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Irecv(&inverse_hessian_method, 1, MPI_INT, rank-1, 3, MPI_COMM_WORLD, &req[2]);

                MPI_Waitall(3, req, MPI_STATUS_IGNORE);

                break;

            case(int)TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:

                MPI_Irecv(&damping_parameter_factor, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[0]);

                MPI_Waitall(1, req, MPI_STATUS_IGNORE);

                break;

            default:
                break;

           case(int)TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:

               MPI_Irecv(&damping_parameter_factor, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[0]);

               MPI_Waitall(1, req, MPI_STATUS_IGNORE);

               break;

           default:
               break;
        }

        MPI_Irecv(&return_minimum_selection_error_model, 1, MPI_INT, rank-1, 4, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&minimum_parameters_increment_norm, 1, MPI_DOUBLE, rank-1, 5, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&minimum_loss_decrease, 1, MPI_DOUBLE, rank-1, 6, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&loss_goal, 1, MPI_DOUBLE, rank-1, 7, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(&gradient_norm_goal, 1, MPI_DOUBLE, rank-1, 8, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(&maximum_selection_error_increases, 1, MPI_INT, rank-1, 9, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(&maximum_iterations_number, 1, MPI_INT, rank-1, 10, MPI_COMM_WORLD, &req[6]);
        MPI_Irecv(&maximum_time, 1, MPI_INT, rank-1, 11, MPI_COMM_WORLD, &req[7]);
        MPI_Irecv(&reserve_parameters_norm_history, 1, MPI_INT, rank-1, 12, MPI_COMM_WORLD, &req[8]);
        MPI_Irecv(&reserve_training_loss_history, 1, MPI_INT, rank-1, 13, MPI_COMM_WORLD, &req[9]);
        MPI_Irecv(&reserve_selection_error_history, 1, MPI_INT, rank-1, 14, MPI_COMM_WORLD, &req[10]);
        MPI_Irecv(&reserve_gradient_norm_history, 1, MPI_INT, rank-1, 15, MPI_COMM_WORLD, &req[11]);

        MPI_Waitall(12, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Send(&original_training_method, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

        MPI_Request req[12];

        switch(original_training_method)
        {
            case(int)TrainingStrategy::GRADIENT_DESCENT:

                MPI_Isend(&training_rate_method, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&loss_tolerance, 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Waitall(2, req, MPI_STATUS_IGNORE);

                break;

            case(int)TrainingStrategy::CONJUGATE_GRADIENT:

                MPI_Isend(&training_rate_method, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&loss_tolerance, 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Isend(&training_direction_method, 1, MPI_INT, rank+1, 3, MPI_COMM_WORLD, &req[2]);

                MPI_Waitall(3, req, MPI_STATUS_IGNORE);

                break;

            case(int)TrainingStrategy::QUASI_NEWTON_METHOD:

                MPI_Isend(&training_rate_method, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&loss_tolerance, 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Isend(&inverse_hessian_method, 1, MPI_INT, rank+1, 3, MPI_COMM_WORLD, &req[2]);

                MPI_Waitall(3, req, MPI_STATUS_IGNORE);

                break;

            case(int)TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:

                MPI_Isend(&damping_parameter_factor, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[0]);

                MPI_Waitall(1, req, MPI_STATUS_IGNORE);

                break;

            default:
                break;

           case(int)TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:

               MPI_Isend(&damping_parameter_factor, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[0]);

               MPI_Waitall(1, req, MPI_STATUS_IGNORE);

               break;

           default:
               break;
        }

        MPI_Isend(&return_minimum_selection_error_model, 1, MPI_INT, rank+1, 4, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&minimum_parameters_increment_norm, 1, MPI_DOUBLE, rank+1, 5, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(&minimum_loss_decrease, 1, MPI_DOUBLE, rank+1, 6, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&loss_goal, 1, MPI_DOUBLE, rank+1, 7, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(&gradient_norm_goal, 1, MPI_DOUBLE, rank+1, 8, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(&maximum_selection_error_increases, 1, MPI_INT, rank+1, 9, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(&maximum_iterations_number, 1, MPI_INT, rank+1, 10, MPI_COMM_WORLD, &req[6]);
        MPI_Isend(&maximum_time, 1, MPI_INT, rank+1, 11, MPI_COMM_WORLD, &req[7]);
        MPI_Isend(&reserve_parameters_norm_history, 1, MPI_INT, rank+1, 12, MPI_COMM_WORLD, &req[8]);
        MPI_Isend(&reserve_training_loss_history, 1, MPI_INT, rank+1, 13, MPI_COMM_WORLD, &req[9]);
        MPI_Isend(&reserve_selection_error_history, 1, MPI_INT, rank+1, 14, MPI_COMM_WORLD, &req[10]);
        MPI_Isend(&reserve_gradient_norm_history, 1, MPI_INT, rank+1, 15, MPI_COMM_WORLD, &req[11]);

        MPI_Waitall(12, req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Set variables

    set_training_method((TrainingStrategy::TrainingMethod)original_training_method);

    switch(original_training_method)
    {
        case(int)TrainingStrategy::GRADIENT_DESCENT:

            gradient_descent_pointer->get_learning_rate_algorithm_pointer()->set_training_rate_method((LearningRateAlgorithm::LearningRateMethod)training_rate_method);
            gradient_descent_pointer->get_learning_rate_algorithm_pointer()->set_loss_tolerance(loss_tolerance);

            gradient_descent_pointer->set_return_minimum_selection_error_neural_network(return_minimum_selection_error_model == 1);
            gradient_descent_pointer->set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
            gradient_descent_pointer->set_minimum_loss_decrease(minimum_loss_decrease);
            gradient_descent_pointer->set_loss_goal(loss_goal);
            gradient_descent_pointer->set_gradient_norm_goal(gradient_norm_goal);
            gradient_descent_pointer->set_maximum_selection_error_increases(maximum_selection_error_increases);
            gradient_descent_pointer->set_maximum_iterations_number(maximum_iterations_number);
            gradient_descent_pointer->set_maximum_time(maximum_time);
            gradient_descent_pointer->set_reserve_parameters_norm_history(reserve_parameters_norm_history == 1);
            gradient_descent_pointer->get_reserve_error_history(reserve_training_loss_history == 1);
            gradient_descent_pointer->set_reserve_selection_error_history(reserve_selection_error_history == 1);
            gradient_descent_pointer->set_reserve_gradient_norm_history(reserve_gradient_norm_history == 1);

            break;

        case(int)TrainingStrategy::CONJUGATE_GRADIENT:

            conjugate_gradient_pointer->set_training_direction_method((ConjugateGradient::TrainingDirectionMethod)training_direction_method);

            conjugate_gradient_pointer->get_learning_rate_algorithm_pointer()->set_training_rate_method((LearningRateAlgorithm::LearningRateMethod)training_rate_method);
            conjugate_gradient_pointer->get_learning_rate_algorithm_pointer()->set_loss_tolerance(loss_tolerance);

            conjugate_gradient_pointer->set_return_minimum_selection_error_neural_network(return_minimum_selection_error_model == 1);
            conjugate_gradient_pointer->set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
            conjugate_gradient_pointer->set_minimum_loss_decrease(minimum_loss_decrease);
            conjugate_gradient_pointer->set_loss_goal(loss_goal);
            conjugate_gradient_pointer->set_gradient_norm_goal(gradient_norm_goal);
            conjugate_gradient_pointer->set_maximum_selection_error_increases(maximum_selection_error_increases);
            conjugate_gradient_pointer->set_maximum_iterations_number(maximum_iterations_number);
            conjugate_gradient_pointer->set_maximum_time(maximum_time);
            conjugate_gradient_pointer->set_reserve_parameters_norm_history(reserve_parameters_norm_history == 1);
            conjugate_gradient_pointer->get_reserve_error_history(reserve_training_loss_history == 1);
            conjugate_gradient_pointer->set_reserve_selection_error_history(reserve_selection_error_history == 1);
            conjugate_gradient_pointer->set_reserve_gradient_norm_history(reserve_gradient_norm_history == 1);

            break;

        case(int)TrainingStrategy::QUASI_NEWTON_METHOD:

            quasi_Newton_method_pointer->set_inverse_Hessian_approximation_method((QuasiNewtonMethod::InverseHessianApproximationMethod)inverse_hessian_method);

            quasi_Newton_method_pointer->get_learning_rate_algorithm_pointer()->set_training_rate_method((LearningRateAlgorithm::LearningRateMethod)training_rate_method);
            quasi_Newton_method_pointer->get_learning_rate_algorithm_pointer()->set_loss_tolerance(loss_tolerance);

            quasi_Newton_method_pointer->set_return_minimum_selection_error_neural_network(return_minimum_selection_error_model == 1);
            quasi_Newton_method_pointer->set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
            quasi_Newton_method_pointer->set_minimum_loss_decrease(minimum_loss_decrease);
            quasi_Newton_method_pointer->set_loss_goal(loss_goal);
            quasi_Newton_method_pointer->set_gradient_norm_goal(gradient_norm_goal);
            quasi_Newton_method_pointer->set_maximum_selection_error_increases(maximum_selection_error_increases);
            quasi_Newton_method_pointer->set_maximum_iterations_number(maximum_iterations_number);
            quasi_Newton_method_pointer->set_maximum_time(maximum_time);
            quasi_Newton_method_pointer->set_reserve_parameters_norm_history(reserve_parameters_norm_history == 1);
            quasi_Newton_method_pointer->get_reserve_error_history(reserve_training_loss_history == 1);
            quasi_Newton_method_pointer->set_reserve_selection_error_history(reserve_selection_error_history == 1);
            quasi_Newton_method_pointer->set_reserve_gradient_norm_history(reserve_gradient_norm_history == 1);

            break;

        case(int)TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:

            Levenberg_Marquardt_algorithm_pointer->set_damping_parameter_factor(damping_parameter_factor);

            Levenberg_Marquardt_algorithm_pointer->set_return_minimum_selection_error_neural_network(return_minimum_selection_error_model == 1);
            Levenberg_Marquardt_algorithm_pointer->set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
            Levenberg_Marquardt_algorithm_pointer->set_minimum_loss_decrease(minimum_loss_decrease);
            Levenberg_Marquardt_algorithm_pointer->set_loss_goal(loss_goal);
            Levenberg_Marquardt_algorithm_pointer->set_gradient_norm_goal(gradient_norm_goal);
            Levenberg_Marquardt_algorithm_pointer->set_maximum_selection_error_increases(maximum_selection_error_increases);
            Levenberg_Marquardt_algorithm_pointer->set_maximum_iterations_number(maximum_iterations_number);
            Levenberg_Marquardt_algorithm_pointer->set_maximum_time(maximum_time);
            Levenberg_Marquardt_algorithm_pointer->set_reserve_parameters_norm_history(reserve_parameters_norm_history == 1);
            Levenberg_Marquardt_algorithm_pointer->get_reserve_error_history(reserve_training_loss_history == 1);
            Levenberg_Marquardt_algorithm_pointer->set_reserve_selection_error_history(reserve_selection_error_history == 1);
            Levenberg_Marquardt_algorithm_pointer->set_reserve_gradient_norm_history(reserve_gradient_norm_history == 1);

            break;

       case(int)TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:

           stochastic_gradient_descent_pointer->set_damping_parameter_factor(damping_parameter_factor);

           stochastic_gradient_descent_pointer->set_return_minimum_selection_error_neural_network(return_minimum_selection_loss_model == 1);
           stochastic_gradient_descent_pointer->set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
           stochastic_gradient_descent_pointer->set_minimum_loss_increase(minimum_loss_decrease);
           stochastic_gradient_descent_pointer->set_loss_goal(loss_goal);
           stochastic_gradient_descent_pointer->set_gradient_norm_goal(gradient_norm_goal);
           stochastic_gradient_descent_pointer->set_maximum_selection_error_increases(maximum_selection_loss_increases);
           stochastic_gradient_descent_pointer->set_maximum_iterations_number(maximum_iterations_number);
           stochastic_gradient_descent_pointer->set_maximum_time(maximum_time);
           stochastic_gradient_descent_pointer->set_reserve_parameters_norm_history(reserve_parameters_norm_history == 1);
           stochastic_gradient_descent_pointer->get_reserve_error_history(reserve_training_loss_history == 1);
           stochastic_gradient_descent_pointer->set_reserve_selection_error_history(reserve_selection_error_history == 1);
           stochastic_gradient_descent_pointer->set_reserve_gradient_norm_history(reserve_gradient_norm_history == 1);

           break;

       case(int)TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION:

           adaptive_moment_estimation_pointer->set_damping_parameter_factor(damping_parameter_factor);

           adaptive_moment_estimation_pointer->set_return_minimum_selection_error_neural_network(return_minimum_selection_loss_model == 1);
           adaptive_moment_estimation_pointer->set_minimum_parameters_increment_norm(minimum_parameters_increment_norm);
           adaptive_moment_estimation_pointer->set_minimum_loss_increase(minimum_loss_decrease);
           adaptive_moment_estimation_pointer->set_loss_goal(loss_goal);
           adaptive_moment_estimation_pointer->set_gradient_norm_goal(gradient_norm_goal);
           adaptive_moment_estimation_pointer->set_maximum_selection_error_increases(maximum_selection_loss_increases);
           adaptive_moment_estimation_pointer->set_maximum_iterations_number(maximum_iterations_number);
           adaptive_moment_estimation_pointer->set_maximum_time(maximum_time);
           adaptive_moment_estimation_pointer->set_reserve_parameters_norm_history(reserve_parameters_norm_history == 1);
           adaptive_moment_estimation_pointer->get_reserve_error_history(reserve_training_loss_history == 1);
           adaptive_moment_estimation_pointer->set_reserve_selection_error_history(reserve_selection_error_history == 1);
           adaptive_moment_estimation_pointer->set_reserve_gradient_norm_history(reserve_gradient_norm_history == 1);

           break;



        default:
            break;
    }

    if(rank != 0)
    {
        set_display(false);
    }
}
#endif

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

/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R F O R M A N C E   F U N C T I O N A L   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "performance_functional.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.  
/// It creates a performance functional object with all pointers initialized to NULL. 
/// It also initializes all the rest of class members to their default values.

PerformanceFunctional::PerformanceFunctional(void)
 : neural_network_pointer(NULL)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    set_error_type(NORMALIZED_SQUARED_ERROR);
    set_regularization_type(NO_REGULARIZATION);

    set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a performance functional object associated to a neural network object. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

PerformanceFunctional::PerformanceFunctional(NeuralNetwork* new_neural_network_pointer)
 : neural_network_pointer(new_neural_network_pointer)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    set_error_type(NORMALIZED_SQUARED_ERROR);
    set_regularization_type(NO_REGULARIZATION);

   set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a performance functional object associated to a neural network and a data set objects. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

PerformanceFunctional::PerformanceFunctional(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : neural_network_pointer(new_neural_network_pointer)
 , data_set_pointer(new_data_set_pointer)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
   set_error_type(NORMALIZED_SQUARED_ERROR);
   set_regularization_type(NO_REGULARIZATION);   

   set_default();

}


// NEURAL NETWORK AND MATHEMATICAL MODEL CONSTRUCTOR

/// Neural network and mathematical model constructor. 
/// It creates a performance functional object associated to a neural network and a mathematical model objects. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.

PerformanceFunctional::PerformanceFunctional(NeuralNetwork* new_neural_network_pointer, MathematicalModel* new_mathematical_model_pointer)
 : neural_network_pointer(new_neural_network_pointer)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(new_mathematical_model_pointer)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    set_error_type(NORMALIZED_SQUARED_ERROR);
    set_regularization_type(NO_REGULARIZATION);

   set_default();
}


// NEURAL NETWORK, MATHEMATICAL MODEL AND DATA SET CONSTRUCTOR

/// Neural network, mathematical model and data set constructor. 
/// It creates a performance functional object associated to a neural network, a mathematical model and a data set objects. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.
/// @param new_data_set_pointer Pointer to a data set object.

PerformanceFunctional::PerformanceFunctional(NeuralNetwork* new_neural_network_pointer, MathematicalModel* new_mathematical_model_pointer, DataSet* new_data_set_pointer)
 : neural_network_pointer(new_neural_network_pointer)
 , data_set_pointer(new_data_set_pointer)
 , mathematical_model_pointer(new_mathematical_model_pointer)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    set_error_type(NORMALIZED_SQUARED_ERROR);
    set_regularization_type(NO_REGULARIZATION);

   set_default();
}


// USER OBJECTIVE TERM CONSTRUCTOR

/// Objective term constructor. 
/// It creates a performance functional object with a given objective functional.
/// The rest of pointers are initialized to NULL. 
/// The other members are set to their default values, but the error term type, which is set to USER_PERFORMANCE_TERM. 

PerformanceFunctional::PerformanceFunctional(ErrorTerm* new_user_error_pointer)
 : neural_network_pointer(NULL)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(new_user_error_pointer)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    error_type = USER_ERROR;
    set_regularization_type(NO_REGULARIZATION);

   set_default();
}


// FILE CONSTRUCTOR

/// File constructor. 
/// It creates a performance functional object by loading its members from an XML-type file.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param file_name Name of performance functional file.

PerformanceFunctional::PerformanceFunctional(const std::string& file_name)
 : neural_network_pointer(NULL)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    set_error_type(NORMALIZED_SQUARED_ERROR);
    set_regularization_type(NO_REGULARIZATION);

   set_default();

   load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a performance functional object by loading its members from an XML document->
/// @param performance_functional_document Pointer to a TinyXML document containing the performance functional data.

PerformanceFunctional::PerformanceFunctional(const tinyxml2::XMLDocument& performance_functional_document)
 : neural_network_pointer(NULL)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    set_error_type(NORMALIZED_SQUARED_ERROR);
    set_regularization_type(NO_REGULARIZATION);

   set_default();

   from_XML(performance_functional_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing performance functional object. 
/// @param other_performance_functional Performance functional object to be copied.
/// @todo

PerformanceFunctional::PerformanceFunctional(const PerformanceFunctional& other_performance_functional)
 : neural_network_pointer(NULL)
 , data_set_pointer(NULL)
 , mathematical_model_pointer(NULL)
 , sum_squared_error_pointer(NULL)
 , mean_squared_error_pointer(NULL)
 , root_mean_squared_error_pointer(NULL)
 , normalized_squared_error_pointer(NULL)
 , Minkowski_error_pointer(NULL)
 , cross_entropy_error_pointer(NULL)
 , weighted_squared_error_pointer(NULL)
 , roc_area_error_pointer(NULL)
 , user_error_pointer(NULL)
 , neural_parameters_norm_pointer(NULL)
 , outputs_integrals_pointer(NULL)
 , user_regularization_pointer(NULL)
{
    neural_network_pointer = other_performance_functional.neural_network_pointer;
    data_set_pointer = other_performance_functional.data_set_pointer;
    mathematical_model_pointer = other_performance_functional.mathematical_model_pointer;

   error_type = other_performance_functional.error_type;
   regularization_type = other_performance_functional.regularization_type;

   // Objective

    switch(error_type)
    {
        case NO_ERROR:
        {
            // Do nothing
        }
        break;

        case SUM_SQUARED_ERROR:
        {
            sum_squared_error_pointer = new SumSquaredError(*other_performance_functional.sum_squared_error_pointer);
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            mean_squared_error_pointer = new MeanSquaredError(*other_performance_functional.mean_squared_error_pointer);
        }
        break;

        case ROOT_MEAN_SQUARED_ERROR:
        {
            root_mean_squared_error_pointer = new RootMeanSquaredError(*other_performance_functional.root_mean_squared_error_pointer);
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            normalized_squared_error_pointer = new NormalizedSquaredError(*other_performance_functional.normalized_squared_error_pointer);
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            weighted_squared_error_pointer = new WeightedSquaredError(*other_performance_functional.weighted_squared_error_pointer);
        }
        break;

        case ROC_AREA_ERROR:
        {
            roc_area_error_pointer = new RocAreaError(*other_performance_functional.roc_area_error_pointer);
        }
        break;

        case MINKOWSKI_ERROR:
        {
            Minkowski_error_pointer = new MinkowskiError(*other_performance_functional.Minkowski_error_pointer);
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            cross_entropy_error_pointer = new CrossEntropyError(*other_performance_functional.cross_entropy_error_pointer);
        }
        break;

        case USER_ERROR:
        {
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "Copy constructor.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

   // Regularization

    switch(regularization_type)
    {
        case NO_REGULARIZATION:
        {
            // Do nothing
        }
        break;

        case NEURAL_PARAMETERS_NORM:
        {
            neural_parameters_norm_pointer = new NeuralParametersNorm(*other_performance_functional.neural_parameters_norm_pointer);
        }
        break;

        case OUTPUTS_INTEGRALS:
        {
            outputs_integrals_pointer = new OutputsIntegrals(*other_performance_functional.outputs_integrals_pointer);
        }
        break;

        case USER_REGULARIZATION:
        {
            //user_regularization_pointer = new ErrorTerm(*other_performance_functional.user_regularization_pointer);
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "Copy constructor.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

   display = other_performance_functional.display;  
}


// DESTRUCTOR

/// Destructor.
/// It deletes the objective, regularization and constraints terms. 

PerformanceFunctional::~PerformanceFunctional(void)
{
   // Delete error terms

   delete sum_squared_error_pointer;
   delete mean_squared_error_pointer;
   delete root_mean_squared_error_pointer;
   delete normalized_squared_error_pointer;
   delete Minkowski_error_pointer;
   delete cross_entropy_error_pointer;
   delete weighted_squared_error_pointer;
   delete outputs_integrals_pointer;
   delete user_error_pointer;

    // Delete regularization terms

   delete neural_parameters_norm_pointer;
   delete outputs_integrals_pointer;
   delete user_regularization_pointer;
}


// METHODS


// bool has_neural_network(void) const method

/// Returns true if this performance functional has a neural network associated,
/// and false otherwise.

bool PerformanceFunctional::has_neural_network(void) const
{
    if(neural_network_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// bool has_mathematical_model(void) const method

/// Returns true if this performance functional has a mathematical model associated,
/// and false otherwise.

bool PerformanceFunctional::has_mathematical_model(void) const
{
    if(mathematical_model_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// bool has_data_set(void) const method

/// Returns true if this performance functional has a data set associated,
/// and false otherwise.

bool PerformanceFunctional::has_data_set(void) const
{
    if(data_set_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// bool has_selection(void) const method

/// Returns true if this performance functional has a selection method defined,
/// and false otherwise.

bool PerformanceFunctional::has_selection(void) const
{
    if(!data_set_pointer)
    {
        return(false);
    }
    else
    {
        const size_t selection_instances_number = data_set_pointer->get_instances().count_selection_instances_number();

        if(selection_instances_number == 0)
        {
            return(false);
        }
    }

    return(true);
}


// bool is_sum_squared_terms(void) const method

/// Returns true if the performance functional can be expressed as the sum of squared terms.
/// Only those performance functionals are suitable for the Levenberg-Marquardt training algorithm.

bool PerformanceFunctional::is_sum_squared_terms(void) const
{
    if(error_type == ROOT_MEAN_SQUARED_ERROR
    || error_type == MINKOWSKI_ERROR
    || error_type == CROSS_ENTROPY_ERROR)
    {
        return(false);
    }

    if(regularization_type == NEURAL_PARAMETERS_NORM
    || regularization_type == OUTPUTS_INTEGRALS)
    {
        return(false);
    }

    return(true);
}


// void check_neural_network(void) const method

/// Throws an exception if no neural network is associated to the performance functional.

void PerformanceFunctional::check_neural_network(void) const
{
    if(!neural_network_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "void check_neural_network(void) const.\n"
              << "Pointer to neural network is NULL.\n";

       throw std::logic_error(buffer.str());
    }
}


// void check_performance_terms(void) const method

/// Throws an exception if the performance functional has not got any
/// objective, regularization or constraints terms.

void PerformanceFunctional::check_performance_terms(void) const
{
    if(error_type == NO_ERROR
    && regularization_type == NO_REGULARIZATION)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PerformanceFunctional class.\n"
               << "void check_performance_terms(void) const method.\n"
               << "None objective, regularization or constraints terms are used.\n";

        throw std::logic_error(buffer.str());

    }
}


// SumSquaredError* get_sum_squared_error_pointer(void) const method

/// Returns a pointer to the sum squared error which is used as objective.
/// If that object does not exists, an exception is thrown.

SumSquaredError* PerformanceFunctional::get_sum_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!sum_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "SumSquaredError* get_sum_squared_error_pointer(void) const method.\n"
              << "Pointer to sum squared error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(sum_squared_error_pointer);
}


// MeanSquaredError* get_mean_squared_error_pointer(void) const method

/// Returns a pointer to the mean squared error which is used as objective.
/// If that object does not exists, an exception is thrown.

MeanSquaredError* PerformanceFunctional::get_mean_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!mean_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "MeanSquaredError* get_mean_squared_error_pointer(void) const method.\n"
              << "Pointer to mean squared error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(mean_squared_error_pointer);
}


// RootMeanSquaredError* get_root_mean_squared_error_pointer(void) const method

/// Returns a pointer to the root mean squared error which is used as objective.
/// If that object does not exists, an exception is thrown.

RootMeanSquaredError* PerformanceFunctional::get_root_mean_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!root_mean_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "RootMeanSquaredError* get_root_mean_squared_error_pointer(void) const method.\n"
              << "Pointer to root mean squared error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(root_mean_squared_error_pointer);
}


// NormalizedSquaredError* get_normalized_squared_error_pointer(void) const method

/// Returns a pointer to the normalized squared error which is used as objective.
/// If that object does not exists, an exception is thrown.

NormalizedSquaredError* PerformanceFunctional::get_normalized_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!normalized_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "NormalizedSquaredError* get_normalized_squared_error_pointer(void) const method.\n"
              << "Pointer to normalized squared error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(normalized_squared_error_pointer);
}


// MinkowskiError* get_Minkowski_error_pointer(void) const method

/// Returns a pointer to the Minkowski error which is used as objective.
/// If that object does not exists, an exception is thrown.

MinkowskiError* PerformanceFunctional::get_Minkowski_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!Minkowski_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "MinkowskiError* get_Minkowski_error_pointer(void) const method.\n"
              << "Pointer to Minkowski error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(Minkowski_error_pointer);
}


// CrossEntropyError* get_cross_entropy_error_pointer(void) const method

/// Returns a pointer to the cross entropy error which is used as objective.
/// If that object does not exists, an exception is thrown.

CrossEntropyError* PerformanceFunctional::get_cross_entropy_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!cross_entropy_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "CrossEntropyError* get_cross_entropy_error_pointer(void) const method.\n"
              << "Pointer to cross entropy error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(cross_entropy_error_pointer);
}


// WeightedSquaredError* get_weighted_squared_error_pointer(void) const method

/// Returns a pointer to the weighted squared error which is used as objective.
/// If that object does not exists, an exception is thrown.

WeightedSquaredError* PerformanceFunctional::get_weighted_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!cross_entropy_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "WeightedSquaredError* get_weighted_squared_error_pointer(void) const method.\n"
              << "Pointer to weighted squared error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(weighted_squared_error_pointer);
}


// RocAreaError* get_roc_area_error_pointer(void) const method

/// Returns a pointer to the ROC area error which is used as objective.
/// If that object does not exists, an exception is thrown.

RocAreaError* PerformanceFunctional::get_roc_area_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!roc_area_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "RocAreaError* get_roc_area_error_pointer(void) const method.\n"
              << "Pointer to ROC area error objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(roc_area_error_pointer);
}



// OutputsIntegrals* get_outputs_integrals_pointer(void) const method

/// Returns a pointer to the outputs integrals which is used as objective.
/// If that object does not exists, an exception is thrown.


OutputsIntegrals* PerformanceFunctional::get_outputs_integrals_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!outputs_integrals_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "OutputsIntegrals* get_outputs_integrals_pointer(void) const method.\n"
              << "Pointer to outputs integrals objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(outputs_integrals_pointer);
}


// ErrorTerm* get_user_error_pointer(void) const method

/// Returns a pointer to the user performance term which is used as objective.
/// If that object does not exists, an exception is thrown.

ErrorTerm* PerformanceFunctional::get_user_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!user_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "ErrorTerm* get_user_error_pointer(void) const method.\n"
              << "Pointer to user objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(user_error_pointer);
}


// NeuralParametersNorm* get_neural_parameters_norm_pointer(void) const method

/// Returns a pointer to the neural parameters norm functional which is used as regularization.
/// If that object does not exists, an exception is thrown.

NeuralParametersNorm* PerformanceFunctional::get_neural_parameters_norm_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!neural_parameters_norm_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "NeuralParametersNorm* get_neural_parameters_norm_pointer(void) const method.\n"
              << "Pointer to neural parameters norm regularization is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(neural_parameters_norm_pointer);
}


// RegularizationTerm* get_user_regularization_pointer(void) const method

/// Returns a pointer to the user regularization functional.
/// If that object does not exists, an exception is thrown.

RegularizationTerm* PerformanceFunctional::get_user_regularization_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!user_regularization_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: PerformanceFunctional class.\n"
              << "ErrorTerm* get_user_regularization_pointer(void) const method.\n"
              << "Pointer to user regularization is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(user_regularization_pointer);
}


// const ErrorType& get_error_type(void) const method

/// Returns the type of objective term used in the performance functional expression.

const PerformanceFunctional::ErrorType& PerformanceFunctional::get_error_type(void) const
{
   return(error_type);
}


// const RegularizationType& get_regularization_type(void) const method

/// Returns the type of regularization term used in the performance functional expression.

const PerformanceFunctional::RegularizationType& PerformanceFunctional::get_regularization_type(void) const
{
   return(regularization_type);
}


// std::string write_error_type(void) const

/// Returns a string with the type of objective term used in the performance functional expression.

std::string PerformanceFunctional::write_error_type(void) const
{
   if(error_type == NO_ERROR)
   {
      return("NO_ERROR");
   }
   else if(error_type == SUM_SQUARED_ERROR)
   {
      return("SUM_SQUARED_ERROR");
   }
   else if(error_type == MEAN_SQUARED_ERROR)
   {
      return("MEAN_SQUARED_ERROR");
   }
   else if(error_type == ROOT_MEAN_SQUARED_ERROR)
   {
      return("ROOT_MEAN_SQUARED_ERROR");
   }
   else if(error_type == NORMALIZED_SQUARED_ERROR)
   {
      return("NORMALIZED_SQUARED_ERROR");
   }
   else if(error_type == WEIGHTED_SQUARED_ERROR)
   {
      return("WEIGHTED_SQUARED_ERROR");
   }
   else if(error_type == MINKOWSKI_ERROR)
   {
      return("MINKOWSKI_ERROR");
   }
   else if(error_type == CROSS_ENTROPY_ERROR)
   {
      return("CROSS_ENTROPY_ERROR");
   }
   else if(error_type == USER_ERROR)
   {
      return("USER_ERROR");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "std::string write_error_type(void) const method.\n"
             << "Unknown error type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string write_regularization_type(void) const method

/// Returns a string with the type of regularization term used in the performance functional expression.

std::string PerformanceFunctional::write_regularization_type(void) const
{
   if(regularization_type == NO_REGULARIZATION)
   {
      return("NO_REGULARIZATION");
   }
   else if(regularization_type == NEURAL_PARAMETERS_NORM)
   {
      return("NEURAL_PARAMETERS_NORM");
   }
   else if(regularization_type == OUTPUTS_INTEGRALS)
   {
      return("OUTPUTS_INTEGRALS");
   }
   else if(regularization_type == USER_REGULARIZATION)
   {
      return("USER_REGULARIZATION_REGULARIZATION");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "std::string write_regularization_type(void) const method.\n"
             << "Unknown regularization type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string write_error_type_text(void) const

/// Returns a string in text format with the type of objective term used in the performance functional expression.

std::string PerformanceFunctional::write_error_type_text(void) const
{
   if(error_type == NO_ERROR)
   {
      return("no error");
   }
   else if(error_type == SUM_SQUARED_ERROR)
   {
      return("sum squared error");
   }
   else if(error_type == MEAN_SQUARED_ERROR)
   {
      return("mean squared error");
   }
   else if(error_type == ROOT_MEAN_SQUARED_ERROR)
   {
      return("root mean squared error");
   }
   else if(error_type == NORMALIZED_SQUARED_ERROR)
   {
      return("normalized squared error");
   }
   else if(error_type == WEIGHTED_SQUARED_ERROR)
   {
      return("weighted squared error");
   }
   else if(error_type == MINKOWSKI_ERROR)
   {
      return("Minkowski error");
   }
   else if(error_type == CROSS_ENTROPY_ERROR)
   {
      return("cross entropy error");
   }
   else if(error_type == USER_ERROR)
   {
      return("user error");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "std::string write_error_type_text(void) const method.\n"
             << "Unknown error type.\n";

      throw std::logic_error(buffer.str());
   }
}


// std::string write_regularization_type_text(void) const method

/// Returns a string in text format with the type of regularization term used in the performance functional expression.

std::string PerformanceFunctional::write_regularization_type_text(void) const
{
   if(regularization_type == NO_REGULARIZATION)
   {
      return("no regularization");
   }
   else if(regularization_type == NEURAL_PARAMETERS_NORM)
   {
      return("neural parameters norm");
   }
   else if(regularization_type == OUTPUTS_INTEGRALS)
   {
      return("outputs integrals");
   }
   else if(regularization_type == USER_REGULARIZATION)
   {
      return("user regularization");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "std::string write_regularization_type_text(void) const method.\n"
             << "Unknown regularization type.\n";

      throw std::logic_error(buffer.str());
   }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages
/// from this class can't be displayed on the screen.

const bool& PerformanceFunctional::get_display(void) const
{
   return(display);
}


// void set_neural_network_pointer(NeuralNetwork*) method

/// Sets a pointer to a multilayer perceptron object which is to be associated to the performance functional.
/// @param new_neural_network_pointer Pointer to a neural network object to be associated to the performance functional.

void PerformanceFunctional::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;

   // Objective

    switch(error_type)
    {
        case NO_ERROR:
        {
            // Do nothing
        }
        break;

        case SUM_SQUARED_ERROR:
        {
            sum_squared_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            mean_squared_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case ROOT_MEAN_SQUARED_ERROR:
        {
            root_mean_squared_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            normalized_squared_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            weighted_squared_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case MINKOWSKI_ERROR:
        {
            Minkowski_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            cross_entropy_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case USER_ERROR:
        {
            user_error_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "void set_neural_network_pointer(NeuralNetwork*) method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

   // Regularization

    switch(regularization_type)
    {
        case NO_REGULARIZATION:
        {
            // Do nothing
        }
        break;

        case NEURAL_PARAMETERS_NORM:
        {
            neural_parameters_norm_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case OUTPUTS_INTEGRALS:
        {
            outputs_integrals_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        case USER_REGULARIZATION:
        {
            user_regularization_pointer->set_neural_network_pointer(new_neural_network_pointer);
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "void set_neural_network_pointer(NeuralNetwork*) method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }
}


// void set_mathematical_model_pointer(MathematicalModel*) method

/// Sets a new mathematical model on which it will be measured the performance functional.
/// @param new_mathematical_model_pointer Pointer to an external mathematical model object.

void PerformanceFunctional::set_mathematical_model_pointer(MathematicalModel* new_mathematical_model_pointer)
{
   mathematical_model_pointer = new_mathematical_model_pointer;
}


// void set_data_set_pointer(DataSet*) method

/// Sets a new data set on which it will be measured the performance functional.
/// @param new_data_set_pointer Pointer to an external data set object.

void PerformanceFunctional::set_data_set_pointer(DataSet* new_data_set_pointer)
{
   data_set_pointer = new_data_set_pointer;

   // Objective

    switch(error_type)
    {
        case NO_ERROR:
        {
            // Do nothing
        }
        break;

        case SUM_SQUARED_ERROR:
        {
            sum_squared_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            mean_squared_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        case ROOT_MEAN_SQUARED_ERROR:
        {
            root_mean_squared_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            normalized_squared_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            weighted_squared_error_pointer->set_data_set_pointer(new_data_set_pointer);

            if(new_data_set_pointer == NULL)
            {
                set_default();
            }
            else if(new_data_set_pointer->empty())
            {
                set_default();
            }
            else
            {
                weighted_squared_error_pointer->set_weights();
            }
        }
        break;

        case MINKOWSKI_ERROR:
        {
            Minkowski_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            cross_entropy_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        case USER_ERROR:
        {
            user_error_pointer->set_data_set_pointer(new_data_set_pointer);
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "void set_data_set_pointer(DataSet*) method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }
}


// void set_user_error_pointer(ErrorTerm*) method

/// Sets the error term to be a specialized one provided by the user.
/// @param new_user_error_pointer Pointer to a performance term object.

void PerformanceFunctional::set_user_error_pointer(ErrorTerm* new_user_error_pointer)
{
    destruct_error_term();

    error_type = USER_ERROR;

    user_error_pointer = new_user_error_pointer;
}


// void set_user_regularization_pointer(RegularizationTerm*) method

/// Sets the regularization term to be a specialized one provided by the user.
/// @param new_user_regularization_pointer Pointer to a regularization term object.

void PerformanceFunctional::set_user_regularization_pointer(RegularizationTerm* new_user_regularization_pointer)
{
    destruct_regularization_term();

    regularization_type = USER_REGULARIZATION;

    user_regularization_pointer = new_user_regularization_pointer;
}


// void set_default(void) method

/// Sets the members of the performance functional object to their default values.

void PerformanceFunctional::set_default(void)
{
   display = true;
}


// void set_error_type(const std::string&) method

/// Sets a new type for the error term from a string.
/// @param new_error_type String with the type of objective term.

void PerformanceFunctional::set_error_type(const std::string& new_error_type)
{
   if(new_error_type == "NO_ERROR")
   {
      set_error_type(NO_ERROR);
   }
   else if(new_error_type == "SUM_SQUARED_ERROR")
   {
      set_error_type(SUM_SQUARED_ERROR);
   }
   else if(new_error_type == "MEAN_SQUARED_ERROR")
   {
      set_error_type(MEAN_SQUARED_ERROR);
   }
   else if(new_error_type == "ROOT_MEAN_SQUARED_ERROR")
   {
      set_error_type(ROOT_MEAN_SQUARED_ERROR);
   }
   else if(new_error_type == "NORMALIZED_SQUARED_ERROR")
   {
      set_error_type(NORMALIZED_SQUARED_ERROR);
   }
   else if(new_error_type == "WEIGHTED_SQUARED_ERROR")
   {
      set_error_type(WEIGHTED_SQUARED_ERROR);
   }
   else if(new_error_type == "ROC_AREA_ERROR")
   {
      set_error_type(ROC_AREA_ERROR);
   }
   else if(new_error_type == "MINKOWSKI_ERROR")
   {
      set_error_type(MINKOWSKI_ERROR);
   }
   else if(new_error_type == "CROSS_ENTROPY_ERROR")
   {
      set_error_type(CROSS_ENTROPY_ERROR);
   }
   else if(new_error_type == "USER_ERROR")
   {
      set_error_type(USER_ERROR);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "void set_error_type(const std::string&) method.\n"
             << "Unknown objective type: " << new_error_type << ".\n";

      throw std::logic_error(buffer.str());
   }   
}


// void set_regularization_type(const std::string&) method

/// Sets a new type for the regularization term from a string.
/// @param new_regularization_type String with the type of regularization term.

void PerformanceFunctional::set_regularization_type(const std::string& new_regularization_type)
{
   if(new_regularization_type == "NO_REGULARIZATION")
   {
      set_regularization_type(NO_REGULARIZATION);
   }
   else if(new_regularization_type == "NEURAL_PARAMETERS_NORM")
   {
      set_regularization_type(NEURAL_PARAMETERS_NORM);
   }
   else if(new_regularization_type == "OUTPUTS_INTEGRALS")
   {
      set_regularization_type(OUTPUTS_INTEGRALS);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "void set_regularization_type(const std::string&) method.\n"
             << "Unknown regularization type: " << new_regularization_type << ".\n";

      throw std::logic_error(buffer.str());
   }   
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void PerformanceFunctional::set_display(const bool& new_display)
{
   display = new_display;
}


// void set_error_type(const ErrorType&) method

/// Creates a new objective term inside the performance functional of a given performance term type.
/// @param new_error_type Type of objective term to be created.

void PerformanceFunctional::set_error_type(const ErrorType& new_error_type)
{
    destruct_error_term();

   error_type = new_error_type;

   switch(new_error_type)
   {
      case NO_ERROR:
      {
         // Do nothing
      }
      break;

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

      case ROOT_MEAN_SQUARED_ERROR:
      {
         root_mean_squared_error_pointer = new RootMeanSquaredError(neural_network_pointer, data_set_pointer);
      }
      break;

      case NORMALIZED_SQUARED_ERROR:
      {
         normalized_squared_error_pointer = new NormalizedSquaredError(neural_network_pointer, data_set_pointer);
      }
      break;

       case WEIGHTED_SQUARED_ERROR:
       {
          weighted_squared_error_pointer = new WeightedSquaredError(neural_network_pointer, data_set_pointer);
       }
       break;

       case ROC_AREA_ERROR:
       {
          roc_area_error_pointer = new RocAreaError(neural_network_pointer, data_set_pointer);
       }
       break;

      case MINKOWSKI_ERROR:
      {
         Minkowski_error_pointer = new MinkowskiError(neural_network_pointer, data_set_pointer);
      }
      break;

      case CROSS_ENTROPY_ERROR:
      {
         cross_entropy_error_pointer = new CrossEntropyError(neural_network_pointer, data_set_pointer);
      }
      break;

      case USER_ERROR:
      {
         //user_error_pointer = NULL;
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                << "void set_error_type(const ErrorType&) method.\n"
                << "Unknown error type.\n";
 
         throw std::logic_error(buffer.str());	     
      }
      break;
   }
}


// void set_regularization_type(const RegularizationType&) method

/// Creates a new regularization term inside the performance functional of a given performance term type.
/// @param new_regularization_type Type of regularization term to be created.

void PerformanceFunctional::set_regularization_type(const RegularizationType& new_regularization_type)
{
    destruct_regularization_term();

   regularization_type = new_regularization_type;

   switch(regularization_type)
   {
      case NO_REGULARIZATION:
      {
         // Do nothing
      }
      break;

      case NEURAL_PARAMETERS_NORM:
      {
         neural_parameters_norm_pointer = new NeuralParametersNorm(neural_network_pointer);
      }
      break;

      case OUTPUTS_INTEGRALS:
      {
         outputs_integrals_pointer = new OutputsIntegrals(neural_network_pointer);
      }
      break;

      case USER_REGULARIZATION:
      {
      //   regularization_pointer = NULL;
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                << "void set_regularization_type(const RegularizationType&) method.\n"
                << "Unknown regularization type.\n";
 
         throw std::logic_error(buffer.str());	     
      }
      break;
   }
}


// void destruct_error_term(void) method

/// This method deletes the error term object. 
/// It also sets the error term type to NONE and the corresponding flag to false. 

void PerformanceFunctional::destruct_error_term(void)
{
    delete sum_squared_error_pointer;
    delete mean_squared_error_pointer;
    delete root_mean_squared_error_pointer;
    delete normalized_squared_error_pointer;
    delete Minkowski_error_pointer;
    delete cross_entropy_error_pointer;
    delete outputs_integrals_pointer;
    delete user_error_pointer;

    sum_squared_error_pointer = NULL;
    mean_squared_error_pointer = NULL;
    root_mean_squared_error_pointer = NULL;
    normalized_squared_error_pointer = NULL;
    Minkowski_error_pointer = NULL;
    cross_entropy_error_pointer = NULL;
    outputs_integrals_pointer = NULL;
    user_error_pointer = NULL;

   error_type = NO_ERROR;
}


// void destruct_regularization_term(void) method

/// This method deletes the regularization term object. 
/// It also sets the regularization term type to NONE and the corresponding flag to false. 

void PerformanceFunctional::destruct_regularization_term(void)
{
    delete neural_parameters_norm_pointer;
    delete outputs_integrals_pointer;
    delete user_regularization_pointer;

    neural_parameters_norm_pointer = NULL;
    outputs_integrals_pointer = NULL;
    user_regularization_pointer = NULL;

   regularization_type = NO_REGULARIZATION;
}


// void destruct_all_terms(void) method

/// This method destructs the objective, regularization and constraints terms. 

void PerformanceFunctional::destruct_all_terms(void)
{
   destruct_error_term();
   destruct_regularization_term();
}


// double calculate_error(void) const method

/// Returns the objective evaluation,
/// according to the respective objective type used in the performance functional expression.

double PerformanceFunctional::calculate_error(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    double objective = 0.0;

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
            objective = sum_squared_error_pointer->calculate_error();
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             objective = mean_squared_error_pointer->calculate_error();
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             objective = root_mean_squared_error_pointer->calculate_error();
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             objective = normalized_squared_error_pointer->calculate_error();
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             objective = weighted_squared_error_pointer->calculate_error();
         }
         break;

         case ROC_AREA_ERROR:
         {
             objective = roc_area_error_pointer->calculate_error();
         }
         break;

         case MINKOWSKI_ERROR:
         {
             objective = Minkowski_error_pointer->calculate_error();
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             objective = cross_entropy_error_pointer->calculate_error();
         }
         break;

         case USER_ERROR:
         {
             objective = user_error_pointer->calculate_error();
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "double calculate_error(void) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(objective);
}


// double calculate_error(const Vector<double>&) const method

/// Returns the objective evaluation,
/// according to the respective objective type used in the performance functional expression.

double PerformanceFunctional::calculate_error(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    double objective = 0.0;

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
            objective = sum_squared_error_pointer->calculate_error(parameters);
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             objective = mean_squared_error_pointer->calculate_error(parameters);
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             objective = root_mean_squared_error_pointer->calculate_error(parameters);
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             objective = normalized_squared_error_pointer->calculate_error(parameters);
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             objective = weighted_squared_error_pointer->calculate_error(parameters);
         }
         break;

         case ROC_AREA_ERROR:
         {
             objective = roc_area_error_pointer->calculate_error(parameters);
         }
         break;

         case MINKOWSKI_ERROR:
         {
             objective = Minkowski_error_pointer->calculate_error(parameters);
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             objective = cross_entropy_error_pointer->calculate_error(parameters);
         }
         break;

         case USER_ERROR:
         {
             objective = user_error_pointer->calculate_error(parameters);
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "double calculate_error(const Vector<double>&) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(objective);
}


// double calculate_regularization(void) const method

/// Returns the regularization evaluation,
/// according to the respective regularization type used in the performance functional expression.

double PerformanceFunctional::calculate_regularization(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    double regularization = 0.0;

    switch(regularization_type)
    {
        case NO_REGULARIZATION:
        {
            // Do nothing
        }
        break;

        case NEURAL_PARAMETERS_NORM:
        {
            regularization = neural_parameters_norm_pointer->calculate_regularization();
        }
        break;

        case OUTPUTS_INTEGRALS:
        {
            regularization = outputs_integrals_pointer->calculate_regularization();
        }
        break;

        case USER_REGULARIZATION:
        {
            regularization = user_regularization_pointer->calculate_regularization();
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "double calculate_regularization(void) const method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

    return(regularization);
}


// double calculate_regularization(const Vector<double>&) const method

/// Returns the regularization evaluation,
/// according to the respective regularization type used in the performance functional expression.

double PerformanceFunctional::calculate_regularization(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    double regularization = 0.0;

    switch(regularization_type)
    {
        case NO_REGULARIZATION:
        {
            // Do nothing
        }
        break;

        case NEURAL_PARAMETERS_NORM:
        {
            regularization = neural_parameters_norm_pointer->calculate_regularization(parameters);
        }
        break;

        case OUTPUTS_INTEGRALS:
        {
            regularization = outputs_integrals_pointer->calculate_regularization(parameters);
        }
        break;

        case USER_REGULARIZATION:
        {
            regularization = user_regularization_pointer->calculate_regularization(parameters);
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "double calculate_regularization(const Vector<double>&) const method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

    return(regularization);
}



// Vector<double> calculate_error_terms(void) const method

/// Returns the evaluation of all the error terms,
/// according to the respective objective type used in the performance functional expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.

Vector<double> PerformanceFunctional::calculate_error_terms(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    std::ostringstream buffer;

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    Vector<double> objective_terms(training_instances_number, 0.0);

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
            objective_terms = sum_squared_error_pointer->calculate_terms();
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
            objective_terms = mean_squared_error_pointer->calculate_terms();
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Cannot calculate performance terms for root mean squared error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             objective_terms = normalized_squared_error_pointer->calculate_terms();
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             objective_terms = weighted_squared_error_pointer->calculate_terms();
         }
         break;

         case MINKOWSKI_ERROR:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Cannot calculate performance terms for Minkowski error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Cannot calculate performance terms for cross-entropy error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

         case USER_ERROR:
         {
             objective_terms = user_error_pointer->calculate_terms();
         }
         break;

         default:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(objective_terms);
}


// Matrix<double> calculate_error_terms_Jacobian(void) const method

/// Returns the Jacobian of the error terms function,
/// according to the objective type used in the performance functional expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.
/// The Jacobian elements are the partial derivatives of a single term with respect to a single parameter.
/// The number of rows in the Jacobian matrix are the number of parameters,
/// and the number of columns the number of terms composing the objective.

Matrix<double> PerformanceFunctional::calculate_error_terms_Jacobian(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    std::ostringstream buffer;

    Matrix<double> objective_terms_Jacobian;

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
            objective_terms_Jacobian = sum_squared_error_pointer->calculate_terms_Jacobian();
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
            objective_terms_Jacobian = mean_squared_error_pointer->calculate_terms_Jacobian();
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                    << "Cannot calculate performance terms for root mean squared error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             objective_terms_Jacobian = normalized_squared_error_pointer->calculate_terms_Jacobian();
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             objective_terms_Jacobian = weighted_squared_error_pointer->calculate_terms_Jacobian();
         }
         break;

         case MINKOWSKI_ERROR:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                    << "Cannot calculate performance terms for Minkowski error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

     case CROSS_ENTROPY_ERROR:
     {
         buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                << "Cannot calculate performance terms for cross-entropy error objective.\n";

         throw std::logic_error(buffer.str());
     }
     break;


         case USER_ERROR:
         {
             objective_terms_Jacobian = user_error_pointer->calculate_terms_Jacobian();
         }
         break;

         default:
         {
             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    return(objective_terms_Jacobian);
}


// Vector<double> calculate_error_gradient(void) const method

/// Returns the gradient of the objective, according to the objective type.
/// That gradient is the vector of partial derivatives of the objective with respect to the parameters.
/// The size is thus the number of parameters.

Vector<double> PerformanceFunctional::calculate_error_gradient(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Vector<double> gradient(parameters_number, 0.0);

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
             gradient = sum_squared_error_pointer->calculate_gradient();
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             gradient = mean_squared_error_pointer->calculate_gradient();
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             gradient = root_mean_squared_error_pointer->calculate_gradient();
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             gradient = normalized_squared_error_pointer->calculate_gradient();
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             gradient = weighted_squared_error_pointer->calculate_gradient();
         }
         break;

         case ROC_AREA_ERROR:
         {
             gradient = roc_area_error_pointer->calculate_gradient();
         }
         break;

         case MINKOWSKI_ERROR:
         {
             gradient = Minkowski_error_pointer->calculate_gradient();
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             gradient = cross_entropy_error_pointer->calculate_gradient();
         }
         break;

         case USER_ERROR:
         {
             gradient = user_error_pointer->calculate_gradient();
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_error_gradient(void) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(gradient);
}


// Vector<double> calculate_error_gradient(const Vector<double>&) const method

/// Returns the gradient of the objective, according to the objective type.
/// That gradient is the vector of partial derivatives of the objective with respect to the parameters.
/// The size is thus the number of parameters.

Vector<double> PerformanceFunctional::calculate_error_gradient(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Vector<double> gradient(parameters_number, 0.0);

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
             //gradient = sum_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             //gradient = mean_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             //gradient = root_mean_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             //gradient = normalized_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             //gradient = weighted_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case MINKOWSKI_ERROR:
         {
             //gradient = Minkowski_error_pointer->calculate_gradient(parameters);
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             //gradient = cross_entropy_error_pointer->calculate_gradient(parameters);
         }
         break;

         case USER_ERROR:
         {
             gradient = user_error_pointer->calculate_gradient(parameters);
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_error_gradient(const Vector<double>&) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(gradient);
}


// Vector<double> calculate_regularization_gradient(void) const method

/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters.

Vector<double> PerformanceFunctional::calculate_regularization_gradient(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Vector<double> gradient(parameters_number, 0.0);

    // Regularization

     switch(regularization_type)
     {
         case NO_REGULARIZATION:
         {
             // Do nothing
         }
         break;

         case NEURAL_PARAMETERS_NORM:
         {
             gradient = neural_parameters_norm_pointer->calculate_gradient();
         }
         break;

         case OUTPUTS_INTEGRALS:
         {
             gradient = outputs_integrals_pointer->calculate_gradient();
         }
         break;

         case USER_REGULARIZATION:
         {
             gradient = user_regularization_pointer->calculate_gradient();
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_regularization_gradient(void) const method.\n"
                    << "Unknown regularization type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    return(gradient);
}


// Vector<double> calculate_regularization_gradient(const Vector<double>&) const method

/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters.

Vector<double> PerformanceFunctional::calculate_regularization_gradient(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Vector<double> gradient(parameters_number, 0.0);

    // Regularization

     switch(regularization_type)
     {
         case NO_REGULARIZATION:
         {
             // Do nothing
         }
         break;

         case NEURAL_PARAMETERS_NORM:
         {
             //gradient = neural_parameters_norm_pointer->calculate_gradient(parameters);
         }
         break;

         case OUTPUTS_INTEGRALS:
         {
             //gradient = outputs_integrals_pointer->calculate_gradient(parameters);
         }
         break;

         case USER_REGULARIZATION:
         {
             gradient = user_regularization_pointer->calculate_gradient(parameters);
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Vector<double> calculate_regularization_gradient(const Vector<double>&) const method.\n"
                    << "Unknown regularization type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    return(gradient);
}


// Matrix<double> calculate_error_Hessian(void) const method

/// Returns the Hessian of the objective, according to the objective type.
/// That Hessian is the matrix of second partial derivatives of the objective with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.

Matrix<double> PerformanceFunctional::calculate_error_Hessian(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
             Hessian = sum_squared_error_pointer->calculate_Hessian();
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             Hessian = mean_squared_error_pointer->calculate_Hessian();
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             Hessian = root_mean_squared_error_pointer->calculate_Hessian();
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             Hessian = normalized_squared_error_pointer->calculate_Hessian();
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             Hessian = weighted_squared_error_pointer->calculate_Hessian();
         }
         break;

         case MINKOWSKI_ERROR:
         {
             Hessian = Minkowski_error_pointer->calculate_Hessian();
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             Hessian = cross_entropy_error_pointer->calculate_Hessian();
         }
         break;

         case USER_ERROR:
         {
             Hessian = user_error_pointer->calculate_Hessian();
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "Matrix<double> Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_error_Hessian(void) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    return(Hessian);
}


// Matrix<double> calculate_error_Hessian(const Vector<double>&) const method

/// Returns the Hessian of the objective, according to the objective type.
/// That Hessian is the matrix of second partial derivatives of the objective with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.
/// @todo

Matrix<double> PerformanceFunctional::calculate_error_Hessian(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
             Hessian = sum_squared_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             //Hessian = mean_squared_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             //Hessian = root_mean_squared_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             //Hessian = normalized_squared_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             //Hessian = weighted_squared_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case MINKOWSKI_ERROR:
         {
             //Hessian = Minkowski_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             //Hessian = cross_entropy_error_pointer->calculate_Hessian(parameters);
         }
         break;

         case USER_ERROR:
         {
             //Hessian = user_error_pointer->calculate_Hessian(parameters);
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "Matrix<double> Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_error_Hessian(const Vector<double>&) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    return(Hessian);
}


// Matrix<double> calculate_regularization_Hessian(void) const method

/// Returns the Hessian of the regularization, according to the regularization type.
/// That Hessian is the matrix of second partial derivatives of the regularization with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.

Matrix<double> PerformanceFunctional::calculate_regularization_Hessian(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

    // Regularization

     switch(regularization_type)
     {
         case NO_REGULARIZATION:
         {
             // Do nothing
         }
         break;

         case NEURAL_PARAMETERS_NORM:
         {
             Hessian = neural_parameters_norm_pointer->calculate_Hessian();
         }
         break;

         case OUTPUTS_INTEGRALS:
         {
             Hessian = outputs_integrals_pointer->calculate_Hessian();
         }
         break;

         case USER_REGULARIZATION:
         {
             Hessian = user_regularization_pointer->calculate_Hessian();
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_regularization_Hessian(void) const method.\n"
                    << "Unknown regularization type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(Hessian);
}


// Matrix<double> calculate_regularization_Hessian(const Vector<double>&) const method

/// Returns the Hessian of the regularization, according to the regularization type.
/// That Hessian is the matrix of second partial derivatives of the regularization with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.
/// @todo

Matrix<double> PerformanceFunctional::calculate_regularization_Hessian(const Vector<double>&) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

    // Regularization

     switch(regularization_type)
     {
         case NO_REGULARIZATION:
         {
             // Do nothing
         }
         break;

         case NEURAL_PARAMETERS_NORM:
         {
             //Hessian = neural_parameters_norm_pointer->calculate_Hessian(parameters);
         }
         break;

         case OUTPUTS_INTEGRALS:
         {
             //Hessian = outputs_integrals_pointer->calculate_Hessian(parameters);
         }
         break;

         case USER_REGULARIZATION:
         {
             //Hessian = user_regularization_pointer->calculate_Hessian(parameters);
         }
         break;

         default:
         {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "Matrix<double> calculate_regularization_Hessian(const Vector<double>&) const method.\n"
                    << "Unknown regularization type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(Hessian);
}


// double calculate_performance(void) const method

/// Calculates the evaluation value of the performance functional,
/// as the sum of the objective, regularization and constraints functionals.

double PerformanceFunctional::calculate_performance(void) const 
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_performance_terms();

   #endif

   return(calculate_error() + calculate_regularization());
}


// double calculate_performance(const Vector<double>&) const method

/// Returns the performance of a neural network for a given vector of parameters.
/// It does not set that vector of parameters to the neural network. 
/// @param parameters Vector of parameters for the neural network associated to the performance functional.

double PerformanceFunctional::calculate_performance(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_performance_terms();

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "double calculate_performance(const Vector<double>&) method.\n"
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   return(calculate_error(parameters) + calculate_regularization(parameters));
}


// double calculate_selection_error(void) const method

/// Returns the evaluation of the error term on the selection instances of the associated data set.

double PerformanceFunctional::calculate_selection_error(void) const
{
    double selection_error = 0.0;

    switch(error_type)
    {
        case NO_ERROR:
        {
            // Do nothing
        }
        break;

        case SUM_SQUARED_ERROR:
        {
            selection_error = sum_squared_error_pointer->calculate_selection_error();
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            selection_error = mean_squared_error_pointer->calculate_selection_error();
        }
        break;

        case ROOT_MEAN_SQUARED_ERROR:
        {
            selection_error = root_mean_squared_error_pointer->calculate_selection_error();
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            selection_error = normalized_squared_error_pointer->calculate_selection_error();
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            selection_error = weighted_squared_error_pointer->calculate_selection_error();
        }
        break;

        case ROC_AREA_ERROR:
        {
            selection_error = roc_area_error_pointer->calculate_selection_error();
        }
        break;

        case MINKOWSKI_ERROR:
        {
            selection_error = Minkowski_error_pointer->calculate_selection_error();
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            selection_error = cross_entropy_error_pointer->calculate_selection_error();
        }
        break;

        case USER_ERROR:
        {
            selection_error = user_error_pointer->calculate_selection_error();
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "double calculate_selection_error(void) const method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

    return(selection_error);
}


// double calculate_selection_performance(void) const method method

/// Calculates the selection performance,
/// as the sum of the objective and the regularization terms. 

double PerformanceFunctional::calculate_selection_performance(void) const 
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_performance_terms();

   #endif

   return(calculate_selection_error());
}


// Vector<double> calculate_gradient(void) const method

/// Returns the performance function gradient, as the sum of the objective and the regularization gradient vectors.

Vector<double> PerformanceFunctional::calculate_gradient(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_performance_terms();

   #endif

   return(calculate_error_gradient() + calculate_regularization_gradient());
}


// Vector<double> calculate_gradient(const Vector<double>&) const method

/// Returns the performance gradient for a given vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of parameters for the neural network associated to the performance functional.

Vector<double> PerformanceFunctional::calculate_gradient(const Vector<double>& parameters) const
{
   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_performance_terms();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   const size_t size = parameters.size();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "Vector<double> calculate_gradient(const Vector<double>&) const method.\n"
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }
   
   #endif
    
   return(calculate_error_gradient(parameters) + calculate_regularization_gradient(parameters));
}



// Matrix<double> calculate_Hessian(void) const method

/// Returns the default objective function Hessian matrix,
/// which is computed as the sum of the objective, regularization and constraints Hessians.

Matrix<double> PerformanceFunctional::calculate_Hessian(void) const
{
    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    check_performance_terms();

    #endif

    return(calculate_error_Hessian() + calculate_regularization_Hessian());
}


// Vector<double> calculate_Hessian(const Vector<double>&) const method

/// Returns which would be the objective function Hessian of a neural network for an
/// hypothetical vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated
/// to this performance functional.

Matrix<double> PerformanceFunctional::calculate_Hessian(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_performance_terms();

   const size_t size = parameters.size();
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "double calculate_Hessian(const Vector<double>&) method.\n"
             << "Size must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   return(calculate_error_Hessian(parameters) + calculate_regularization_Hessian(parameters));
}


// Vector<double> calculate_terms(void) const method

/// Evaluates the objective, regularization and constraints terms functions,
/// and returns the total performance terms as the assembly of that three vectors.

Vector<double> PerformanceFunctional::calculate_terms(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     check_neural_network();

     check_performance_terms();

    #endif

    const Vector<double> objective_terms = calculate_error_terms();

    return(objective_terms);
}


// Matrix<double> calculate_terms_Jacobian(void) const method

/// @todo

Matrix<double> PerformanceFunctional::calculate_terms_Jacobian(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     check_neural_network();

     check_performance_terms();

    #endif

    const Matrix<double> objective_terms_Jacobian = calculate_error_terms_Jacobian();

//    const Matrix<double> regularization_terms_Jacobian = calculate_regularization_terms_Jacobian();

//    const Matrix<double> constraints_terms_Jacobian = calculate_constraints_terms_Jacobian();

//    Matrix<double> terms_Jacobian;

//    if(!objective_terms_Jacobian.empty())
//    {
//        terms_Jacobian = objective_terms_Jacobian;
//    }

    return(objective_terms_Jacobian);
}


// Matrix<double> calculate_inverse_Hessian(void) const method

/// Returns inverse matrix of the Hessian.
/// It first computes the Hessian matrix and then computes its inverse. 
/// @todo

Matrix<double> PerformanceFunctional::calculate_inverse_Hessian(void) const
{  
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

    check_neural_network();

    check_performance_terms();

   #endif

   const Matrix<double> Hessian = calculate_Hessian();
         
   return(Hessian.calculate_LU_inverse());
}


// Vector<double> calculate_vector_dot_Hessian(Vector<double>) const method

/// Returns the default product of some vector with the objective function Hessian matrix, which is
/// computed using numerical differentiation.
/// @param vector Vector in the dot product. 
/// @todo

Vector<double> PerformanceFunctional::calculate_vector_dot_Hessian(const Vector<double>& vector) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     check_neural_network();

     check_performance_terms();

    #endif


   // Control sentence

   const size_t size = vector.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "Vector<double> calculate_vector_dot_Hessian(Vector<double>) method.\n"
             << "Size of vector must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Calculate vector Hessian product

   Vector<double> vector_Hessian_product(parameters_number);

   return(vector_Hessian_product);
}


// ZeroOrderperformance calculate_zero_order_performance(void) const method

/// Returns a zero order performance structure, which just contains the performance value of the performance function.

PerformanceFunctional::ZeroOrderperformance PerformanceFunctional::calculate_zero_order_performance(void) const
{
   ZeroOrderperformance zero_order_performance;

   zero_order_performance.performance = calculate_performance();

   return(zero_order_performance);
}


// FirstOrderperformance calculate_first_order_performance(void) const method

/// Returns a first order performance structure, which contains the value and the gradient of the performance function.

PerformanceFunctional::FirstOrderperformance PerformanceFunctional::calculate_first_order_performance(void) const
{
   FirstOrderperformance first_order_performance;

   first_order_performance.performance = calculate_performance();
   first_order_performance.gradient = calculate_gradient();

   return(first_order_performance);
}


// SecondOrderperformance calculate_second_order_performance(void) const method

/// Returns a second order performance structure, which contains the value, the gradient and the Hessian of the performance function.

PerformanceFunctional::SecondOrderperformance PerformanceFunctional::calculate_second_order_performance(void) const
{
   SecondOrderperformance second_order_performance;

   second_order_performance.performance = calculate_performance();
   second_order_performance.gradient = calculate_gradient();
   second_order_performance.Hessian = calculate_Hessian();

   return(second_order_performance);
}


// double calculate_zero_order_Taylor_approximation(const Vector<double>&) const method

/// Returns the Taylor approximation of the performance function at some point near the parameters.
/// The order of the approximation here is zero, i.e., only the performance value is used. 

double PerformanceFunctional::calculate_zero_order_Taylor_approximation(const Vector<double>&) const 
{
   return(calculate_performance());
}


// double calculate_first_order_Taylor_approximation(const Vector<double>&) const method

/// Returns the Taylor approximation of the performance function at some point near the parameters.
/// The order of the approximation here is one, i.e., both the performance value and the performance gradient are used. 
/// @param parameters Approximation point. 

double PerformanceFunctional::calculate_first_order_Taylor_approximation(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t parameters_size = parameters.size();
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(parameters_size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "double calculate_first_order_Taylor_approximation(const Vector<double>&) const method.\n"
             << "Size of potential parameters must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   const Vector<double> original_parameters = neural_network_pointer->arrange_parameters();

   const double performance = calculate_performance();
   const Vector<double> gradient = calculate_gradient();

   const double first_order_Taylor_approximation = performance + gradient.dot(parameters-parameters);

   return(first_order_Taylor_approximation);
}


// double calculate_second_order_Taylor_approximation(const Vector<double>&) const method

/// Returns the Taylor approximation of the performance function at some point near the parameters.
/// The order of the approximation here is two, i.e., the performance value, the performance gradient and the performance Hessian are used. 
/// @param parameters Approximation point. 

double PerformanceFunctional::calculate_second_order_Taylor_approximation(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t parameters_size = parameters.size();
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(parameters_size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "double calculate_second_order_Taylor_approximation(const Vector<double>&) const method.\n"
             << "Size of potential parameters must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Neural network stuff 

   const Vector<double> original_parameters = neural_network_pointer->arrange_parameters();
   const Vector<double> parameters_difference = parameters - parameters;

   // Performance functioal stuff

   const double performance = calculate_performance();
   const Vector<double> gradient = calculate_gradient();
   const Matrix<double> Hessian = calculate_Hessian();
   
   const double second_order_Taylor_approximation = performance 
   + gradient.dot(parameters_difference) 
   + parameters_difference.dot(Hessian).dot(parameters_difference)/2.0;

   return(second_order_Taylor_approximation);
}


// double calculate_performance(const Vector<double>&, const double&) const method

/// Returns the value of the performance function at some step along some direction.
/// @param direction Direction vector.
/// @param rate Step value. 

double PerformanceFunctional::calculate_performance(const Vector<double>& direction, const double& rate) const
{
   const Vector<double> parameters = neural_network_pointer->arrange_parameters();
   const Vector<double> increment = direction*rate;

   return(calculate_performance(parameters + increment));
}


// double calculate_performance_derivative(const Vector<double>&, const double&) const method

/// Returns the derivative of the performance function at some step along some direction.
/// @param direction Direction vector.
/// @param rate Step value. 

double PerformanceFunctional::calculate_performance_derivative(const Vector<double>& direction, const double& rate) const
{
    if(direction == 0.0)
    {
        return(0.0);
    }

    const Vector<double> parameters = neural_network_pointer->arrange_parameters();
    const Vector<double> potential_parameters = parameters + direction*rate;

   const Vector<double> gradient = calculate_gradient(potential_parameters);

   const Vector<double> normalized_direction = direction/direction.calculate_norm();

   return(gradient.dot(normalized_direction));
}


// double calculate_performance_second_derivative(const Vector<double>&, double) const method

/// Returns the second derivative of the performance function at some step along some direction.
/// @param direction Direction vector.
/// @param rate Step value. 

double PerformanceFunctional::calculate_performance_second_derivative(const Vector<double>& direction, const double& rate) const
{
    if(direction == 0.0)
    {
        return(0.0);
    }

    const Vector<double> parameters = neural_network_pointer->arrange_parameters();
    const Vector<double> potential_parameters = parameters + direction*rate;

   const Matrix<double> Hessian = calculate_Hessian(potential_parameters);

   const Vector<double> normalized_direction = direction/direction.calculate_norm();

   return(normalized_direction.dot(Hessian).dot(normalized_direction));
}


// tinyxml2::XMLDocument* to_XML(void) const method 

/// Serializes a default performance functional object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* PerformanceFunctional::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Performance functional

   tinyxml2::XMLElement* performance_functional_element = document->NewElement("PerformanceFunctional");

   document->InsertFirstChild(performance_functional_element);

   // Objective

   switch(error_type)
   {
      case NO_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

           objective_element->SetAttribute("Type", "NO_ERROR");
      }
      break;

      case SUM_SQUARED_ERROR:
      {                
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

           objective_element->SetAttribute("Type", "SUM_SQUARED_ERROR");

           const tinyxml2::XMLDocument* sum_squared_error_document = sum_squared_error_pointer->to_XML();

           const tinyxml2::XMLElement* sum_squared_error_element = sum_squared_error_document->FirstChildElement("SumSquaredError");

           DeepClone(objective_element, sum_squared_error_element, document, NULL);

           delete sum_squared_error_document;
      }
      break;

      case MEAN_SQUARED_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

            objective_element->SetAttribute("Type", "MEAN_SQUARED_ERROR");

            const tinyxml2::XMLDocument* mean_squared_error_document = mean_squared_error_pointer->to_XML();

            const tinyxml2::XMLElement* mean_squared_error_element = mean_squared_error_document->FirstChildElement("MeanSquaredError");

            DeepClone(objective_element, mean_squared_error_element, document, NULL);

            delete mean_squared_error_document;
      }
      break;

      case ROOT_MEAN_SQUARED_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

            objective_element->SetAttribute("Type", "ROOT_MEAN_SQUARED_ERROR");

            const tinyxml2::XMLDocument* root_mean_squared_error_document = root_mean_squared_error_pointer->to_XML();

            const tinyxml2::XMLElement* root_mean_squared_error_element = root_mean_squared_error_document->FirstChildElement("RootMeanSquaredError");

            DeepClone(objective_element, root_mean_squared_error_element, document, NULL);

            delete root_mean_squared_error_document;
      }
      break;

      case NORMALIZED_SQUARED_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

           objective_element->SetAttribute("Type", "NORMALIZED_SQUARED_ERROR");

           const tinyxml2::XMLDocument* normalized_squared_error_document = normalized_squared_error_pointer->to_XML();

           const tinyxml2::XMLElement* normalized_squared_error_element = normalized_squared_error_document->FirstChildElement("NormalizedSquaredError");

           DeepClone(objective_element, normalized_squared_error_element, document, NULL);

           delete normalized_squared_error_document;
      }
      break;

       case WEIGHTED_SQUARED_ERROR:
       {
            tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
            performance_functional_element->LinkEndChild(objective_element);

            objective_element->SetAttribute("Type", "WEIGHTED_SQUARED_ERROR");

            const tinyxml2::XMLDocument* weighted_squared_error_document = weighted_squared_error_pointer->to_XML();

            const tinyxml2::XMLElement* weighted_squared_error_element = weighted_squared_error_document->FirstChildElement("WeightedSquaredError");

            DeepClone(objective_element, weighted_squared_error_element, document, NULL);

            delete weighted_squared_error_document;
       }
       break;

      case MINKOWSKI_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

           objective_element->SetAttribute("Type", "MINKOWSKI_ERROR");

           const tinyxml2::XMLDocument* Minkowski_error_document = Minkowski_error_pointer->to_XML();

           const tinyxml2::XMLElement* Minkowski_error_element = Minkowski_error_document->FirstChildElement("MinkowskiError");

           DeepClone(objective_element, Minkowski_error_element, document, NULL);

           delete Minkowski_error_document;
      }
      break;

      case CROSS_ENTROPY_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           performance_functional_element->LinkEndChild(objective_element);

           objective_element->SetAttribute("Type", "CROSS_ENTROPY_ERROR");

           const tinyxml2::XMLDocument* cross_entropy_error_document = cross_entropy_error_pointer->to_XML();

           const tinyxml2::XMLElement* cross_entropy_error_element = cross_entropy_error_document->FirstChildElement("CrossEntropyError");

           DeepClone(objective_element, cross_entropy_error_element, document, NULL);

           delete cross_entropy_error_document;
      }
      break;

      case USER_ERROR:
      {
         // Do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
                << "Unknown error type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Regularization

   switch(regularization_type)
   {
      case NO_REGULARIZATION:
      {
           tinyxml2::XMLElement* regularization_element = document->NewElement("Regularization");
           performance_functional_element->LinkEndChild(regularization_element);

           regularization_element->SetAttribute("Type", "NO_REGULARIZATION");
      }
      break;

      case NEURAL_PARAMETERS_NORM:
      {
           tinyxml2::XMLElement* regularization_element = document->NewElement("Regularization");
           performance_functional_element->LinkEndChild(regularization_element);

           regularization_element->SetAttribute("Type", "NEURAL_PARAMETERS_NORM");

           const tinyxml2::XMLDocument* neural_parameters_norm_document = neural_parameters_norm_pointer->to_XML();

           const tinyxml2::XMLElement* neural_parameters_norm_element = neural_parameters_norm_document->FirstChildElement("NeuralParametersNorm");

           DeepClone(regularization_element, neural_parameters_norm_element, document, NULL);

           delete neural_parameters_norm_document;
      }
      break;

      case OUTPUTS_INTEGRALS:
      {
           tinyxml2::XMLElement* regularization_element = document->NewElement("OUTPUTS_INTEGRALS");
           performance_functional_element->LinkEndChild(regularization_element);

           regularization_element->SetAttribute("Type", "OUTPUTS_INTEGRALS");

           const tinyxml2::XMLDocument* outputs_integrals_document = outputs_integrals_pointer->to_XML();

           const tinyxml2::XMLElement* outputs_integrals_element = outputs_integrals_document->FirstChildElement("OutputsIntegrals");

           DeepClone(regularization_element, outputs_integrals_element, document, NULL);

           delete outputs_integrals_document;
      }
      break;

      case USER_REGULARIZATION:
      {
          // Do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
                << "Unknown regularization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Display

//   tinyxml2::XMLElement* display_element = document->NewElement("Display");
//   performance_functional_element->LinkEndChild(display_element);

//   buffer.str("");
//   buffer << display;

//   tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//   display_element->LinkEndChild(display_text);

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void PerformanceFunctional::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("PerformanceFunctional");

    // Objective

    switch(error_type)
    {
       case NO_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "NO_ERROR");

            file_stream.CloseElement();
       }
       break;

       case SUM_SQUARED_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "SUM_SQUARED_ERROR");

            sum_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case MEAN_SQUARED_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "MEAN_SQUARED_ERROR");

            mean_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case ROOT_MEAN_SQUARED_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "ROOT_MEAN_SQUARED_ERROR");

            root_mean_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case NORMALIZED_SQUARED_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "NORMALIZED_SQUARED_ERROR");

            normalized_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

        case WEIGHTED_SQUARED_ERROR:
        {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "WEIGHTED_SQUARED_ERROR");

            weighted_squared_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
        }
        break;

       case MINKOWSKI_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "MINKOWSKI_ERROR");

            Minkowski_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case CROSS_ENTROPY_ERROR:
       {
            file_stream.OpenElement("Objective");

            file_stream.PushAttribute("Type", "CROSS_ENTROPY_ERROR");

            cross_entropy_error_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case USER_ERROR:
       {
          // Do nothing
       }
       break;

       default:
       {
          std::ostringstream buffer;

          file_stream.CloseElement();

          buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                 << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
                 << "Unknown error type.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }

    // Regularization

    switch(regularization_type)
    {
       case NO_REGULARIZATION:
       {
            file_stream.OpenElement("Regularization");

            file_stream.PushAttribute("Type", "NO_REGULARIZATION");

            file_stream.CloseElement();
       }
       break;

       case NEURAL_PARAMETERS_NORM:
       {
            file_stream.OpenElement("Regularization");

            file_stream.PushAttribute("Type", "NEURAL_PARAMETERS_NORM");

            neural_parameters_norm_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case OUTPUTS_INTEGRALS:
       {
            file_stream.OpenElement("Regularization");

            file_stream.PushAttribute("Type", "OUTPUTS_INTEGRALS");

            outputs_integrals_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case USER_REGULARIZATION:
       {
           // Do nothing
       }
       break;

       default:
       {
          std::ostringstream buffer;

          file_stream.CloseElement();

          buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                 << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
                 << "Unknown regularization type.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }

    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Sets the performance functional member data from an XML document.
/// @param document Pointer to a TinyXML document with the performance functional data.

void PerformanceFunctional::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* performance_functional_element = document.FirstChildElement("PerformanceFunctional");

    if(!performance_functional_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PerformanceFunctional class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Performance functional element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

   // Objective type

   const tinyxml2::XMLElement* objective_element = performance_functional_element->FirstChildElement("Objective");

   if(objective_element)
   {
       const std::string new_error_type = objective_element->Attribute("Type");

       set_error_type(new_error_type);

       switch(error_type)
       {
          case NO_ERROR:
          {
             // Do nothing
          }
          break;

          case SUM_SQUARED_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("SumSquaredError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               sum_squared_error_pointer->from_XML(new_document);
          }
          break;

          case MEAN_SQUARED_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("MeanSquaredError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               mean_squared_error_pointer->from_XML(new_document);
          }
          break;

          case ROOT_MEAN_SQUARED_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("RootMeanSquaredError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               root_mean_squared_error_pointer->from_XML(new_document);
          }
          break;

          case NORMALIZED_SQUARED_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("NormalizedSquaredError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               normalized_squared_error_pointer->from_XML(new_document);
          }
          break;

          case WEIGHTED_SQUARED_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("WeightedSquaredError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               weighted_squared_error_pointer->from_XML(new_document);
          }
          break;

          case MINKOWSKI_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("MinkowskiError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               Minkowski_error_pointer->from_XML(new_document);
          }
          break;

          case CROSS_ENTROPY_ERROR:
          {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("CrossEntropyError");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, objective_element, &new_document, NULL);

               cross_entropy_error_pointer->from_XML(new_document);
          }
          break;

          case USER_ERROR:
          {
             //user_error_pointer = NULL;
          }
          break;

          default:
          {
             std::ostringstream buffer;

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
          }
          break;
       }
   }    

   // Regularization type

   const tinyxml2::XMLElement* regularization_element = performance_functional_element->FirstChildElement("Regularization");

   if(regularization_element)
   {
      const std::string new_regularization_type = regularization_element->Attribute("Type");

      set_regularization_type(new_regularization_type);

      switch(regularization_type)
      {
         case NO_REGULARIZATION:
         {
            // Do nothing
         }
         break;

         case NEURAL_PARAMETERS_NORM:
         {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("NeuralParametersNorm");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, regularization_element, &new_document, NULL);

               neural_parameters_norm_pointer->from_XML(new_document);
         }
         break;

         case OUTPUTS_INTEGRALS:
         {
               tinyxml2::XMLDocument new_document;

               tinyxml2::XMLElement* element_clone = new_document.NewElement("OutputsIntegrals");
               new_document.InsertFirstChild(element_clone);

               DeepClone(element_clone, regularization_element, &new_document, NULL);

               outputs_integrals_pointer->from_XML(new_document);
         }
         break;

         case USER_REGULARIZATION:
         {
             // Do nothing
         }
         break;

         default:
         {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
         }
         break;
      }

    }
    // Display

   const tinyxml2::XMLElement* display_element = performance_functional_element->FirstChildElement("Display");

   if(display_element)
   {
      std::string new_display_string = display_element->GetText();           

      try
      {
         set_display(new_display_string != "0");
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;		 
      }
   }
}


// std::string to_string(void) method

/// Writes to a string the members of the performance functional object in text format.

std::string PerformanceFunctional::to_string(void) const
{
    std::ostringstream buffer;

    buffer << "Performance functional\n"
           << "Objective type: " << write_error_type() << "\n";

    // Objective

     switch(error_type)
     {
         case NO_ERROR:
         {
             // Do nothing
         }
         break;

         case SUM_SQUARED_ERROR:
         {
             buffer << sum_squared_error_pointer->to_string();
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
             buffer << mean_squared_error_pointer->to_string();
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
             buffer << root_mean_squared_error_pointer->to_string();
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
             buffer << normalized_squared_error_pointer->to_string();
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
             buffer << weighted_squared_error_pointer->to_string();
         }
         break;

         case ROC_AREA_ERROR:
         {
             buffer << roc_area_error_pointer->to_string();
         }
         break;

         case MINKOWSKI_ERROR:
         {
             buffer << Minkowski_error_pointer->to_string();
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             buffer << cross_entropy_error_pointer->to_string();
         }
         break;

         case USER_ERROR:
         {
             buffer << user_error_pointer->to_string();
         }
         break;

         default:
         {
             buffer.str("");

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "std::string to_string(void) method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    // Regularization

     buffer << "Regularization type: " << write_regularization_type() << "\n";

     switch(regularization_type)
     {
         case NO_REGULARIZATION:
         {
             // Do nothing
         }
         break;

         case NEURAL_PARAMETERS_NORM:
         {
             buffer << neural_parameters_norm_pointer->to_string();
         }
         break;

         case OUTPUTS_INTEGRALS:
         {
             buffer << outputs_integrals_pointer->to_string();
         }
         break;

         case USER_REGULARIZATION:
         {
             buffer << user_regularization_pointer->to_string();
         }
         break;

         default:
         {
             buffer.str("");

             buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                    << "std::string to_string(void) method.\n"
                    << "Unknown regularization type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

    //buffer << "Display:" << display << "\n";

    return(buffer.str());
}


// void save(const std::string&) const method

/// Saves to a XML-type file a string representation of the performance functional object.
/// @param file_name Name of XML-type performance functional file. 

void PerformanceFunctional::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   // Declaration

//   TiXmlDeclaration* declaration = new TiXmlDeclaration("1.0", "", "");
//   document->LinkEndChild(declaration);

   // Performance functional

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a default performance functional XML-type file.
/// @param file_name Name of default XML-type performance functional file. 

void PerformanceFunctional::load(const std::string& file_name)
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      buffer << "OpenNN Exception: PerformanceFunctional class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}


// std::string write_information(void) method

/// Returns any useful information about the objective function during training.
/// By default it is an empty string.

std::string PerformanceFunctional::write_information(void)  
{
   std::ostringstream buffer;
   
   // Objective

    switch(error_type)
    {
        case NO_ERROR:
        {
            // Do nothing
        }
        break;

        case SUM_SQUARED_ERROR:
        {
            buffer << sum_squared_error_pointer->write_information();
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            buffer << mean_squared_error_pointer->write_information();
        }
        break;

        case ROOT_MEAN_SQUARED_ERROR:
        {
            buffer << root_mean_squared_error_pointer->write_information();
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            buffer << normalized_squared_error_pointer->write_information();
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            buffer << weighted_squared_error_pointer->write_information();
        }
        break;

        case ROC_AREA_ERROR:
        {
            buffer << roc_area_error_pointer->write_information();
        }
        break;

        case MINKOWSKI_ERROR:
        {
            buffer << Minkowski_error_pointer->write_information();
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            buffer << cross_entropy_error_pointer->write_information();
        }
        break;

        case USER_ERROR:
        {
            buffer << user_error_pointer->write_information();
        }
        break;

        default:
        {
            buffer.str("");

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "std::string write_information(void) method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

   // Regularization

    switch(regularization_type)
    {
        case NO_REGULARIZATION:
        {
            // Do nothing
        }
        break;

        case NEURAL_PARAMETERS_NORM:
        {
            buffer << neural_parameters_norm_pointer->write_information();
        }
        break;

        case OUTPUTS_INTEGRALS:
        {
            buffer << outputs_integrals_pointer->write_information();
        }
        break;

        case USER_REGULARIZATION:
        {
            buffer << user_regularization_pointer->write_information();
        }
        break;

        default:
        {
            buffer.str("");

            buffer << "OpenNN Exception: PerformanceFunctional class.\n"
                   << "std::string write_information(void) method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }


   return(buffer.str());
}


// void print(void) const method

/// Print the members of this object to the standard output.

void PerformanceFunctional::print(void) const
{
    std::cout << to_string();
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

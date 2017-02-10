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

#include "loss_index.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.  
/// It creates a loss functional object with all pointers initialized to NULL. 
/// It also initializes all the rest of class members to their default values.

LossIndex::LossIndex(void)
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
/// It creates a loss functional object associated to a neural network object. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer)
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
 , user_error_pointer(NULL)
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
/// It creates a loss functional object associated to a neural network and a data set objects. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
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
/// It creates a loss functional object associated to a neural network and a mathematical model objects. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer, MathematicalModel* new_mathematical_model_pointer)
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
/// It creates a loss functional object associated to a neural network, a mathematical model and a data set objects. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer, MathematicalModel* new_mathematical_model_pointer, DataSet* new_data_set_pointer)
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
/// It creates a loss functional object with a given objective functional.
/// The rest of pointers are initialized to NULL. 
/// The other members are set to their default values, but the error term type, which is set to USER_PERFORMANCE_TERM. 

LossIndex::LossIndex(ErrorTerm* new_user_error_pointer)
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
/// It creates a loss functional object by loading its members from an XML-type file.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param file_name Name of loss functional file.

LossIndex::LossIndex(const std::string& file_name)
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
/// It creates a loss functional object by loading its members from an XML document->
/// @param loss_index_document Pointer to a TinyXML document containing the loss functional data.

LossIndex::LossIndex(const tinyxml2::XMLDocument& loss_index_document)
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

   from_XML(loss_index_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing loss functional object. 
/// @param other_loss_index Loss index object to be copied.
/// @todo

LossIndex::LossIndex(const LossIndex& other_loss_index)
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
    neural_network_pointer = other_loss_index.neural_network_pointer;
    data_set_pointer = other_loss_index.data_set_pointer;
    mathematical_model_pointer = other_loss_index.mathematical_model_pointer;

   error_type = other_loss_index.error_type;
   regularization_type = other_loss_index.regularization_type;

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
            sum_squared_error_pointer = new SumSquaredError(*other_loss_index.sum_squared_error_pointer);
        }
        break;

        case MEAN_SQUARED_ERROR:
        {
            mean_squared_error_pointer = new MeanSquaredError(*other_loss_index.mean_squared_error_pointer);
        }
        break;

        case ROOT_MEAN_SQUARED_ERROR:
        {
            root_mean_squared_error_pointer = new RootMeanSquaredError(*other_loss_index.root_mean_squared_error_pointer);
        }
        break;

        case NORMALIZED_SQUARED_ERROR:
        {
            normalized_squared_error_pointer = new NormalizedSquaredError(*other_loss_index.normalized_squared_error_pointer);
        }
        break;

        case WEIGHTED_SQUARED_ERROR:
        {
            weighted_squared_error_pointer = new WeightedSquaredError(*other_loss_index.weighted_squared_error_pointer);
        }
        break;

        case ROC_AREA_ERROR:
        {
            roc_area_error_pointer = new RocAreaError(*other_loss_index.roc_area_error_pointer);
        }
        break;

        case MINKOWSKI_ERROR:
        {
            Minkowski_error_pointer = new MinkowskiError(*other_loss_index.Minkowski_error_pointer);
        }
        break;

        case CROSS_ENTROPY_ERROR:
        {
            cross_entropy_error_pointer = new CrossEntropyError(*other_loss_index.cross_entropy_error_pointer);
        }
        break;

        case USER_ERROR:
        {
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: LossIndex class.\n"
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
            neural_parameters_norm_pointer = new NeuralParametersNorm(*other_loss_index.neural_parameters_norm_pointer);
        }
        break;

        case OUTPUTS_INTEGRALS:
        {
            outputs_integrals_pointer = new OutputsIntegrals(*other_loss_index.outputs_integrals_pointer);
        }
        break;

        case USER_REGULARIZATION:
        {
            //user_regularization_pointer = new ErrorTerm(*other_loss_index.user_regularization_pointer);
        }
        break;

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "Copy constructor.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

   display = other_loss_index.display;  
}


// DESTRUCTOR

/// Destructor.
/// It deletes the objective, regularization and constraints terms. 

LossIndex::~LossIndex(void)
{
   // Delete error terms

   delete sum_squared_error_pointer;
   delete mean_squared_error_pointer;
   delete root_mean_squared_error_pointer;
   delete normalized_squared_error_pointer;
   delete Minkowski_error_pointer;
   delete cross_entropy_error_pointer;
   delete weighted_squared_error_pointer;
   delete roc_area_error_pointer;
   delete user_error_pointer;

    // Delete regularization terms

   delete neural_parameters_norm_pointer;
   delete outputs_integrals_pointer;
   delete user_regularization_pointer;
}


// METHODS


// bool has_neural_network(void) const method

/// Returns true if this loss functional has a neural network associated,
/// and false otherwise.

bool LossIndex::has_neural_network(void) const
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

/// Returns true if this loss functional has a mathematical model associated,
/// and false otherwise.

bool LossIndex::has_mathematical_model(void) const
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

/// Returns true if this loss functional has a data set associated,
/// and false otherwise.

bool LossIndex::has_data_set(void) const
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

/// Returns true if this loss functional has a selection method defined,
/// and false otherwise.

bool LossIndex::has_selection(void) const
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

/// Returns true if the loss functional can be expressed as the sum of squared terms.
/// Only those loss functionals are suitable for the Levenberg-Marquardt training algorithm.

bool LossIndex::is_sum_squared_terms(void) const
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

/// Throws an exception if no neural network is associated to the loss functional.

void LossIndex::check_neural_network(void) const
{
    if(!neural_network_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "void check_neural_network(void) const.\n"
              << "Pointer to neural network is NULL.\n";

       throw std::logic_error(buffer.str());
    }
}


// void check_error_terms(void) const method

/// Throws an exception if the loss functional has not got any
/// objective, regularization or constraints terms.

void LossIndex::check_error_terms(void) const
{
    if(error_type == NO_ERROR
    && regularization_type == NO_REGULARIZATION)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void check_error_terms(void) const method.\n"
               << "None objective, regularization or constraints terms are used.\n";

        throw std::logic_error(buffer.str());

    }
}


// SumSquaredError* get_sum_squared_error_pointer(void) const method

/// Returns a pointer to the sum squared error which is used as objective.
/// If that object does not exists, an exception is thrown.

SumSquaredError* LossIndex::get_sum_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!sum_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

MeanSquaredError* LossIndex::get_mean_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!mean_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

RootMeanSquaredError* LossIndex::get_root_mean_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!root_mean_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

NormalizedSquaredError* LossIndex::get_normalized_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!normalized_squared_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

MinkowskiError* LossIndex::get_Minkowski_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!Minkowski_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

CrossEntropyError* LossIndex::get_cross_entropy_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!cross_entropy_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

WeightedSquaredError* LossIndex::get_weighted_squared_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!cross_entropy_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

RocAreaError* LossIndex::get_roc_area_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!roc_area_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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


OutputsIntegrals* LossIndex::get_outputs_integrals_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!outputs_integrals_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "OutputsIntegrals* get_outputs_integrals_pointer(void) const method.\n"
              << "Pointer to outputs integrals objective is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(outputs_integrals_pointer);
}


// ErrorTerm* get_user_error_pointer(void) const method

/// Returns a pointer to the user error term which is used as objective.
/// If that object does not exists, an exception is thrown.

ErrorTerm* LossIndex::get_user_error_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!user_error_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

NeuralParametersNorm* LossIndex::get_neural_parameters_norm_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!neural_parameters_norm_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
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

RegularizationTerm* LossIndex::get_user_regularization_pointer(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(!user_regularization_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: LossIndex class.\n"
              << "ErrorTerm* get_user_regularization_pointer(void) const method.\n"
              << "Pointer to user regularization is NULL.\n";

       throw std::logic_error(buffer.str());
     }

     #endif

    return(user_regularization_pointer);
}


// const ErrorType& get_error_type(void) const method

/// Returns the type of objective term used in the loss functional expression.

const LossIndex::ErrorType& LossIndex::get_error_type(void) const
{
   return(error_type);
}


// const RegularizationType& get_regularization_type(void) const method

/// Returns the type of regularization term used in the loss functional expression.

const LossIndex::RegularizationType& LossIndex::get_regularization_type(void) const
{
   return(regularization_type);
}


// std::string write_error_type(void) const

/// Returns a string with the type of objective term used in the loss functional expression.

std::string LossIndex::write_error_type(void) const
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

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "std::string write_error_type(void) const method.\n"
             << "Unknown error type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string write_regularization_type(void) const method

/// Returns a string with the type of regularization term used in the loss functional expression.

std::string LossIndex::write_regularization_type(void) const
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

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "std::string write_regularization_type(void) const method.\n"
             << "Unknown regularization type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string write_error_type_text(void) const

/// Returns a string in text format with the type of objective term used in the loss functional expression.

std::string LossIndex::write_error_type_text(void) const
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

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "std::string write_error_type_text(void) const method.\n"
             << "Unknown error type.\n";

      throw std::logic_error(buffer.str());
   }
}


// std::string write_regularization_type_text(void) const method

/// Returns a string in text format with the type of regularization term used in the loss functional expression.

std::string LossIndex::write_regularization_type_text(void) const
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

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "std::string write_regularization_type_text(void) const method.\n"
             << "Unknown regularization type.\n";

      throw std::logic_error(buffer.str());
   }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages
/// from this class can't be displayed on the screen.

const bool& LossIndex::get_display(void) const
{
   return(display);
}


// void set_neural_network_pointer(NeuralNetwork*) method

/// Sets a pointer to a multilayer perceptron object which is to be associated to the loss functional.
/// @param new_neural_network_pointer Pointer to a neural network object to be associated to the loss functional.

void LossIndex::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
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

            buffer << "OpenNN Exception: LossIndex class.\n"
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

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void set_neural_network_pointer(NeuralNetwork*) method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }
}


// void set_mathematical_model_pointer(MathematicalModel*) method

/// Sets a new mathematical model on which it will be measured the loss functional.
/// @param new_mathematical_model_pointer Pointer to an external mathematical model object.

void LossIndex::set_mathematical_model_pointer(MathematicalModel* new_mathematical_model_pointer)
{
   mathematical_model_pointer = new_mathematical_model_pointer;
}


// void set_data_set_pointer(DataSet*) method

/// Sets a new data set on which it will be measured the loss functional.
/// @param new_data_set_pointer Pointer to an external data set object.

void LossIndex::set_data_set_pointer(DataSet* new_data_set_pointer)
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

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void set_data_set_pointer(DataSet*) method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }
}


// void set_user_error_pointer(ErrorTerm*) method

/// Sets the error term to be a specialized one provided by the user.
/// @param new_user_error_pointer Pointer to a error term object.

void LossIndex::set_user_error_pointer(ErrorTerm* new_user_error_pointer)
{
    destruct_error_term();

    error_type = USER_ERROR;

    user_error_pointer = new_user_error_pointer;
}


// void set_user_regularization_pointer(RegularizationTerm*) method

/// Sets the regularization term to be a specialized one provided by the user.
/// @param new_user_regularization_pointer Pointer to a regularization term object.

void LossIndex::set_user_regularization_pointer(RegularizationTerm* new_user_regularization_pointer)
{
    destruct_regularization_term();

    regularization_type = USER_REGULARIZATION;

    user_regularization_pointer = new_user_regularization_pointer;
}


// void set_default(void) method

/// Sets the members of the loss functional object to their default values.

void LossIndex::set_default(void)
{
   display = true;
}

#ifdef __OPENNN_MPI__
// void set_MPI(DataSet*, NeuralNetwork*, const LossIndex*) method

///
/// @param new_data_set
/// @param new_neural_network
/// @param loss_index

void LossIndex::set_MPI(DataSet* new_data_set, NeuralNetwork* new_neural_network, const LossIndex* loss_index)
{
    set_data_set_pointer(new_data_set);
    set_neural_network_pointer(new_neural_network);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int original_error_type;
    int original_regularization_type;

    double positives_weight, negatives_weight;
    double minkowski_parameter;

    double regularization_weight;

    if(rank == 0)
    {
        // Variables to send initialization

        original_error_type = (int)loss_index->get_error_type();
        original_regularization_type = (int)loss_index->get_regularization_type();

        if(loss_index->get_error_type() == LossIndex::WEIGHTED_SQUARED_ERROR)
        {
            positives_weight = loss_index->get_weighted_squared_error_pointer()->get_positives_weight();
            negatives_weight = loss_index->get_weighted_squared_error_pointer()->get_negatives_weight();
        }
        else if(loss_index->get_error_type() == LossIndex::MINKOWSKI_ERROR)
        {
            minkowski_parameter = loss_index->get_Minkowski_error_pointer()->get_Minkowski_parameter();
        }

        if(loss_index->get_regularization_type() == LossIndex::NEURAL_PARAMETERS_NORM)
        {
            regularization_weight = loss_index->get_neural_parameters_norm_pointer()->get_neural_parameters_norm_weight();
        }
    }

    // Send variables

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank > 0)
    {
        MPI_Request req[2];

        MPI_Irecv(&original_error_type, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&original_regularization_type, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);

        if(original_error_type == (int)LossIndex::WEIGHTED_SQUARED_ERROR)
        {
            MPI_Irecv(&positives_weight, 1, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(&negatives_weight, 1, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &req[1]);

            MPI_Waitall(2, req, MPI_STATUS_IGNORE);
        }
        else if(original_error_type == (int)LossIndex::MINKOWSKI_ERROR)
        {
            MPI_Irecv(&minkowski_parameter, 1, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &req[0]);

            MPI_Waitall(1, req, MPI_STATUS_IGNORE);
        }

        if(original_regularization_type == (int)LossIndex::NEURAL_PARAMETERS_NORM)
        {
            MPI_Recv(&regularization_weight, 1, MPI_DOUBLE, rank-1, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if(rank < size-1)
    {
        MPI_Request req[4];

        MPI_Isend(&original_error_type, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&original_regularization_type, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);

        if(original_error_type == (int)LossIndex::WEIGHTED_SQUARED_ERROR)
        {
            MPI_Isend(&positives_weight, 1, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &req[0]);
            MPI_Isend(&negatives_weight, 1, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &req[1]);

            MPI_Waitall(2, req, MPI_STATUS_IGNORE);
        }
        else if(original_error_type == (int)LossIndex::MINKOWSKI_ERROR)
        {
            MPI_Isend(&minkowski_parameter, 1, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &req[0]);

            MPI_Waitall(1, req, MPI_STATUS_IGNORE);
        }

        if(original_regularization_type == (int)LossIndex::NEURAL_PARAMETERS_NORM)
        {
            MPI_Send(&regularization_weight, 1, MPI_DOUBLE, rank+1, 5, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    set_error_type((LossIndex::ErrorType)original_error_type);
    set_regularization_type((LossIndex::RegularizationType)original_regularization_type);

    if(original_error_type == (int)LossIndex::WEIGHTED_SQUARED_ERROR)
    {
        get_weighted_squared_error_pointer()->set_positives_weight(positives_weight);
        get_weighted_squared_error_pointer()->set_negatives_weight(negatives_weight);
    }
    else if(original_error_type == (int)LossIndex::MINKOWSKI_ERROR)
    {
        get_Minkowski_error_pointer()->set_Minkowski_parameter(minkowski_parameter);
    }

    if(original_regularization_type == (int)LossIndex::NEURAL_PARAMETERS_NORM)
    {
        get_neural_parameters_norm_pointer()->set_neural_parameters_norm_weight(regularization_weight);
    }
}
#endif

// void set_error_type(const std::string&) method

/// Sets a new type for the error term from a string.
/// @param new_error_type String with the type of objective term.

void LossIndex::set_error_type(const std::string& new_error_type)
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

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "void set_error_type(const std::string&) method.\n"
             << "Unknown objective type: " << new_error_type << ".\n";

      throw std::logic_error(buffer.str());
   }   
}


// void set_regularization_type(const std::string&) method

/// Sets a new type for the regularization term from a string.
/// @param new_regularization_type String with the type of regularization term.

void LossIndex::set_regularization_type(const std::string& new_regularization_type)
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

      buffer << "OpenNN Exception: LossIndex class.\n"
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

void LossIndex::set_display(const bool& new_display)
{
   display = new_display;
}


// void set_error_type(const ErrorType&) method

/// Creates a new objective term inside the loss functional of a given error term type.
/// @param new_error_type Type of objective term to be created.

void LossIndex::set_error_type(const ErrorType& new_error_type)
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

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void set_error_type(const ErrorType&) method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
            break;
    }
}


// void set_regularization_type(const RegularizationType&) method

/// Creates a new regularization term inside the loss functional of a given error term type.
/// @param new_regularization_type Type of regularization term to be created.

void LossIndex::set_regularization_type(const RegularizationType& new_regularization_type)
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

         buffer << "OpenNN Exception: LossIndex class.\n"
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

void LossIndex::destruct_error_term(void)
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

void LossIndex::destruct_regularization_term(void)
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

void LossIndex::destruct_all_terms(void)
{
   destruct_error_term();
   destruct_regularization_term();
}


// double calculate_error(void) const method

/// Returns the objective evaluation,
/// according to the respective objective type used in the loss functional expression.

double LossIndex::calculate_error(void) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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
/// according to the respective objective type used in the loss functional expression.

double LossIndex::calculate_error(const Vector<double>& parameters) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "double calculate_error(const Vector<double>&) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(objective);
}

#ifdef __OPENNN_MPI__

double LossIndex::calculate_error_MPI(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int training_instances_number = 0;
    int local_training_instances_number = 0;

    if(has_data_set() && data_set_pointer->has_data())
    {
        local_training_instances_number = (int)data_set_pointer->get_instances_pointer()->count_training_instances_number();
    }
    MPI_Allreduce(&local_training_instances_number, &training_instances_number, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(training_instances_number == 0)
    {
        return 0.0;
    }

    size = std::min(size,training_instances_number);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int* ranks = (int*)malloc(size*sizeof(int));

    for(int i = 0; i < size; i++)
    {
        ranks[i] = i;
    }

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group error_group;
    MPI_Group_incl(world_group, size, ranks, &error_group);

    // Create a new communicator based on the group
    MPI_Comm current_comm;
    MPI_Comm_create(MPI_COMM_WORLD, error_group, &current_comm);

    double global_objective = 0.0;

    double local_objective = 0.0;

    double normalization_coefficient = 0.0;

    // Objective

    if(rank < size)
    {
         switch(error_type)
         {
             case NO_ERROR:
             {
                 // Do nothing
             }
             break;

             case SUM_SQUARED_ERROR:
             {
                local_objective = sum_squared_error_pointer->calculate_error();
             }
             break;

             case MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_objective = temp_loss_index.calculate_error();

                 local_objective /= training_instances_number;
             }
             break;

             case ROOT_MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_objective = temp_loss_index.calculate_error();

                 local_objective /= training_instances_number;
             }
             break;

             case NORMALIZED_SQUARED_ERROR:
             {
                 const size_t targets_number = data_set_pointer->get_variables_pointer()->count_targets_number();

                 Vector<double> training_target_data_mean(targets_number, 0.0);

                 const Vector<double> local_training_target_data_mean = data_set_pointer->arrange_training_target_data().calculate_rows_sum();

                 MPI_Allreduce(local_training_target_data_mean.data(), training_target_data_mean.data(), (int)targets_number, MPI_DOUBLE, MPI_SUM, current_comm);

                 training_target_data_mean /= training_instances_number;

                 const Vector<double> error_normalization = get_normalized_squared_error_pointer()->calculate_error_normalization(training_target_data_mean);

                 MPI_Reduce(&error_normalization[1], &normalization_coefficient, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

                 local_objective = error_normalization[0];
             }
             break;

             case WEIGHTED_SQUARED_ERROR:
             {
                 const Vector<size_t> targets_indices = data_set_pointer->get_variables_pointer()->arrange_targets_indices();

                 const int local_training_negatives = (int)data_set_pointer->calculate_training_negatives(targets_indices[0]);

                 int negatives = 0;
                 MPI_Allreduce(&local_training_negatives, &negatives, 1, MPI_INT, MPI_SUM, current_comm);

                 normalization_coefficient = negatives*weighted_squared_error_pointer->get_negatives_weight()*0.5;
                 local_objective = weighted_squared_error_pointer->calculate_error(normalization_coefficient);
             }
             break;

             case ROC_AREA_ERROR: // todo
             {
    //             local_objective = roc_area_error_pointer->calculate_error();
             }
             break;

             case MINKOWSKI_ERROR:
             {
                 local_objective = Minkowski_error_pointer->calculate_error();
             }
             break;

             case CROSS_ENTROPY_ERROR:
             {
                 local_objective = cross_entropy_error_pointer->calculate_error_unnormalized();

                 local_objective /= training_instances_number;
             }
             break;

             case USER_ERROR:
             {
                 local_objective = user_error_pointer->calculate_error();
             }
             break;

             default:
             {
                 std::ostringstream buffer;

                 buffer << "OpenNN Exception: LossIndex class.\n"
                        << "double calculate_error_MPI(void) const method.\n"
                        << "Unknown error type.\n";

                 MPI_Abort(MPI_COMM_WORLD, 1);

                 throw std::logic_error(buffer.str());
             }
             break;
         }

         MPI_Reduce(&local_objective, &global_objective, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

         MPI_Barrier(current_comm);
    }
     if(rank == 0)
     {
         if(error_type == (int)LossIndex::ROOT_MEAN_SQUARED_ERROR)
         {
             global_objective = sqrt(global_objective);
         }
         else if(error_type == (int)LossIndex::NORMALIZED_SQUARED_ERROR)
         {
             if(normalization_coefficient < 1.0e-99)
             {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
                       << "double calculate_selection_loss(void) const method.\n"
                       << "Normalization coefficient is zero.\n"
                       << "Unuse constant target variables or choose another error functional. ";

                MPI_Abort(MPI_COMM_WORLD, 1);

                throw std::logic_error(buffer.str());
             }

             global_objective /= normalization_coefficient;
         }
     }

     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Comm_free(&current_comm);
     MPI_Group_free(&error_group);
     free(ranks);

     return(global_objective);
}

double LossIndex::calculate_error_MPI(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    int initialized;

    MPI_Initialized(&initialized);

    if(!initialized)
    {
      MPI_Init(NULL, NULL);
    }

    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int training_instances_number = 0;
    int local_training_instances_number = 0;

    if(has_data_set() && data_set_pointer->has_data())
    {
        local_training_instances_number = (int)data_set_pointer->get_instances_pointer()->count_training_instances_number();
    }
    MPI_Allreduce(&local_training_instances_number, &training_instances_number, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    size = std::min(size,training_instances_number);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int* ranks = (int*)malloc(size*sizeof(int));

    for(int i = 0; i < size; i++)
    {
        ranks[i] = i;
    }

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group error_group;
    MPI_Group_incl(world_group, size, ranks, &error_group);

    // Create a new communicator based on the group
    MPI_Comm current_comm;
    MPI_Comm_create(MPI_COMM_WORLD, error_group, &current_comm);

    double global_objective = 0.0;

    double local_objective = 0.0;

    double normalization_coefficient = 0.0;

    // Objective

    if(rank < size)
    {
         switch(error_type)
         {
             case NO_ERROR:
             {
                 // Do nothing
             }
             break;

             case SUM_SQUARED_ERROR:
             {
                local_objective = sum_squared_error_pointer->calculate_error(parameters);
             }
             break;

             case MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_objective = temp_loss_index.calculate_error(parameters);

                 local_objective /= training_instances_number;
             }
             break;

             case ROOT_MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_objective = temp_loss_index.calculate_error(parameters);

                 local_objective /= training_instances_number;
             }
             break;

             case NORMALIZED_SQUARED_ERROR:
             {
                 const size_t targets_number = data_set_pointer->get_variables_pointer()->count_targets_number();

                 Vector<double> training_target_data_mean(targets_number, 0.0);

                 const Vector<double> local_training_target_data_mean = data_set_pointer->arrange_training_target_data().calculate_rows_sum();

                 MPI_Allreduce(local_training_target_data_mean.data(), training_target_data_mean.data(), (int)targets_number, MPI_DOUBLE, MPI_SUM, current_comm);

                 training_target_data_mean /= training_instances_number;

                 const Vector<double> error_normalization = get_normalized_squared_error_pointer()->calculate_error_normalization(parameters, training_target_data_mean);

                 MPI_Reduce(&error_normalization[1], &normalization_coefficient, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

                 local_objective = error_normalization[0];
             }
             break;

             case WEIGHTED_SQUARED_ERROR:
             {
                 const Vector<size_t> targets_indices = data_set_pointer->get_variables_pointer()->arrange_targets_indices();

                 const int local_training_negatives = (int)data_set_pointer->calculate_training_negatives(targets_indices[0]);

                 int negatives = 0;
                 MPI_Allreduce(&local_training_negatives, &negatives, 1, MPI_INT, MPI_SUM, current_comm);

                 normalization_coefficient = negatives*weighted_squared_error_pointer->get_negatives_weight()*0.5;
                 local_objective = weighted_squared_error_pointer->calculate_error(parameters, normalization_coefficient);
             }
             break;

             case ROC_AREA_ERROR: // todo
             {
    //             local_objective = roc_area_error_pointer->calculate_error();
             }
             break;

             case MINKOWSKI_ERROR:
             {
                 local_objective = Minkowski_error_pointer->calculate_error(parameters);
             }
             break;

             case CROSS_ENTROPY_ERROR:
             {
                 local_objective = cross_entropy_error_pointer->calculate_error_unnormalized(parameters);

                 local_objective /= training_instances_number;
             }
             break;

             case USER_ERROR:
             {
                 local_objective = user_error_pointer->calculate_error(parameters);
             }
             break;

             default:
             {
                 std::ostringstream buffer;

                 buffer << "OpenNN Exception: LossIndex class.\n"
                        << "double calculate_error_MPI(const Vector<double>&) const method.\n"
                        << "Unknown error type.\n";

                 MPI_Abort(MPI_COMM_WORLD, 1);

                 throw std::logic_error(buffer.str());
             }
             break;
         }

         MPI_Reduce(&local_objective, &global_objective, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);


         MPI_Barrier(current_comm);
    }

     if(rank == 0)
     {
         if(error_type == (int)LossIndex::ROOT_MEAN_SQUARED_ERROR)
         {
             global_objective = sqrt(global_objective);
         }
         else if(error_type == (int)LossIndex::NORMALIZED_SQUARED_ERROR)
         {
             if(normalization_coefficient < 1.0e-99)
             {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
                       << "double calculate_selection_loss(void) const method.\n"
                       << "Normalization coefficient is zero.\n"
                       << "Unuse constant target variables or choose another error functional. ";

                MPI_Abort(MPI_COMM_WORLD, 1);

                throw std::logic_error(buffer.str());
             }

             global_objective /= normalization_coefficient;
         }
     }

     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Comm_free(&current_comm);
     MPI_Group_free(&error_group);
     free(ranks);

     return(global_objective);
}
#endif


// double calculate_regularization(void) const method

/// Returns the regularization evaluation,
/// according to the respective regularization type used in the loss functional expression.

double LossIndex::calculate_regularization(void) const
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

            buffer << "OpenNN Exception: LossIndex class.\n"
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
/// according to the respective regularization type used in the loss functional expression.

double LossIndex::calculate_regularization(const Vector<double>& parameters) const
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

            buffer << "OpenNN Exception: LossIndex class.\n"
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
/// according to the respective objective type used in the loss functional expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.

Vector<double> LossIndex::calculate_error_terms(void) const
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
             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Cannot calculate error terms for root mean squared error objective.\n";

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
             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Cannot calculate error terms for Minkowski error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Vector<double> calculate_error_terms(void) const method.\n"
                    << "Cannot calculate error terms for cross-entropy error objective.\n";

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
             buffer << "OpenNN Exception: LossIndex class.\n"
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
/// according to the objective type used in the loss functional expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.
/// The Jacobian elements are the partial derivatives of a single term with respect to a single parameter.
/// The number of rows in the Jacobian matrix are the number of parameters,
/// and the number of columns the number of terms composing the objective.

Matrix<double> LossIndex::calculate_error_terms_Jacobian(void) const
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
             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                    << "Cannot calculate error terms for root mean squared error objective.\n";

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
             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                    << "Cannot calculate error terms for Minkowski error objective.\n";

             throw std::logic_error(buffer.str());
         }
         break;

     case CROSS_ENTROPY_ERROR:
     {
         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Matrix<double> calculate_error_terms_Jacobian(void) const method.\n"
                << "Cannot calculate error terms for cross-entropy error objective.\n";

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
             buffer << "OpenNN Exception: LossIndex class.\n"
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

Vector<double> LossIndex::calculate_error_gradient(void) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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

Vector<double> LossIndex::calculate_error_gradient(const Vector<double>& parameters) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Vector<double> calculate_error_gradient(const Vector<double>&) const method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(gradient);
}

#ifdef __OPENNN_MPI__

Vector<double> LossIndex::calculate_error_gradient_MPI(void) const
{

    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int training_instances_number = 0;
    int local_training_instances_number = 0;

    if(has_data_set() && data_set_pointer->has_data())
    {
        local_training_instances_number = (int)data_set_pointer->get_instances_pointer()->count_training_instances_number();
    }
    MPI_Allreduce(&local_training_instances_number, &training_instances_number, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    size = std::min(size,training_instances_number);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int* ranks = (int*)malloc(size*sizeof(int));

    for(int i = 0; i < size; i++)
    {
        ranks[i] = i;
    }

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group error_group;
    MPI_Group_incl(world_group, size, ranks, &error_group);

    // Create a new communicator based on the group
    MPI_Comm current_comm;
    MPI_Comm_create(MPI_COMM_WORLD, error_group, &current_comm);

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Vector<double> local_gradient(parameters_number, 0.0);
    Vector<double> global_gradient(parameters_number, 0.0);

    double normalization_coefficient = 0.0;
    double error = 0.0;

    if(error_type == ROOT_MEAN_SQUARED_ERROR)
    {
        error = calculate_error_MPI();

        MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Objective

    if(rank < size)
    {
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
                 local_gradient = sum_squared_error_pointer->calculate_gradient();
             }
             break;

             case MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_gradient = temp_loss_index.get_sum_squared_error_pointer()->calculate_gradient();

                 local_gradient /= training_instances_number;
             }
             break;

             case ROOT_MEAN_SQUARED_ERROR:
             {
                 local_gradient = root_mean_squared_error_pointer->calculate_gradient(training_instances_number, error);
             }
             break;

             case NORMALIZED_SQUARED_ERROR:
             {
                 const size_t targets_number = data_set_pointer->get_variables_pointer()->count_targets_number();

                 Vector<double> training_target_data_mean(targets_number, 0.0);

                 const Vector<double> local_training_target_data_mean = data_set_pointer->arrange_training_target_data().calculate_rows_sum();

                 MPI_Allreduce(local_training_target_data_mean.data(), training_target_data_mean.data(), (int)targets_number, MPI_DOUBLE, MPI_SUM, current_comm);

                 training_target_data_mean /= training_instances_number;

                 Vector<double> error_normalization = normalized_squared_error_pointer->calculate_gradient_normalization(training_target_data_mean);

                 const double local_normalization_coefficient = error_normalization[error_normalization.size() - 1];

                 MPI_Reduce(&local_normalization_coefficient, &normalization_coefficient, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

                 error_normalization.pop_back();
                 local_gradient = error_normalization;
             }
             break;

             case WEIGHTED_SQUARED_ERROR:
             {
                 const Vector<size_t> targets_indices = data_set_pointer->get_variables_pointer()->arrange_targets_indices();

                 const int local_training_negatives = (int)data_set_pointer->calculate_training_negatives(targets_indices[0]);

                 int negatives = 0;
                 MPI_Allreduce(&local_training_negatives, &negatives, 1, MPI_INT, MPI_SUM, current_comm);

                 normalization_coefficient = negatives*weighted_squared_error_pointer->get_negatives_weight()*0.5;

                 local_gradient = weighted_squared_error_pointer->calculate_gradient_with_normalization(normalization_coefficient);
             }
             break;

             case ROC_AREA_ERROR: // todo
             {
//                 local_gradient = roc_area_error_pointer->calculate_gradient();
             }
             break;

             case MINKOWSKI_ERROR:
             {
                 local_gradient = Minkowski_error_pointer->calculate_gradient();
             }
             break;

             case CROSS_ENTROPY_ERROR: // todo
             {
                 const int local_training_instances_number = (int)data_set_pointer->get_instances_pointer()->count_training_instances_number();
                 int training_instances_number = 0;

                 MPI_Allreduce(&local_training_instances_number, &training_instances_number, 1, MPI_INT, MPI_SUM, current_comm);

                 local_gradient = cross_entropy_error_pointer->calculate_gradient_unnormalized();

                 local_gradient /= training_instances_number;
             }
             break;

             case USER_ERROR:
             {
                 local_gradient = user_error_pointer->calculate_gradient();
             }
             break;

             default:
             {
                 std::ostringstream buffer;

                 buffer << "OpenNN Exception: LossIndex class.\n"
                        << "Vector<double> calculate_error_gradient_MPI(void) const method.\n"
                        << "Unknown error type.\n";

                 MPI_Abort(MPI_COMM_WORLD, 1);

                 throw std::logic_error(buffer.str());
             }
             break;
         }

         MPI_Reduce(local_gradient.data(), global_gradient.data(), (int)parameters_number, MPI_DOUBLE, MPI_SUM, 0, current_comm);

         MPI_Barrier(current_comm);
    }

    if(rank == 0)
    {
        if(error_type == (int)LossIndex::NORMALIZED_SQUARED_ERROR)
        {
            if(normalization_coefficient < 1.0e-99)
            {
               std::ostringstream buffer;

               buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
                      << "double calculate_selection_loss(void) const method.\n"
                      << "Normalization coefficient is zero.\n"
                      << "Unuse constant target variables or choose another error functional. ";

               MPI_Abort(MPI_COMM_WORLD, 1);

               throw std::logic_error(buffer.str());
            }

            global_gradient /= normalization_coefficient;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&current_comm);
    MPI_Group_free(&error_group);
    free(ranks);

    return(global_gradient);

}

Vector<double> LossIndex::calculate_error_gradient_MPI(const Vector<double>& parameters) const
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
//             gradient = sum_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case MEAN_SQUARED_ERROR:
         {
//             gradient = mean_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case ROOT_MEAN_SQUARED_ERROR:
         {
//             gradient = root_mean_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case NORMALIZED_SQUARED_ERROR:
         {
//             gradient = normalized_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case WEIGHTED_SQUARED_ERROR:
         {
//             gradient = weighted_squared_error_pointer->calculate_gradient(parameters);
         }
         break;

         case ROC_AREA_ERROR:
         {
//             gradient = roc_area_error_pointer->calculate_gradient(parameters);
         }
         break;

         case MINKOWSKI_ERROR:
         {
//             gradient = Minkowski_error_pointer->calculate_gradient(parameters);
         }
         break;

         case CROSS_ENTROPY_ERROR:
         {
//             gradient = cross_entropy_error_pointer->calculate_gradient(parameters);
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

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Vector<double> calculate_error_gradient_MPI(const Vector<double>&) const method.\n"
                    << "Unknown error type.\n";

             MPI_Abort(MPI_COMM_WORLD, 1);

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(gradient);
}
#endif

// Vector<double> calculate_regularization_gradient(void) const method

/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters.

Vector<double> LossIndex::calculate_regularization_gradient(void) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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

Vector<double> LossIndex::calculate_regularization_gradient(const Vector<double>& parameters) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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

Matrix<double> LossIndex::calculate_error_Hessian(void) const
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

             buffer << "Matrix<double> Exception: LossIndex class.\n"
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

Matrix<double> LossIndex::calculate_error_Hessian(const Vector<double>& parameters) const
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

             buffer << "Matrix<double> Exception: LossIndex class.\n"
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

Matrix<double> LossIndex::calculate_regularization_Hessian(void) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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

Matrix<double> LossIndex::calculate_regularization_Hessian(const Vector<double>&) const
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

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "Matrix<double> calculate_regularization_Hessian(const Vector<double>&) const method.\n"
                    << "Unknown regularization type.\n";

             throw std::logic_error(buffer.str());
         }
         break;
     }

     return(Hessian);
}


// double calculate_loss(void) const method

/// Calculates the evaluation value of the loss functional,
/// as the sum of the objective, regularization and constraints functionals.

double LossIndex::calculate_loss(void) const 
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_error_terms();

   #endif

#ifdef __OPENNN_MPI__

#ifdef __OPENNN_DEBUG__

    int initialized;

    MPI_Initialized(&initialized);

    if(!initialized)
    {
      MPI_Init(NULL, NULL);
    }

#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double loss = 0.0;
    double error = 0.0;
    double regularization = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        error = calculate_error_MPI();

        regularization = calculate_regularization();

        loss = error + regularization;
    }
    else
    {
        calculate_error_MPI();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&loss, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return(loss);

#else
   return(calculate_error() + calculate_regularization());
#endif
}


// double calculate_loss(const Vector<double>&) const method

/// Returns the loss of a neural network for a given vector of parameters.
/// It does not set that vector of parameters to the neural network. 
/// @param parameters Vector of parameters for the neural network associated to the loss functional.

double LossIndex::calculate_loss(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_error_terms();

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "double calculate_loss(const Vector<double>&) method.\n"
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

#ifdef __OPENNN_MPI__

#ifdef __OPENNN_DEBUG__

    int initialized;

    MPI_Initialized(&initialized);

    if(!initialized)
    {
      MPI_Init(NULL, NULL);
    }

#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double loss = 0.0;
    double error = 0.0;
    double regularization = 0.0;

    if(rank == 0)
    {
        error = calculate_error_MPI(parameters);

        regularization = calculate_regularization(parameters);

        loss = error + regularization;
    }
    else
    {
        calculate_error_MPI(parameters);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&loss, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return(loss);

#else
   return(calculate_error(parameters) + calculate_regularization(parameters));
#endif
}


// double calculate_selection_error(void) const method

/// Returns the evaluation of the error term on the selection instances of the associated data set.

double LossIndex::calculate_selection_error(void) const
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

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "double calculate_selection_error(void) const method.\n"
                   << "Unknown error type.\n";

            throw std::logic_error(buffer.str());
        }
        break;
    }

    return(selection_error);
}

#ifdef __OPENNN_MPI__

double LossIndex::calculate_selection_error_MPI(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int selection_instances_number = 0;
    int local_selection_instances_number = 0;

    if(has_data_set() && data_set_pointer->has_data())
    {
        local_selection_instances_number = (int)data_set_pointer->get_instances_pointer()->count_selection_instances_number();
    }
    MPI_Allreduce(&local_selection_instances_number, &selection_instances_number, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(selection_instances_number == 0)
    {
        return 0.0;
    }

    size = std::min(size,selection_instances_number);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int* ranks = (int*)malloc(size*sizeof(int));

    for(int i = 0; i < size; i++)
    {
        ranks[i] = i;
    }

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group error_group;
    MPI_Group_incl(world_group, size, ranks, &error_group);

    // Create a new communicator based on the group
    MPI_Comm current_comm;
    MPI_Comm_create(MPI_COMM_WORLD, error_group, &current_comm);


    double global_selection_objective = 0.0;

    double local_selection_objective = 0.0;

    double normalization_coefficient = 0.0;

    // Objective

    if(rank < size)
    {
         switch(error_type)
         {
             case NO_ERROR:
             {
                 // Do nothing
             }
             break;

             case SUM_SQUARED_ERROR:
             {
                local_selection_objective = sum_squared_error_pointer->calculate_error();
             }
             break;

             case MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_selection_objective = temp_loss_index.calculate_selection_error();

                 local_selection_objective /= selection_instances_number;
             }
             break;

             case ROOT_MEAN_SQUARED_ERROR:
             {
                 LossIndex temp_loss_index(*this);
                 temp_loss_index.set_error_type(LossIndex::SUM_SQUARED_ERROR);

                 local_selection_objective = temp_loss_index.calculate_selection_error();

                 local_selection_objective /= selection_instances_number;
             }
             break;

             case NORMALIZED_SQUARED_ERROR:
             {
                 const size_t targets_number = data_set_pointer->get_variables_pointer()->count_targets_number();

                 Vector<double> selection_target_data_mean(targets_number, 0.0);

                 const Vector<double> local_selection_target_data_mean = data_set_pointer->arrange_selection_target_data().calculate_rows_sum();

                 MPI_Allreduce(local_selection_target_data_mean.data(), selection_target_data_mean.data(), (int)targets_number, MPI_DOUBLE, MPI_SUM, current_comm);

                 selection_target_data_mean /= selection_instances_number;

                 const Vector<double> error_normalization = get_normalized_squared_error_pointer()->calculate_selection_error_normalization(selection_target_data_mean);

                 MPI_Reduce(&error_normalization[1], &normalization_coefficient, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

                 local_selection_objective = error_normalization[0];
             }
             break;

             case WEIGHTED_SQUARED_ERROR:
             {
                 const Vector<size_t> targets_indices = data_set_pointer->get_variables_pointer()->arrange_targets_indices();

                 const int local_selection_negatives = (int)data_set_pointer->calculate_selection_negatives(targets_indices[0]);

                 int negatives = 0;
                 MPI_Allreduce(&local_selection_negatives, &negatives, 1, MPI_INT, MPI_SUM, current_comm);

                 normalization_coefficient = negatives*weighted_squared_error_pointer->get_negatives_weight()*0.5;
                 local_selection_objective = weighted_squared_error_pointer->calculate_selection_error(normalization_coefficient);
             }
             break;

             case ROC_AREA_ERROR: // todo
             {
    //             local_objective = roc_area_error_pointer->calculate_error();
             }
             break;

             case MINKOWSKI_ERROR:
             {
                 local_selection_objective = Minkowski_error_pointer->calculate_selection_error();
             }
             break;

             case CROSS_ENTROPY_ERROR:
             {
                 local_selection_objective = cross_entropy_error_pointer->calculate_selection_error_unnormalized();

                 local_selection_objective /= selection_instances_number;
             }
             break;

             case USER_ERROR:
             {
                 local_selection_objective = user_error_pointer->calculate_selection_error();
             }
             break;

             default:
             {
                 std::ostringstream buffer;

                 buffer << "OpenNN Exception: LossIndex class.\n"
                        << "double calculate_selection_error_MPI(void) const method.\n"
                        << "Unknown error type.\n";

                 MPI_Abort(MPI_COMM_WORLD, 1);

                 throw std::logic_error(buffer.str());
             }
             break;
         }

         MPI_Reduce(&local_selection_objective, &global_selection_objective, 1, MPI_DOUBLE, MPI_SUM, 0, current_comm);

         MPI_Barrier(current_comm);
    }

     if(rank == 0)
     {
         if(error_type == (int)LossIndex::ROOT_MEAN_SQUARED_ERROR)
         {
             global_selection_objective = sqrt(global_selection_objective);
         }
         else if(error_type == (int)LossIndex::NORMALIZED_SQUARED_ERROR)
         {
             if(normalization_coefficient < 1.0e-99)
             {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
                       << "double calculate_selection_loss(void) const method.\n"
                       << "Normalization coefficient is zero.\n"
                       << "Unuse constant target variables or choose another error functional. ";

                MPI_Abort(MPI_COMM_WORLD, 1);

                throw std::logic_error(buffer.str());
             }

             global_selection_objective /= normalization_coefficient;
         }
     }

     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Comm_free(&current_comm);
     MPI_Group_free(&error_group);
     free(ranks);

     return(global_selection_objective);

}
#endif

// double calculate_selection_loss(void) const method method

/// Calculates the selection loss,
/// as the sum of the objective and the regularization terms. 

double LossIndex::calculate_selection_loss(void) const 
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    check_error_terms();

    #endif

 #ifdef __OPENNN_MPI__

 #ifdef __OPENNN_DEBUG__

     int initialized;

     MPI_Initialized(&initialized);

     if(!initialized)
     {
       MPI_Init(NULL, NULL);
     }

 #endif

     int rank;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

     double error = 0.0;

     MPI_Barrier(MPI_COMM_WORLD);

     if(rank == 0)
     {
         error = calculate_selection_error_MPI();
     }
     else
     {
         calculate_selection_error_MPI();
     }

     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     return(error);

 #else
    return(calculate_selection_error());
 #endif
}


// Vector<double> calculate_gradient(void) const method

/// Returns the loss function gradient, as the sum of the objective and the regularization gradient vectors.

Vector<double> LossIndex::calculate_gradient(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_error_terms();

   #endif

#ifdef __OPENNN_MPI__

#ifdef __OPENNN_DEBUG__

    int initialized;

    MPI_Initialized(&initialized);

    if(!initialized)
    {
      MPI_Init(NULL, NULL);
    }

#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int parameters_number = (int)neural_network_pointer->count_parameters_number();

    Vector<double> gradient(parameters_number);

    Vector<double> error_gradient;
    Vector<double> regularization_gradient;

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        error_gradient = calculate_error_gradient_MPI();

        regularization_gradient = calculate_regularization_gradient();

        gradient = error_gradient + regularization_gradient;
    }
    else
    {
        calculate_error_gradient_MPI();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(gradient.data(), parameters_number, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return(gradient);

#else
    return(calculate_error_gradient() + calculate_regularization_gradient());
#endif
}


// Vector<double> calculate_gradient(const Vector<double>&) const method

/// Returns the loss gradient for a given vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of parameters for the neural network associated to the loss functional.

Vector<double> LossIndex::calculate_gradient(const Vector<double>& parameters) const
{
   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_error_terms();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   const size_t size = parameters.size();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
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

Matrix<double> LossIndex::calculate_Hessian(void) const
{
    #ifdef __OPENNN_DEBUG__

    check_neural_network();

    check_error_terms();

    #endif

    return(calculate_error_Hessian() + calculate_regularization_Hessian());
}


// Vector<double> calculate_Hessian(const Vector<double>&) const method

/// Returns which would be the objective function Hessian of a neural network for an
/// hypothetical vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated
/// to this loss functional.

Matrix<double> LossIndex::calculate_Hessian(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    check_neural_network();

    check_error_terms();

   const size_t size = parameters.size();
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "double calculate_Hessian(const Vector<double>&) method.\n"
             << "Size must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   return(calculate_error_Hessian(parameters) + calculate_regularization_Hessian(parameters));
}


// Vector<double> calculate_terms(void) const method

/// Evaluates the objective, regularization and constraints terms functions,
/// and returns the total error terms as the assembly of that three vectors.

Vector<double> LossIndex::calculate_terms(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     check_neural_network();

     check_error_terms();

    #endif

    const Vector<double> objective_terms = calculate_error_terms();

    return(objective_terms);
}


// Matrix<double> calculate_terms_Jacobian(void) const method

/// @todo

Matrix<double> LossIndex::calculate_terms_Jacobian(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     check_neural_network();

     check_error_terms();

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

Matrix<double> LossIndex::calculate_inverse_Hessian(void) const
{  
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

    check_neural_network();

    check_error_terms();

   #endif

   const Matrix<double> Hessian = calculate_Hessian();
         
   return(Hessian.calculate_LU_inverse());
}


// Vector<double> calculate_vector_dot_Hessian(Vector<double>) const method

/// Returns the default product of some vector with the objective function Hessian matrix, which is
/// computed using numerical differentiation.
/// @param vector Vector in the dot product. 
/// @todo

Vector<double> LossIndex::calculate_vector_dot_Hessian(const Vector<double>& vector) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

     check_neural_network();

     check_error_terms();

    #endif


   // Control sentence

   const size_t size = vector.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector<double> calculate_vector_dot_Hessian(Vector<double>) method.\n"
             << "Size of vector must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Calculate vector Hessian product

   Vector<double> vector_Hessian_product(parameters_number);

   return(vector_Hessian_product);
}


// ZeroOrderloss calculate_zero_order_loss(void) const method

/// Returns a zero order loss structure, which just contains the loss value of the loss function.

LossIndex::ZeroOrderloss LossIndex::calculate_zero_order_loss(void) const
{
   ZeroOrderloss zero_order_loss;

   zero_order_loss.loss = calculate_loss();

   return(zero_order_loss);
}


// FirstOrderloss calculate_first_order_loss(void) const method

/// Returns a first order loss structure, which contains the value and the gradient of the loss function.

LossIndex::FirstOrderloss LossIndex::calculate_first_order_loss(void) const
{
   FirstOrderloss first_order_loss;

   first_order_loss.loss = calculate_loss();
   first_order_loss.gradient = calculate_gradient();

   return(first_order_loss);
}


// SecondOrderloss calculate_second_order_loss(void) const method

/// Returns a second order loss structure, which contains the value, the gradient and the Hessian of the loss function.

LossIndex::SecondOrderloss LossIndex::calculate_second_order_loss(void) const
{
   SecondOrderloss second_order_loss;

   second_order_loss.loss = calculate_loss();
   second_order_loss.gradient = calculate_gradient();
   second_order_loss.Hessian = calculate_Hessian();

   return(second_order_loss);
}


// double calculate_zero_order_Taylor_approximation(const Vector<double>&) const method

/// Returns the Taylor approximation of the loss function at some point near the parameters.
/// The order of the approximation here is zero, i.e., only the loss value is used. 

double LossIndex::calculate_zero_order_Taylor_approximation(const Vector<double>&) const 
{
   return(calculate_loss());
}


// double calculate_first_order_Taylor_approximation(const Vector<double>&) const method

/// Returns the Taylor approximation of the loss function at some point near the parameters.
/// The order of the approximation here is one, i.e., both the loss value and the loss gradient are used. 
/// @param parameters Approximation point. 

double LossIndex::calculate_first_order_Taylor_approximation(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t parameters_size = parameters.size();
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(parameters_size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "double calculate_first_order_Taylor_approximation(const Vector<double>&) const method.\n"
             << "Size of potential parameters must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   const Vector<double> original_parameters = neural_network_pointer->arrange_parameters();

   const double loss = calculate_loss();
   const Vector<double> gradient = calculate_gradient();

   const double first_order_Taylor_approximation = loss + gradient.dot(parameters-parameters);

   return(first_order_Taylor_approximation);
}


// double calculate_second_order_Taylor_approximation(const Vector<double>&) const method

/// Returns the Taylor approximation of the loss function at some point near the parameters.
/// The order of the approximation here is two, i.e., the loss value, the loss gradient and the loss Hessian are used. 
/// @param parameters Approximation point. 

double LossIndex::calculate_second_order_Taylor_approximation(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t parameters_size = parameters.size();
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(parameters_size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "double calculate_second_order_Taylor_approximation(const Vector<double>&) const method.\n"
             << "Size of potential parameters must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Neural network stuff 

   const Vector<double> original_parameters = neural_network_pointer->arrange_parameters();
   const Vector<double> parameters_difference = parameters - parameters;

   // Performance functioal stuff

   const double loss = calculate_loss();
   const Vector<double> gradient = calculate_gradient();
   const Matrix<double> Hessian = calculate_Hessian();
   
   const double second_order_Taylor_approximation = loss 
   + gradient.dot(parameters_difference) 
   + parameters_difference.dot(Hessian).dot(parameters_difference)/2.0;

   return(second_order_Taylor_approximation);
}


// double calculate_loss(const Vector<double>&, const double&) const method

/// Returns the value of the loss function at some step along some direction.
/// @param direction Direction vector.
/// @param rate Step value. 

double LossIndex::calculate_loss(const Vector<double>& direction, const double& rate) const
{
   const Vector<double> parameters = neural_network_pointer->arrange_parameters();
   const Vector<double> increment = direction*rate;

   return(calculate_loss(parameters + increment));
}


// double calculate_loss_derivative(const Vector<double>&, const double&) const method

/// Returns the derivative of the loss function at some step along some direction.
/// @param direction Direction vector.
/// @param rate Step value. 

double LossIndex::calculate_loss_derivative(const Vector<double>& direction, const double& rate) const
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


// double calculate_loss_second_derivative(const Vector<double>&, double) const method

/// Returns the second derivative of the loss function at some step along some direction.
/// @param direction Direction vector.
/// @param rate Step value. 

double LossIndex::calculate_loss_second_derivative(const Vector<double>& direction, const double& rate) const
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

/// Serializes a default loss functional object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* LossIndex::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Loss index

   tinyxml2::XMLElement* loss_index_element = document->NewElement("LossIndex");

   document->InsertFirstChild(loss_index_element);

   // Objective

   switch(error_type)
   {
      case NO_ERROR:
      {
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           loss_index_element->LinkEndChild(objective_element);

           objective_element->SetAttribute("Type", "NO_ERROR");
      }
      break;

      case SUM_SQUARED_ERROR:
      {                
           tinyxml2::XMLElement* objective_element = document->NewElement("Objective");
           loss_index_element->LinkEndChild(objective_element);

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
           loss_index_element->LinkEndChild(objective_element);

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
           loss_index_element->LinkEndChild(objective_element);

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
           loss_index_element->LinkEndChild(objective_element);

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
            loss_index_element->LinkEndChild(objective_element);

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
           loss_index_element->LinkEndChild(objective_element);

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
           loss_index_element->LinkEndChild(objective_element);

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

         buffer << "OpenNN Exception: LossIndex class.\n"
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
           loss_index_element->LinkEndChild(regularization_element);

           regularization_element->SetAttribute("Type", "NO_REGULARIZATION");
      }
      break;

      case NEURAL_PARAMETERS_NORM:
      {
           tinyxml2::XMLElement* regularization_element = document->NewElement("Regularization");
           loss_index_element->LinkEndChild(regularization_element);

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
           loss_index_element->LinkEndChild(regularization_element);

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

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
                << "Unknown regularization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Display

//   tinyxml2::XMLElement* display_element = document->NewElement("Display");
//   loss_index_element->LinkEndChild(display_element);

//   buffer.str("");
//   buffer << display;

//   tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//   display_element->LinkEndChild(display_text);

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the loss index object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LossIndex::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("LossIndex");

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

          buffer << "OpenNN Exception: LossIndex class.\n"
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

          buffer << "OpenNN Exception: LossIndex class.\n"
                 << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
                 << "Unknown regularization type.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }

    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Sets the loss functional member data from an XML document.
/// @param document Pointer to a TinyXML document with the loss functional data.

void LossIndex::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* loss_index_element = document.FirstChildElement("LossIndex");

    if(!loss_index_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Loss index element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

   // Objective type

   const tinyxml2::XMLElement* objective_element = loss_index_element->FirstChildElement("Objective");

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

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                    << "Unknown error type.\n";

             throw std::logic_error(buffer.str());
          }
          break;
       }
   }    

   // Regularization type

   const tinyxml2::XMLElement* regularization_element = loss_index_element->FirstChildElement("Regularization");

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

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Unknown regularization type.\n";

            throw std::logic_error(buffer.str());
         }
         break;
      }

    }
    // Display

   const tinyxml2::XMLElement* display_element = loss_index_element->FirstChildElement("Display");

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

/// Writes to a string the members of the loss functional object in text format.

std::string LossIndex::to_string(void) const
{
    std::ostringstream buffer;

    buffer << "Loss index\n"
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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

             buffer << "OpenNN Exception: LossIndex class.\n"
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

/// Saves to a XML-type file a string representation of the loss functional object.
/// @param file_name Name of XML-type loss functional file. 

void LossIndex::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   // Declaration

//   TiXmlDeclaration* declaration = new TiXmlDeclaration("1.0", "", "");
//   document->LinkEndChild(declaration);

   // Loss index

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a default loss functional XML-type file.
/// @param file_name Name of default XML-type loss functional file. 

void LossIndex::load(const std::string& file_name)
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      buffer << "OpenNN Exception: LossIndex class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}


// std::string write_information(void) method

/// Returns any useful information about the objective function during training.
/// By default it is an empty string.

std::string LossIndex::write_information(void)  
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

            buffer << "OpenNN Exception: LossIndex class.\n"
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

            buffer << "OpenNN Exception: LossIndex class.\n"
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

void LossIndex::print(void) const
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

/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T H R E S H O L D   S E L E C T I O N   A L G O R I T H M   C L A S S                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "threshold_selection_algorithm.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

ThresholdSelectionAlgorithm::ThresholdSelectionAlgorithm()
    : training_strategy_pointer(NULL)
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

ThresholdSelectionAlgorithm::ThresholdSelectionAlgorithm(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/*/// @param file_name Name of XML order selection file.*/

ThresholdSelectionAlgorithm::ThresholdSelectionAlgorithm(const string&)
    : training_strategy_pointer(NULL)
{
    //load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/*/// @param threshold_selection_document Pointer to a TinyXML document containing the threshold selection algorithm data.*/

ThresholdSelectionAlgorithm::ThresholdSelectionAlgorithm(const tinyxml2::XMLDocument& )
    : training_strategy_pointer(NULL)
{
    //from_XML(order_selection_document);
}


// DESTRUCTOR

/// Destructor.

ThresholdSelectionAlgorithm::~ThresholdSelectionAlgorithm()
{
}


// METHODS

// TrainingStrategy* get_training_strategy_pointer() const method

/// Returns a pointer to the training strategy object.

TrainingStrategy* ThresholdSelectionAlgorithm::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is NULL.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}

// bool has_training_strategy() const method

/// Returns true if this threshold selection algorithm has a training strategy associated, and false otherwise.

bool ThresholdSelectionAlgorithm::has_training_strategy() const
{
    if(training_strategy_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}

// const bool& get_reserve_binary_classification_tests_data() const method

/// Returns true if the binary classification test are to be reserved, and false otherwise.

const bool& ThresholdSelectionAlgorithm::get_reserve_binary_classification_tests_data() const
{
    return(reserve_binary_classification_tests_data);
}

// const bool& get_reserve_function_data() const method

/// Returns true if the function values to optimize are to be reserved, and false otherwise.

const bool& ThresholdSelectionAlgorithm::get_reserve_function_data() const
{
    return(reserve_function_data);
}

// const bool& get_display() const method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& ThresholdSelectionAlgorithm::get_display() const
{
    return(display);
}

// void set_training_strategy_pointer(TrainingStrategy*) method

/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void ThresholdSelectionAlgorithm::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


// void set_default() method

/// Sets the members of the threshold selection object to their default values.

void ThresholdSelectionAlgorithm::set_default()
{
    // MEMBERS

    display = true;

    reserve_binary_classification_tests_data = false;
    reserve_function_data = true;
}

// void set_reserve_binary_classification_tests_data(const bool&) method

/// Sets the reserve flag for the binary classification test.
/// @param new_reserve_binary_classification_tests_data Flag value

void ThresholdSelectionAlgorithm::set_reserve_binary_classification_tests_data(const bool& new_reserve_binary_classification_tests_data)
{
    reserve_binary_classification_tests_data = new_reserve_binary_classification_tests_data;
}

// void set_reserve_function_data(const bool&) method

/// Sets the reserve flag for the function data.
/// @param new_reserve_function_data Flag value

void ThresholdSelectionAlgorithm::set_reserve_function_data(const bool& new_reserve_function_data)
{
    reserve_function_data = new_reserve_function_data;
}

// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ThresholdSelectionAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}

// Errors calculation methods

/// Returns the confusion matrix of a neural network on the testing instances of a data set.
/// If the number of outputs is one, the size of the confusion matrix is two.
/// If the number of outputs is greater than one, the size of the confusion matrix is the number of outputs.

Matrix<size_t> ThresholdSelectionAlgorithm::calculate_confusion(const double& decision_threshold) const
{
    #ifdef __OPENNN_DEBUG__

    check();
    #endif

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
              << "Matrix<size_t> calculate_confusion(const double&) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

       throw logic_error(buffer.str());
    }


    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class." << endl
               << "Matrix<size_t> calculate_confusion(const double&) const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    if(outputs_number != variables.count_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class." << endl
              << "Matrix<size_t> calculate_confusion(const double&) const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    if(outputs_number != 1)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class." << endl
              << "Matrix<size_t> calculate_confusion(const double&) const method." << endl
              << "Number of outputs in neural network must be equal to 1." << endl;

       throw logic_error(buffer.str());
    }
    #endif

     const Matrix<double> input_data = data_set_pointer->arrange_selection_input_data();
     const Matrix<double> target_data = data_set_pointer->arrange_selection_target_data();

     const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

     const size_t rows_number = target_data.get_rows_number();

     Matrix<size_t> confusion(2, 2);

     size_t true_positive = 0;
     size_t false_negative = 0;
     size_t false_positive = 0;
     size_t true_negative = 0;

     for(size_t i = 0; i < rows_number; i++)
     {
         if(decision_threshold == 0.0 && target_data(i,0) == 0.0 )
         {
             false_positive++;

         }
         else if(decision_threshold == 0.0 && target_data(i,0) == 1.0)
         {
             true_positive++;

         }
         else if(target_data(i,0) >= decision_threshold && output_data(i,0) >= decision_threshold)
         {
             // True positive

             true_positive++;

         }
         else if(target_data(i,0) >= decision_threshold && output_data(i,0) < decision_threshold)
         {
             // False negative

             false_negative++;

         }
         else if(target_data(i,0) < decision_threshold && output_data(i,0) >= decision_threshold)
         {
             // False positive

             false_positive++;

         }
         else if(target_data(i,0) < decision_threshold && output_data(i,0) < decision_threshold)
         {
             // True negative

             true_negative++;
         }
     }

     confusion(0,0) = true_positive;
     confusion(0,1) = false_negative;
     confusion(1,0) = false_positive;
     confusion(1,1) = true_negative;

     if(confusion.calculate_sum() != rows_number)
     {
         ostringstream buffer;

         buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
                << "Matrix<size_t> calculate_confusion(const double&) const method.\n"
                << "Number of elements in confusion matrix must be equal to number of testing instances.\n";

         throw logic_error(buffer.str());
     }

     return(confusion);
}

/// Returns the results of a binary classification test in a single vector.
/// The size of that vector is fifteen.

Vector<double> ThresholdSelectionAlgorithm::calculate_binary_classification_test(const Matrix<size_t>& confusion) const
{
#ifdef __OPENNN_DEBUG__

    check();

    const size_t rows = confusion.get_rows_number();
    const size_t columns = confusion.get_columns_number();

    if(rows != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "Matrix<size_t> calculate_binary_classification_test(const Matrix<size_t>&) const method.\n"
               << "Number of rows in confusion matrix must be equal to two.\n";

        throw logic_error(buffer.str());
    }

    if(columns != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "Matrix<size_t> calculate_binary_classification_test(const Matrix<size_t>&) const method.\n"
               << "Number of columns in confusion matrix must be equal to two.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t true_positive = confusion(0,0);
    const size_t false_positive = confusion(1,0);
    const size_t false_negative = confusion(0,1);
    const size_t true_negative = confusion(1,1);

    // Classification accuracy

    double classification_accuracy;

    if(true_positive + true_negative + false_positive + false_negative == 0)
    {
        classification_accuracy = 0.0;
    }
    else
    {
        classification_accuracy =(double)(true_positive + true_negative)/(double)(true_positive + true_negative + false_positive + false_negative);
    }

    // Error rate

    double error_rate;

    if(true_positive + true_negative + false_positive + false_negative == 0)
    {
        error_rate = 0.0;
    }
    else
    {
        error_rate =(double)(false_positive + false_negative)/(double)(true_positive + true_negative + false_positive + false_negative);
    }

    // Sensitivity

    double sensitivity;

    if(true_positive + false_negative == 0)
    {
        sensitivity = 0.0;
    }
    else
    {
        sensitivity =(double)true_positive/(double)(true_positive + false_negative);
    }

    // Specificity

    double specificity;

    if(true_negative + false_positive == 0)
    {
        specificity = 0.0;
    }
    else
    {
        specificity =(double)true_negative/(double)(true_negative + false_positive);
    }

    // Precision

    double precision;

    if(true_positive + false_positive == 0)
    {
        precision = 0.0;
    }
    else
    {
       precision =(double) true_positive /(double)(true_positive + false_positive);
    }

    // Positive likelihood

    double positive_likelihood;

    if(classification_accuracy == 1.0)
    {
        positive_likelihood = 1.0;
    }
    else if(1.0 - specificity == 0.0)
    {
        positive_likelihood = 0.0;
    }
    else
    {
        positive_likelihood = sensitivity/(1.0 - specificity);
    }

    // Negative likelihood

    double negative_likelihood;

    if(classification_accuracy == 1.0)
    {
        negative_likelihood = 1.0;
    }
    else if(1.0 - sensitivity == 0.0)
    {
        negative_likelihood = 0.0;
    }
    else
    {
        negative_likelihood = specificity/(1.0 - sensitivity);
    }

    // F1 score

    double F1_score;

    if(2*true_positive + false_positive + false_negative == 0)
    {
        F1_score = 0.0;
    }
    else
    {
        F1_score =(double) 2*true_positive/(double)(2*true_positive + false_positive + false_negative);
    }

    // False positive rate

    double false_positive_rate;

    if(false_positive + true_negative == 0)
    {
        false_positive_rate = 0.0;
    }
    else
    {
        false_positive_rate =(double) false_positive/(double)(false_positive + true_negative);
    }

    // False discovery rate

    double false_discovery_rate;

    if(false_positive + true_positive == 0)
    {
        false_discovery_rate = 0.0;
    }
    else
    {
        false_discovery_rate =(double) false_positive /(double)(false_positive + true_positive);
    }

    // False negative rate

    double false_negative_rate;

    if(false_negative + true_positive == 0)
    {
        false_negative_rate = 0.0;
    }
    else
    {
        false_negative_rate =(double) false_negative /(double)(false_negative + true_positive);
    }

    // Negative predictive value

    double negative_predictive_value;

    if(true_negative + false_negative == 0)
    {
        negative_predictive_value = 0.0;
    }
    else
    {
        negative_predictive_value =(double) true_negative/(double)(true_negative + false_negative);
    }

    //Matthews correlation coefficient

    double Matthews_correlation_coefficient;

    if((true_positive + false_positive) *(true_positive + false_negative) *(true_negative + false_positive) *(true_negative + false_negative) == 0)
    {
        Matthews_correlation_coefficient = 0.0;
    }
    else
    {
        Matthews_correlation_coefficient =(double)(true_positive * true_negative - false_positive * false_negative) /(double) sqrt((true_positive + false_positive) *(true_positive + false_negative) *(true_negative + false_positive) *(true_negative + false_negative));
    }

    //Informedness

    double informedness = sensitivity + specificity - 1;

    //Markedness

    double markedness;

    if(true_negative + false_positive == 0)
    {
        markedness = precision - 1;
    }
    else
    {
        markedness = precision +(double) true_negative/(double)(true_negative + false_positive) - 1;
    }

    //Arrange vector

    Vector<double> binary_classification_test(15);

    binary_classification_test[0] = classification_accuracy;
    binary_classification_test[1] = error_rate;
    binary_classification_test[2] = sensitivity;
    binary_classification_test[3] = specificity;
    binary_classification_test[4] = precision;
    binary_classification_test[5] = positive_likelihood;
    binary_classification_test[6] = negative_likelihood;
    binary_classification_test[7] = F1_score;
    binary_classification_test[8] = false_positive_rate;
    binary_classification_test[9] = false_discovery_rate;
    binary_classification_test[10] = false_negative_rate;
    binary_classification_test[11] = negative_predictive_value;
    binary_classification_test[12] = Matthews_correlation_coefficient;
    binary_classification_test[13] = informedness;
    binary_classification_test[14] = markedness;

    return(binary_classification_test);
}

// void check() const method

/// Checks that the different pointers needed for performing the threshold selection are not NULL.

void ThresholdSelectionAlgorithm::check() const
{
    // Training algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is NULL.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to loss functional is NULL.\n";

        throw logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is NULL.\n";

        throw logic_error(buffer.str());
    }

    const ProbabilisticLayer* probabilistic_layer_pointer = neural_network_pointer->get_probabilistic_layer_pointer();

    if(!probabilistic_layer_pointer)
    {
        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to probabilistic layer is NULL.\n";

        throw logic_error(buffer.str());
    }

    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is NULL.\n";

        throw logic_error(buffer.str());
    }

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.count_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: ThresholdSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
    }

}


// string write_stopping_condition() const method

/// Return a string with the stopping condition of the ThresholdSelectionResults.

string ThresholdSelectionAlgorithm::ThresholdSelectionResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case PerfectConfusionMatrix:
    {
        return("PerfectConfusionMatrix");
    }
    case AlgorithmFinished:
    {
        return("AlgorithmFinished");
    }
    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ThresholdSelectionResults struct.\n"
               << "string write_stopping_condition() const method.\n"
               << "Unknown stopping condition type.\n";

        throw logic_error(buffer.str());

        break;
    }
    }

}


// string object_to_string() const method

/// Returns a string representation of the current threshold selection results structure.

string ThresholdSelectionAlgorithm::ThresholdSelectionResults::object_to_string() const
{
   ostringstream buffer;

   // Threshold history

   if(!threshold_data.empty())
   {
     buffer << "% Threshold history:\n"
            << threshold_data.to_row_matrix() << "\n";
   }

   // Binary classification test history

   if(!binary_classification_test_data.empty())
   {
     buffer << "% Binary classification test history:\n"
            << binary_classification_test_data.to_row_matrix() << "\n";
   }

   // Function history

   if(!function_data.empty())
   {
     buffer << "% Function history:\n"
            << function_data.to_row_matrix() << "\n";
   }

   // Final threshold

   buffer << "% Final threshold:\n"
          << final_threshold << "\n";

   // Final binary classification test

   buffer << "% Final function value:\n"
          << final_function_value << "\n";

   // Stopping condition

   buffer << "% Stopping condition\n"
          << write_stopping_condition() << "\n";

   // Iterations number

   buffer << "% Number of iterations:\n"
          << iterations_number << "\n";

   return(buffer.str());
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

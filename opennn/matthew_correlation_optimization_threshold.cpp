/********************************************************************************************************************/
/*                                                                                                                  */
/*   OpenNN: Open Neural Networks Library                                                                           */
/*   www.opennn.net                                                                                                 */
/*                                                                                                                  */
/*   M A T T H E W   C O R R E L A T I O N   O P T I M I Z A T I O N   T H R E S H O L D   C L A S S                */
/*                                                                                                                  */
/*   Fernando Gomez                                                                                                 */
/*   Artelnics - Making intelligent use of data                                                                     */
/*   fernandogomez@artelnics.com                                                                                    */
/*                                                                                                                  */
/********************************************************************************************************************/


// OpenNN includes

#include "matthew_correlation_optimization_threshold.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

MatthewCorrelationOptimizationThreshold::MatthewCorrelationOptimizationThreshold(void)
    : ThresholdSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

MatthewCorrelationOptimizationThreshold::MatthewCorrelationOptimizationThreshold(TrainingStrategy* new_training_strategy_pointer)
    : ThresholdSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}

// XML CONSTRUCTOR

/// XML constructor.
/// @param matthew_correlation_optimization_document Pointer to a TinyXML document containing the matthew correlation optimization data.

MatthewCorrelationOptimizationThreshold::MatthewCorrelationOptimizationThreshold(const tinyxml2::XMLDocument& matthew_correlation_optimization_document)
    : ThresholdSelectionAlgorithm(matthew_correlation_optimization_document)
{
    from_XML(matthew_correlation_optimization_document);
}

// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML matthew correlation optimization file.

MatthewCorrelationOptimizationThreshold::MatthewCorrelationOptimizationThreshold(const std::string& file_name)
    : ThresholdSelectionAlgorithm(file_name)
{
    load(file_name);
}



// DESTRUCTOR

/// Destructor.

MatthewCorrelationOptimizationThreshold::~MatthewCorrelationOptimizationThreshold(void)
{
}

// METHODS

// const double& get_minimum_threshold(void) const method

/// Returns the minimum threshold of the algorithm.

const double& MatthewCorrelationOptimizationThreshold::get_minimum_threshold(void) const
{
    return(minimum_threshold);
}

// const double& get_maximum_threshold(void) const method

/// Returns the maximum threshold of the algorithm.

const double& MatthewCorrelationOptimizationThreshold::get_maximum_threshold(void) const
{
    return(maximum_threshold);
}

// const double& get_step(void) const method

/// Returns the step for the sucesive iterations of the algorithm.

const double& MatthewCorrelationOptimizationThreshold::get_step(void) const
{
    return(step);
}

// void set_default(void) method

/// Sets the members of the matthew correlation optimization object to their default values.

void MatthewCorrelationOptimizationThreshold::set_default(void)
{
    minimum_threshold = 0.0;

    maximum_threshold = 1.0;

    step = 0.001;
}

// void set_minimum_threshold(const double&) method

/// Sets the minimum value of the threshold selection algotihm.
/// @param new_minimum_threshold Minimum threshold for the algorithm.

void MatthewCorrelationOptimizationThreshold::set_minimum_threshold(const double& new_minimum_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_threshold <= 0 || new_minimum_threshold >= 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MatthewCorrelationOptimizationThreshold class.\n"
               << "void set_minimum_threshold(const double&) method.\n"
               << "Minimum threshold must be between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    minimum_threshold = new_minimum_threshold;
}

// void set_maximum_threshold(const double&) method

/// Sets the maximum value of the threshold selection algotihm.
/// @param new_maximum_threshold Maximum threshold for the algorithm.

void MatthewCorrelationOptimizationThreshold::set_maximum_threshold(const double& new_maximum_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_threshold <= 0 || new_maximum_threshold >= 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MatthewCorrelationOptimizationThreshold class.\n"
               << "void set_maximum_threshold(const double&) method.\n"
               << "Maximum threshold must be between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_threshold = new_maximum_threshold;
}

// void set_step(const double&) method

/// Sets the step between two iterations of the threshold selection algotihm.
/// @param new_step Difference of threshold between two consecutive iterations.

void MatthewCorrelationOptimizationThreshold::set_step(const double& new_step)
{
#ifdef __OPENNN_DEBUG__

    if(new_step <= 0 || new_step >= 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MatthewCorrelationOptimizationThreshold class.\n"
               << "void set_step(const double&) method.\n"
               << "Step must be between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    step = new_step;
}

// MatthewCorrelationOptimizationThresholdResults* perform_order_selection(void) method

/// Perform the decision threshold selection optimizing the Matthew correlation.

MatthewCorrelationOptimizationThreshold::MatthewCorrelationOptimizationThresholdResults* MatthewCorrelationOptimizationThreshold::perform_threshold_selection(void)
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    MatthewCorrelationOptimizationThresholdResults* results = new MatthewCorrelationOptimizationThresholdResults();

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    double current_threshold = minimum_threshold;

    Matrix<size_t> current_confusion;

    Vector<double> current_binary_classification_test;

    double current_matthew_correlation;

    double optimum_threshold;

    Vector<double> optimal_binary_classification_test(15,1);

    double optimum_matthew_correlation = 0.0;

    size_t iterations = 0;

    bool end = false;

    while (!end)
    {
        current_confusion = calculate_confusion(current_threshold);
        current_binary_classification_test = calculate_binary_classification_test(current_confusion);

        current_matthew_correlation = current_binary_classification_test[12];

        results->threshold_data.push_back(current_threshold);

        if(reserve_binary_classification_tests_data)
        {
            results->binary_classification_test_data.push_back(current_binary_classification_test);
        }

        if(reserve_function_data)
        {
            results->function_data.push_back(current_matthew_correlation);
        }

        if (current_matthew_correlation > optimum_matthew_correlation ||
            (current_matthew_correlation == optimum_matthew_correlation && current_binary_classification_test[1] < optimal_binary_classification_test[1]))
        {
            optimum_matthew_correlation = current_matthew_correlation;
            optimum_threshold = current_threshold;
            optimal_binary_classification_test.set(current_binary_classification_test);
        }

        iterations++;

        if (current_confusion(0,1) == 0 && current_confusion(1,0) == 0)
        {
            end = true;

            if(display)
            {
                std::cout << "Perfect confusion matrix reached." << std::endl;
            }

            results->stopping_condition = ThresholdSelectionAlgorithm::PerfectConfusionMatrix;
        }
        else if (current_threshold == maximum_threshold)
        {
            end = true;

            if(display)
            {
                std::cout << "Algorithm finished \n";
            }

            results->stopping_condition = ThresholdSelectionAlgorithm::AlgorithmFinished;
        }

        if (display)
        {
            std::cout << "Iteration: " << iterations << std::endl;
            std::cout << "Current threshold: " << current_threshold << std::endl;
            std::cout << "Current error: " << current_binary_classification_test[1] << std::endl;
            std::cout << "Current sensitivity: " << current_binary_classification_test[2] << std::endl;
            std::cout << "Current specifity: " << current_binary_classification_test[3] << std::endl;
            std::cout << "Current Matthew correlation: " << current_binary_classification_test[12] << std::endl;
            std::cout << "Confusion matrix: " << std::endl
                      << current_confusion << std::endl;
            std::cout << std::endl;
        }

        current_threshold = fmin(maximum_threshold, current_threshold + step);

    }

    if (display)
    {
        std::cout << "Optimum threshold: " << optimum_threshold << std::endl;
        std::cout << "Optimal error: " << optimal_binary_classification_test[1] << std::endl;
    }

    results->iterations_number = iterations;
    results->final_threshold = optimum_threshold;
    results->final_function_value = optimum_matthew_correlation;

    neural_network_pointer->get_probabilistic_layer_pointer()->set_decision_threshold(optimum_threshold);

    return(results);
}

// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> MatthewCorrelationOptimizationThreshold::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

    // Minimum threshold

    labels.push_back("Minimum threshold");

    buffer.str("");
    buffer << minimum_threshold;

    values.push_back(buffer.str());

    // Maximum threshold

    labels.push_back("Maximum threshold");

    buffer.str("");
    buffer << maximum_threshold;

    values.push_back(buffer.str());

   // Step

   labels.push_back("Step");

   buffer.str("");
   buffer << step;

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the matthew correlation optimization parameters, the stopping criteria
/// and other user stuff concerning the matthew correlation optimization object.

tinyxml2::XMLDocument* MatthewCorrelationOptimizationThreshold::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("MatthewCorrelationOptimizationThreshold");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Minimum threshold
   {
   element = document->NewElement("MinimumThreshold");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_threshold;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum threshold
   {
   element = document->NewElement("MaximumThreshold");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_threshold;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Step
   {
   element = document->NewElement("Step");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << step;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Performance calculation method
//   {
//   element = document->NewElement("PerformanceCalculationMethod");
//   root_element->LinkEndChild(element);

//   text = document->NewText(write_loss_calculation_method().c_str());
//   element->LinkEndChild(text);
//   }

   // Reserve parameters data
//   {
//   element = document->NewElement("ReserveParametersData");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_parameters_data;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Reserve minimal parameters
//   {
//   element = document->NewElement("ReserveMinimalParameters");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_minimal_parameters;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Display
//   {
//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the Matthew's correlation optimization threshold object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void MatthewCorrelationOptimizationThreshold::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("MatthewCorrelationOptimizationThreshold");

    // Minimum threshold

    file_stream.OpenElement("MinimumThreshold");

    buffer.str("");
    buffer << minimum_threshold;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum threshold

    file_stream.OpenElement("MaximumThreshold");

    buffer.str("");
    buffer << maximum_threshold;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Step

    file_stream.OpenElement("Step");

    buffer.str("");
    buffer << step;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve function data

    file_stream.OpenElement("ReserveFunctionData");

    buffer.str("");
    buffer << reserve_function_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    //file_stream.CloseElement();
}

// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this matthew correlation optimization object.
/// @param document TinyXML document containing the member data.

void MatthewCorrelationOptimizationThreshold::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MatthewCorrelationOptimizationThreshold");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MatthewCorrelationOptimizationThreshold class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "MatthewCorrelationOptimizationThreshold element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Minimum threshold
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumThreshold");

        if(element)
        {
           const double new_minimum_threshold = atof(element->GetText());

           try
           {
              set_minimum_threshold(new_minimum_threshold);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Maximum threshold
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumThreshold");

        if(element)
        {
           const double new_maximum_threshold = atof(element->GetText());

           try
           {
              set_maximum_threshold(new_maximum_threshold);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Step
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Step");

        if(element)
        {
           const double new_step = atof(element->GetText());

           try
           {
              set_step(new_step);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Reserve function data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveFunctionData");

        if(element)
        {
            const std::string new_reserve_function_data = element->GetText();

            try
            {
                set_reserve_function_data(new_reserve_function_data != "0");
            }
            catch(const std::logic_error& e)
            {
               std::cout << e.what() << std::endl;
            }
        }
    }

    // Display
//    {
//        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

//        if(element)
//        {
//           const std::string new_display = element->GetText();

//           try
//           {
//              set_display(new_display != "0");
//           }
//           catch(const std::logic_error& e)
//           {
//              std::cout << e.what() << std::endl;
//           }
//        }
//    }
}

// void save(const std::string&) const method

/// Saves to a XML-type file the members of the matthew correlation optimization object.
/// @param file_name Name of matthew correlation optimization XML-type file.

void MatthewCorrelationOptimizationThreshold::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a matthew correlation optimization object from a XML-type file.
/// @param file_name Name of matthew correlation optimization XML-type file.

void MatthewCorrelationOptimizationThreshold::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: MatthewCorrelationOptimizationThreshold class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
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

/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N F O R M E D N E S S   O P T I M I Z A T I O N   T H R E S H O L D   C L A S S                          */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "informedness_optimization_threshold.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

InformednessOptimizationThreshold::InformednessOptimizationThreshold(void)
    : ThresholdSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

InformednessOptimizationThreshold::InformednessOptimizationThreshold(TrainingStrategy* new_training_strategy_pointer)
    : ThresholdSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}

// XML CONSTRUCTOR

/// XML constructor.
/// @param incremental_order_document Pointer to a TinyXML document containing the incremental order data.

InformednessOptimizationThreshold::InformednessOptimizationThreshold(const tinyxml2::XMLDocument& incremental_order_document)
    : ThresholdSelectionAlgorithm(incremental_order_document)
{
    from_XML(incremental_order_document);
}

// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML incremental order file.

InformednessOptimizationThreshold::InformednessOptimizationThreshold(const std::string& file_name)
    : ThresholdSelectionAlgorithm(file_name)
{
    load(file_name);
}



// DESTRUCTOR

/// Destructor.

InformednessOptimizationThreshold::~InformednessOptimizationThreshold(void)
{
}

// METHODS


// const double& get_step(void) const method

/// Returns the step for the sucesive iterations of the algorithm.

const double& InformednessOptimizationThreshold::get_step(void) const
{
    return(step);
}

// const size_t& get_maximum_selection_failures(void) const method

/// Returns the maximum number of selection failures in the model order selection algorithm.

const size_t& InformednessOptimizationThreshold::get_maximum_selection_failures(void) const
{
    return(maximum_selection_failures);
}

// void set_default(void) method

/// Sets the members of the model selection object to their default values:

void InformednessOptimizationThreshold::set_default(void)
{
    step = 0.001;

    maximum_selection_failures = 10;
}


// void set_step(const double&) method

/// Sets the step between two iterations of the threshold selection algotihm.
/// @param new_step Difference of threshold between two consecutive iterations.

void InformednessOptimizationThreshold::set_step(const double& new_step)
{
#ifdef __OPENNN_DEBUG__

    if(new_step <= 0 || new_step >= 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InformednessOptimizationThreshold class.\n"
               << "void set_step(const double&) method.\n"
               << "Step must be between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    step = new_step;
}

// void set_maximum_selection_failures(const size_t&) method

/// Sets the maximum selection failures for the Incremental order selection algorithm.
/// @param new_maximum_performance_failures Maximum number of selection failures in the Incremental order selection algorithm.

void InformednessOptimizationThreshold::set_maximum_selection_failures(const size_t& new_maximum_performance_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_performance_failures <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InformednessOptimizationThreshold class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_performance_failures;
}

// InformednessOptimizationThresholdResults* perform_order_selection(void) method

/// Perform the decision threshold selection optimizing the informedness.

InformednessOptimizationThreshold::InformednessOptimizationThresholdResults* InformednessOptimizationThreshold::perform_threshold_selection(void)
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    InformednessOptimizationThresholdResults* results = new InformednessOptimizationThresholdResults();

    const PerformanceFunctional* performance_functional_pointer = training_strategy_pointer->get_performance_functional_pointer();

    NeuralNetwork* neural_network_pointer = performance_functional_pointer->get_neural_network_pointer();

    double current_threshold = step;

    Matrix<size_t> current_confusion;

    Vector<double> current_binary_classification_test;

    double current_informedness;

    double optimum_threshold;

    Vector<double> optimal_binary_classification_test(15,1);

    double optimum_informedness = 0.0;

    size_t iterations = 0;

    bool end = false;

    while (!end)
    {
        current_confusion = calculate_confusion(current_threshold);
        current_binary_classification_test = calculate_binary_classification_test(current_confusion);

        current_informedness = current_binary_classification_test[13];

        results->threshold_data.push_back(current_threshold);

        if(reserve_binary_classification_tests_data)
        {
            results->binary_classification_test_data.push_back(current_binary_classification_test);
        }

        if (current_informedness > optimum_informedness ||
            (current_informedness == optimum_informedness && current_binary_classification_test[1] < optimal_binary_classification_test[1]))
        {
            optimum_informedness = current_informedness;
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
        }else if (current_threshold == 1)
        {
            end = true;

            if(display)
            {
                std::cout << "Algorithm finished \n";
            }

            results->stopping_condition = ThresholdSelectionAlgorithm::AlgorithmFinished;
        }else if (iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum number of iterations reached." << std::endl;
            }

            results->stopping_condition = ThresholdSelectionAlgorithm::MaximumIterations;
        }

        if (display)
        {
            std::cout << "Iteration: " << iterations << std::endl;
            std::cout << "Current threshold: " << current_threshold << std::endl;
            std::cout << "Current error: " << current_binary_classification_test[1] << std::endl;
            std::cout << "Current sensitivity: " << current_binary_classification_test[2] << std::endl;
            std::cout << "Current specifity: " << current_binary_classification_test[3] << std::endl;
            std::cout << "Current Informedness: " << current_binary_classification_test[13] << std::endl;
            std::cout << "Confusion matrix: " << std::endl
                      << current_confusion << std::endl;
            std::cout << std::endl;
        }

        current_threshold = fmin(1, current_threshold + step);

    }

    if (display)
    {
        std::cout << "Optimum threshold: " << optimum_threshold << std::endl;
        std::cout << "Optimal error: " << optimal_binary_classification_test[1] << std::endl;
    }

    results->iterations_number = iterations;
    results->final_threshold = optimum_threshold;
    results->final_binary_classification_test = optimal_binary_classification_test;

    neural_network_pointer->get_probabilistic_layer_pointer()->set_decision_threshold(optimum_threshold);

    return(results);
}

// Matrix<std::string> to_string_matrix(void) const method

// the most representative

Matrix<std::string> InformednessOptimizationThreshold::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

   // Step

   labels.push_back("Step");

   buffer.str("");
   buffer << step;

   values.push_back(buffer.str());

   // Maximum selection failures

   labels.push_back("Maximum selection failures");

   buffer.str("");
   buffer << maximum_selection_failures;

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

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the incremental order parameters, the stopping criteria
/// and other user stuff concerning the incremental order object.

tinyxml2::XMLDocument* InformednessOptimizationThreshold::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("InformednessOptimizationThreshold");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

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

//   text = document->NewText(write_performance_calculation_method().c_str());
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


   // Maximum iterations
//   {
//   element = document->NewElement("MaximumIterationsNumber");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << maximum_iterations_number;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Maximum selection failures
   {
   element = document->NewElement("MaximumSelectionFailures");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_failures;

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


   return(document);
}

// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this incremental order object.
/// @param document TinyXML document containing the member data.

void InformednessOptimizationThreshold::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("InformednessOptimizationThreshold");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InformednessOptimizationThreshold class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InformednessOptimizationThreshold element is NULL.\n";

        throw std::logic_error(buffer.str());
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
           const double new_maximum_time = atoi(element->GetText());

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

    // Maximum selection failures
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionFailures");

        if(element)
        {
           const size_t new_maximum_selection_failures = atoi(element->GetText());

           try
           {
              set_maximum_selection_failures(new_maximum_selection_failures);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }
}

// void save(const std::string&) const method

/// Saves to a XML-type file the members of the incremental order object.
/// @param file_name Name of incremental order XML-type file.

void InformednessOptimizationThreshold::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a incremental order object from a XML-type file.
/// @param file_name Name of incremental order XML-type file.

void InformednessOptimizationThreshold::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: InformednessOptimizationThreshold class.\n"
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

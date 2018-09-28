/********************************************************************************************************************/
/*                                                                                                                  */
/*   OpenNN: Open Neural Networks Library                                                                           */
/*   www.opennn.net                                                                                                 */
/*                                                                                                                  */
/*   R O C   C U R V E   O P T I M I Z A T I O N   T H R E S H O L D   C L A S S                                    */
/*                                                                                                                  */
/*   Fernando Gomez                                                                                                 */
/*   Artificial Intelligence Techniques SL                                                                     */
/*   fernandogomez@artelnics.com                                                                                    */
/*                                                                                                                  */
/********************************************************************************************************************/


// OpenNN includes

#include "roc_curve_optimization_threshold.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

ROCCurveOptimizationThreshold::ROCCurveOptimizationThreshold()
    : ThresholdSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

ROCCurveOptimizationThreshold::ROCCurveOptimizationThreshold(TrainingStrategy* new_training_strategy_pointer)
    : ThresholdSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}

// XML CONSTRUCTOR

/// XML constructor.
/// @param roc_curve_optimization_document Pointer to a TinyXML document containing the roc curve optimization data.

ROCCurveOptimizationThreshold::ROCCurveOptimizationThreshold(const tinyxml2::XMLDocument& roc_curve_optimization_document)
    : ThresholdSelectionAlgorithm(roc_curve_optimization_document)
{
    from_XML(roc_curve_optimization_document);
}

// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML roc curve optimization file.

ROCCurveOptimizationThreshold::ROCCurveOptimizationThreshold(const string& file_name)
    : ThresholdSelectionAlgorithm(file_name)
{
    load(file_name);
}



// DESTRUCTOR

/// Destructor.

ROCCurveOptimizationThreshold::~ROCCurveOptimizationThreshold()
{
}

// METHODS

// const double& get_minimum_threshold() const method

/// Returns the minimum threshold of the algorithm.

const double& ROCCurveOptimizationThreshold::get_minimum_threshold() const
{
    return(minimum_threshold);
}

// const double& get_maximum_threshold() const method

/// Returns the maximum threshold of the algorithm.

const double& ROCCurveOptimizationThreshold::get_maximum_threshold() const
{
    return(maximum_threshold);
}

// const double& get_step() const method

/// Returns the step for the sucesive iterations of the algorithm.

const double& ROCCurveOptimizationThreshold::get_step() const
{
    return(step);
}

// void set_default() method

/// Sets the members of the model selection object to their default values:

void ROCCurveOptimizationThreshold::set_default()
{
    minimum_threshold = 0.0;

    maximum_threshold = 1.0;

    step = 0.001;
}

// void set_minimum_threshold(const double&) method

/// Sets the minimum value of the threshold selection algotihm.
/// @param new_minimum_threshold Minimum threshold for the algorithm.

void ROCCurveOptimizationThreshold::set_minimum_threshold(const double& new_minimum_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_threshold <= 0 || new_minimum_threshold >= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ROCCurveOptimizationThreshold class.\n"
               << "void set_minimum_threshold(const double&) method.\n"
               << "Minimum threshold must be between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_threshold = new_minimum_threshold;
}

// void set_maximum_threshold(const double&) method

/// Sets the maximum value of the threshold selection algotihm.
/// @param new_maximum_threshold Maximum threshold for the algorithm.

void ROCCurveOptimizationThreshold::set_maximum_threshold(const double& new_maximum_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_threshold <= 0 || new_maximum_threshold >= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ROCCurveOptimizationThreshold class.\n"
               << "void set_maximum_threshold(const double&) method.\n"
               << "Maximum threshold must be between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_threshold = new_maximum_threshold;
}

// void set_step(const double&) method

/// Sets the step between two iterations of the threshold selection algotihm.
/// @param new_step Difference of threshold between two consecutive iterations.

void ROCCurveOptimizationThreshold::set_step(const double& new_step)
{
#ifdef __OPENNN_DEBUG__

    if(new_step <= 0 || new_step >= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ROCCurveOptimizationThreshold class.\n"
               << "void set_step(const double&) method.\n"
               << "Step must be between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    step = new_step;
}

// ROCCurveOptimizationThresholdResults* perform_order_selection() method

/// Perform the decision threshold selection optimizing the ROC curve distance.

ROCCurveOptimizationThreshold::ROCCurveOptimizationThresholdResults* ROCCurveOptimizationThreshold::perform_threshold_selection()
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    ROCCurveOptimizationThresholdResults* results = new ROCCurveOptimizationThresholdResults();

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    double current_threshold = minimum_threshold;

    Matrix<size_t> current_confusion;

    Vector<double> current_binary_classification_test;

    double current_roc_curve_distance;

    double optimum_threshold;

    Vector<double> optimal_binary_classification_test(15,1);

    double optimum_roc_curve_distance = 1.0;

    size_t iterations = 0;

    bool end = false;

    while(!end)
    {
        current_confusion = calculate_confusion(current_threshold);
        current_binary_classification_test = calculate_binary_classification_test(current_confusion);

        current_roc_curve_distance =(1-current_binary_classification_test[3])*(1-current_binary_classification_test[3]) +
                                    (current_binary_classification_test[2]-1)*(current_binary_classification_test[2]-1);

        current_roc_curve_distance = sqrt(current_roc_curve_distance);

        results->threshold_data.push_back(current_threshold);

        if(reserve_binary_classification_tests_data)
        {
            results->binary_classification_test_data.push_back(current_binary_classification_test);
        }

        if(reserve_function_data)
        {
            results->function_data.push_back(current_roc_curve_distance);
        }

        if(current_roc_curve_distance < optimum_roc_curve_distance ||
           (current_roc_curve_distance == optimum_roc_curve_distance && current_binary_classification_test[1] < optimal_binary_classification_test[1]))
        {
            optimum_roc_curve_distance = current_roc_curve_distance;
            optimum_threshold = current_threshold;
            optimal_binary_classification_test.set(current_binary_classification_test);
        }

        iterations++;

        if(current_confusion(0,1) == 0 && current_confusion(1,0) == 0)
        {
            end = true;

            if(display)
            {
                cout << "Perfect confusion matrix reached." << endl;
            }

            results->stopping_condition = ThresholdSelectionAlgorithm::PerfectConfusionMatrix;
        }
        else if(current_threshold == maximum_threshold)
        {
            end = true;

            if(display)
            {
                cout << "Algorithm finished \n";
            }

            results->stopping_condition = ThresholdSelectionAlgorithm::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iterations << endl;
            cout << "Current threshold: " << current_threshold << endl;
            cout << "Current error: " << current_binary_classification_test[1] << endl;
            cout << "Current sensitivity: " << current_binary_classification_test[2] << endl;
            cout << "Current specifity: " << current_binary_classification_test[3] << endl;
            cout << "Current ROC curve distance: " << current_roc_curve_distance << endl;
            cout << "Confusion matrix: " << endl
                      << current_confusion << endl;
            cout << endl;
        }

        current_threshold = fmin(maximum_threshold, current_threshold + step);

    }

    if(display)
    {
        cout << "Optimum threshold: " << optimum_threshold << endl;
        cout << "Optimal error: " << optimal_binary_classification_test[1] << endl;
    }

    results->iterations_number = iterations;
    results->final_threshold = optimum_threshold;
    results->final_function_value = optimum_roc_curve_distance;

    neural_network_pointer->get_probabilistic_layer_pointer()->set_decision_threshold(optimum_threshold);

    return(results);
}

// Matrix<string> to_string_matrix() const method

/// Writes as matrix of strings the most representative atributes.

Matrix<string> ROCCurveOptimizationThreshold::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

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

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "");
   string_matrix.set_column(1, values, "");

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML() const method

/// Prints to the screen the roc curve optimization parameters, the stopping criteria
/// and other user stuff concerning the roc curve optimization object.

tinyxml2::XMLDocument* ROCCurveOptimizationThreshold::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("ROCCurveOptimizationThreshold");

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

/// Serializes the ROC curve optimization threshold object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ROCCurveOptimizationThreshold::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("ROCCurveOptimizationThreshold");

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

/// Deserializes a TinyXML document into this roc curve optimization object.
/// @param document TinyXML document containing the member data.

void ROCCurveOptimizationThreshold::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("ROCCurveOptimizationThreshold");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ROCCurveOptimizationThreshold class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ROCCurveOptimizationThreshold element is NULL.\n";

        throw logic_error(buffer.str());
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
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
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
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
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
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Reserve function data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveFunctionData");

        if(element)
        {
            const string new_reserve_function_data = element->GetText();

            try
            {
                set_reserve_function_data(new_reserve_function_data != "0");
            }
            catch(const logic_error& e)
            {
               cout << e.what() << endl;
            }
        }
    }

    // Display
//    {
//        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

//        if(element)
//        {
//           const string new_display = element->GetText();

//           try
//           {
//              set_display(new_display != "0");
//           }
//           catch(const logic_error& e)
//           {
//              cout << e.what() << endl;
//           }
//        }
//    }

}

// void save(const string&) const method

/// Saves to a XML-type file the members of the roc curve optimization object.
/// @param file_name Name of roc curve optimization XML-type file.

void ROCCurveOptimizationThreshold::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads a roc curve optimization object from a XML-type file.
/// @param file_name Name of roc curve optimization XML-type file.

void ROCCurveOptimizationThreshold::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ROCCurveOptimizationThreshold class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
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

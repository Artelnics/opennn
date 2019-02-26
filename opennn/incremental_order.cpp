/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N C R E M E N T A L   O R D E R   C L A S S                                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "incremental_order.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

IncrementalOrder::IncrementalOrder()
    : OrderSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a gradient descent object.

IncrementalOrder::IncrementalOrder(TrainingStrategy* new_training_strategy_pointer)
    : OrderSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}

// XML CONSTRUCTOR

/// XML constructor.
/// @param incremental_order_document Pointer to a TinyXML document containing the incremental order data.

IncrementalOrder::IncrementalOrder(const tinyxml2::XMLDocument& incremental_order_document)
    : OrderSelectionAlgorithm(incremental_order_document)
{
    from_XML(incremental_order_document);
}

// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML incremental order file.

IncrementalOrder::IncrementalOrder(const string& file_name)
    : OrderSelectionAlgorithm(file_name)
{
    load(file_name);
}



// DESTRUCTOR

/// Destructor.

IncrementalOrder::~IncrementalOrder()
{
}

// METHODS


/// Returns the number of the hidden perceptrons pointed in each iteration of the Incremental algorithm.

const size_t& IncrementalOrder::get_step() const
{
    return(step);
}


/// Returns the maximum number of selection failures in the model order selection algorithm.

const size_t& IncrementalOrder::get_maximum_selection_failures() const
{
    return(maximum_selection_failures);
}


/// Sets the members of the model selection object to their default values:

void IncrementalOrder::set_default()
{
    step = 1;

    maximum_selection_failures = 10;
}


/// Sets the number of the hidden perceptrons pointed in each iteration of the Incremental algorithm in the model order selection process.
/// @param new_step number of hidden perceptrons pointed.

void IncrementalOrder::set_step(const size_t& new_step)
{
#ifdef __OPENNN_DEBUG__

    if(new_step <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void set_step(const size_t&) method.\n"
               << "New_step(" << new_step << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_step >(maximum_order-minimum_order))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void set_step(const size_t&) method.\n"
               << "New_step must be less than the distance between maximum_order and minimum_order(" << maximum_order-minimum_order << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    step = new_step;
}


/// Sets the maximum selection failures for the Incremental order selection algorithm.
/// @param new_maximum_loss_failures Maximum number of selection failures in the Incremental order selection algorithm.

void IncrementalOrder::set_maximum_selection_failures(const size_t& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}


/// Perform the order selection with the Incremental method.

IncrementalOrder::IncrementalOrderResults* IncrementalOrder::perform_order_selection()
{
    IncrementalOrderResults* results = new IncrementalOrderResults();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    Vector<double> loss(2);
    double prev_selection_error = numeric_limits<double>::max();

    size_t optimal_order = 0;
    Vector<double> optimum_parameters;
    double optimum_training_loss = 0.0;
    double optimum_selection_error = 0.0;

    Vector<double> parameters_history_row;
    double current_training_loss, current_selection_error;

    size_t order = minimum_order;
    size_t iterations = 0;
    size_t selection_failures = 0;

    bool end = false;

    time_t beginning_time, current_time;
    double elapsed_time = 0.0;

    if(display)
    {
        cout << "Performing Incremental order selection..." << endl;
        cout.flush();
    }

    time(&beginning_time);

    while(!end)
    {
        loss = perform_model_evaluation(order);

        current_training_loss = loss[0];
        current_selection_error = loss[1];

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        results->order_data.push_back(order);

        if(reserve_error_data)
        {
            results->loss_data.push_back(current_training_loss);
        }

        if(reserve_selection_error_data)
        {
            results->selection_error_data.push_back(current_selection_error);
        }

        if(reserve_parameters_data)
        {
            parameters_history_row = get_parameters_order(order);
            results->parameters_data.push_back(parameters_history_row);
        }

        if(iterations == 0
        ||(optimum_selection_error > current_selection_error
        && fabs(optimum_selection_error - current_selection_error) > tolerance))
        {
            optimal_order = order;
            optimum_training_loss = current_training_loss;
            optimum_selection_error = current_selection_error;
            optimum_parameters = get_parameters_order(optimal_order);

        }
        else if(prev_selection_error < current_selection_error)
        {
            selection_failures++;
        }

        prev_selection_error = current_selection_error;
        iterations++;

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                cout << "Maximum time reached." << endl;
            }

            results->stopping_condition = IncrementalOrder::MaximumTime;            
        }
        else if(loss[1] <= selection_error_goal)
        {
            end = true;

            if(display)
            {
                cout << "Selection loss reached." << endl;
            }

            results->stopping_condition = IncrementalOrder::SelectionErrorGoal;

        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                cout << "Maximum number of iterations reached." << endl;
            }

            results->stopping_condition = IncrementalOrder::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display)
            {
                cout << "Maximum selection failures("<<selection_failures<<") reached." << endl;
            }

            results->stopping_condition = IncrementalOrder::MaximumSelectionFailures;
        }
        else if(order == maximum_order)
        {
            end = true;

            if(display)
            {
                cout << "Algorithm finished" << endl;
            }

            results->stopping_condition = IncrementalOrder::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iterations << endl
                      << "Hidden neurons number: " << order << endl
                      << "Training loss: " << loss[0] << endl
                      << "Selection error: " << loss[1] << endl
                      << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
        }


        if(!end)
        {
            order = min(maximum_order, order+step);
        }
    }

    if(display)
    {
        cout << endl
                  << "Optimal order: " << optimal_order <<  endl
                  << "Optimum selection error: " << optimum_selection_error << endl
                  << "Corresponding training loss: " << optimum_training_loss << endl;
    }

    const size_t last_hidden_layer = multilayer_perceptron_pointer->get_layers_number()-2;

    multilayer_perceptron_pointer->set_layer_perceptrons_number(last_hidden_layer, optimal_order);

    multilayer_perceptron_pointer->set_parameters(optimum_parameters);

#ifdef __OPENNN_MPI__
    neural_network_pointer->set_multilayer_perceptron_pointer(multilayer_perceptron_pointer);
#endif

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimum_parameters;
    }

    results->optimal_order = optimal_order;
    results->final_selection_error = optimum_selection_error;
    results->final_loss = perform_model_evaluation(optimal_order)[0];
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    return(results);
}


/// Writes as matrix of strings the most representative atributes.

Matrix<string> IncrementalOrder::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

   // Minimum order

   labels.push_back("Minimum order");

   buffer.str("");
   buffer << minimum_order;

   values.push_back(buffer.str());

   // Maximum order

   labels.push_back("Maximum order");

   buffer.str("");
   buffer << maximum_order;

   values.push_back(buffer.str());

   // Step

   labels.push_back("Step");

   buffer.str("");
   buffer << step;

   values.push_back(buffer.str());

   // Trials number

   labels.push_back("Trials number");

   buffer.str("");
   buffer << trials_number;

   values.push_back(buffer.str());

   // Tolerance

   labels.push_back("Tolerance");

   buffer.str("");
   buffer << tolerance;

   values.push_back(buffer.str());

   // Selection loss goal

   labels.push_back("Selection loss goal");

   buffer.str("");
   buffer << selection_error_goal;

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

   // Plot training error history

   labels.push_back("Plot training error history");

   buffer.str("");

   if(reserve_error_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot selection error history

   labels.push_back("Plot selection error history");

   buffer.str("");

   if(reserve_selection_error_data)
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

    return(string_matrix);
}


/// Prints to the screen the incremental order parameters, the stopping criteria
/// and other user stuff concerning the incremental order object.

tinyxml2::XMLDocument* IncrementalOrder::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("IncrementalOrder");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Minimum order
   {
   element = document->NewElement("MinimumOrder");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_order;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum order
   {
   element = document->NewElement("MaximumOrder");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_order;

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

   // Parameters assays number
   {
   element = document->NewElement("TrialsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << trials_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Performance calculation method
//   {
//   element = document->NewElement("LossCalculationMethod");
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

   // Tolerance
   {
   element = document->NewElement("Tolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << tolerance;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // selection error goal
   {
   element = document->NewElement("SelectionErrorGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << selection_error_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

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

   // Reserve loss data
   {
   element = document->NewElement("ReserveErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_error_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection error data
   {
   element = document->NewElement("ReserveSelectionErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_error_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   return(document);
}


/// Serializes the incremental order object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void IncrementalOrder::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("IncrementalOrder");

    // Minimum order

    file_stream.OpenElement("MinimumOrder");

    buffer.str("");
    buffer << minimum_order;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum order

    file_stream.OpenElement("MaximumOrder");

    buffer.str("");
    buffer << maximum_order;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Step

    file_stream.OpenElement("Step");

    buffer.str("");
    buffer << step;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Parameters assays number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Tolerance

    file_stream.OpenElement("Tolerance");

    buffer.str("");
    buffer << tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // selection error goal

    file_stream.OpenElement("SelectionErrorGoal");

    buffer.str("");
    buffer << selection_error_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection failures

    file_stream.OpenElement("MaximumSelectionFailures");

    buffer.str("");
    buffer << maximum_selection_failures;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve loss data

    file_stream.OpenElement("ReserveErrorHistory");

    buffer.str("");
    buffer << reserve_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error data

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this incremental order object.
/// @param document TinyXML document containing the member data.

void IncrementalOrder::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("IncrementalOrder");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "IncrementalOrder element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Minimum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumOrder");

        if(element)
        {
           const size_t new_minimum_order = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              minimum_order = new_minimum_order;
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Maximum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumOrder");

        if(element)
        {
           const size_t new_maximum_order = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              maximum_order = new_maximum_order;
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Step
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Step");

        if(element)
        {
           const size_t new_step = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              set_step(new_step);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Parameters assays number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
           const size_t new_trials_number = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              set_trials_number(new_trials_number);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Performance calculation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossCalculationMethod");

        if(element)
        {
           const string new_loss_calculation_method = element->GetText();

           try
           {
              set_loss_calculation_method(new_loss_calculation_method);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve parameters data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersData");

        if(element)
        {
           const string new_reserve_parameters_data = element->GetText();

           try
           {
              set_reserve_parameters_data(new_reserve_parameters_data != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveErrorHistory");

        if(element)
        {
           const string new_reserve_error_data = element->GetText();

           try
           {
              set_reserve_error_data(new_reserve_error_data != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve selection error data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
           const string new_reserve_selection_error_data = element->GetText();

           try
           {
              set_reserve_selection_error_data(new_reserve_selection_error_data != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve minimal parameters
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMinimalParameters");

        if(element)
        {
           const string new_reserve_minimal_parameters = element->GetText();

           try
           {
              set_reserve_minimal_parameters(new_reserve_minimal_parameters != "0");
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

    // selection error goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionErrorGoal");

        if(element)
        {
           const double new_selection_error_goal = atof(element->GetText());

           try
           {
              set_selection_error_goal(new_selection_error_goal);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Maximum iterations number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumIterationsNumber");

        if(element)
        {
           const size_t new_maximum_iterations_number = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              set_maximum_iterations_number(new_maximum_iterations_number);
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
           const double new_maximum_time = atoi(element->GetText());

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

    // Tolerance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Tolerance");

        if(element)
        {
           const double new_tolerance = atof(element->GetText());

           try
           {
              set_tolerance(new_tolerance);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Maximum selection failures
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionFailures");

        if(element)
        {
           const size_t new_maximum_selection_failures = static_cast<size_t>(atoi(element->GetText()));

           try
           {
              set_maximum_selection_failures(new_maximum_selection_failures);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }
}


/// Saves to a XML-type file the members of the incremental order object.
/// @param file_name Name of incremental order XML-type file.

void IncrementalOrder::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


/// Loads a incremental order object from a XML-type file.
/// @param file_name Name of incremental order XML-type file.

void IncrementalOrder::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: IncrementalOrder class.\n"
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

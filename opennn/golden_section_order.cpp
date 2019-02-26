/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G O L D E N   S E C T I O N    O R D E R   C L A S S                                                       */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "golden_section_order.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

GoldenSectionOrder::GoldenSectionOrder()
    : OrderSelectionAlgorithm()
{
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a gradient descent object.

GoldenSectionOrder::GoldenSectionOrder(TrainingStrategy* new_training_strategy_pointer)
    : OrderSelectionAlgorithm(new_training_strategy_pointer)
{
}

// XML CONSTRUCTOR

/// XML constructor.
/// @param golden_section_order_document Pointer to a TinyXML document containing the golden section order data.

GoldenSectionOrder::GoldenSectionOrder(const tinyxml2::XMLDocument& golden_section_order_document)
    : OrderSelectionAlgorithm(golden_section_order_document)
{
    from_XML(golden_section_order_document);
}

// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML golden section order file.

GoldenSectionOrder::GoldenSectionOrder(const string& file_name)
    : OrderSelectionAlgorithm(file_name)
{
    load(file_name);
}

// DESTRUCTOR

/// Destructor.

GoldenSectionOrder::~GoldenSectionOrder()
{
}

// METHODS

// GoldenSectionOrderResults* perform_order_selection() method

/// Perform the order selection with the golden section method.

GoldenSectionOrder::GoldenSectionOrderResults* GoldenSectionOrder::perform_order_selection()
{
    GoldenSectionOrderResults* results = new GoldenSectionOrderResults();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    Vector<double> mu_loss(2);
    Vector<double> ln_loss(2);

    Vector<double> a_parameters;
    Vector<double> ln_parameters;
    Vector<double> mu_parameters;
    Vector<double> b_parameters;

    size_t optimal_order;
    Vector<double> optimum_parameters;
    Vector<double> optimum_loss(2);

    bool end = false;
    Vector<double> minimums(4);
    double minimum;
    size_t iterations = 0;

    double current_training_loss, current_selection_error;

    time_t beginning_time, current_time;
    double elapsed_time;

    size_t a = minimum_order;
    size_t b = maximum_order;
    size_t ln = static_cast<size_t>(a + (1. - 0.618)*(b - a));
    size_t mu = static_cast<size_t>(a + 0.618*(b - a));

    if(display)
    {
        cout << "Performing order selection with golden section method..." << endl;
    }

    time(&beginning_time);

    mu_loss = perform_model_evaluation(mu);
    current_training_loss = mu_loss[0];
    current_selection_error = mu_loss[1];
    mu_parameters = get_parameters_order(mu);

    results->order_data.push_back(mu);

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
        results->parameters_data.push_back(mu_parameters);
    }

    ln_loss = perform_model_evaluation(ln);
    current_training_loss = ln_loss[0];
    current_selection_error = ln_loss[1];
    ln_parameters = get_parameters_order(ln);

    results->order_data.push_back(ln);

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
        results->parameters_data.push_back(ln_parameters);
    }

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        cout << "Initial values: " << endl;
        cout << "a = " << a << " ln = " << ln << " mu = " << mu << " b = " << b << endl;
        cout << "ln final training loss: " << ln_loss[0] << endl;
        cout << "ln final selection error: " << ln_loss[1] << endl;
        cout << "mu final training loss: " << mu_loss[0] << endl;
        cout << "mu final selection error: " << mu_loss[1] << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    if((ln == mu) ||(ln > mu) ||(mu < ln))
    {
        end = true;

        if(display)
        {
            cout << "Algorithm finished " << endl;
        }

        results->stopping_condition = GoldenSectionOrder::AlgorithmFinished;
    }

    while(!end){

        if(ln_loss[1] < mu_loss[1]
        || fabs(ln_loss[1] - mu_loss[1]) < tolerance)
        {
            b = mu;
            mu = ln;
            mu_loss = ln_loss;
            ln = static_cast<size_t>(a+ (1.-0.618)*(b-a));

            ln_loss = perform_model_evaluation(ln);
            current_training_loss = ln_loss[0];
            current_selection_error = ln_loss[1];
            ln_parameters = get_parameters_order(ln);

            results->order_data.push_back(ln);

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
                results->parameters_data.push_back(ln_parameters);
            }

        }
        else
        {
            a = ln;
            ln = mu;
            ln_loss = mu_loss;
            mu = static_cast<size_t>(a+0.618*(b-a));

            mu_loss = perform_model_evaluation(mu);
            current_training_loss = mu_loss[0];
            current_selection_error = mu_loss[1];
            mu_parameters = get_parameters_order(mu);

            results->order_data.push_back(mu);

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
                results->parameters_data.push_back(mu_parameters);
            }

        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        iterations++;

        // Stopping criteria

        if((ln == mu) ||(ln > mu))
        {
            end = true;

            if(display)
            {
                cout << "Algorithm finished " << endl;
            }

            results->stopping_condition = GoldenSectionOrder::AlgorithmFinished;
        }
        else if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                cout << "Maximum time reached." << endl;
            }

            results->stopping_condition = GoldenSectionOrder::MaximumTime;
        }
        else if(min(ln_loss[1],mu_loss[1]) <= selection_error_goal)
        {
            end = true;

            if(display)
            {
                cout << "Selection loss reached." << endl;
            }

            results->stopping_condition = GoldenSectionOrder::SelectionErrorGoal;
        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                cout << "Maximum number of iterations reached." << endl;
            }

            results->stopping_condition = GoldenSectionOrder::MaximumIterations;
        }

        if(display && !end)
        {

            cout << "Iteration: " << iterations << endl;
            cout << "a = " << a << " ln = " << ln << " mu = " << mu << " b = " << b << endl;
            cout << "ln final training loss: " << ln_loss[0] << endl;
            cout << "ln final selection error: " << ln_loss[1] << endl;
            cout << "mu final training loss: " << mu_loss[0] << endl;
            cout << "mu final selection error: " << mu_loss[1] << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
        }
    }

    minimums[0] = perform_model_evaluation(a)[1];
    a_parameters = get_parameters_order(a);

    minimums[1] = perform_model_evaluation(ln)[1];
    ln_parameters = get_parameters_order(ln);

    minimums[2] = perform_model_evaluation(mu)[1];
    mu_parameters = get_parameters_order(mu);

    minimums[3] = perform_model_evaluation(b)[1];
    b_parameters = get_parameters_order(b);

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        cout << "Iteration: " << iterations << endl;
        cout << "a = " << a << " ln = " << ln << " mu = " << mu << " b = " << b << endl;
        cout << "a final training loss: " << perform_model_evaluation(a)[0] << endl;
        cout << "a final selection error: " << perform_model_evaluation(a)[1] << endl;
        cout << "ln final training loss: " << ln_loss[0] << endl;
        cout << "ln final selection error: " << ln_loss[1] << endl;
        cout << "mu final training loss: " << mu_loss[0] << endl;
        cout << "mu final selection error: " << mu_loss[1] << endl;
        cout << "b final training loss: " << perform_model_evaluation(b)[0] << endl;
        cout << "b final selection error: " << perform_model_evaluation(b)[1] << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    minimum = minimums.calculate_minimum();

    if(fabs(minimums[0] - minimum) < tolerance)
    {
        optimal_order = a;

        optimum_parameters = a_parameters;

        optimum_loss[0] = perform_model_evaluation(a)[0];
        optimum_loss[1] = minimums[0];

    }
    else if(fabs(minimums[1] - minimum) < tolerance)
    {
        optimal_order = ln;

        optimum_parameters = ln_parameters;

        optimum_loss[0] = perform_model_evaluation(ln)[0];
        optimum_loss[1] = minimums[1];

    }
    else if(fabs(minimums[2] - minimum) < tolerance)
    {
        optimal_order = mu;

        optimum_parameters = mu_parameters;

        optimum_loss[0] = perform_model_evaluation(mu)[0];
        optimum_loss[1] = minimums[2];
    }
    else
    {
        optimal_order = b;

        optimum_parameters = b_parameters;

        optimum_loss[0] = perform_model_evaluation(b)[0];
        optimum_loss[1] = minimums[3];
    }

    if(display)
    {
        cout << "Optimal order: " << optimal_order << endl;
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
    results->final_loss = optimum_loss[0];
    results->final_selection_error = optimum_loss[1];
    results->elapsed_time = elapsed_time;
    results->iterations_number = iterations;

    return(results);
}


/// Prints to the screen the order selection parameters, the stopping criteria
/// and other user stuff concerning the golden section order object.

tinyxml2::XMLDocument* GoldenSectionOrder::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("GoldenSectionOrder");

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
   {
   element = document->NewElement("LossCalculationMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_loss_calculation_method().c_str());
   element->LinkEndChild(text);
   }

   // Reserve parameters data
   {
   element = document->NewElement("ReserveParametersData");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_parameters_data;

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

   // Reserve minimal parameters
   {
   element = document->NewElement("ReserveMinimalParameters");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_minimal_parameters;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Display
   {
   element = document->NewElement("Display");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << display;

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
   {
   element = document->NewElement("MaximumIterationsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_iterations_number;

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

   // Tolerance
   {
   element = document->NewElement("Tolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << tolerance;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   return(document);
}


/// Serializes the golden section object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GoldenSectionOrder::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("GoldenSectionOrder");

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

    // Parameters assays number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Performance calculation method

    file_stream.OpenElement("LossCalculationMethod");

    file_stream.PushText(write_loss_calculation_method().c_str());

    file_stream.CloseElement();

    // Reserve parameters data

    file_stream.OpenElement("ReserveParametersData");

    buffer.str("");
    buffer << reserve_parameters_data;

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

    // Reserve minimal parameters

    file_stream.OpenElement("ReserveMinimalParameters");

    buffer.str("");
    buffer << reserve_minimal_parameters;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // selection error goal

    file_stream.OpenElement("SelectionErrorGoal");

    buffer.str("");
    buffer << selection_error_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations

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

    // Tolerance

    file_stream.OpenElement("Tolerance");

    buffer.str("");
    buffer << tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this golden section order object.
/// @param document TinyXML document containing the member data.

void GoldenSectionOrder::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GoldenSectionOrder");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GoldenSectionOrder element is nullptr.\n";

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

}


/// Saves to a XML-type file the members of the golden section order object.
/// @param file_name Name of golden section order XML-type file.

void GoldenSectionOrder::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


/// Loads a golden section order object from a XML-type file.
/// @param file_name Name of golden section order XML-type file.

void GoldenSectionOrder::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: GoldenSectionOrder class.\n"
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

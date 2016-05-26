/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N C R E M E N T A L   O R D E R   C L A S S                                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "incremental_order.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

IncrementalOrder::IncrementalOrder(void)
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

IncrementalOrder::IncrementalOrder(const std::string& file_name)
    : OrderSelectionAlgorithm(file_name)
{
    load(file_name);
}



// DESTRUCTOR

/// Destructor.

IncrementalOrder::~IncrementalOrder(void)
{
}

// METHODS


// const size_t& get_step(void) const method

/// Returns the number of the hidden perceptrons pointed in each iteration of the Incremental algorithm.

const size_t& IncrementalOrder::get_step(void) const
{
    return(step);
}

// const size_t& get_maximum_selection_failures(void) const method

/// Returns the maximum number of selection failures in the model order selection algorithm.

const size_t& IncrementalOrder::get_maximum_selection_failures(void) const
{
    return(maximum_selection_failures);
}

// void set_default(void) method

/// Sets the members of the model selection object to their default values:

void IncrementalOrder::set_default(void)
{
    step = 1;

    maximum_selection_failures = 10;
}

// void set_step(const size_t&) method

/// Sets the number of the hidden perceptrons pointed in each iteration of the Incremental algorithm in the model order selection process.
/// @param new_step number of hidden perceptrons pointed.

void IncrementalOrder::set_step(const size_t& new_step)
{
#ifdef __OPENNN_DEBUG__

    if(new_step <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void set_step(const size_t&) method.\n"
               << "New_step (" << new_step << ") must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(new_step > (maximum_order-minimum_order))
    {
        std::ostringstream buffer;
        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void set_step(const size_t&) method.\n"
               << "New_step must be less than the distance between maximum_order and minimum_order (" << maximum_order-minimum_order << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    step = new_step;
}

// void set_maximum_selection_failures(const size_t&) method

/// Sets the maximum selection failures for the Incremental order selection algorithm.
/// @param new_maximum_performance_failures Maximum number of selection failures in the Incremental order selection algorithm.

void IncrementalOrder::set_maximum_selection_failures(const size_t& new_maximum_performance_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_performance_failures <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_performance_failures;
}

// IncrementalOrderResults* perform_order_selection(void) method

/// Perform the order selection with the Incremental method.

IncrementalOrder::IncrementalOrderResults* IncrementalOrder::perform_order_selection(void)
{
    IncrementalOrderResults* results = new IncrementalOrderResults();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_performance_functional_pointer()->get_neural_network_pointer();
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    Vector<double> performance(2);
    double prev_selection_performance = 1.0e99;

    size_t optimal_order;
    Vector<double> optimum_parameters;
    double optimum_training_performance;
    double optimum_selection_performance;

    Vector<double> parameters_history_row;
    double current_training_performance, current_selection_performance;

    size_t order = minimum_order;
    size_t iterations = 0;
    size_t selection_failures = 0;

    bool end = false;

    time_t beginning_time, current_time;
    double elapsed_time;

    if(display)
    {
        std::cout << "Performing Incremental order selection..." << std::endl;
        std::cout.flush();
    }

    time(&beginning_time);

    while (!end)
    {
        performance = perform_model_evaluation(order);
        current_training_performance = performance[0];
        current_selection_performance = performance[1];

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        results->order_data.push_back(order);

        if(reserve_performance_data)
        {
            results->performance_data.push_back(current_training_performance);
        }

        if(reserve_selection_performance_data)
        {
            results->selection_performance_data.push_back(current_selection_performance);
        }

        if(reserve_parameters_data)
        {
            parameters_history_row = get_parameters_order(order);
            results->parameters_data.push_back(parameters_history_row);
        }

        if(iterations == 0
        || (optimum_selection_performance > current_selection_performance
        && fabs(optimum_selection_performance - current_selection_performance) > tolerance))
        {
            optimal_order = order;
            optimum_training_performance = current_training_performance;
            optimum_selection_performance = current_selection_performance;
            optimum_parameters = get_parameters_order(optimal_order);

        }else if(prev_selection_performance < current_selection_performance)
        {
            selection_failures++;
        }

        prev_selection_performance = current_selection_performance;
        iterations++;

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum time reached." << std::endl;
            }

            results->stopping_condition = IncrementalOrder::MaximumTime;            
        }
        else if(performance[1] <= selection_performance_goal)
        {
            end = true;

            if(display)
            {
                std::cout << "Selection performance reached." << std::endl;
            }

            results->stopping_condition = IncrementalOrder::SelectionPerformanceGoal;

        }else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum number of iterations reached." << std::endl;
            }

            results->stopping_condition = IncrementalOrder::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum selection failures ("<<selection_failures<<") reached." << std::endl;
            }

            results->stopping_condition = IncrementalOrder::MaximumSelectionFailures;
        }
        else if(order == maximum_order)
        {
            end = true;

            if(display)
            {
                std::cout << "Algorithm finished" << std::endl;
            }

            results->stopping_condition = IncrementalOrder::AlgorithmFinished;
        }

        if(display)
        {
            std::cout << "Iteration: " << iterations << std::endl
                      << "Hidden neurons number: " << order << std::endl
                      << "Training performance: " << performance[0] << std::endl
                      << "Selection performance: " << performance[1] << std::endl
                      << "Elapsed time: " << elapsed_time << std::endl;
        }


        if(!end)
        {
            order = std::min(maximum_order, order+step);
        }
    }

    if(display)
    {
        std::cout << std::endl
                  << "Optimal order: " << optimal_order << std:: endl
                  << "Optimum selection performance: " << optimum_selection_performance << std::endl
                  << "Corresponding training performance: " << optimum_training_performance << std::endl;
    }

    const size_t last_hidden_layer = multilayer_perceptron_pointer->get_layers_number()-2;
    const size_t perceptrons_number = multilayer_perceptron_pointer->get_layer_pointer(last_hidden_layer)->get_perceptrons_number();

    if(optimal_order > perceptrons_number)
    {
        multilayer_perceptron_pointer->grow_layer_perceptron(last_hidden_layer,optimal_order-perceptrons_number);
    }
    else
    {
        for (size_t i = 0; i < (perceptrons_number-optimal_order); i++)
        {
            multilayer_perceptron_pointer->prune_layer_perceptron(last_hidden_layer,0);
        }
    }

    multilayer_perceptron_pointer->set_parameters(optimum_parameters);

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimum_parameters;
    }

    results->optimal_order = optimal_order;
    results->final_selection_performance = optimum_selection_performance;
    results->final_performance = perform_model_evaluation(optimal_order)[0];
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    return(results);
}

// Matrix<std::string> to_string_matrix(void) const method

// the most representative

Matrix<std::string> IncrementalOrder::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

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

   // Selection performance goal

   labels.push_back("Selection performance goal");

   buffer.str("");
   buffer << selection_performance_goal;

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

   // Plot training performance history

   labels.push_back("Plot training performance history");

   buffer.str("");
   buffer << reserve_performance_data;

   values.push_back(buffer.str());

   // Plot selection performance history

   labels.push_back("Plot selection performance history");

   buffer.str("");
   buffer << reserve_selection_performance_data;

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

tinyxml2::XMLDocument* IncrementalOrder::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("IncrementalOrder");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

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

   // Tolerance
   {
   element = document->NewElement("Tolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << tolerance;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Selection performance goal
   {
   element = document->NewElement("SelectionPerformanceGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << selection_performance_goal;

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

   // Reserve performance data
   {
   element = document->NewElement("ReservePerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_performance_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection performance data
   {
   element = document->NewElement("ReserveSelectionPerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_performance_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void IncrementalOrder::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

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

    // Selection performance goal

    file_stream.OpenElement("SelectionPerformanceGoal");

    buffer.str("");
    buffer << selection_performance_goal;

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

    // Reserve performance data

    file_stream.OpenElement("ReservePerformanceHistory");

    buffer.str("");
    buffer << reserve_performance_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection performance data

    file_stream.OpenElement("ReserveSelectionPerformanceHistory");

    buffer.str("");
    buffer << reserve_selection_performance_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this incremental order object.
/// @param document TinyXML document containing the member data.

void IncrementalOrder::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("IncrementalOrder");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "IncrementalOrder element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Minimum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumOrder");

        if(element)
        {
           const size_t new_minimum_order = atoi(element->GetText());

           try
           {
              minimum_order = new_minimum_order;
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Maximum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumOrder");

        if(element)
        {
           const size_t new_maximum_order = atoi(element->GetText());

           try
           {
              maximum_order = new_maximum_order;
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Parameters assays number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
           const size_t new_trials_number = atoi(element->GetText());

           try
           {
              set_trials_number(new_trials_number);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Performance calculation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PerformanceCalculationMethod");

        if(element)
        {
           const std::string new_performance_calculation_method = element->GetText();

           try
           {
              set_performance_calculation_method(new_performance_calculation_method);
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
           const size_t new_step = atoi(element->GetText());

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

    // Reserve parameters data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersData");

        if(element)
        {
           const std::string new_reserve_parameters_data = element->GetText();

           try
           {
              set_reserve_parameters_data(new_reserve_parameters_data != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Reserve performance data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

        if(element)
        {
           const std::string new_reserve_performance_data = element->GetText();

           try
           {
              set_reserve_performance_data(new_reserve_performance_data != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Reserve selection performance data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionPerformanceHistory");

        if(element)
        {
           const std::string new_reserve_selection_performance_data = element->GetText();

           try
           {
              set_reserve_selection_performance_data(new_reserve_selection_performance_data != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Reserve minimal parameters
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMinimalParameters");

        if(element)
        {
           const std::string new_reserve_minimal_parameters = element->GetText();

           try
           {
              set_reserve_minimal_parameters(new_reserve_minimal_parameters != "0");
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

    // Selection performance goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionPerformanceGoal");

        if(element)
        {
           const double new_selection_performance_goal = atof(element->GetText());

           try
           {
              set_selection_performance_goal(new_selection_performance_goal);
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

void IncrementalOrder::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a incremental order object from a XML-type file.
/// @param file_name Name of incremental order XML-type file.

void IncrementalOrder::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: IncrementalOrder class.\n"
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

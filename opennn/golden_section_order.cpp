/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G O L D E N   S E C T I O N    O R D E R   C L A S S                                                       */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "golden_section_order.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

GoldenSectionOrder::GoldenSectionOrder(void)
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

GoldenSectionOrder::GoldenSectionOrder(const std::string& file_name)
    : OrderSelectionAlgorithm(file_name)
{
    load(file_name);
}

// DESTRUCTOR

/// Destructor.

GoldenSectionOrder::~GoldenSectionOrder(void)
{
}

// METHODS

// GoldenSectionOrderResults* perform_order_selection(void) method

/// Perform the order selection with the golden section method.

GoldenSectionOrder::GoldenSectionOrderResults* GoldenSectionOrder::perform_order_selection(void)
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

    double current_training_loss, current_selection_loss;

    time_t beginning_time, current_time;
    double elapsed_time;

    size_t a = minimum_order;
    size_t b = maximum_order;
    size_t ln = (int)(a+(1.-0.618)*(b-a));
    size_t mu = (int)(a+0.618*(b-a));

    if(display)
    {
        std::cout << "Performing order selection with golden section method..." << std::endl;
    }

    time(&beginning_time);

    mu_loss = perform_model_evaluation(mu);
    current_training_loss = mu_loss[0];
    current_selection_loss = mu_loss[1];
    mu_parameters = get_parameters_order(mu);

    results->order_data.push_back(mu);

    if(reserve_loss_data)
    {
        results->loss_data.push_back(current_training_loss);
    }

    if(reserve_selection_loss_data)
    {
        results->selection_loss_data.push_back(current_selection_loss);
    }

    if(reserve_parameters_data)
    {
        results->parameters_data.push_back(mu_parameters);
    }

    ln_loss = perform_model_evaluation(ln);
    current_training_loss = ln_loss[0];
    current_selection_loss = ln_loss[1];
    ln_parameters = get_parameters_order(ln);

    results->order_data.push_back(ln);

    if(reserve_loss_data)
    {
        results->loss_data.push_back(current_training_loss);
    }

    if(reserve_selection_loss_data)
    {
        results->selection_loss_data.push_back(current_selection_loss);
    }

    if(reserve_parameters_data)
    {
        results->parameters_data.push_back(ln_parameters);
    }

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        std::cout << "Initial values: " << std::endl;
        std::cout << "a = " << a << "  ln = " << ln << " mu = " << mu << " b = " << b << std::endl;
        std::cout << "ln final training loss: " << ln_loss[0] << std::endl;
        std::cout << "ln final selection loss: " << ln_loss[1] << std::endl;
        std::cout << "mu final training loss: " << mu_loss[0] << std::endl;
        std::cout << "mu final selection loss: " << mu_loss[1] << std::endl;
        std::cout << "Elapsed time: " << elapsed_time << std::endl;
    }

    if((ln == mu) || (ln > mu) || (mu < ln))
    {
        end = true;

        if(display)
        {
            std::cout << "Algorithm finished " << std::endl;
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
            ln = (int)(a+(1.-0.618)*(b-a));

            ln_loss = perform_model_evaluation(ln);
            current_training_loss = ln_loss[0];
            current_selection_loss = ln_loss[1];
            ln_parameters = get_parameters_order(ln);

            results->order_data.push_back(ln);

            if(reserve_loss_data)
            {
                results->loss_data.push_back(current_training_loss);
            }

            if(reserve_selection_loss_data)
            {
                results->selection_loss_data.push_back(current_selection_loss);
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
            mu = (int)(a+0.618*(b-a));

            mu_loss = perform_model_evaluation(mu);
            current_training_loss = mu_loss[0];
            current_selection_loss = mu_loss[1];
            mu_parameters = get_parameters_order(mu);

            results->order_data.push_back(mu);

            if(reserve_loss_data)
            {
                results->loss_data.push_back(current_training_loss);
            }

            if(reserve_selection_loss_data)
            {
                results->selection_loss_data.push_back(current_selection_loss);
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

        if((ln == mu) || (ln > mu))
        {
            end = true;

            if(display)
            {
                std::cout << "Algorithm finished " << std::endl;
            }

            results->stopping_condition = GoldenSectionOrder::AlgorithmFinished;
        }
        else if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum time reached." << std::endl;
            }

            results->stopping_condition = GoldenSectionOrder::MaximumTime;
        }
        else if(std::min(ln_loss[1],mu_loss[1]) <= selection_loss_goal)
        {
            end = true;

            if(display)
            {
                std::cout << "Selection loss reached." << std::endl;
            }

            results->stopping_condition = GoldenSectionOrder::SelectionLossGoal;
        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum number of iterations reached." << std::endl;
            }

            results->stopping_condition = GoldenSectionOrder::MaximumIterations;
        }

        if(display && !end)
        {

            std::cout << "Iteration: " << iterations << std::endl;
            std::cout << "a = " << a << "  ln = " << ln << " mu = " << mu << " b = " << b << std::endl;
            std::cout << "ln final training loss: " << ln_loss[0] << std::endl;
            std::cout << "ln final selection loss: " << ln_loss[1] << std::endl;
            std::cout << "mu final training loss: " << mu_loss[0] << std::endl;
            std::cout << "mu final selection loss: " << mu_loss[1] << std::endl;
            std::cout << "Elapsed time: " << elapsed_time << std::endl;
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
        std::cout << "Iteration: " << iterations << std::endl;
        std::cout << "a = " << a << "  ln = " << ln << " mu = " << mu << " b = " << b << std::endl;
        std::cout << "a final training loss: " << perform_model_evaluation(a)[0] << std::endl;
        std::cout << "a final selection loss: " << perform_model_evaluation(a)[1] << std::endl;
        std::cout << "ln final training loss: " << ln_loss[0] << std::endl;
        std::cout << "ln final selection loss: " << ln_loss[1] << std::endl;
        std::cout << "mu final training loss: " << mu_loss[0] << std::endl;
        std::cout << "mu final selection loss: " << mu_loss[1] << std::endl;
        std::cout << "b final training loss: " << perform_model_evaluation(b)[0] << std::endl;
        std::cout << "b final selection loss: " << perform_model_evaluation(b)[1] << std::endl;
        std::cout << "Elapsed time: " << elapsed_time << std::endl;
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
        std::cout << "Optimal order: " << optimal_order << std::endl;
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
            multilayer_perceptron_pointer->prune_layer_perceptron(last_hidden_layer,0);
    }

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
    results->final_selection_loss = optimum_loss[1];
    results->elapsed_time = elapsed_time;
    results->iterations_number = iterations;

    return(results);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the order selection parameters, the stopping criteria
/// and other user stuff concerning the golden section order object.

tinyxml2::XMLDocument* GoldenSectionOrder::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("GoldenSectionOrder");

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
   element = document->NewElement("PerformanceCalculationMethod");
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
   element = document->NewElement("ReservePerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_loss_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection loss data
   {
   element = document->NewElement("ReserveSelectionLossHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_loss_data;

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

   // selection loss goal
   {
   element = document->NewElement("SelectionLossGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << selection_loss_goal;

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


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the golden section object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GoldenSectionOrder::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

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

    file_stream.OpenElement("PerformanceCalculationMethod");

    file_stream.PushText(write_loss_calculation_method().c_str());

    file_stream.CloseElement();

    // Reserve parameters data

    file_stream.OpenElement("ReserveParametersData");

    buffer.str("");
    buffer << reserve_parameters_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve loss data

    file_stream.OpenElement("ReservePerformanceHistory");

    buffer.str("");
    buffer << reserve_loss_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection loss data

    file_stream.OpenElement("ReserveSelectionLossHistory");

    buffer.str("");
    buffer << reserve_selection_loss_data;

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

    // selection loss goal

    file_stream.OpenElement("SelectionLossGoal");

    buffer.str("");
    buffer << selection_loss_goal;

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


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this golden section order object.
/// @param document TinyXML document containing the member data.

void GoldenSectionOrder::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GoldenSectionOrder");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GoldenSectionOrder element is NULL.\n";

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
           const std::string new_loss_calculation_method = element->GetText();

           try
           {
              set_loss_calculation_method(new_loss_calculation_method);
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

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

        if(element)
        {
           const std::string new_reserve_loss_data = element->GetText();

           try
           {
              set_reserve_loss_data(new_reserve_loss_data != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Reserve selection loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

        if(element)
        {
           const std::string new_reserve_selection_loss_data = element->GetText();

           try
           {
              set_reserve_selection_loss_data(new_reserve_selection_loss_data != "0");
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

    // selection loss goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionLossGoal");

        if(element)
        {
           const double new_selection_loss_goal = atof(element->GetText());

           try
           {
              set_selection_loss_goal(new_selection_loss_goal);
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

}

// void save(const std::string&) const method

/// Saves to a XML-type file the members of the golden section order object.
/// @param file_name Name of golden section order XML-type file.

void GoldenSectionOrder::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a golden section order object from a XML-type file.
/// @param file_name Name of golden section order XML-type file.

void GoldenSectionOrder::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GoldenSectionOrder class.\n"
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

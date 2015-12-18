/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
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

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_performance_functional_pointer()->get_neural_network_pointer();
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Vector<double> mu_performance(2);
    Vector<double> ln_performance(2);

    Vector<double> a_parameters;
    Vector<double> ln_parameters;
    Vector<double> mu_parameters;
    Vector<double> b_parameters;

    bool end = false;
    Vector<double> history_row(2);
    Vector<double> parameters_history_row;
    Vector<double> minimums(4);
    double minimum;
    size_t iterations = 0;

    double current_training_performance, current_generalization_performance;

    time_t beginning_time, current_time;
    double elapsed_time;

    size_t a = minimum_order;
    size_t b = maximum_order;
    size_t ln = (int)(a+(1.-0.618)*(b-a));
    size_t mu = (int)(a+0.618*(b-a));

    if (display)
        std::cout << "Performing order selection with golden section method..." << std::endl;

    time(&beginning_time);

    mu_performance = calculate_performances(mu);
    current_training_performance = mu_performance[0];
    current_generalization_performance = mu_performance[1];
    mu_parameters = get_parameters_order(mu);

    if (reserve_performance_data)
    {
        history_row[0] = (double)mu;
        history_row[1] = current_training_performance;
        results->performance_data.push_back(history_row);
    }

    if (reserve_generalization_performance_data)
    {
        history_row[0] = (double)mu;
        history_row[1] = current_generalization_performance;
        results->generalization_performance_data.push_back(history_row);
    }

    if (reserve_parameters_data)
    {
        parameters_history_row = get_parameters_order(mu);
        parameters_history_row.insert(parameters_history_row.begin(),(double)mu);
        results->parameters_data.push_back(parameters_history_row);
    }

    ln_performance = calculate_performances(ln);
    current_training_performance = ln_performance[0];
    current_generalization_performance = ln_performance[1];
    ln_parameters = get_parameters_order(ln);

    if (reserve_performance_data)
    {
        history_row[0] = (double)ln;
        history_row[1] = current_training_performance;
        results->performance_data.push_back(history_row);
    }

    if (reserve_generalization_performance_data)
    {
        history_row[0] = (double)ln;
        history_row[1] = current_generalization_performance;
        results->generalization_performance_data.push_back(history_row);
    }

    if (reserve_parameters_data)
    {
        parameters_history_row = get_parameters_order(ln);
        parameters_history_row.insert(parameters_history_row.begin(),(double)ln);
        results->parameters_data.push_back(parameters_history_row);
    }

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if (display)
    {
        std::cout << "Initial values : " << std::endl;
        std::cout << "a = " << a << "  ln = " << ln << " mu = " << mu << " b = " << b << std::endl;
        std::cout << "ln final training performance : " << ln_performance[0] << std::endl;
        std::cout << "ln final generalization performance : " << ln_performance[1] << std::endl;
        std::cout << "mu final training performance : " << mu_performance[0] << std::endl;
        std::cout << "mu final generalization performance : " << mu_performance[1] << std::endl;
        std::cout << "Elapsed time : " << elapsed_time << std::endl;

    }

    if ((ln == mu) || (ln > mu) || (mu < ln)){
        end = true;
        if (display)
            std::cout << "Algorithm finished " << std::endl;
        results->stopping_condition = GoldenSectionOrder::AlgorithmFinished;
    }

    while(!end){

        if (ln_performance[1] < mu_performance[1]
        || fabs(ln_performance[1] - mu_performance[1]) < tolerance)
        {
            b = mu;
            mu = ln;
            mu_performance = ln_performance;
            ln = (int)(a+(1.-0.618)*(b-a));

            ln_performance = calculate_performances(ln);
            current_training_performance = ln_performance[0];
            current_generalization_performance = ln_performance[1];
            ln_parameters = get_parameters_order(ln);

            if (reserve_performance_data)
            {
                history_row[0] = (double)ln;
                history_row[1] = current_training_performance;
                results->performance_data.push_back(history_row);
            }

            if (reserve_generalization_performance_data)
            {
                history_row[0] = (double)ln;
                history_row[1] = current_generalization_performance;
                results->generalization_performance_data.push_back(history_row);
            }

            if (reserve_parameters_data)
            {
                parameters_history_row = get_parameters_order(ln);
                parameters_history_row.insert(parameters_history_row.begin(),(double)ln);
                results->parameters_data.push_back(parameters_history_row);
            }

        }else
        {
            a = ln;
            ln = mu;
            ln_performance = mu_performance;
            mu = (int)(a+0.618*(b-a));

            mu_performance = calculate_performances(mu);
            current_training_performance = mu_performance[0];
            current_generalization_performance = mu_performance[1];
            mu_parameters = get_parameters_order(mu);

            if (reserve_performance_data)
            {
                history_row[0] = (double)mu;
                history_row[1] = current_training_performance;
                results->performance_data.push_back(history_row);
            }

            if (reserve_generalization_performance_data)
            {
                history_row[0] = (double)mu;
                history_row[1] = current_generalization_performance;
                results->generalization_performance_data.push_back(history_row);
            }

            if (reserve_parameters_data)
            {
                parameters_history_row = get_parameters_order(mu);
                parameters_history_row.insert(parameters_history_row.begin(),(double)mu);
                results->parameters_data.push_back(parameters_history_row);
            }

        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        iterations++;

        // Stopping criteria

        if ((ln == mu) || (ln > mu)){
            end = true;
            if (display)
                std::cout << "Algorithm finished " << std::endl;
            results->stopping_condition = GoldenSectionOrder::AlgorithmFinished;
        }else if (elapsed_time > maximum_time)
        {
            end = true;
            if (display)
                std::cout << "Maximum time reached." << std::endl;
            results->stopping_condition = GoldenSectionOrder::MaximumTime;
        }else if (fmin(ln_performance[1],mu_performance[1]) < generalization_performance_goal)
        {
            end = true;
            if (display)
                std::cout << "Generalization performance reached." << std::endl;
            results->stopping_condition = GoldenSectionOrder::GeneralizationPerformanceGoal;
        }else if (iterations > maximum_iterations_number)
        {
            end = true;
            if (display)
                std::cout << "Maximum number of iterations reached." << std::endl;
            results->stopping_condition = GoldenSectionOrder::MaximumIterations;
        }

        if (display && !end)
        {

            std::cout << "Iteration : " << iterations << std::endl;
            std::cout << "a = " << a << "  ln = " << ln << " mu = " << mu << " b = " << b << std::endl;
            std::cout << "ln final training performance : " << ln_performance[0] << std::endl;
            std::cout << "ln final generalization performance : " << ln_performance[1] << std::endl;
            std::cout << "mu final training performance : " << mu_performance[0] << std::endl;
            std::cout << "mu final generalization performance : " << mu_performance[1] << std::endl;
            std::cout << "Elapsed time : " << elapsed_time << std::endl;
        }
    }

    minimums[0] = calculate_performances(a)[1];
    a_parameters = get_parameters_order(a);

    minimums[1] = calculate_performances(ln)[1];
    ln_parameters = get_parameters_order(ln);

    minimums[2] = calculate_performances(mu)[1];
    mu_parameters = get_parameters_order(mu);

    minimums[3] = calculate_performances(b)[1];
    b_parameters = get_parameters_order(b);

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if (display)
    {
        std::cout << "Iteration : " << iterations << std::endl;
        std::cout << "a = " << a << "  ln = " << ln << " mu = " << mu << " b = " << b << std::endl;
        std::cout << "a final training performance : " << calculate_performances(a)[0] << std::endl;
        std::cout << "a final generalization performance : " << calculate_performances(a)[1] << std::endl;
        std::cout << "ln final training performance : " << ln_performance[0] << std::endl;
        std::cout << "ln final generalization performance : " << ln_performance[1] << std::endl;
        std::cout << "mu final training performance : " << mu_performance[0] << std::endl;
        std::cout << "mu final generalization performance : " << mu_performance[1] << std::endl;
        std::cout << "b final training performance : " << calculate_performances(b)[0] << std::endl;
        std::cout << "b final generalization performance : " << calculate_performances(b)[1] << std::endl;
        std::cout << "Elapsed time : " << elapsed_time << std::endl;
    }

    minimum = minimums.calculate_minimum();

    if (fabs(minimums[0] - minimum) < tolerance)
    {
        if (display)
            std::cout << "Optimal order : " << a << std::endl;

        multilayer_perceptron_pointer->set(inputs_number, a, outputs_number);
        multilayer_perceptron_pointer->set_parameters(a_parameters);

        if (reserve_minimal_parameters)
            results->minimal_parameters = a_parameters;

        results->optimal_order = a;
        results->final_generalization_performance = minimums[0];
        results->final_performance = calculate_performances(a)[0];

    }else if (fabs(minimums[1] - minimum) < tolerance)
    {
        if (display)
            std::cout << "Optimal order : " << ln << std::endl;

        multilayer_perceptron_pointer->set(inputs_number, ln, outputs_number);
        multilayer_perceptron_pointer->set_parameters(ln_parameters);

        if (reserve_minimal_parameters)
            results->minimal_parameters = ln_parameters;

        results->optimal_order = ln;
        results->final_generalization_performance = minimums[1];
        results->final_performance = calculate_performances(ln)[0];

    }else if(fabs(minimums[2] - minimum) < tolerance)
    {
        if (display)
            std::cout << "Optimal order : " << mu << std::endl;

        multilayer_perceptron_pointer->set(inputs_number, mu, outputs_number);
        multilayer_perceptron_pointer->set_parameters(mu_parameters);

        if (reserve_minimal_parameters)
            results->minimal_parameters = mu_parameters;

        results->optimal_order = mu;
        results->final_generalization_performance = minimums[2];
        results->final_performance = calculate_performances(mu)[0];

    }else
    {
        if (display)
            std::cout << "Optimal order : " << b << std::endl;

        multilayer_perceptron_pointer->set(inputs_number, b, outputs_number);
        multilayer_perceptron_pointer->set_parameters(b_parameters);

        if (reserve_minimal_parameters)
            results->minimal_parameters = b_parameters;

        results->optimal_order = b;
        results->final_generalization_performance = minimums[3];
        results->final_performance = calculate_performances(b)[0];

    }

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

   text = document->NewText(write_performance_calculation_method().c_str());
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

   // Reserve performance data
   {
   element = document->NewElement("ReservePerformanceData");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_performance_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve generalization performance data
   {
   element = document->NewElement("ReserveGeneralizationPerformanceData");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_generalization_performance_data;

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

   // Generalization performance goal
   {
   element = document->NewElement("GeneralizationPerformanceGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << generalization_performance_goal;

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceData");

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

    // Reserve generalization performance data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGeneralizationPerformanceData");

        if(element)
        {
           const std::string new_reserve_generalization_performance_data = element->GetText();

           try
           {
              set_reserve_generalization_performance_data(new_reserve_generalization_performance_data != "0");
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

    // Generalization performance goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("GeneralizationPerformanceGoal");

        if(element)
        {
           const double new_generalization_performance_goal = atof(element->GetText());

           try
           {
              set_generalization_performance_goal(new_generalization_performance_goal);
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

   if (document.LoadFile(file_name.c_str()))
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



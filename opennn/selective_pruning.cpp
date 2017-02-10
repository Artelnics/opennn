/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S E L E C T I V E  P R U N I N G   C L A S S                                                               */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "selective_pruning.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

SelectivePruning::SelectivePruning(void)
    : InputsSelectionAlgorithm()
{
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

SelectivePruning::SelectivePruning(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelectionAlgorithm(new_training_strategy_pointer)
{
}


// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML selective pruning file.

SelectivePruning::SelectivePruning(const std::string& file_name)
    : InputsSelectionAlgorithm(file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/// @param selective_pruning_document Pointer to a TinyXML document containing the selective pruning data.

SelectivePruning::SelectivePruning(const tinyxml2::XMLDocument& selective_pruning_document)
    : InputsSelectionAlgorithm(selective_pruning_document)
{
    from_XML(selective_pruning_document);
}


// DESTRUCTOR

/// Destructor.

SelectivePruning::~SelectivePruning(void)
{
}

// METHODS

// const size_t& get_minimum_inputs_number(void) const method

/// Returns the minimum number of inputs in the selective pruning selection algorithm.

const size_t& SelectivePruning::get_minimum_inputs_number(void) const
{
    return(minimum_inputs_number);
}

// const size_t& get_maximum_selection_failures(void) const method

/// Returns the maximum number of selection failures in the selective pruning algorithm.

const size_t& SelectivePruning::get_maximum_selection_failures(void) const
{
    return(maximum_selection_failures);
}

// void set_default(void) method

/// Sets the members of the selective pruning object to their default values.

void SelectivePruning::set_default(void)
{
    size_t inputs_number ;

    if(training_strategy_pointer == NULL
            || !training_strategy_pointer->has_loss_index())
    {
        maximum_selection_failures = 3;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer()->get_inputs_number();
        maximum_selection_failures = (size_t)std::max(3.,inputs_number/5.);
    }

    minimum_inputs_number = 1;
}

// void set_minimum_inputs_number(const size_t&) method

/// Sets the minimum inputs for the selective pruning algorithm.
/// @param new_minimum_inputs_number Minimum number of inputs in the selective pruning algorithm.

void SelectivePruning::set_minimum_inputs_number(const size_t& new_minimum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_inputs_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: SelectivePruning class.\n"
               << "void set_minimum_inputs_number(const size_t&) method.\n"
               << "Minimum inputs number must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}

// void set_maximum_selection_failures(const size_t&) method

/// Sets the maximum selection failures for the selective pruning algorithm.
/// @param new_maximum_loss_failures Maximum number of selection failures in the selective pruning algorithm.

void SelectivePruning::set_maximum_selection_failures(const size_t& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: SelectivePruning class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}

// SelectivePruningResults* perform_inputs_selection(void) method

/// Perform the inputs selection with the selective pruning method.

SelectivePruning::SelectivePruningResults* SelectivePruning::perform_inputs_selection(void)
{

#ifdef __OPENNN_DEBUG__

    check();

#endif

    SelectivePruningResults* results = new SelectivePruningResults();

    size_t original_index;

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Variables* variables = data_set_pointer->get_variables_pointer();

    const size_t inputs_number = variables->count_inputs_number();

    Vector< Statistics<double> > original_statistics;
    ScalingLayer::ScalingMethod original_scaling_method;

    bool has_scaling_layer = neural_network_pointer->has_scaling_layer();

    if(has_scaling_layer)
    {
        original_statistics = neural_network_pointer->get_scaling_layer_pointer()->get_statistics();
        original_scaling_method = neural_network_pointer->get_scaling_layer_pointer()->get_scaling_method();
    }

    Vector<bool> current_inputs(inputs_number, true);

    size_t current_inputs_number = inputs_number;

    Vector<Variables::Use> current_uses = variables->arrange_uses();
    const Vector<Variables::Use> original_uses = current_uses;

    Vector<double> current_parameters;

    double optimum_selection_error = 1e10;
    double optimum_loss_error;
    Vector<bool> optimal_inputs;
    Vector<double> optimal_parameters;

    Vector<double> final(2);
    Vector<double> history_row;

    double current_training_loss, current_selection_loss;
    size_t best_ratio_index;
    double best_error;

    bool end = false;
    size_t iterations = 0;
    size_t index = 0;

    time_t beginning_time, current_time;
    double elapsed_time;

    error_ratios.set(inputs_number, 0.0);

    if(display)
    {
        std::cout << "Performing selective pruning selection..." << std::endl;
        std::cout << "Performing initial training..." << std::endl;
    }

    final = perform_model_evaluation(current_inputs);

    current_training_loss = final[0];
    current_selection_loss = final[1];

    results->inputs_data.push_back(current_inputs);

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
        history_row = neural_network_pointer->arrange_parameters();
        results->parameters_data.push_back(history_row);
    }

    if(display)
    {
        std::cout << "Initial values: " << std::endl;

        std::cout << "Current inputs: " << variables->arrange_inputs_name().to_string() << std::endl;
        std::cout << "Number of inputs: " << current_inputs.count_occurrences(true) << std::endl;
        std::cout << "Training loss: " << current_training_loss << std::endl;
        std::cout << "Selection loss: " << current_selection_loss << std::endl;

        std::cout << std::endl;
    }

    time(&beginning_time);

    while(!end)
    {
        current_inputs_number = current_inputs.count_occurrences(true);

        current_parameters = neural_network_pointer->arrange_parameters();

        index = 0;

        while(error_ratios[index] == 1e20)
        {
            index++;
        }

        for (size_t i = 0; i < current_inputs_number; i++)
        {
            if (error_ratios[i+index] == 1e20)
            {
                index++;
            }

            neural_network_pointer->prune_input(i);

            error_ratios[i+index] = loss_index_pointer->calculate_selection_error();

            neural_network_pointer->grow_input();

            neural_network_pointer->set_parameters(current_parameters);
        }

        best_ratio_index = error_ratios.calculate_minimal_index();

        best_error = error_ratios[best_ratio_index];

        index = best_ratio_index;

        if (error_ratios[best_ratio_index] < current_selection_loss)
        {
            original_index = get_input_index(original_uses, best_ratio_index);

            current_uses[original_index] = Variables::Unused;

            current_inputs[best_ratio_index] = false;

            variables->set_uses(current_uses);

            for(size_t i = 0; i < best_ratio_index; i++)
            {
                if(error_ratios[i] == 1e20)
                {
                    index--;
                }
            }

            neural_network_pointer->prune_input(index);
        }

        current_training_loss = loss_index_pointer->calculate_loss();
        current_selection_loss = loss_index_pointer->calculate_selection_error();
        error_ratios[best_ratio_index] = 1e20;

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        iterations++;

        results->inputs_data.push_back(current_inputs);

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
            history_row = get_parameters_inputs(current_inputs);
            results->parameters_data.push_back(history_row);
        }

        // STOPPING CRITERIA

        if(!end && elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum time reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::MaximumTime;
        }
        else if(final[1] < selection_loss_goal)
        {
            end = true;

            if(display)
            {
                std::cout << "Selection loss reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::SelectionLossGoal;
        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum number of iterations reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::MaximumIterations;
        }
        else if(current_inputs.count_occurrences(true) <= minimum_inputs_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Minimum inputs (" << minimum_inputs_number << ") reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::MinimumInputs;
        }
        else if(current_inputs.count_occurrences(true) == 1 || best_error >= current_selection_loss)
        {
            end = true;

            if(display)
            {
                std::cout << "Algorithm finished" << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::AlgorithmFinished;
        }

        if(display)
        {
            std::cout << "Iteration: " << iterations << std::endl;

            if(current_inputs_number > current_inputs.count_occurrences(true))
                std::cout << "Remove input: " << variables->arrange_names()[original_index] << std::endl;

            std::cout << "Current inputs: " << variables->arrange_inputs_name().to_string() << std::endl;
            std::cout << "Number of inputs: " << current_inputs.count_occurrences(true) << std::endl;
            std::cout << "Training loss: " << current_training_loss << std::endl;
            std::cout << "Selection loss: " << current_selection_loss << std::endl;
            std::cout << "Elapsed time: " << elapsed_time << std::endl;

            std::cout << std::endl;
        }
    }

    optimal_inputs = current_inputs;

    optimal_parameters = neural_network_pointer->arrange_parameters();

    optimum_loss_error = loss_index_pointer->calculate_loss();
    optimum_selection_error = loss_index_pointer->calculate_selection_error();

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }

    results->optimal_inputs = optimal_inputs;
    results->final_selection_loss = current_selection_loss;
    results->final_loss = current_training_loss;
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    Vector< Statistics<double> > statistics;

    for (size_t i = 0; i < optimal_inputs.size(); i++)
    {
        original_index = get_input_index(original_uses, i);

        if(optimal_inputs[i] == 1)
        {
            current_uses[original_index] = Variables::Input;
            if(has_scaling_layer)
            {
                statistics.push_back(original_statistics[i]);
            }
        }
        else
        {
            current_uses[original_index] = Variables::Unused;
        }
    }
    variables->set_uses(current_uses);

    set_neural_inputs(optimal_inputs);
    neural_network_pointer->set_parameters(optimal_parameters);
    neural_network_pointer->get_inputs_pointer()->set_names(variables->arrange_inputs_name());

    if(has_scaling_layer)
    {
        ScalingLayer scaling_layer(statistics);
        scaling_layer.set_scaling_method(original_scaling_method);
        neural_network_pointer->set_scaling_layer(scaling_layer);
    }

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        std::cout << "Optimal inputs: " << neural_network_pointer->get_inputs_pointer()->arrange_names().to_string() << std::endl
                  << "Optimal number of inputs: " << optimal_inputs.count_occurrences(true) << std::endl
                  << "Optimum training loss: " << optimum_loss_error << std::endl
                  << "Optimum selection loss: " << optimum_selection_error << std::endl
                  << "Elapsed time: " << elapsed_time << std::endl;
    }

    return results;
}


// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> SelectivePruning::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

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
   buffer << selection_loss_goal;

   values.push_back(buffer.str());

   // Maximum selection failures

   labels.push_back("Maximum selection failures");

   buffer.str("");
   buffer << maximum_selection_failures;

   values.push_back(buffer.str());

   // Minimum inputs number

   labels.push_back("Minimum inputs number");

   buffer.str("");
   buffer << minimum_inputs_number;

   values.push_back(buffer.str());

   // Minimum correlation

   labels.push_back("Minimum correlation");

   buffer.str("");
   buffer << minimum_correlation;

   values.push_back(buffer.str());

   // Maximum correlation

   labels.push_back("Maximum correlation");

   buffer.str("");
   buffer << maximum_correlation;

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

   // Plot training loss history

   labels.push_back("Plot training loss history");

   buffer.str("");
   buffer << reserve_loss_data;

   values.push_back(buffer.str());

   // Plot selection loss history

   labels.push_back("Plot selection loss history");

   buffer.str("");
   buffer << reserve_selection_loss_data;

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}

// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the selective pruning parameters, the stopping criteria
/// and other user stuff concerning the selective pruning object.

tinyxml2::XMLDocument* SelectivePruning::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("SelectivePruning");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Regression
//   {
//   element = document->NewElement("Approximation");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << approximation;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Trials number
   {
   element = document->NewElement("TrialsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << trials_number;

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

   // selection loss goal
   {
   element = document->NewElement("SelectionLossGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << selection_loss_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum selection failures
   {
   element = document->NewElement("MaximumSelectionFailures");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_failures;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Minimum inputs number
   {
   element = document->NewElement("MinimumInputsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_inputs_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Minimum correlation
   {
   element = document->NewElement("MinimumCorrelation");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_correlation;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum correlation
   {
   element = document->NewElement("MaximumCorrelation");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_correlation;

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

// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this selective pruning object.
/// @param document TinyXML document containing the member data.

void SelectivePruning::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SelectivePruning");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: SelectivePruning class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "SelectivePruning element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Regression
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Approximation");

        if(element)
        {
            const std::string new_approximation = element->GetText();

            try
            {
               set_approximation(new_approximation != "0");
            }
            catch(const std::logic_error& e)
            {
               std::cout << e.what() << std::endl;
            }
        }
    }

    // Trials number
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

    // Maximum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumCorrelation");

        if(element)
        {
           const double new_maximum_correlation = atof(element->GetText());

           try
           {
              set_maximum_correlation(new_maximum_correlation);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
    }

    // Minimum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumCorrelation");

        if(element)
        {
           const double new_minimum_correlation = atof(element->GetText());

           try
           {
              set_minimum_correlation(new_minimum_correlation);
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

    // Minimum inputs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumInputsNumber");

        if(element)
        {
           const size_t new_minimum_inputs_number = atoi(element->GetText());

           try
           {
              set_minimum_inputs_number(new_minimum_inputs_number);
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

/// Saves to a XML-type file the members of the selective pruning object.
/// @param file_name Name of selective pruning XML-type file.

void SelectivePruning::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a selective pruning object from a XML-type file.
/// @param file_name Name of selective pruning XML-type file.

void SelectivePruning::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: SelectivePruning class.\n"
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

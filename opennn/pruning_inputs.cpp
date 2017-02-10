/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R U N I N G   I N P U T S   C L A S S                                                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "pruning_inputs.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

PruningInputs::PruningInputs(void)
    : InputsSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

PruningInputs::PruningInputs(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML pruning inputs file.

PruningInputs::PruningInputs(const std::string& file_name)
    : InputsSelectionAlgorithm(file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/// @param pruning_inputs_document Pointer to a TinyXML document containing the pruning inputs data.

PruningInputs::PruningInputs(const tinyxml2::XMLDocument& pruning_inputs_document)
    : InputsSelectionAlgorithm(pruning_inputs_document)
{
    from_XML(pruning_inputs_document);
}


// DESTRUCTOR

/// Destructor.

PruningInputs::~PruningInputs(void)
{
}

// METHODS

// const size_t& get_minimum_inputs_number(void) const method

/// Returns the minimum number of inputs in the pruning inputs selection algorithm.

const size_t& PruningInputs::get_minimum_inputs_number(void) const
{
    return(minimum_inputs_number);
}

// const size_t& get_maximum_selection_failures(void) const method

/// Returns the maximum number of selection failures in the pruning inputs algorithm.

const size_t& PruningInputs::get_maximum_selection_failures(void) const
{
    return(maximum_selection_failures);
}

// void set_default(void) method

/// Sets the members of the pruning inputs object to their default values.

void PruningInputs::set_default(void)
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

/// Sets the minimum inputs for the pruning inputs algorithm.
/// @param new_minimum_inputs_number Minimum number of inputs in the pruning inputs algorithm.

void PruningInputs::set_minimum_inputs_number(const size_t& new_minimum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_inputs_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_minimum_inputs_number(const size_t&) method.\n"
               << "Minimum inputs number must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}

// void set_maximum_selection_failures(const size_t&) method

/// Sets the maximum selection failures for the pruning inputs algorithm.
/// @param new_maximum_loss_failures Maximum number of selection failures in the pruning inputs algorithm.

void PruningInputs::set_maximum_selection_failures(const size_t& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}

// PruningInputsResults* perform_inputs_selection(void) method

/// Perform the inputs selection with the pruning inputs method.

PruningInputs::PruningInputsResults* PruningInputs::perform_inputs_selection(void)
{

#ifdef __OPENNN_DEBUG__

    check();

#endif

    PruningInputsResults* results = new PruningInputsResults();

    size_t index;

    size_t original_index;

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Variables* variables = data_set_pointer->get_variables_pointer();

    const size_t inputs_number = variables->count_inputs_number();

    const size_t targets_number = variables->count_targets_number();

    Vector< Statistics<double> > original_statistics;
    ScalingLayer::ScalingMethod original_scaling_method;

    bool has_scaling_layer = neural_network_pointer->has_scaling_layer();

    if(has_scaling_layer)
    {
        original_statistics = neural_network_pointer->get_scaling_layer_pointer()->get_statistics();
        original_scaling_method = neural_network_pointer->get_scaling_layer_pointer()->get_scaling_method();
    }

    Vector<bool> current_inputs(inputs_number, true);

    Vector<Variables::Use> current_uses = variables->arrange_uses();
    const Vector<Variables::Use> original_uses = current_uses;

    double optimum_selection_error = 1e10;
    double optimum_loss_error;
    Vector<bool> optimal_inputs;
    Vector<double> optimal_parameters;

    Vector<double> final_correlations;

    Vector<double> final(2);
    Vector<double> history_row;

    double current_training_loss, current_selection_loss;
    double previous_selection_loss;

    bool end = false;
    size_t iterations = 0;
    size_t selection_failures = 0;

    time_t beginning_time, current_time;
    double elapsed_time;

    if(display)
    {
        std::cout << "Performing pruning inputs selection..." << std::endl;
        std::cout << std::endl << "Calculating correlations..." << std::endl;
    }

    final_correlations = calculate_final_correlations();

    if(display)
    {
        std::cout << "Correlations : \n";

        for(size_t i = 0; i < final_correlations.size(); i++)
        {
            original_index = get_input_index(original_uses, i);

            std::cout << "Input: " << variables->arrange_names()[original_index] << "; Correlation: " << final_correlations[i] << std::endl;

            if(i == 9)
            {
                std::cout << "...\n";
                break;
            }
        }
    }

    set_neural_inputs(current_inputs);

    current_inputs.set(inputs_number, true);

    time(&beginning_time);

    for(size_t i = 0; i < final_correlations.size(); i++)
    {
        if(final_correlations[i] <= minimum_correlation*targets_number &&
           current_inputs.count_occurrences(true) > 1)
        {
            final_correlations[i] = 1e20;

            original_index = get_input_index(original_uses, i);

            current_uses[original_index] = Variables::Unused;

            current_inputs[i] = false;

            variables->set_uses(current_uses);

            if(display)
            {
                std::cout << "Remove input "<< variables->arrange_names()[original_index] << std::endl;
            }

            if(current_inputs.count_occurrences(true) <= minimum_inputs_number)
            {
                end = true;

                if(display)
                {
                    std::cout << "Minimum inputs ("<< minimum_inputs_number <<") reached." << std::endl;
                }

                results->stopping_condition = InputsSelectionAlgorithm::MinimumInputs;

                final = perform_model_evaluation(current_inputs);

                optimal_inputs = current_inputs;
                optimum_loss_error = final[0];
                optimum_selection_error = final[1];

                results->inputs_data.push_back(current_inputs);

                if(reserve_loss_data)
                {
                    results->loss_data.push_back(optimum_loss_error);
                }

                if(reserve_selection_loss_data)
                {
                    results->selection_loss_data.push_back(optimum_selection_error);
                }

                if(reserve_parameters_data)
                {
                    history_row = get_parameters_inputs(current_inputs);

                    results->parameters_data.push_back(history_row);
                }

                break;
            }

        }
    }

    if(display)
    {
        std::cout << std::endl;
    }

    while(!end)
    {
        index = final_correlations.calculate_minimal_index();

        if(iterations != 0 && final_correlations[index] >= maximum_correlation*targets_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Correlation goal reached." << std::endl << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::CorrelationGoal;
        }
        else if(iterations != 0)
        {

            final_correlations[index] = 1e20;

            original_index = get_input_index(original_uses, index);

            current_uses[original_index] = Variables::Unused;

            current_inputs[index] = false;

            variables->set_uses(current_uses);
        }

        if(!end)
        {
            final = perform_model_evaluation(current_inputs);

            current_training_loss = final[0];
            current_selection_loss = final[1];

            if(fabs(optimum_selection_error - current_selection_loss) > tolerance &&
                    optimum_selection_error > current_selection_loss)
            {
                optimal_inputs = current_inputs;
                optimum_loss_error = current_training_loss;
                optimum_selection_error = current_selection_loss;
            }
            else if(previous_selection_loss < current_selection_loss)
            {
                selection_failures++;
            }

            time(&current_time);
            elapsed_time = difftime(current_time, beginning_time);

            previous_selection_loss = current_selection_loss;
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
            else if(selection_failures >= maximum_selection_failures)
            {
                end = true;

                if(display)
                {
                    std::cout << "Maximum selection failures (" << selection_failures << ") reached." << std::endl;
                }

                results->stopping_condition = InputsSelectionAlgorithm::MaximumSelectionFailures;
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
            else if(current_inputs.count_occurrences(true) == 1)
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

                if(iterations != 1)
                {
                    std::cout << "Remove input: " << variables->arrange_names()[original_index] << std::endl;
                }

                std::cout << "Current inputs: " << variables->arrange_inputs_name().to_string() << std::endl;
                std::cout << "Number of inputs: " << current_inputs.count_occurrences(true) << std::endl;
                std::cout << "Training loss: " << final[0] << std::endl;
                std::cout << "Selection loss: " << final[1] << std::endl;
                std::cout << "Elapsed time: " << elapsed_time << std::endl;

                std::cout << std::endl;
            }
        }
    }

    optimal_parameters = get_parameters_inputs(optimal_inputs);
    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }


    results->optimal_inputs = optimal_inputs;
    results->final_selection_loss = optimum_selection_error;
    results->final_loss = perform_model_evaluation(optimal_inputs)[0];
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

    if(neural_network_pointer->has_inputs())
    {
        neural_network_pointer->get_inputs_pointer()->set_names(variables->arrange_inputs_name());
    }

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
        if(neural_network_pointer->has_inputs())
        {
            std::cout << "Optimal inputs: " << neural_network_pointer->get_inputs_pointer()->arrange_names().to_string() << std::endl;
        }

        std::cout << "Optimal number of inputs: " << optimal_inputs.count_occurrences(true) << std::endl;
        std::cout << "Optimum training loss: " << optimum_loss_error << std::endl;
        std::cout << "Optimum selection loss: " << optimum_selection_error << std::endl;
        std::cout << "Elapsed time: " << elapsed_time << std::endl;;
    }

    return results;
}


// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> PruningInputs::to_string_matrix(void) const
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

   if(reserve_loss_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot selection loss history

   labels.push_back("Plot selection loss history");

   buffer.str("");

   if(reserve_selection_loss_data)
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

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}

// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the pruning inputs parameters, the stopping criteria
/// and other user stuff concerning the pruning inputs object.

tinyxml2::XMLDocument* PruningInputs::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("PruningInputs");

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


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the pruning inputs object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PruningInputs::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("PruningInputs");

    // Trials number

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

    // selection loss goal

    file_stream.OpenElement("SelectionLossGoal");

    buffer.str("");
    buffer << selection_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection failures

    file_stream.OpenElement("MaximumSelectionFailures");

    buffer.str("");
    buffer << maximum_selection_failures;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum inputs number

    file_stream.OpenElement("MinimumInputsNumber");

    buffer.str("");
    buffer << minimum_inputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum correlation

    file_stream.OpenElement("MinimumCorrelation");

    buffer.str("");
    buffer << minimum_correlation;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum correlation

    file_stream.OpenElement("MaximumCorrelation");

    buffer.str("");
    buffer << maximum_correlation;

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


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this pruning inputs object.
/// @param document TinyXML document containing the member data.

void PruningInputs::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("PruningInputs");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PruningInputs element is NULL.\n";

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

/// Saves to a XML-type file the members of the pruning inputs object.
/// @param file_name Name of pruning inputs XML-type file.

void PruningInputs::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a pruning inputs object from a XML-type file.
/// @param file_name Name of pruning inputs XML-type file.

void PruningInputs::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PruningInputs class.\n"
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

/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R O W I N G   I N P U T S   C L A S S                                                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "growing_inputs.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

GrowingInputs::GrowingInputs()
    : InputsSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

GrowingInputs::GrowingInputs(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML genetic algorithm file.

GrowingInputs::GrowingInputs(const string& file_name)
    : InputsSelectionAlgorithm(file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/// @param genetic_algorithm_document Pointer to a TinyXML document containing the genetic algorithm data.

GrowingInputs::GrowingInputs(const tinyxml2::XMLDocument& genetic_algorithm_document)
    : InputsSelectionAlgorithm(genetic_algorithm_document)
{
    from_XML(genetic_algorithm_document);
}


// DESTRUCTOR

/// Destructor.

GrowingInputs::~GrowingInputs()
{
}

// METHODS

// const size_t& get_maximum_inputs_number() const method

/// Returns the maximum number of inputs in the growing inputs selection algorithm.

const size_t& GrowingInputs::get_maximum_inputs_number() const
{
    return(maximum_inputs_number);
}

// const size_t& get_minimum_inputs_number() const method

/// Returns the minimum number of inputs in the growing inputs selection algorithm.

const size_t& GrowingInputs::get_minimum_inputs_number() const
{
    return(minimum_inputs_number);
}

// const size_t& get_maximum_selection_failures() const method

/// Returns the maximum number of selection failures in the growing inputs selection algorithm.

const size_t& GrowingInputs::get_maximum_selection_failures() const
{
    return(maximum_selection_failures);
}

// void set_default() method

/// Sets the members of the growing inputs object to their default values.

void GrowingInputs::set_default()
{
    size_t inputs_number ;

    if(training_strategy_pointer == NULL || !training_strategy_pointer->has_loss_index())
    {
        maximum_selection_failures = 3;

        maximum_inputs_number = 100;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer()->get_inputs_number();
        maximum_selection_failures =(size_t)max(3.,inputs_number/5.);

        maximum_inputs_number = inputs_number;
    }

    minimum_inputs_number = 1;
}

// void set_maximum_inputs_number(const size_t&) method

/// Sets the maximum inputs number for the growing inputs selection algorithm.
/// @param new_maximum_inputs_number Maximum inputs number in the growing inputs selection algorithm.

void GrowingInputs::set_maximum_inputs_number(const size_t& new_maximum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_inputs_number <= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_inputs_number = new_maximum_inputs_number;
}

// void set_minimum_inputs_number(const size_t&) method

/// Sets the minimum inputs number for the growing inputs selection algorithm.
/// @param new_minimum_inputs_number Minimum inputs number in the growing inputs selection algorithm.

void GrowingInputs::set_minimum_inputs_number(const size_t& new_minimum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_inputs_number <= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_maximum_inputs_number(const size_t&) method.\n"
               << "Minimum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}

// void set_maximum_selection_failures(const size_t&) method

/// Sets the maximum selection failures for the growing inputs selection algorithm.
/// @param new_maximum_loss_failures Maximum number of selection failures in the growing inputs selection algorithm.

void GrowingInputs::set_maximum_selection_failures(const size_t& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}

// GrowingInputsResults* perform_inputs_selection() method

/// Perform the inputs selection with the growing inputs method.

GrowingInputs::GrowingInputsResults* GrowingInputs::perform_inputs_selection()
{

#ifdef __OPENNN_DEBUG__

    check();

#endif

    GrowingInputsResults* results = new GrowingInputsResults();

    size_t index;

    size_t original_index;

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Variables* variables_pointer = data_set_pointer->get_variables_pointer();

    const size_t inputs_number = variables_pointer->count_inputs_number();

    const size_t targets_number = variables_pointer->count_targets_number();

    Vector<bool> inputs_selection(inputs_number, true);

    Vector<Variables::Use> current_uses = variables_pointer->arrange_uses();

    const Vector<Variables::Use> original_uses = current_uses;

    double optimum_selection_error = 1.0e10;
    double optimum_training_error = 1.0e10;

    Vector<bool> optimal_inputs;
    Vector<double> optimal_parameters;

    Vector<double> final(2);
    Vector<double> history_row;

    double current_training_loss, current_selection_loss;
    double previous_selection_loss = -1;

    bool flag_input = false;

    bool end = false;

    size_t iteration = 0;

    size_t selection_failures = 0;

    time_t beginning_time, current_time;

    double elapsed_time;

    if(display) cout << "Performing growing inputs selection..." << endl;

    Vector<double> total_input_correlations = data_set_pointer->calculate_total_input_correlations();

    if(display)
    {
        cout << "Top correlations:" << endl;

        data_set_pointer->print_top_input_target_correlations(10);
    }

    current_uses = current_uses.replace_value(Variables::Input, Variables::Unused);

    inputs_selection.set(inputs_number, false);

    time(&beginning_time);

    index = total_input_correlations.calculate_maximal_index();

    if(total_input_correlations[index] >= 0.9999*targets_number)
    {
        original_index = get_input_index(original_uses, index);

        current_uses[original_index] = Variables::Input;

        inputs_selection[index] = true;

        variables_pointer->set_uses(current_uses);

        if(display)
        {
            cout << "Maximal correlation(" << total_input_correlations[index] << ") is nearly 1. \n"
                 << "The problem is linear separable with the input " << variables_pointer->arrange_names()[original_index] << ".\n\n" ;
        }

        final = perform_model_evaluation(inputs_selection);

        optimal_inputs = inputs_selection;
        optimum_selection_error = final[0];
        optimum_training_error = final[1];
        optimal_parameters = get_parameters_inputs(inputs_selection);

        results->inputs_data.push_back(inputs_selection);

        if(reserve_loss_data) results->loss_data.push_back(optimum_selection_error);

        if(reserve_selection_loss_data) results->selection_loss_data.push_back(optimum_training_error);

        if(reserve_parameters_data) results->parameters_data.push_back(optimal_parameters);

        results->stopping_condition = InputsSelectionAlgorithm::AlgorithmFinished;

        end = true;
    }
    else
    {
        for(size_t i = 0; i < minimum_inputs_number; i++)
        {
            index = total_input_correlations.calculate_maximal_index();

            total_input_correlations[index] = -1;

            original_index = get_input_index(original_uses, index);

            current_uses[original_index] = Variables::Input;

            inputs_selection[index] = true;

            variables_pointer->set_uses(current_uses);

            flag_input = true;
        }

        for(size_t i = 0; i < total_input_correlations.size(); i++)
        {
            if(total_input_correlations[i] >= maximum_correlation*targets_number)
            {
                total_input_correlations[i] = -1;

                original_index = get_input_index(original_uses, i);

                current_uses[original_index] = Variables::Input;

                inputs_selection[i] = true;

                variables_pointer->set_uses(current_uses);

                if(display)
                {
                    cout << "Added input "<< variables_pointer->arrange_names()[original_index] << endl;
                }

                if(inputs_selection.count_equal_to(true) >= maximum_inputs_number)
                {
                    end = true;

                    if(display)
                    {
                        cout << "Maximum inputs("<< maximum_inputs_number <<") reached." << endl;
                    }

                    results->stopping_condition = InputsSelectionAlgorithm::MaximumInputs;

                    final = perform_model_evaluation(inputs_selection);

                    optimal_inputs = inputs_selection;
                    optimum_training_error = final[0];
                    optimum_selection_error = final[1];

                    results->inputs_data.push_back(inputs_selection);

                    if(reserve_loss_data)
                    {
                        results->loss_data.push_back(optimum_training_error);
                    }

                    if(reserve_selection_loss_data)
                    {
                        results->selection_loss_data.push_back(optimum_selection_error);
                    }

                    if(reserve_parameters_data)
                    {
                        history_row = get_parameters_inputs(inputs_selection);

                        results->parameters_data.push_back(history_row);
                    }

                    break;
                }

                flag_input = true;
            }
        }
    }

    while(!end)
    {
        if(display) cout << "Iteration " << iteration << endl;

        if(!flag_input)
        {
            index = total_input_correlations.calculate_maximal_index();

            if(total_input_correlations[index] <= minimum_correlation*targets_number)
            {
                end = true;

                if(display) cout << "Correlation goal reached." << endl << endl;

                results->stopping_condition = InputsSelectionAlgorithm::CorrelationGoal;

                if(inputs_selection.count_equal_to(true) == 0)
                {
                    original_index = get_input_index(original_uses, index);

                    current_uses[original_index] = Variables::Input;

                    inputs_selection[index] = true;

                    variables_pointer->set_uses(current_uses);
                }

                optimal_inputs = inputs_selection;
            }
            else
            {
                total_input_correlations[index] = -1;

                original_index = get_input_index(original_uses, index);

                current_uses[original_index] = Variables::Input;

                inputs_selection[index] = true;

                variables_pointer->set_uses(current_uses);
            }
        }

        final = perform_model_evaluation(inputs_selection);

        current_training_loss = final[0];
        current_selection_loss = final[1];

        if(fabs(optimum_selection_error - current_selection_loss) < tolerance ||
                optimum_selection_error > current_selection_loss)
        {
            optimal_inputs = inputs_selection;
            optimum_training_error = current_training_loss;
            optimum_selection_error = current_selection_loss;
        }
        else if(optimum_selection_error < current_selection_loss)
        {
            selection_failures++;
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        previous_selection_loss = current_selection_loss;
        iteration++;

        results->inputs_data.push_back(inputs_selection);

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
            history_row = get_parameters_inputs(inputs_selection);

            results->parameters_data.push_back(history_row);
        }

        // STOPPING CRITERIA

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display) cout << "Maximum time reached." << endl;

            results->stopping_condition = InputsSelectionAlgorithm::MaximumTime;
        }
        else if(final[1] <= selection_loss_goal)
        {
            end = true;

            if(display) cout << "Selection loss reached." << endl;

            results->stopping_condition = InputsSelectionAlgorithm::SelectionLossGoal;
        }
        else if(iteration >= maximum_iterations_number)
        {
            end = true;

            if(display) cout << "Maximum number of iterations reached." << endl;

            results->stopping_condition = InputsSelectionAlgorithm::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display) cout << "Maximum selection failures("<<selection_failures<<") reached." << endl;

            results->stopping_condition = InputsSelectionAlgorithm::MaximumSelectionFailures;
        }
        else if(inputs_selection.count_equal_to(true) >= maximum_inputs_number)
        {
            end = true;

            if(display) cout << "Maximum inputs("<< maximum_inputs_number <<") reached." << endl;

            results->stopping_condition = InputsSelectionAlgorithm::MaximumInputs;
        }
        else if(inputs_selection.count_equal_to(true) == inputs_selection.size())
        {
            end = true;

            if(display) cout << "Algorithm finished" << endl;

            results->stopping_condition = InputsSelectionAlgorithm::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iteration << endl;

            if(!flag_input) cout << "Add input: " << variables_pointer->arrange_names()[original_index] << endl;

            cout << "Current inputs: " <<  variables_pointer->arrange_inputs_name().vector_to_string() << endl;
            cout << "Number of inputs: " << inputs_selection.count_equal_to(true) << endl;
            cout << "Training loss: " << final[0] << endl;
            cout << "Selection loss: " << final[1] << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            cout << endl;
        }

        flag_input = false;
    }

    optimal_parameters = get_parameters_inputs(optimal_inputs);

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }

    results->optimal_inputs = optimal_inputs;
    results->final_selection_loss = optimum_selection_error;
    results->final_loss = optimum_training_error;
    results->iterations_number = iteration;
    results->elapsed_time = elapsed_time;

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        if(neural_network_pointer->has_inputs())
        {
            cout << "Optimal inputs: " << variables_pointer->arrange_inputs_name().vector_to_string() << endl;
        }

        cout << "Optimal number of inputs: " << inputs_selection.count_equal_to(true) << endl;
        cout << "Optimum training loss: " << optimum_training_error << endl;
        cout << "Optimum selection loss: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    // Set data set stuff

    variables_pointer->set_uses(current_uses);

    neural_network_pointer->set_inputs(optimal_inputs);

    neural_network_pointer->set_parameters(optimal_parameters);

    results->inputs_selection = inputs_selection;

    return results;
}


// Matrix<string> to_string_matrix() const method

/// Writes as matrix of strings the most representative atributes.

Matrix<string> GrowingInputs::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

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

   // Maximum inputs number

   labels.push_back("Maximum inputs number");

   buffer.str("");
   buffer << maximum_inputs_number;

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

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return(string_matrix);
}

// tinyxml2::XMLDocument* to_XML() const method

/// Prints to the screen the growing inputs parameters, the stopping criteria
/// and other user stuff concerning the growing inputs object.

tinyxml2::XMLDocument* GrowingInputs::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Order Selection algorithm

    tinyxml2::XMLElement* root_element = document->NewElement("GrowingInputs");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = NULL;
    tinyxml2::XMLText* text = NULL;

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

    // Maximum inputs number
    {
        element = document->NewElement("MaximumInputsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_inputs_number;

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

    // Display
//    {
//        element = document->NewElement("Display");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << display;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the growing inputs object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GrowingInputs::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("GrowingInputs");

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

    // Maximum inputs number

    file_stream.OpenElement("MaximumInputsNumber");

    buffer.str("");
    buffer << maximum_inputs_number;

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

/// Deserializes a TinyXML document into this growing inputs object.
/// @param document TinyXML document containing the member data.

void GrowingInputs::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GrowingInputs");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GrowingInputs element is NULL.\n";

        throw logic_error(buffer.str());
    }

    // Regression
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Approximation");

        if(element)
        {
            const string new_regression = element->GetText();

            try
            {
                set_approximation(new_regression != "0");
            }
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
            }
        }
    }

    // Performance calculation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PerformanceCalculationMethod");

        if(element)
        {
            const string new_loss_calculation_method = element->GetText();

            try
            {
                set_loss_calculation_method(new_loss_calculation_method);
            }
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
                cout << e.what() << endl;
            }
        }
    }

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

        if(element)
        {
            const string new_reserve_loss_data = element->GetText();

            try
            {
                set_reserve_loss_data(new_reserve_loss_data != "0");
            }
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
            }
        }
    }

    // Reserve selection loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

        if(element)
        {
            const string new_reserve_selection_loss_data = element->GetText();

            try
            {
                set_reserve_selection_loss_data(new_reserve_selection_loss_data != "0");
            }
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
                cout << e.what() << endl;
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
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
                cout << e.what() << endl;
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
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
            }
        }
    }

    // Maximum inputs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumInputsNumber");

        if(element)
        {
            const size_t new_maximum_inputs_number = atoi(element->GetText());

            try
            {
                set_maximum_inputs_number(new_maximum_inputs_number);
            }
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
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
            catch(const logic_error& e)
            {
                cout << e.what() << endl;
            }
        }
    }

}

// void save(const string&) const method

/// Saves to a XML-type file the members of the growing inputs object.
/// @param file_name Name of growing inputs XML-type file.

void GrowingInputs::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


// void load(const string&) method

/// Loads a growing inputs object from a XML-type file.
/// @param file_name Name of growing inputs XML-type file.

void GrowingInputs::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
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

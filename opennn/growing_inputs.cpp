//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "growing_inputs.h"

namespace OpenNN {

/// Default constructor.

GrowingInputs::GrowingInputs()
    : InputsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

GrowingInputs::GrowingInputs(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// File constructor.
/// @param file_name Name of XML genetic algorithm file.

GrowingInputs::GrowingInputs(const string& file_name)
    : InputsSelection(file_name)
{
    load(file_name);
}


/// XML constructor.
/// @param genetic_algorithm_document Pointer to a TinyXML document containing the genetic algorithm data.

GrowingInputs::GrowingInputs(const tinyxml2::XMLDocument& genetic_algorithm_document)
    : InputsSelection(genetic_algorithm_document)
{
    from_XML(genetic_algorithm_document);
}


/// Destructor.

GrowingInputs::~GrowingInputs()
{
}


/// Returns the maximum number of inputs in the growing inputs selection algorithm.

const size_t& GrowingInputs::get_maximum_inputs_number() const
{
    return(maximum_inputs_number);
}


/// Returns the minimum number of inputs in the growing inputs selection algorithm.

const size_t& GrowingInputs::get_minimum_inputs_number() const
{
    return(minimum_inputs_number);
}


/// Returns the maximum number of selection failures in the growing inputs selection algorithm.

const size_t& GrowingInputs::get_maximum_selection_failures() const
{
    return(maximum_selection_failures);
}


/// Sets the members of the growing inputs object to their default values.

void GrowingInputs::set_default()
{
    maximum_selection_failures = 3;

    if(training_strategy_pointer == nullptr || !training_strategy_pointer->has_neural_network())
    {
        maximum_inputs_number = 100;
    }
    else
    {
        training_strategy_pointer->get_neural_network_pointer()->get_display();

        const size_t inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();

        maximum_selection_failures = static_cast<size_t>(max(3.,inputs_number/5.));

        maximum_inputs_number = inputs_number;
    }

    minimum_inputs_number = 1;
}


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


/// Sets the minimum inputs number for the growing inputs selection algorithm.
/// @param new_minimum_inputs_number Minimum inputs number in the growing inputs selection algorithm.

void GrowingInputs::set_minimum_inputs_number(const size_t& new_minimum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_inputs_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_minimum_inputs_number(const size_t&) method.\n"
               << "Minimum inputs number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}


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


/// Perform the inputs selection with the growing inputs method.

GrowingInputs::GrowingInputsResults* GrowingInputs::perform_inputs_selection()
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    GrowingInputsResults* results = new GrowingInputsResults();

    if(display) cout << "Performing growing inputs selection..." << endl;

    // Loss index Stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    double optimum_training_error = numeric_limits<double>::max();
    double optimum_selection_error = numeric_limits<double>::max();

    double previus_selection_error = numeric_limits<double>::max();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const size_t inputs_number = data_set_pointer->get_input_columns_number();

    const size_t used_columns_number = data_set_pointer->get_used_columns_number();

    const Vector<string> used_columns_names = data_set_pointer->get_used_columns_names();

    const Matrix<double> correlations = data_set_pointer->calculate_input_target_columns_correlations_double();

    const Vector<double> total_correlations = absolute_value(correlations.calculate_rows_sum());

    const Vector<size_t> correlations_descending_indices = total_correlations.sort_descending_indices();

    data_set_pointer->set_input_columns_unused();

    // Neural network

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();

    // Optimization algorithm

    double current_training_error = 0.0;
    double current_selection_error = 0.0;

    Vector<double> current_parameters;

    Vector<size_t> current_columns_indices;

    Vector<size_t> optimal_columns_indices;

    Vector<double> optimal_parameters;

    size_t selection_failures = 0;

    time_t beginning_time, current_time;
    double elapsed_time = 0.0;

    time(&beginning_time);

    bool end_algorithm = false;

    // Model selection

    if(used_columns_number < maximum_epochs_number) maximum_epochs_number = used_columns_number;

    for(size_t epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        const size_t column_index = correlations_descending_indices[epoch];

        const string column_name = used_columns_names[column_index];

        data_set_pointer->set_column_use(column_name, DataSet::Input);

        current_columns_indices.push_back(column_index);

        const size_t input_variables_number = data_set_pointer->get_input_variables_number();

        data_set_pointer->set_input_variables_dimensions({input_variables_number});

        neural_network_pointer->set_inputs_number(input_variables_number);

        // Trial

        double optimum_selection_error_trial = numeric_limits<double>::max();
        double optimum_training_error_trial = numeric_limits<double>::max();
        Vector<double> optimum_parameters_trial;

        for(size_t i = 0; i < trials_number; i++)
        {
            OptimizationAlgorithm::Results training_results = training_strategy_pointer->perform_training();

            double current_training_error_trial = training_results.final_training_error;
            double current_selection_error_trial = training_results.final_selection_error;
            Vector<double> current_parameters_trial = training_results.final_parameters;

            if(display)
            {
                cout << endl << "Trial number: " << i << endl;
                cout << "Training error: " << current_training_error_trial << endl;
                cout << "Selection error: " << current_selection_error_trial << endl;
                cout << "Stopping condition: " << training_results.write_stopping_condition() << endl << endl;
            }

            if(current_selection_error_trial < optimum_selection_error_trial)
            {
                optimum_parameters_trial = current_parameters_trial;
                optimum_selection_error_trial = current_selection_error_trial;
                optimum_training_error_trial = current_training_error_trial;
            }
        }

        current_selection_error = optimum_selection_error_trial;
        current_training_error = optimum_training_error_trial;
        current_parameters = optimum_parameters_trial;

        if(current_selection_error < optimum_selection_error)
        {
            optimal_columns_indices = current_columns_indices;
            optimal_parameters = current_parameters;
            optimum_selection_error = current_selection_error;
            optimum_training_error = current_training_error;
        }
        else if (previus_selection_error < current_selection_error)
        {
            selection_failures++;
        }

        previus_selection_error = current_selection_error;

        time(&current_time);

        elapsed_time = difftime(current_time,beginning_time);

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end_algorithm = true;

            if(display) cout << "Maximum time reached." << endl;

            results->stopping_condition = InputsSelection::MaximumTime;
        }
        else if(current_selection_error <= selection_error_goal)
        {
            end_algorithm = true;

            if(display) cout << "Selection loss reached." << endl;

            results->stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(epoch >= maximum_epochs_number)
        {
            end_algorithm = true;

            if(display) cout << "Maximum number of iterations reached." << endl;

            results->stopping_condition = InputsSelection::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end_algorithm = true;

            if(display) cout << "Maximum selection failures("<<selection_failures<<") reached." << endl;

            results->stopping_condition = InputsSelection::MaximumSelectionFailures;
        }
        else if(current_columns_indices.size() > maximum_inputs_number)
        {
            end_algorithm = true;

            if(display) cout << "Maximum inputs("<< maximum_inputs_number <<") reached." << endl;

            results->stopping_condition = InputsSelection::MaximumInputs;
        }
        else if(current_columns_indices.size() == inputs_number)
        {
            end_algorithm = true;

            if(display) cout << "Algorithm finished" << endl;

            results->stopping_condition = InputsSelection::AlgorithmFinished;
        }       

        if(display)
        {
            cout << "Iteration: " << epoch << endl;

            if(end_algorithm == false) cout << "Add input: " << data_set_pointer->get_variable_name(column_index) << endl;

            cout << "Current inputs: " <<  data_set_pointer->get_input_variables_names().vector_to_string() << endl;
            cout << "Number of inputs: " << current_columns_indices.size() << endl;
            cout << "Training loss: " << current_training_error << endl;
            cout << "Selection error: " << current_selection_error << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            cout << endl;
        }

        if(end_algorithm == true) break;
    }

    // Save results

    results->optimal_inputs_indices = optimal_columns_indices;
    results->final_selection_error = optimum_selection_error;
    results->final_training_error = optimum_training_error;
//    results->iterations_number = iteration;
    results->elapsed_time = elapsed_time;
    results->minimal_parameters = optimal_parameters;

    // Set Data set stuff

    data_set_pointer->set_input_columns_unused();

    const size_t optimal_inputs_number = optimal_columns_indices.size();

    for(size_t i = 0; i< optimal_inputs_number; i++)
    {
        size_t optimal_input_index = optimal_columns_indices[i];

        data_set_pointer->set_column_use(optimal_input_index,DataSet::Input);
    }

    data_set_pointer->set_input_variables_dimensions({optimal_inputs_number});

    // Set Neural network stuff

    neural_network_pointer->set_inputs_number(optimal_inputs_number);

    neural_network_pointer->set_parameters(optimal_parameters);

    if(display)
    {
        cout << "Optimal inputs: " << data_set_pointer->get_input_variables_names().vector_to_string() << endl;
        cout << "Optimal number of inputs: " << optimal_columns_indices.size() << endl;
        cout << "Optimum training error: " << optimum_training_error << endl;
        cout << "Optimum selection error: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    return results;


/*
    Vector<DataSet::VariableUse> current_uses(columns_uses);
   Vector<DataSet::VariableUse> optimum_uses = current_uses;


    // Optimization algorithm stuff

    size_t index;

    size_t original_index = 0;

    Vector<bool> inputs_selection(inputs_number, true);

    Vector<bool> optimal_inputs;
    Vector<double> optimal_parameters;

    Vector<double> selection_parameters(2);
    Vector<double> history_row;

    double current_training_error;
    double current_selection_error;

    double previus_selection_error;

    bool flag_input = false;

    bool end = false;

    size_t iterations = 0;

    size_t selection_failures = 0;

    time_t beginning_time, current_time;

    double elapsed_time = 0.0;


   if(display)
    {
        cout << "Top correlations:" << endl;

//        data_set_pointer->print_top_inputs_targets_correlations(10);
    }

    current_uses.replace_value(DataSet::Input, DataSet::UnusedVariable);

    inputs_selection.set(inputs_number, false);

    time(&beginning_time);

    // Main loop


//    index = maximal_index(total_input_correlations);

    if(total_input_correlations[index] >= 0.9999*targets_number)
    {
        original_index = get_input_index(original_uses, index);

        current_uses[original_index] = DataSet::Input;

        inputs_selection[index] = true;

        data_set_pointer->set_columns_uses(current_uses);

        optimum_uses = current_uses;

        if(display)
        {
            cout << "Maximal correlation (" << total_input_correlations[index] << ") is nearly 1. \n"
                 << "The problem is linear separable with the input " << data_set_pointer->get_variables_names()[original_index] << ".\n\n";
        }

        neural_network_pointer->set_inputs_number(inputs_selection);

        final = calculate_losses(inputs_selection);

        optimal_inputs = inputs_selection;
        optimum_selection_error = final[0];
        optimum_training_error = final[1];
        optimal_parameters = get_parameters_inputs(inputs_selection);

        results->inputs_data.push_back(inputs_selection);

        if(reserve_error_data) results->loss_data.push_back(optimum_selection_error);

        if(reserve_selection_error_data) results->selection_error_data.push_back(optimum_training_error);

        results->stopping_condition = InputsSelection::AlgorithmFinished;

        end = true;
    }
    else
    {
        for(size_t i = 0; i < minimum_inputs_number; i++)
        {
            index = maximal_index(total_input_correlations);

            total_input_correlations[index] = -1;

            original_index = get_input_index(original_uses, index);

            current_uses[original_index] = DataSet::Input;

            inputs_selection[index] = true;

            data_set_pointer->set_columns_uses(current_uses);

            flag_input = true;
        }

        for(size_t i = 0; i < total_input_correlations.size(); i++)
        {
            if(total_input_correlations[i] >= maximum_correlation*targets_number)
            {
                total_input_correlations[i] = -1;

                original_index = get_input_index(original_uses, i);

                current_uses[original_index] = DataSet::Input;

                inputs_selection[i] = true;

                data_set_pointer->set_columns_uses(current_uses);

                if(display)
                {
                    cout << "Added input "<< data_set_pointer->get_variables_names()[original_index] << endl;
                }

                if(inputs_selection.count_equal_to(true) >= maximum_inputs_number)
                {
                    end = true;

                    if(display)
                    {
                        cout << "Maximum inputs("<< maximum_inputs_number <<") reached." << endl;
                    }

                    results->stopping_condition = InputsSelection::MaximumInputs;

                    final = calculate_losses(inputs_selection);

                    optimal_inputs = inputs_selection;
                    optimum_training_error = final[0];
                    optimum_selection_error = final[1];

                    results->inputs_data.push_back(inputs_selection);

                    if(reserve_error_data)
                    {
                        results->loss_data.push_back(optimum_training_error);
                    }

                    if(reserve_selection_error_data)
                    {
                        results->selection_error_data.push_back(optimum_selection_error);
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
            index = maximal_index(total_input_correlations);

            if(total_input_correlations[index] <= minimum_correlation*targets_number)
            {
                end = true;

                if(display) cout << "Correlation goal reached." << endl << endl;

                results->stopping_condition = InputsSelection::CorrelationGoal;

                if(inputs_selection.count_equal_to(true) == 0)
                {
                    original_index = get_input_index(original_uses, index);

                    current_uses[original_index] = DataSet::Input;

                    inputs_selection[index] = true;

                    data_set_pointer->set_columns_uses(current_uses);
                }

                optimal_inputs = inputs_selection;
            }
            else
            {
                total_input_correlations[index] = -1;

                original_index = get_input_index(original_uses, index);

                current_uses[original_index] = DataSet::Input;

                inputs_selection[index] = true;

                data_set_pointer->set_columns_uses(current_uses);
            }
        }

        final = calculate_losses(inputs_selection);

        current_training_loss = final[0];
        current_selection_error = final[1];

        // OPTIMUM

        if(abs(optimum_selection_error - current_selection_error) < tolerance ||
                optimum_selection_error > current_selection_error)
        {
            optimal_inputs = inputs_selection;
            optimum_training_error = current_training_loss;
            optimum_selection_error = current_selection_error;
            optimum_uses = current_uses;
        }
        else if(optimum_selection_error < current_selection_error)
        {
            selection_failures++;
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

//        previous_selection_error = current_selection_error;
        iteration++;

        results->inputs_data.push_back(inputs_selection);

        if(reserve_error_data)
        {
            results->loss_data.push_back(current_training_loss);
        }

        if(reserve_selection_error_data)
        {
            results->selection_error_data.push_back(current_selection_error);
        }

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display) cout << "Maximum time reached." << endl;

            results->stopping_condition = InputsSelection::MaximumTime;
        }
        else if(final[1] <= selection_error_goal)
        {
            end = true;

            if(display) cout << "Selection loss reached." << endl;

            results->stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(iteration >= maximum_iterations_number)
        {
            end = true;

            if(display) cout << "Maximum number of iterations reached." << endl;

            results->stopping_condition = InputsSelection::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display) cout << "Maximum selection failures("<<selection_failures<<") reached." << endl;

            results->stopping_condition = InputsSelection::MaximumSelectionFailures;
        }
        else if(inputs_selection.count_equal_to(true) >= maximum_inputs_number)
        {
            end = true;

            if(display) cout << "Maximum inputs("<< maximum_inputs_number <<") reached." << endl;

            results->stopping_condition = InputsSelection::MaximumInputs;
        }
        else if(inputs_selection.count_equal_to(true) == inputs_selection.size())
        {
            end = true;

            if(display) cout << "Algorithm finished" << endl;

            results->stopping_condition = InputsSelection::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iteration << endl;

            if(!flag_input) cout << "Add input: " << data_set_pointer->get_variables_names()[original_index] << endl;

            cout << "Current inputs: " <<  data_set_pointer->get_input_variables_names().vector_to_string() << endl;
            cout << "Number of inputs: " << inputs_selection.count_equal_to(true) << endl;
            cout << "Training loss: " << final[0] << endl;
            cout << "Selection error: " << final[1] << endl;
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
    results->final_selection_error = optimum_selection_error;
    results->final_loss = optimum_training_error;
    results->iterations_number = iteration;
    results->elapsed_time = elapsed_time;

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    // Set data set stuff

    data_set_pointer->set_columns_uses(optimum_uses);

    neural_network_pointer->set_inputs_number(optimal_inputs);

    neural_network_pointer->set_parameters(optimal_parameters);

    results->selected_inputs = optimal_inputs;

    if(display)
    {
        cout << "Optimal inputs: " << data_set_pointer->get_input_variables_names().vector_to_string() << endl;
        cout << "Optimal number of inputs: " << optimal_inputs.count_equal_to(true) << endl;
        cout << "Optimum training error: " << optimum_training_error << endl;
        cout << "Optimum selection error: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }
*/
//    return results;
}


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
   buffer << selection_error_goal;

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
   buffer << maximum_epochs_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Plot training loss history

   labels.push_back("Plot training loss history");

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

    return string_matrix;
}


/// Prints to the screen the growing inputs parameters, the stopping criteria
/// and other user stuff concerning the growing inputs object.

tinyxml2::XMLDocument* GrowingInputs::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Order Selection algorithm

    tinyxml2::XMLElement* root_element = document->NewElement("GrowingInputs");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = nullptr;
    tinyxml2::XMLText* text = nullptr;

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

    // selection error goal
    {
        element = document->NewElement("SelectionErrorGoal");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << selection_error_goal;

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
        element = document->NewElement("MaximumEpochsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_epochs_number;

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
        element = document->NewElement("ReserveTrainingErrorHistory");
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

    // Display
//    {
//        element = document->NewElement("Display");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << display;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    return document;
}


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

    file_stream.OpenElement("MaximumEpochsNumber");

    buffer.str("");
    buffer << maximum_epochs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve loss data

    file_stream.OpenElement("ReserveTrainingErrorHistory");

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
               << "GrowingInputs element is nullptr.\n";

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
                cerr << e.what() << endl;
            }
        }
    }

    // Trials number
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

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumEpochsNumber");

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
                cerr << e.what() << endl;
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

    // Minimum inputs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumInputsNumber");

        if(element)
        {
            const size_t new_minimum_inputs_number = static_cast<size_t>(atoi(element->GetText()));

            try
            {
                set_minimum_inputs_number(new_minimum_inputs_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum inputs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumInputsNumber");

        if(element)
        {
            const size_t new_maximum_inputs_number = static_cast<size_t>(atoi(element->GetText()));

            try
            {
                set_maximum_inputs_number(new_maximum_inputs_number);
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


/// Saves to a XML-type file the members of the growing inputs object.
/// @param file_name Name of growing inputs XML-type file.

void GrowingInputs::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


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
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R U N I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pruning_inputs.h"

namespace OpenNN
{

/// Default constructor.

PruningInputs::PruningInputs()
    : InputsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

PruningInputs::PruningInputs(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// Destructor.

PruningInputs::~PruningInputs()
{
}


/// Returns the minimum number of inputs in the pruning inputs selection algorithm.

const Index& PruningInputs::get_minimum_inputs_number() const
{
    return minimum_inputs_number;
}


/// Returns the maximum number of inputs in the pruning inputs selection algorithm.

const Index& PruningInputs::get_maximum_inputs_number() const
{
    return maximum_inputs_number;
}


/// Returns the maximum number of selection failures in the pruning inputs algorithm.

const Index& PruningInputs::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Sets the members of the pruning inputs object to their default values.

void PruningInputs::set_default()
{
    Index inputs_number;

    if(training_strategy_pointer == nullptr || !training_strategy_pointer->has_neural_network())
    {
        maximum_selection_failures = 100;

        maximum_inputs_number = 20;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();
        maximum_selection_failures = 100;//static_cast<Index>(max(3.,inputs_number/5.));

        maximum_inputs_number = inputs_number;
    }

    minimum_inputs_number = 1;

    minimum_correlation = 0.0;

    trials_number = 3;

    maximum_epochs_number = 1000;

    maximum_time = 3600.0;
}


/// Sets the minimum inputs for the pruning inputs algorithm.
/// @param new_minimum_inputs_number Minimum number of inputs in the pruning inputs algorithm.

void PruningInputs::set_minimum_inputs_number(const Index& new_minimum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_inputs_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_minimum_inputs_number(const Index&) method.\n"
               << "Minimum inputs number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}


/// Sets the maximum inputs for the pruning inputs algorithm.
/// @param new_maximum_inputs_number Maximum number of inputs in the pruning inputs algorithm.

void PruningInputs::set_maximum_inputs_number(const Index& new_maximum_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_inputs_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_maximum_inputs_number(const Index&) method.\n"
               << "Maximum inputs number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_inputs_number = new_maximum_inputs_number;
}


/// Sets the maximum selection failures for the pruning inputs algorithm.
/// @param new_maximum_loss_failures Maximum number of selection failures in the pruning inputs algorithm.

void PruningInputs::set_maximum_selection_failures(const Index& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_maximum_selection_failures(const Index&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}


/// Perform the inputs selection with the pruning inputs method.

PruningInputs::PruningInputsResults* PruningInputs::perform_inputs_selection()
{

#ifdef __OPENNN_DEBUG__

    check();

#endif

    PruningInputsResults* results = new PruningInputsResults();

    if(display)
    {
        cout << "Performing pruning inputs selection..." << endl;
        cout << endl << "Calculating correlations..." << endl;
    }

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    type optimum_training_error = numeric_limits<type>::max();
    type optimum_selection_error = numeric_limits<type>::max();
    type previus_selection_error = numeric_limits<type>::max();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Tensor<Index, 1> original_input_columns_indices = data_set_pointer->get_input_columns_indices();

//    const Tensor<Index, 1> inputs_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> inputs_variables_indices = data_set_pointer->get_input_columns_indices();

    const Index used_columns_number = data_set_pointer->get_used_columns_number();

//    const Tensor<string, 1> used_columns_names = data_set_pointer->get_used_columns_names();
    const Tensor<string, 1> columns_names = data_set_pointer->get_columns_names();
//    const Tensor<string, 1> used_columns_names = data_set_pointer->get_input_variables_names();

    const Tensor<type, 2> correlations = data_set_pointer->calculate_input_target_columns_correlations_values();

    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

    const Tensor<type, 1> total_correlations = correlations.sum(rows_sum).abs();

    Tensor<type, 1> correlations_ascending(total_correlations);

    Tensor<Index, 1> correlations_ascending_indices = original_input_columns_indices;

    sort( correlations_ascending_indices.data(),
          correlations_ascending_indices.data()+original_input_columns_indices.size(),
          [&](Index i,Index j){return total_correlations[i]>total_correlations[j];} );


//    sort(correlations_ascending.data(), correlations_ascending.data() +  correlations_ascending.size(), less<type>());

//    Tensor<Index, 1> correlations_ascending_indices(total_correlations.size());
//    correlations_ascending_indices.setZero();

//    for(Index i = 0; i < total_correlations.size(); i++)
//    {
//        for(Index j = 0; j < correlations_ascending.size(); j++)
//        {
//            if(correlations_ascending(i) == total_correlations(j))
//            {
//                correlations_ascending_indices(i) = j;
//                correlations_ascending_indices(i) = original_input_columns_indices(j);
//                continue;
//            }
//        }
//    }

    // Neural network

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();

    const Tensor<Descriptives, 1> original_input_variables_descriptives = neural_network_pointer->get_scaling_layer_pointer()->get_descriptives();

    const Tensor<ScalingLayer::ScalingMethod, 1> original_scaling_methods = neural_network_pointer->get_scaling_layer_pointer()->get_scaling_methods();

    Tensor<Layer*, 1> trainable_layers = neural_network_pointer->get_trainable_layers_pointers();

    Index trainable_layers_number = trainable_layers.size();

    // Optimization algorithm

    Tensor<Index, 1> current_columns_indices = inputs_variables_indices;

    Tensor<Index, 1> optimal_columns_indices;

    Tensor<type, 1> optimal_parameters;

    Index selection_failures = 0;

    time_t beginning_time, current_time;
    type elapsed_time = 0;

    time(&beginning_time);

    bool end_algorithm = false;

    // Model selection

    if(used_columns_number < maximum_epochs_number) maximum_epochs_number = used_columns_number;

    for(Index iteration = 0; iteration < maximum_epochs_number; iteration++)
    {
        OptimizationAlgorithm::Results training_results;

        Index column_index;
        string column_name;

        if(iteration > 0)
        {
            column_index = correlations_ascending_indices[iteration-1];

            column_name = columns_names[column_index];

            data_set_pointer->set_column_use(column_name, DataSet::UnusedVariable);

            current_columns_indices = delete_result(column_index, current_columns_indices);

            const Index input_variables_number = data_set_pointer->get_input_variables_number();

            data_set_pointer->set_input_variables_dimensions(Tensor<Index, 1> (1).constant(input_variables_number));

            neural_network_pointer->set_inputs_number(input_variables_number);
        }

//        if(iteration == 0)
//        {
//            training_results = training_strategy_pointer->perform_training();

//            current_training_error = training_results.final_training_error;
//            current_selection_error = training_results.final_selection_error;
//            current_parameters = training_results.final_parameters;
//        }
//        else
//        {

        // Trial

        type optimum_selection_error_trial = numeric_limits<type>::max();
        type optimum_training_error_trial = numeric_limits<type>::max();
        Tensor<type, 1> optimum_parameters_trial(neural_network_pointer->get_parameters_number());

        for(Index trial = 0; trial < trials_number; trial ++)
        {
            neural_network_pointer->set_parameters_random();

            training_results = training_strategy_pointer->perform_training();

            type current_training_error_trial = training_results.final_training_error;
            type current_selection_error_trial = training_results.final_selection_error;
            Tensor<type, 1> current_parameters_trial = training_results.final_parameters;

            if(current_selection_error_trial < optimum_selection_error_trial)
            {
                optimum_parameters_trial = current_parameters_trial;
                optimum_selection_error_trial = current_selection_error_trial;
                optimum_training_error_trial = current_training_error_trial;
            }

            //                current_training_error = training_results.final_training_error;
            //                current_selection_error = training_results.final_selection_error;
            //                current_parameters = training_results.final_parameters;

            if(display)
            {
                cout << endl << "Trial number: " << iteration << endl;
                cout << "Training error: " << current_training_error_trial << endl;
                cout << "Selection error: " << current_selection_error_trial << endl;
                cout << "Stopping condition: " << training_results.write_stopping_condition() << endl << endl;
            }
        }

//        }

//        if(display)
//        {
//            cout << endl << "Trial number: " << iteration << endl;
//            cout << "Training error: " << current_training_error << endl;
//            cout << "Selection error: " << current_selection_error << endl;
//            cout << "Stopping condition: " << training_results.write_stopping_condition() << endl << endl;
//        }

        if(iteration == 0
                ||(optimum_selection_error > optimum_selection_error_trial
                   && abs(optimum_selection_error - optimum_selection_error_trial) > tolerance))
        {
            optimal_columns_indices = current_columns_indices;
            optimal_parameters = optimum_parameters_trial;
            optimum_selection_error = optimum_selection_error_trial;
            optimum_training_error = optimum_training_error_trial;
        }
        else if (previus_selection_error < optimum_selection_error_trial)
        {
            selection_failures++;
        }

        previus_selection_error = optimum_selection_error_trial;

        if(reserve_training_error_data)
        {
            results->training_error_data = insert_result(optimum_training_error_trial, results->training_error_data);
        }

        if(reserve_selection_error_data)
        {
            results->selection_error_data = insert_result(optimum_selection_error_trial, results->selection_error_data);
        }

        time(&current_time);

        elapsed_time = static_cast<type>( difftime(current_time,beginning_time));

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end_algorithm = true;

            if(display) cout << "Maximum time reached." << endl;

            results->stopping_condition = InputsSelection::MaximumTime;
        }
        else if(optimum_selection_error_trial <= selection_error_goal)
        {
            end_algorithm = true;

            if(display) cout << "Selection loss reached." << endl;

            results->stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(iteration >= maximum_epochs_number)
        {
            end_algorithm = true;

            if(display) cout << "Maximum number of epochs reached." << endl;

            results->stopping_condition = InputsSelection::MaximumEpochs;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end_algorithm = true;

            if(display) cout << "Maximum selection failures("<<selection_failures<<") reached." << endl;

            results->stopping_condition = InputsSelection::MaximumSelectionFailures;
        }
        else if(current_columns_indices.size() <= minimum_inputs_number)
        {
            //¿?
            end_algorithm = true;

            if(display) cout << "Minimum inputs("<< minimum_inputs_number <<") reached." << endl;

            results->stopping_condition = InputsSelection::MaximumInputs;
        }
        else if(current_columns_indices.size() == 1)
        {
            end_algorithm = true;

            if(display) cout << "Algorithm finished" << endl;

            results->stopping_condition = InputsSelection::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iteration << endl;

            if(end_algorithm == false && iteration != 0) cout << "Pruning input: " << data_set_pointer->get_variable_name(column_index) << endl;

            cout << "Current inputs: " << endl <<  data_set_pointer->get_input_variables_names().cast<string>() << endl << endl;
            cout << "Number of inputs: " << current_columns_indices.size() << endl;
            cout << "Training error: " << optimum_training_error_trial << endl;
            cout << "Selection error: " << optimum_selection_error_trial << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            cout << endl;
        }

        if(end_algorithm == true)
        {
            // Save results

            results->optimal_inputs_indices = optimal_columns_indices;
            results->final_selection_error = optimum_selection_error;
            results->final_training_error = optimum_training_error;
            results->iterations_number = iteration + 1;
            results->elapsed_time = write_elapsed_time(elapsed_time);
            results->minimal_parameters = optimal_parameters;

            break;
        }
    }

    // Set Data set stuff

    data_set_pointer->set_input_columns_unused();

    const Index optimal_inputs_number = optimal_columns_indices.size();

    for(Index i = 0; i < optimal_inputs_number; i++)
    {
        const Index optimal_input_index = optimal_columns_indices[i];

        data_set_pointer->set_column_use(optimal_input_index, DataSet::Input);
    }

    const Index optimal_input_variables_number = data_set_pointer->get_input_variables_names().size();

    data_set_pointer->set_input_variables_dimensions(Tensor<Index, 1> (1).constant(optimal_input_variables_number));

    // Set Neural network stuff

    neural_network_pointer->set_inputs_number(optimal_input_variables_number);

    neural_network_pointer->set_parameters(optimal_parameters);

    neural_network_pointer->set_inputs_names(data_set_pointer->get_input_variables_names());

    Tensor<Descriptives, 1> new_input_descriptives(optimal_input_variables_number);
    Tensor<ScalingLayer::ScalingMethod, 1> new_scaling_methods(optimal_input_variables_number);

    Index descriptive_index = 0;
    Index unused = 0;

    for(Index i = 0; i < original_input_columns_indices.size(); i++)
    {
        const Index current_column_index = original_input_columns_indices(i);

        if(data_set_pointer->get_column_use(current_column_index) == DataSet::Input)
        {
            if(data_set_pointer->get_column_type(current_column_index) != DataSet::ColumnType::Categorical)
            {
                new_input_descriptives(descriptive_index) = original_input_variables_descriptives(descriptive_index + unused);
                new_scaling_methods(descriptive_index) = original_scaling_methods(descriptive_index + unused);
                descriptive_index++;
            }
            else
            {
                for(Index j = 0; j < data_set_pointer->get_columns()[current_column_index].get_categories_number(); j++)
                {
                    new_input_descriptives(descriptive_index) = original_input_variables_descriptives(descriptive_index + unused);
                    new_scaling_methods(descriptive_index) = original_scaling_methods(descriptive_index + unused);
                    descriptive_index++;
                }
            }
        }
        else if(data_set_pointer->get_column_use(current_column_index) == DataSet::UnusedVariable)
        {
            if(data_set_pointer->get_column_type(current_column_index) != DataSet::ColumnType::Categorical) unused ++;
            else
            {
                for(Index j = 0; j < data_set_pointer->get_columns()[current_column_index].get_categories_number(); j++) unused ++;
            }
        }
    }
    neural_network_pointer->get_scaling_layer_pointer()->set_descriptives(new_input_descriptives);
    neural_network_pointer->get_scaling_layer_pointer()->set_scaling_methods(new_scaling_methods);

    if(display)
    {
        cout << "Optimal inputs: " << endl << data_set_pointer->get_input_variables_names().cast<string>() << endl << endl;
        cout << "Optimal number of inputs: " << optimal_input_variables_number << endl;
        cout << "Optimum training error: " << optimum_training_error << endl;
        cout << "Optimum selection error: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    return results;
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> PruningInputs::to_string_matrix() const
{

    ostringstream buffer;

    Tensor<string, 1> labels(11);
    Tensor<string, 1> values(11);

    // Trials number

    labels(0) = "Trials number";

    buffer.str("");
    buffer << trials_number;

    values(0) = buffer.str();

    // Tolerance

    labels(1) = "Tolerance";

    buffer.str("");
    buffer << tolerance;

    values(1) = buffer.str();

    // Selection loss goal

    labels(2) = "Selection loss goal";

    buffer.str("");
    buffer << selection_error_goal;

    values(2) = buffer.str();

    // Maximum selection failures

    labels(3) = "Maximum selection failures";

    buffer.str("");
    buffer << maximum_selection_failures;

    values(3) = buffer.str();

    // Minimum inputs number

    labels(4) = "Minimum inputs number";

    buffer.str("");
    buffer << minimum_inputs_number;

    values(4) = buffer.str();

    // Minimum correlation

    labels(5) = "Minimum correlation";

    buffer.str("");
    buffer << minimum_correlation;

    values(5) = buffer.str();

    // Maximum correlation

    labels(6) = "Maximum correlation";

    buffer.str("");
    buffer << maximum_correlation;

    values(6) = buffer.str();

    // Maximum iterations number

    labels(7) = "Maximum iterations number";

    buffer.str("");
    buffer << maximum_epochs_number;

    values(7) = buffer.str();

    // Maximum time

    labels(8) = "Maximum time";

    buffer.str("");
    buffer << maximum_time;

    values(8) = buffer.str();

    // Plot training loss history

    labels(9) = "Plot training loss history";

    buffer.str("");

    if(reserve_training_error_data)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(9) = buffer.str();

    // Plot selection error history

    labels(10) = "Plot selection error hitory";

    buffer.str("");

    if(reserve_selection_error_data)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(10) = buffer.str();

    const Index rows_number = labels.size();
    const Index columns_number = 2;

    Tensor<string, 2> string_matrix(rows_number, columns_number);

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

     return string_matrix;
}


/// Serializes the pruning inputs object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PruningInputs::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("PruningInputs");

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
    buffer << reserve_training_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error data

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this pruning inputs object.
/// @param document TinyXML document containing the member data.

void PruningInputs::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("PruningInputs");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PruningInputs element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Regression
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Approximation");

        if(element)
        {
            const string new_approximation = element->GetText();

            try
            {
                set_approximation(new_approximation != "0");
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
            const Index new_trials_number = static_cast<Index>(atoi(element->GetText()));

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
            const string new_reserve_training_error_data = element->GetText();

            try
            {
                set_reserve_training_error_data(new_reserve_training_error_data != "0");
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
            const type new_selection_error_goal = static_cast<type>(atof(element->GetText()));

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
            const Index new_maximum_iterations_number = static_cast<Index>(atoi(element->GetText()));

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
            const type new_maximum_correlation = static_cast<type>(atof(element->GetText()));

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
            const type new_minimum_correlation = static_cast<type>(atof(element->GetText()));

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
            const type new_maximum_time = static_cast<type>( atoi(element->GetText()));

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
            const type new_tolerance = static_cast<type>(atof(element->GetText()));

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
            const Index new_minimum_inputs_number = static_cast<Index>(atoi(element->GetText()));

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
            const Index new_maximum_inputs_number = static_cast<Index>(atoi(element->GetText()));

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
            const Index new_maximum_selection_failures = static_cast<Index>(atoi(element->GetText()));

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



/// Saves to a XML-type file the members of the pruning inputs object.
/// @param file_name Name of pruning inputs XML-type file.

void PruningInputs::save(const string& file_name) const
{
//    tinyxml2::XMLDocument* document = to_XML();

//    document->SaveFile(file_name.c_str());

//    delete document;
}




/// Loads a pruning inputs object from a XML-type file.
/// @param file_name Name of pruning inputs XML-type file.

void PruningInputs::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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

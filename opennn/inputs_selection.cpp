//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "inputs_selection.h"

namespace OpenNN
{


/// Default constructor.

InputsSelection::InputsSelection()
    : training_strategy_pointer(nullptr)
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a trainig strategy object.

InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
{
    set_default();
}


/// Destructor.

InputsSelection::~InputsSelection()
{
}


/// Returns whether the problem is of function regression type.

const bool& InputsSelection::get_approximation() const
{
    return approximation;
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* InputsSelection::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return training_strategy_pointer;
}


/// Returns true if this inputs selection algorithm has a training strategy associated, and false otherwise.

bool InputsSelection::has_training_strategy() const
{
    if(training_strategy_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the number of trials for each network architecture.

const Index& InputsSelection::get_trials_number() const
{
    return trials_number;
}


/// Returns true if the loss index losses are to be reserved, and false otherwise.

const bool& InputsSelection::get_reserve_training_error_data() const
{
    return reserve_training_error_data;
}


/// Returns true if the selection losses are to be reserved, and false otherwise.

const bool& InputsSelection::get_reserve_selection_error_data() const
{
    return reserve_selection_error_data;
}


/// Returns true if the parameters vector of the neural network with minimum selection error is to be reserved,
/// and false otherwise.

const bool& InputsSelection::get_reserve_minimal_parameters() const
{
    return reserve_minimal_parameters;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& InputsSelection::get_display() const
{
    return display;
}


/// Returns the goal for the selection error in the inputs selection algorithm.

const type& InputsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


/// Returns the maximum number of iterations in the inputs selection algorithm.

const Index& InputsSelection::get_maximum_iterations_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum time in the inputs selection algorithm.

const type& InputsSelection::get_maximum_time() const
{
    return maximum_time;
}


/// Return the maximum correlation for the algorithm.

const type& InputsSelection::get_maximum_correlation() const
{
    return maximum_correlation;
}


/// Return the minimum correlation for the algorithm.

const type& InputsSelection::get_minimum_correlation() const
{
    return minimum_correlation;
}


/// Return the tolerance of error for the algorithm.

const type& InputsSelection::get_tolerance() const
{
    return tolerance;
}


/// Sets a new regression value.
/// If it is set to true the problem will be taken as a function regression;
/// if it is set to false the problem will be taken as a classification.
/// @param new_approximation Regression value.

void InputsSelection::set_approximation(const bool& new_approximation)
{
    approximation = new_approximation;
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void InputsSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


/// Sets the members of the inputs selection object to their default values.

void InputsSelection::set_default()
{
    trials_number = 1;

    // Results

    reserve_training_error_data = true;
    reserve_selection_error_data = true;
    reserve_minimal_parameters = true;

    // Stopping criteria

    selection_error_goal = 0;

    maximum_epochs_number = 1000;

    maximum_correlation = 1.0;
    minimum_correlation = 0;

    maximum_time = 3600.0;

    tolerance = 0;
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of trials for each set of parameters.

void InputsSelection::set_trials_number(const Index& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_trials_number(const Index&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets the reserve flag for the loss data.
/// @param new_reserve_error_data Flag value.

void InputsSelection::set_reserve_training_error_data(const bool& new_reserve_training_error_data)
{
    reserve_training_error_data = new_reserve_training_error_data;
}


/// Sets the reserve flag for the selection error data.
/// @param new_reserve_selection_error_data Flag value.

void InputsSelection::set_reserve_selection_error_data(const bool& new_reserve_selection_error_data)
{
    reserve_selection_error_data = new_reserve_selection_error_data;
}


/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void InputsSelection::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void InputsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the selection error goal for the inputs selection algorithm.
/// @param new_selection_error_goal Goal of the selection error.

void InputsSelection::set_selection_error_goal(const type& new_selection_error_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_selection_error_goal(const type&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum iterations number for the inputs selection algorithm.
/// @param new_maximum_iterations_number Maximum number of epochs.

void InputsSelection::set_maximum_iterations_number(const Index& new_maximum_iterations_number)
{
    maximum_epochs_number = new_maximum_iterations_number;
}


/// Sets the maximum time for the inputs selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void InputsSelection::set_maximum_time(const type& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Sets the maximum value for the correlations in the inputs selection algorithm.
/// @param new_maximum_correlation Maximum value of the correlations.

void InputsSelection::set_maximum_correlation(const type& new_maximum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_correlation < 0 || new_maximum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_maximum_correlation(const type&) method.\n"
               << "Maximum correlation must be comprised between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_correlation = new_maximum_correlation;
}


/// Sets the minimum value for the correlations in the inputs selection algorithm.
/// @param new_minimum_correlation Minimum value of the correlations.

void InputsSelection::set_minimum_correlation(const type& new_minimum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_correlation < 0 || new_minimum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_minimum_correlation(const type&) method.\n"
               << "Minimum correaltion must be comprised between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_correlation = new_minimum_correlation;
}


/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void InputsSelection::set_tolerance(const type& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_tolerance(const type&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


/// Returns the minimum of the loss and selection error in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Tensor<type, 1> InputsSelection::calculate_losses(const Tensor<bool, 1> & inputs)
{

#ifdef __OPENNN_DEBUG__

    Index count = 0;

    for(Index i = 0; i < inputs.size(); i++)
    {
        if(inputs(i) == true) count++;
    }

    if(count <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Tensor<type, 1> perform_minimum_model_evaluation(Index) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Tensor<type, 1> perform_minimum_model_evaluation(Index) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Neural network

    NeuralNetwork* neural_network = training_strategy_pointer->get_neural_network_pointer();

    // Optimization algorithm

    OptimizationAlgorithm::Results results;

    type optimum_selection_error = numeric_limits<type>::max();
    type optimum_training_error = numeric_limits<type>::max();

    Tensor<type, 1> optimum_parameters;

    Tensor<type, 1> optimum_losses(2);

    bool flag_loss = false;
    bool flag_selection = false;

    // Check population

    for(Index i = 0; i < inputs_history.size(); i++)
    {
        const Tensor<bool, 1> inputs_rows = inputs_history.chip(i,0);

        for(Index j = 0; j < inputs_rows.size(); j++)
        {
            if(inputs_rows(j) != inputs(j)) break;

            optimum_losses[0] = training_error_history[i];
            flag_loss = true;
        }
        /*if(inputs_history(i) == inputs)
        {
            optimum_losses[0] = training_error_history[i];
            flag_loss = true;
        }*/
    }

    for(Index i = 0; i < inputs_history.size(); i++)
    {
        const Tensor<bool, 1> inputs_rows = inputs_history.chip(i,0);

        for(Index j = 0; j < inputs_rows.size(); j++)
        {
            if(inputs_rows(j) != inputs(j)) break;

            optimum_losses(0) = selection_error_history(i);
            flag_selection = true;
        }
        /*
                if(inputs_history[i] == inputs)
                {
                    optimum_losses[1] = selection_error_history[i];
                    flag_selection = true;
                }*/
    }

    if(flag_loss && flag_selection)
    {
        if(display)
        {
            cout << "Training error: " << optimum_losses[0] << endl;
            cout << "Selection error: " << optimum_losses[1] << endl;
            cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
        }
        return optimum_losses;
    }

    neural_network->perturbate_parameters(static_cast<type>(0.001));

    for(Index i = 0; i < trials_number; i++)
    {
        neural_network->set_parameters_random();

        results = training_strategy_pointer->perform_training();

        const type selection_error = results.final_selection_error;
        const type training_error = results.final_training_error;
        const Tensor<type, 1> parameters = results.final_parameters;

        if(display && trials_number != 1)
        {
            cout << "Trial number: " << i << endl;
            cout << "Training error: " << training_error << endl;
            cout << "Selection error: " << selection_error << endl;
            cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
        }

        if(optimum_selection_error > selection_error)
        {
            optimum_selection_error = selection_error;
            optimum_training_error = training_error;
            optimum_parameters = parameters;
        }
    }

    if(display)
    {
        if(trials_number != 1) cout << "Trial number: " << trials_number << endl;
        cout << "Training error: " << optimum_training_error << endl;
        cout << "Selection error: " << optimum_selection_error << endl;
        cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
    }

//    inputs_history = insert_result(inputs, inputs_history);

    training_error_history = insert_result(optimum_training_error, training_error_history);

    selection_error_history = insert_result(optimum_selection_error, selection_error_history);

    parameters_history = insert_result(optimum_parameters, parameters_history);
    /*
        inputs_history.push_back(inputs);

        training_error_history.push_back(optimum_training_error);

        selection_error_history.push_back(optimum_selection_error);

        parameters_history.push_back(optimum_parameters);
    */
    optimum_losses[0] = optimum_training_error;
    optimum_losses[1] = optimum_selection_error;

    return optimum_losses;

//    return Tensor<type, 1>();
}


Tensor<type, 1> InputsSelection::insert_result(const type& value, const Tensor<type, 1>& old_tensor) const
{
    const Index size = old_tensor.size();

    Tensor<type, 1> new_tensor(size+1);

    for(Index i = 0; i < size; i++)
    {
        new_tensor(i) = old_tensor(i);
    }

    new_tensor(size) = value;

    return new_tensor;
}

Tensor<Index, 1> InputsSelection::insert_result(const Index& value, const Tensor<Index, 1>& old_tensor) const
{
    const Index size = old_tensor.size();

    Tensor<Index, 1> new_tensor(size+1);

    for(Index i = 0; i < size; i++)
    {
        new_tensor(i) = old_tensor(i);
    }

    new_tensor(size) = value;

    return new_tensor;
}

Tensor<Index, 1> InputsSelection::delete_result(const Index& value, const Tensor<Index, 1>& old_tensor) const
{
    const Index size = old_tensor.size();

    Tensor<Index, 1> new_tensor(size-1);

    Index index = 0;

    for(Index i = 0; i < size; i++)
    {
        if(old_tensor(i) != value)
        {
            new_tensor(index) = old_tensor(i);

            index++;
        }
    }

    return new_tensor;
}


Tensor< Tensor<type, 1>, 1> InputsSelection::insert_result(const Tensor<type, 1>& value,
                                                           const Tensor< Tensor<type, 1>, 1>& old_tensor) const
{
    const Index size = old_tensor.size();

    Tensor< Tensor<type, 1>, 1> new_tensor(size+1);

    for(Index i = 0; i < size; i++)
    {
        new_tensor(i) = old_tensor(i);
    }

    new_tensor(size) = value;

    return new_tensor;
}


/// Returns the parameters of the neural network if the inputs is in the history.
/// @param inputs Vector of inputs to be trained with.

Tensor<type, 1> InputsSelection::get_parameters_inputs(const Tensor<bool, 1>& inputs) const
{
    /*
    #ifdef __OPENNN_DEBUG__

        if(inputs.count_equal_to(true) <= 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: InputsSelection class.\n"
                   << "Tensor<type, 1> get_parameters_inputs(const Tensor<bool, 1>&) method.\n"
                   << "Inputs must be greater than 1.\n";

            throw logic_error(buffer.str());
        }

    #endif

        Index i;

        Tensor<type, 1> parameters;

        for(i = 0; i < inputs_history.size(); i++)
        {
            if(inputs_history[i] == inputs)
            {
                parameters = parameters_history[i];

                break;
            }
        }

        return parameters;
    */
    return Tensor<type, 1>();
}


/// Return a string with the stopping condition of the training depending on the training method.
/// @param results Results of the perform_training method.

string InputsSelection::write_stopping_condition(const OptimizationAlgorithm::Results& results) const
{
    return results.write_stopping_condition();
}


/// Delete the history of the selection error values.

void InputsSelection::delete_selection_history()
{
    /*
        selection_error_history.set();
    */
}


/// Delete the history of the loss values.

void InputsSelection::delete_loss_history()
{
    /*
        training_error_history.set();
    */
}


/// Delete the history of the parameters of the trained neural networks.

void InputsSelection::delete_parameters_history()
{
    /*
        parameters_history.set();
    */
}


/// Checks that the different pointers needed for performing the inputs selection are not nullptr.

void InputsSelection::check() const
{
    // Optimization algorithm

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network

    if(!loss_index_pointer->has_neural_network())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(neural_network_pointer->is_empty())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Neural network is empty.\n";

        throw logic_error(buffer.str());
    }

    // Data set

    if(!loss_index_pointer->has_data_set())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();



    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection samples is zero.\n";

        throw logic_error(buffer.str());
    }

}


/// Return a string with the stopping condition of the Results

string InputsSelection::Results::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case MaximumTime:
    {
        return "MaximumTime";
    }
    case SelectionErrorGoal:
    {
        return "SelectionErrorGoal";
    }
    case MaximumInputs:
    {
        return "MaximumInputs";
    }
    case MinimumInputs:
    {
        return "MinimumInputs";
    }
    case MaximumEpochs:
    {
        return "MaximumEpochs";
    }
    case MaximumSelectionFailures:
    {
        return "MaximumSelectionFailures";
    }
    case CorrelationGoal:
    {
        return "CorrelationGoal";
    }
    case AlgorithmFinished:
    {
        return "AlgorithmFinished";
    }
    }

    return string();
}


/// Writes the time from seconds in format HH:mm:ss.

const string InputsSelection::write_elapsed_time(const type& time) const
{

#ifdef __OPENNN_DEBUG__

    if(time > static_cast<type>(3600e5))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "const string write_elapsed_time(const type& time) const method.\n"
               << "Time must be lower than 10e5 seconds.\n";

        throw logic_error(buffer.str());
    }

    if(time < static_cast<type>(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "const string write_elapsed_time(const type& time) const method.\n"
               << "Time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }
#endif

    int hours = static_cast<int>(time) / 3600;
    int seconds = static_cast<int>(time) % 3600;
    int minutes = seconds / 60;
    seconds = seconds % 60;

    ostringstream elapsed_time;

    elapsed_time << setfill('0') << setw(2) << hours << ":"
                 << setfill('0') << setw(2) << minutes << ":"
                 << setfill('0') << setw(2) << seconds << endl;

    return elapsed_time.str();
}


/// Return the index of uses where is the(input_number)-th input.
/// @param uses Vector of the uses of the variables.
/// @param input_number Index of the input to find.

Index InputsSelection::get_input_index(const Tensor<DataSet::VariableUse, 1> uses, const Index input_number)
{
#ifdef __OPENNN_DEBUG__

    if(uses.size() < input_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "const Index get_input_index(const Tensor<DataSet::VariableUse, 1>, const Index) method.\n"
               << "Size of uses vector("<< uses.size() <<") must be greater than " <<  input_number << ".\n";

        throw logic_error(buffer.str());
    }
#endif

    Index i = 0;

    Index j = 0;

    while(i < uses.size())
    {
        if(uses[i] == DataSet::Input &&
                input_number == j)
        {
            break;
        }
        else if(uses[i] == DataSet::Input)
        {
            i++;
            j++;
        }
        else
        {
            i++;
        }
    }
    return i;
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

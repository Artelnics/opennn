//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neurons_selection.h"

namespace OpenNN
{

/// Default constructor.

NeuronsSelection::NeuronsSelection()
{
    training_strategy_pointer = nullptr;

    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

NeuronsSelection::NeuronsSelection(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    set_default();
}


/// Destructor.

NeuronsSelection::~NeuronsSelection()
{
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* NeuronsSelection::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return training_strategy_pointer;
}


/// Returns true if this neurons selection algorithm has a training strategy associated, and false otherwise.

bool NeuronsSelection::has_training_strategy() const
{
    if(training_strategy_pointer != nullptr)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the maximum of the hidden perceptrons number used in the neurons selection.

const Index& NeuronsSelection::get_maximum_neurons() const
{
    return maximum_neurons;
}


/// Returns the minimum of the hidden perceptrons number used in the neurons selection.

const Index& NeuronsSelection::get_minimum_neurons() const
{
    return minimum_neurons;
}


/// Returns the number of trials for each network architecture.

const Index& NeuronsSelection::get_trials_number() const
{
    return trials_number;
}


/// Returns true if the loss index losses are to be reserved, and false otherwise.

const bool& NeuronsSelection::get_reserve_training_error_data() const
{
    return reserve_training_error_data;
}


/// Returns true if the loss index selection losses are to be reserved, and false otherwise.

const bool& NeuronsSelection::get_reserve_selection_error_data() const
{
    return reserve_selection_error_data;
}


/// Returns true if the parameters vector of the neural network with minimum selection error is to be reserved, and false otherwise.

const bool& NeuronsSelection::get_reserve_minimal_parameters() const
{
    return reserve_minimal_parameters;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& NeuronsSelection::get_display() const
{
    return display;
}


/// Returns the goal for the selection error in the neurons selection algorithm.

const type& NeuronsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


/// Returns the maximum number of epochs in the neurons selection algorithm.

const Index& NeuronsSelection::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum time in the neurons selection algorithm.

const type& NeuronsSelection::get_maximum_time() const
{
    return maximum_time;
}


/// Return the tolerance of error for the neurons selection algorithm.

const type& NeuronsSelection::get_tolerance() const
{
    return tolerance;
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void NeuronsSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


/// Sets the members of the neurons selection object to their default values.

void NeuronsSelection::set_default()
{
    Index inputs_number;
    Index outputs_number;

    if(training_strategy_pointer == nullptr
            || !training_strategy_pointer->has_neural_network())
    {
        inputs_number = 0;
        outputs_number = 0;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();
        outputs_number = training_strategy_pointer->get_neural_network_pointer()->get_outputs_number();
    }
    // MEMBERS

    minimum_neurons = 1;

    // Heuristic value for the maximum_neurons

    maximum_neurons = 2*(inputs_number + outputs_number);
    trials_number = 1;

    // Neurons selection results

    reserve_training_error_data = true;
    reserve_selection_error_data = true;
    reserve_minimal_parameters = true;

    display = true;

    // Stopping criteria

    selection_error_goal = 0;

    maximum_epochs_number = 1000;
    maximum_time = 3600;

    tolerance = 0;
}


/// Sets the number of the maximum hidden perceptrons for the neurons selection algorithm.
/// @param new_maximum_neurons Maximum number of hidden perceptrons.

void NeuronsSelection::set_maximum_neurons(const Index& new_maximum_neurons)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_neurons <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_neurons(const Index&) method.\n"
               << "maximum_neurons(" << new_maximum_neurons << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_maximum_neurons < minimum_neurons)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_neurons(const Index&) method.\n"
               << "maximum_neurons(" << new_maximum_neurons << ") must be equal or greater than minimum_neurons(" << minimum_neurons << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_neurons = new_maximum_neurons;
}


/// Sets the number of the minimum hidden perceptrons for the neurons selection algorithm.
/// @param new_minimum_neurons Minimum number of hidden perceptrons.

void NeuronsSelection::set_minimum_neurons(const Index& new_minimum_neurons)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_neurons <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_minimum_neurons(const Index&) method.\n"
               << "minimum_neurons(" << new_minimum_neurons << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_minimum_neurons >= maximum_neurons)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_minimum_neurons(const Index&) method.\n"
               << "minimum_neurons(" << new_minimum_neurons << ") must be less than maximum_neurons(" << maximum_neurons << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_neurons = new_minimum_neurons;
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of assays for each set of parameters.

void NeuronsSelection::set_trials_number(const Index& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_trials_number(const Index&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets the reserve flag for the loss data.
/// @param new_reserve_training_error_data Flag value.

void NeuronsSelection::set_reserve_training_error_data(const bool& new_reserve_training_error_data)
{
    reserve_training_error_data = new_reserve_training_error_data;
}


/// Sets the reserve flag for the selection error data.
/// @param new_reserve_selection_error_data Flag value.

void NeuronsSelection::set_reserve_selection_error_data(const bool& new_reserve_selection_error_data)
{
    reserve_selection_error_data = new_reserve_selection_error_data;
}


/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void NeuronsSelection::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void NeuronsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the selection error goal for the neurons selection algorithm.
/// @param new_selection_error_goal Goal of the selection error.

void NeuronsSelection::set_selection_error_goal(const type& new_selection_error_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_selection_error_goal(const type&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum epochs number for the neurons selection algorithm.
/// @param new_maximum_epochs_number Maximum number of epochs.

void NeuronsSelection::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_epochs_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_epochs_number(const Index&) method.\n"
               << "Maximum epochs number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets the maximum time for the neurons selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void NeuronsSelection::set_maximum_time(const type& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void NeuronsSelection::set_tolerance(const type& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_tolerance(const type&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


/// Returns the minimum of the loss and selection error in trials_number trainings.
/// @param neurons_number Number of neurons in the hidden layer to be trained with.

Tensor<type, 1> NeuronsSelection::calculate_losses(const Index& neurons_number, NeuralNetwork& neural_network)
{
#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "Tensor<type, 1> calculate_losses(Index) method.\n"
               << "Number of hidden neurons must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "Tensor<type, 1> calculate_losses(Index) method.\n"
               << "Number of trials must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Neural network

    const Index trainable_layers_number = neural_network.get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network.get_trainable_layers_pointers();

    // Loss index

    type optimum_selection_error = numeric_limits<type>::max();
    type optimum_training_error = numeric_limits<type>::max();
    Tensor<type, 1> optimum_parameters;

    // Optimization algorithm

    OptimizationAlgorithm::Results results;

    Tensor<type, 1> final_losses(2);

    final_losses.setConstant(numeric_limits<type>::max());

    Tensor<type, 1> current_loss(2);

    Tensor<type, 1> final_parameters;

    bool flag_training = false;
    bool flag_selection = false;

    for(Index i = 0; i < neurons_history.size(); i++)
    {
        if(neurons_history[i] == neurons_number)
        {
            final_losses[0] = training_error_history[i];
            flag_training = true;
        }
    }

    for(Index i = 0; i < neurons_history.size(); i++)
    {
        if(neurons_history[i] == neurons_number)
        {
            final_losses[1] = selection_error_history[i];
            flag_selection = true;
        }
    }

    if(flag_training && flag_selection)
    {
        return final_losses;
    }

    trainable_layers_pointers[trainable_layers_number-2]->set_neurons_number(neurons_number); // Fix
    trainable_layers_pointers[trainable_layers_number-1]->set_inputs_number(neurons_number); // Fix

    for(Index i = 0; i < trials_number; i++)
    {
        neural_network.set_parameters_random();

        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy_pointer->perform_training();

        const type current_training_error = optimization_algorithm_results.final_training_error;
        const type current_selection_error = optimization_algorithm_results.final_selection_error;
        const Tensor<type, 1> current_parameters = optimization_algorithm_results.final_parameters;

        if(current_selection_error < optimum_selection_error)
        {
            optimum_training_error = current_training_error;
            optimum_selection_error = current_selection_error;
            optimum_parameters = current_parameters; // Optimum parameters = final_parameters
        }

        if(display)
        {
            cout << "Trial number: " << i << endl;
            cout << "Training error: " << current_training_error << endl;
            cout << "Selection error: " << current_selection_error << endl;
            cout << "Stopping condition: " << optimization_algorithm_results.write_stopping_condition() << endl << endl;
        }
    }

    // Save results

    final_losses[0] = optimum_training_error;
    final_losses[1] = optimum_selection_error;

    neurons_history = insert_result(neurons_number, neurons_history.cast<type>()).cast<Index>();

    training_error_history = insert_result(final_losses(0), training_error_history);

    selection_error_history = insert_result(final_losses(1), selection_error_history);

    parameters_history = insert_result(optimum_parameters, parameters_history);

    return final_losses;
}


Tensor<Index, 1> NeuronsSelection::insert_index_result(const Index& value, const Tensor<Index, 1>& old_tensor) const
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


Tensor<type, 1> NeuronsSelection::insert_result(const type& value, const Tensor<type, 1>& old_tensor) const
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


Tensor< Tensor<type, 1>, 1> NeuronsSelection::insert_result(const Tensor<type, 1>& value, const Tensor< Tensor<type, 1>, 1>& old_tensor) const
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


/// Return a string with the stopping condition of the training depending on the training method.
/// @param results Results of the perform_training method.

string NeuronsSelection::write_stopping_condition(const OptimizationAlgorithm::Results& results) const
{
    return results.write_stopping_condition();
}


/// Delete the history of the selection error values.

void NeuronsSelection::delete_selection_history()
{
    selection_error_history.resize(0);
}


/// Delete the history of the loss values.

void NeuronsSelection::delete_training_error_history()
{

    training_error_history.resize(0);

}


/// Checks that the different pointers needed for performing the neurons selection are not nullptr.

void NeuronsSelection::check() const
{
    // Optimization algorithm

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neural_network_pointer->is_empty())
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw logic_error(buffer.str());
    }


    if(neural_network_pointer->get_layers_number() == 1)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Number of layers in neural network must be greater than 1.\n";

        throw logic_error(buffer.str());
    }

    // Data set

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection samples is zero.\n";

        throw logic_error(buffer.str());
    }

}

/// Writes the time from seconds in format HH:mm:ss.

const string NeuronsSelection::write_elapsed_time(const type& time) const
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


/// Return a string with the stopping condition of the Results

string NeuronsSelection::Results::write_stopping_condition() const
{
    switch(stopping_condition)
    {
        case MaximumTime:
            return "MaximumTime";

        case SelectionErrorGoal:
            return "SelectionErrorGoal";

        case MaximumEpochs:
            return "MaximumEpochs";

        case MaximumSelectionFailures:
            return "MaximumSelectionFailures";

        case AlgorithmFinished:
            return "AlgorithmFinished";
    }

    return string();
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

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "inputs_selection.h"

namespace OpenNN {


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


/// File constructor.

InputsSelection::InputsSelection(const string&)
    : training_strategy_pointer(nullptr)
{
}


/// XML constructor.

InputsSelection::InputsSelection(const tinyxml2::XMLDocument&)
    : training_strategy_pointer(nullptr)
{
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

const size_t& InputsSelection::get_trials_number() const
{
    return trials_number;
}


/// Returns true if the loss index losses are to be reserved, and false otherwise.

const bool& InputsSelection::get_reserve_error_data() const
{
    return(reserve_error_data);
}


/// Returns true if the selection losses are to be reserved, and false otherwise.

const bool& InputsSelection::get_reserve_selection_error_data() const
{
    return(reserve_selection_error_data);
}


/// Returns true if the parameters vector of the neural network with minimum selection error is to be reserved, and false otherwise.

const bool& InputsSelection::get_reserve_minimal_parameters() const
{
    return(reserve_minimal_parameters);
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& InputsSelection::get_display() const
{
    return display;
}


/// Returns the goal for the selection error in the inputs selection algorithm.

const double& InputsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


/// Returns the maximum number of iterations in the inputs selection algorithm.

const size_t& InputsSelection::get_maximum_iterations_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum time in the inputs selection algorithm.

const double& InputsSelection::get_maximum_time() const
{
    return maximum_time;
}


/// Return the maximum correlation for the algorithm.

const double& InputsSelection::get_maximum_correlation() const
{
    return(maximum_correlation);
}


/// Return the minimum correlation for the algorithm.

const double& InputsSelection::get_minimum_correlation() const
{
    return(minimum_correlation);
}


/// Return the tolerance of error for the algorithm.

const double& InputsSelection::get_tolerance() const
{
    return(tolerance);
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

    reserve_error_data = true;
    reserve_selection_error_data = true;
    reserve_minimal_parameters = true;

    display = true;

    // Stopping criteria

    selection_error_goal = 0.0;

    maximum_epochs_number = 1000;

    maximum_correlation = 1.0;
    minimum_correlation = 0.0;

    maximum_time = 10000.0;

    tolerance = 0.0;
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of trials for each set of parameters.

void InputsSelection::set_trials_number(const size_t& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_trials_number(const size_t&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets the reserve flag for the loss data.
/// @param new_reserve_error_data Flag value.

void InputsSelection::set_reserve_error_data(const bool& new_reserve_error_data)
{
    reserve_error_data = new_reserve_error_data;
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

void InputsSelection::set_selection_error_goal(const double& new_selection_error_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_selection_error_goal(const double&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum iterations number for the inputs selection algorithm.
/// @param new_maximum_iterations_number Maximum number of iterations.

void InputsSelection::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
    maximum_epochs_number = new_maximum_iterations_number;
}


/// Sets the maximum time for the inputs selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void InputsSelection::set_maximum_time(const double& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_maximum_time(const double&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Sets the maximum value for the correlations in the inputs selection algorithm.
/// @param new_maximum_correlation Maximum value of the correlations.

void InputsSelection::set_maximum_correlation(const double& new_maximum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_correlation < 0 || new_maximum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_maximum_correlation(const double&) method.\n"
               << "Maximum correlation must be comprised between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_correlation = new_maximum_correlation;
}


/// Sets the minimum value for the correlations in the inputs selection algorithm.
/// @param new_minimum_correlation Minimum value of the correlations.

void InputsSelection::set_minimum_correlation(const double& new_minimum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_correlation < 0 || new_minimum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_minimum_correlation(const double&) method.\n"
               << "Minimum correaltion must be comprised between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_correlation = new_minimum_correlation;
}


/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void InputsSelection::set_tolerance(const double& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_tolerance(const double&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


/// Returns the minimum of the loss and selection error in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelection::calculate_losses(const Vector<bool> & inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Neural network stuff

    NeuralNetwork* neural_network = training_strategy_pointer->get_neural_network_pointer();

    // Optimization algorithm stuff

    OptimizationAlgorithm::Results results;

    double optimum_selection_error = numeric_limits<double>::max();
    double optimum_training_error = numeric_limits<double>::max();

    Vector<double> optimum_parameters;

    Vector<double> optimum_losses(2);

    bool flag_loss = false;
    bool flag_selection = false;

    // Check population

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            optimum_losses[0] = training_error_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            optimum_losses[1] = selection_error_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        if(display)
        {
            cout << "Training loss: " << optimum_losses[0] << endl;
            cout << "Selection error: " << optimum_losses[1] << endl;
            cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
        }
        return optimum_losses;
    }

    neural_network->perturbate_parameters(0.001);

    neural_network->set_inputs_number(inputs);

    for(size_t i = 0; i < trials_number; i++)
    {
        neural_network->randomize_parameters_normal();

        results = training_strategy_pointer->perform_training();

        const double selection_error = results.final_selection_error;
        const double training_error = results.final_training_error;
        const Vector<double> parameters = results.final_parameters;

        if(display && trials_number != 1)
        {
            cout << "Trial number: " << i << endl;
            cout << "Training loss: " << training_error << endl;
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
        cout << "Training loss: " << optimum_training_error << endl;
        cout << "Selection error: " << optimum_selection_error << endl;
        cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
    }

    inputs_history.push_back(inputs);

    training_error_history.push_back(optimum_training_error);

    selection_error_history.push_back(optimum_selection_error);

    parameters_history.push_back(optimum_parameters);

    optimum_losses[0] = optimum_training_error;
    optimum_losses[1] = optimum_selection_error;

    return optimum_losses;
}


/// Returns the mean of the loss and selection error in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelection::perform_mean_model_evaluation(const Vector<bool>&inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_neural_network_pointer();

    OptimizationAlgorithm::Results results;

    Vector<double> mean_final(2);
    mean_final[0] = 0;
    mean_final[1] = 0;

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            mean_final[0] = training_error_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            mean_final[1] = selection_error_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(mean_final);
    }

    neural_network->perturbate_parameters(0.001);

    results = training_strategy_pointer->perform_training();

//    current_loss = get_final_losses(results);

    mean_final[0] = current_loss[0];
    mean_final[1] = current_loss[1];

    final_parameters.set(neural_network->get_parameters());

    for(size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            cout << "Trial number: " << i << endl;
            if(i == 1)
            {
                cout << "Training loss: " << mean_final[0] << endl;
                cout << "Selection error: " << mean_final[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
            }
            else
            {
                cout << "Training loss: " << current_loss[0] << endl;
                cout << "Selection error: " << current_loss[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
            }
        }

        neural_network->randomize_parameters_normal();

        results = training_strategy_pointer->perform_training();

//        current_loss = get_final_losses(results);

        if(!flag_loss)
        {
            mean_final[0] += current_loss[0]/trials_number;
        }

        if(!flag_selection)
        {
            mean_final[1] += current_loss[1]/trials_number;
        }
    }

    if(display)
    {
        cout << "Trial number: " << trials_number << endl;
        cout << "Training loss: " << mean_final[0] << endl;
        cout << "Selection error: " << mean_final[1] << endl;
        cout << "Stopping condition: " << write_stopping_condition(results) << endl << endl;
    }

    inputs_history.push_back(inputs);

    training_error_history.push_back(mean_final[0]);

    selection_error_history.push_back(mean_final[1]);

    parameters_history.push_back(final_parameters);

    return mean_final;
}


/// Returns the parameters of the neural network if the inputs is in the history.
/// @param inputs Vector of inputs to be trained with.

Vector<double> InputsSelection::get_parameters_inputs(const Vector<bool>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "Vector<double> get_parameters_inputs(const Vector<bool>&) method.\n"
               << "Inputs must be greater than 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t i;

    Vector<double> parameters;

    for(i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            parameters = parameters_history[i];

            break;
        }
    }

    return parameters;

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
    selection_error_history.set();
}


/// Delete the history of the loss values.

void InputsSelection::delete_loss_history()
{
    training_error_history.set();
}


/// Delete the history of the parameters of the trained neural networks.

void InputsSelection::delete_parameters_history()
{
    parameters_history.set();
}


/// Checks that the different pointers needed for performing the inputs selection are not nullptr.

void InputsSelection::check() const
{
    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff


    if(!training_strategy_pointer->has_loss_index())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network stuff

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

    // Data set stuff

    if(!loss_index_pointer->has_data_set())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    

    const size_t selection_instances_number = data_set_pointer->get_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

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
        case MaximumIterations:
        {
            return "MaximumIterations";
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


/// Returns a string representation of the current inputs selection results structure.

string InputsSelection::Results::object_to_string() const
{
   ostringstream buffer;

   // Inputs history

   if(!inputs_data.empty())
   {
     buffer << "% Inputs history:\n"
            << inputs_data.to_row_matrix() << "\n";
   }

   // Loss history

   if(!loss_data.empty())
   {
       buffer << "% Loss history:\n"
              << loss_data.to_row_matrix() << "\n";
   }

   // Selection loss history

   if(!selection_error_data.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_error_data.to_row_matrix() << "\n";
   }

   // Minimal parameters

   if(!minimal_parameters.empty())
   {
       buffer << "% Minimal parameters:\n"
              << minimal_parameters << "\n";
   }

   // Stopping condition

   buffer << "% Stopping condition\n"
          << write_stopping_condition() << "\n";

   // Optimum selection error

   if(abs(final_selection_error - 0) > numeric_limits<double>::epsilon())
   {
       buffer << "% Optimum selection error:\n"
              << final_selection_error << "\n";
   }

   // Final training loss

   if(abs(final_training_error - 0) > numeric_limits<double>::epsilon())
   {
       buffer << "% Final training loss:\n"
              << final_training_error << "\n";
   }

   // Optimal input

   if(!optimal_inputs_indices.empty())
   {
       buffer << "% Optimal input:\n"
              << optimal_inputs_indices << "\n";
   }

   // Iterations number


   buffer << "% Number of iterations:\n"
          << iterations_number << "\n";


   // Elapsed time

   buffer << "% Elapsed time:\n"
          << write_elapsed_time(elapsed_time) << "\n";



   return buffer.str();
}


/// Return the index of uses where is the(input_number)-th input.
/// @param uses Vector of the uses of the variables.
/// @param input_number Index of the input to find.

size_t InputsSelection::get_input_index(const Vector<DataSet::VariableUse> uses, const size_t input_number)
{
#ifdef __OPENNN_DEBUG__

    if(uses.size() < input_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "const size_t get_input_index(const Vector<DataSet::VariableUse>, const size_t) method.\n"
               << "Size of uses vector("<< uses.size() <<") must be greater than " <<  input_number << ".\n";

        throw logic_error(buffer.str());
    }
#endif

    size_t i = 0;

    size_t j = 0;

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

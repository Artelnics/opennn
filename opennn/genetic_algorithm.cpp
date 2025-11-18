//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "correlations.h"
#include "dataset.h"
#include "time_series_dataset.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_3d.h"
#include "training_strategy.h"
#include "genetic_algorithm.h"

namespace opennn
{

GeneticAlgorithm::GeneticAlgorithm(const TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
    set_default();
}


const Tensor<bool, 2>& GeneticAlgorithm::get_population() const
{
    return population;
}


const Tensor<type, 1>& GeneticAlgorithm::get_training_errors() const
{
    return training_errors;
}


const Tensor<type, 1>& GeneticAlgorithm::get_selection_errors() const
{
    return selection_errors;
}


const Tensor<type, 1>& GeneticAlgorithm::get_fitness() const
{
    return fitness;
}


const Tensor<bool, 1>& GeneticAlgorithm::get_selection() const
{
    return selection;
}


Index GeneticAlgorithm::get_individuals_number() const
{
    return population.dimension(0);
}


Index GeneticAlgorithm::get_genes_number() const
{
    return original_input_raw_variable_indices.size();
}


const Index& GeneticAlgorithm::get_minimum_inputs_number() const 
{ 
    return minimum_inputs_number; 
}


const Index& GeneticAlgorithm::get_maximum_inputs_number() const
{ 
    return maximum_inputs_number; 
}


const type& GeneticAlgorithm::get_mutation_rate() const
{
    return mutation_rate;
}


const Index& GeneticAlgorithm::get_elitism_size() const
{
    return elitism_size;
}


const string& GeneticAlgorithm::get_initialization_method() const
{
    return initialization_method;
}


void GeneticAlgorithm::set_default()
{
    name = "GeneticAlgorithm";

    if (!training_strategy || !training_strategy->has_neural_network())
        return;

    const Index individuals_number = 40;

    const Dataset* dataset = training_strategy->get_dataset();

    if (!dataset)
        throw runtime_error("Dataset is null");

    original_input_raw_variable_indices = dataset->get_raw_variable_indices("Input");
    original_target_raw_variable_indices = dataset->get_raw_variable_indices("Target");

    const Index genes_number = get_genes_number();

    population.resize(individuals_number, genes_number);

    minimum_inputs_number = 1;
    maximum_inputs_number = genes_number;

    maximum_epochs_number = 100;

    maximum_time = type(3600);

    mutation_rate = type(0.05);

    parameters.resize(individuals_number);

    training_errors.resize(individuals_number);

    selection_errors.resize(individuals_number);

    fitness.resize(individuals_number);
    fitness.setConstant(type(-1.0));

    selection.resize(individuals_number);

    elitism_size = Index(ceil(individuals_number / 4));

    initialization_method = "Correlations";
}


void GeneticAlgorithm::set_minimum_inputs_number(const Index& new_minimum_inputs_number)
{
    minimum_inputs_number = new_minimum_inputs_number;
}


void GeneticAlgorithm::set_maximum_inputs_number(const Index& new_maximum_inputs_number)
{
    const Dataset* dataset = training_strategy ? training_strategy->get_dataset() : nullptr;
    const Index inputs_number = dataset ? dataset->get_raw_variables_number("Input") : 0;

    maximum_inputs_number = (inputs_number == 0)
        ? new_maximum_inputs_number
        : clamp(new_maximum_inputs_number, Index(1), inputs_number);
}


void GeneticAlgorithm::set_population(const Tensor<bool, 2>& new_population)
{
    population = new_population;
}


void GeneticAlgorithm::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void GeneticAlgorithm::set_individuals_number(const Index& new_individuals_number)
{
    if (!training_strategy || !training_strategy->get_dataset())
        throw runtime_error("Training strategy or dataset is null");

    const Index genes_number = get_genes_number();

    population.resize(new_individuals_number, genes_number);
    parameters.resize(new_individuals_number);
    training_errors.resize(new_individuals_number);
    selection_errors.resize(new_individuals_number);
    fitness.resize(new_individuals_number);
    fitness.setConstant(type(-1.0));
    selection.resize(new_individuals_number);

    elitism_size = min(elitism_size, new_individuals_number);
}


void GeneticAlgorithm::set_initialization_method(const string& new_initialization_method)
{
    initialization_method = new_initialization_method;
}


void GeneticAlgorithm::set_mutation_rate(const type& new_mutation_rate)
{
    mutation_rate = clamp(new_mutation_rate, type(0), type(1));
}


void GeneticAlgorithm::set_elitism_size(const Index& new_elitism_size)
{
    elitism_size = clamp<Index>(new_elitism_size, 0, get_individuals_number());
}


void GeneticAlgorithm::set_fitness(const Tensor<type, 1>& new_fitness)
{
    fitness = new_fitness;
}


void GeneticAlgorithm::set_selection(const Tensor<bool, 1>& new_selection)
{
    selection = new_selection;
}


void GeneticAlgorithm::initialize_population()
{
    initialization_method == "Random"
        ? initialize_population_random()
        : initialize_population_correlations();
}


void GeneticAlgorithm::initialize_population_random()
{
    mt19937 gen(rd());

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    population.setConstant(false);

    Tensor<bool, 1> individual_genes(genes_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        individual_genes.setConstant(false);

        const Index true_count = get_random_index(minimum_inputs_number, maximum_inputs_number);

        fill_n(individual_genes.data(), true_count, true);

        shuffle(individual_genes.data(), individual_genes.data() + genes_number, gen);

        population.chip(i, 0) = individual_genes;
    }
}


void GeneticAlgorithm::initialize_population_correlations()
{
    if (minimum_inputs_number > maximum_inputs_number)
        throw runtime_error("GeneticAlgorithm: Minimum inputs number cannot be greater than maximum inputs number.");

    mt19937 gen(rd());

    const Dataset* dataset = training_strategy->get_dataset();

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    Tensor<bool, 1> individual_genes(genes_number);

    population.resize(individuals_number, genes_number);
    population.setConstant(false);

    const Tensor<type, 1> correlations_rank = dataset->calculate_correlations_rank().template cast<type>() + type(1.0);

    const Tensor<type, 0> correlations_sum = correlations_rank.sum();
    const Tensor<type, 1> correlations_cumsum = correlations_rank.cumsum(0);

    const type* begin = correlations_cumsum.data();
    const type* end   = begin + genes_number;

    for (Index i = 0; i < individuals_number; i++)
    {        
        individual_genes.setConstant(false);

        const Index true_count = get_random_index(minimum_inputs_number, maximum_inputs_number);

        while (count(individual_genes.data(), individual_genes.data() + genes_number, true) < true_count)
        {   
            const type arrow = get_random_type(0, correlations_sum());

            const Index j = static_cast<Index>(upper_bound(begin, end, arrow) - begin);

            individual_genes(j) = true;
        }

        if (is_equal(individual_genes, false))
            throw logic_error("All individual genes are false");
        
        population.chip(i, 0) = individual_genes;
    }
}


void GeneticAlgorithm::evaluate_population()
{
    // Training strategy

    TrainingResults training_results;

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Dataset

    Dataset* dataset = training_strategy->get_dataset();
    TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Model selection

    const Index individuals_number = get_individuals_number();

    for(Index i = 0; i < individuals_number; i++)
    {
        const Tensor<bool, 1> individual = population.chip(i, 0);

        if (display) cout << "\nIndividual " << i + 1 << endl;

        const vector<Index> individual_raw_variables_indices = get_raw_variable_indices(individual);

        dataset->set_raw_variable_indices(individual_raw_variables_indices, original_target_raw_variable_indices);

        const Index input_variables_number = dataset->get_variables_number("Input");

        if(time_series_dataset)
        {
            const Index past_time_steps = time_series_dataset->get_past_time_steps();
            neural_network->set_input_dimensions({ past_time_steps, input_variables_number });
            dataset->set_dimensions("Input", { past_time_steps, input_variables_number });

            vector<string> final_feature_names;
            const vector<string> base_names = dataset->get_raw_variable_names("Input");
            const Index time_steps = time_series_dataset->get_past_time_steps();
            final_feature_names.reserve(base_names.size() * time_steps);
            for(const string& base_name : base_names)
            {
                for(Index j = 0; j < time_steps; j++)
                {
                    string name = (base_name.empty() ? "variable" : base_name) + "_lag" + to_string(j);
                    final_feature_names.push_back(name);
                }
            }
            neural_network->set_feature_names(final_feature_names);
        }
        else
        {
            neural_network->set_input_dimensions({input_variables_number});
            dataset->set_dimensions("Input", { input_variables_number });

            neural_network->set_feature_names(dataset->get_variable_names("Input"));
        }

        neural_network->set_parameters_random();

        //Training

        training_strategy->get_loss_index()->set_display(false);
        training_strategy->get_optimization_algorithm()->set_display(false);

        training_results = training_strategy->train();

        neural_network->get_parameters(parameters(i));

        training_errors(i) = training_results.get_training_error();

        selection_errors(i) = training_results.get_selection_error();

        if(display)
            cout << "Training error: " << training_results.get_training_error() << endl
                 << "Selection error: " << training_results.get_selection_error() << endl
                 << "Variables number: " << input_variables_number << endl
                 << "Inputs number: " << dataset->get_raw_variables_number("Input") << endl;

        dataset->set_raw_variable_indices(original_input_raw_variable_indices, original_target_raw_variable_indices);
    }

    const Tensor<type, 0> mean_training_errors = training_errors.mean();
    const Tensor<type, 0> mean_selection_errors = selection_errors.mean();

    mean_training_error = mean_training_errors();
    mean_selection_error = mean_selection_errors();
}


void GeneticAlgorithm::perform_fitness_assignment()
{
    fitness = calculate_rank_greater(selection_errors).cast<type>() + type(1.0);
}


void GeneticAlgorithm::perform_selection()
{
    const Index individuals_number = get_individuals_number();

    const Tensor<type, 0> fitness_sum = fitness.sum();

    selection.setConstant(false);

    const Index individuals_to_be_selected = individuals_number/2;

    if(elitism_size != 0)
        for(Index i = 0; i < individuals_number; i++)
            selection(i) = (fitness(i) - 1 >= 0) && (fitness(i) > (individuals_number - elitism_size));

    const Tensor<type, 1> fitness_cumsum = fitness.cumsum(0);

    const type* begin = fitness_cumsum.data();
    const type* end = begin + individuals_number;

    while (get_selected_individuals_number() < individuals_to_be_selected)
    {
        const type arrow = get_random_type(type(0), fitness_sum());

        const Index i = static_cast<Index>(upper_bound(begin, end, arrow) - begin);

        selection(i) = true;
    }
}


Index GeneticAlgorithm::get_selected_individuals_number() const
{
    return count(selection.data(), selection.data() + selection.size(), 1);
}


vector<Index> GeneticAlgorithm::get_selected_individual_indices() const
{
    const Index selected_individuals_number = get_selected_individuals_number();

    vector<Index> selection_indices(selected_individuals_number);

    Index count = 0;

    for(Index i = 0; i < selection.size(); i++)
        if(selection(i))
            selection_indices[count++] = i;

    return selection_indices;
}


Tensor<bool, 1> GeneticAlgorithm::cross(const Tensor<bool, 1>& parent_1, const Tensor<bool, 1>& parent_2)
{
    /*
    const Index genes_number = get_genes_number();

    Tensor<bool, 1> descendent(genes_number);

    for (Index i = 0; i < genes_number; i++)
        descendent(i) = (parent_1(i) == parent_2(i)) ? parent_1(i) : get_random_bool();

    if (is_equal(descendent, false))
        descendent(get_random_index(0, genes_number - 1)) = true;

    return descendent;
    */
    
    const Index genes_number = get_genes_number();
    Tensor<bool, 1> descendent(genes_number);
    descendent.setConstant(false);
    mt19937 gen(rd());

    vector<Index> intersection, difference;
    for (Index i = 0; i < genes_number; ++i)
        if (parent_1(i) && parent_2(i)) 
            intersection.push_back(i);
        else if (parent_1(i) != parent_2(i))
            difference.push_back(i);

    for (Index idx : intersection)
        descendent(idx) = true;

    Index current_size = intersection.size();

    if (current_size > maximum_inputs_number) 
    {
        shuffle(intersection.begin(), intersection.end(), gen);
        descendent.setConstant(false);
        for(Index i = 0; i < maximum_inputs_number; ++i)
            descendent(intersection[i]) = true;

        return descendent;
    }

    Index target_size = get_random_index(max(minimum_inputs_number, current_size), maximum_inputs_number);

    shuffle(difference.begin(), difference.end(), gen);
    Index genes_to_add = target_size - current_size;
    
    for (Index i = 0; i < genes_to_add && i < difference.size(); ++i)
        descendent(difference[i]) = true;
    
    Index final_count = count(descendent.data(), descendent.data() + genes_number, true);
    if (final_count < minimum_inputs_number) 
    {
        vector<Index> never_true_indices;
        for(Index i = 0; i < genes_number; ++i)
            if(!parent_1(i) && !parent_2(i))
                never_true_indices.push_back(i);

        shuffle(never_true_indices.begin(), never_true_indices.end(), gen);
        Index genes_needed = minimum_inputs_number - final_count;
        for(Index i = 0; i < genes_needed && i < never_true_indices.size(); ++i) {
            descendent(never_true_indices[i]) = true;
        }
    }

    return descendent;
    
}


void GeneticAlgorithm::perform_crossover()
{
    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    Tensor<bool, 2> new_population(individuals_number, genes_number);

    Tensor<bool, 1> parent_1_genes;
    Tensor<bool, 1> parent_2_genes;

    mt19937 g(rd());

    const Index selected_individuals_number = get_selected_individuals_number();

    vector<Index> selected_individual_indices = get_selected_individual_indices();

    if (selected_individuals_number == 0)
        throw logic_error("Cannot perform crossover with zero selected parents.");

    const vector<Index> parent_1_indices = selected_individual_indices;
    vector<Index> parent_2_indices = selected_individual_indices;
    shuffle(parent_2_indices.begin(), parent_2_indices.end(), g);

    Index descendent_index = 0;

    while (descendent_index < individuals_number)
    {
        const Index parent_1_random_index = get_random_element(selected_individual_indices);
        Index parent_2_random_index = get_random_element(selected_individual_indices);

        while (selected_individuals_number > 1 && parent_1_random_index == parent_2_random_index)
            parent_2_random_index = get_random_element(selected_individual_indices);

        const Tensor<bool, 1> parent_1_genes = population.chip(parent_1_random_index, 0);
        const Tensor<bool, 1> parent_2_genes = population.chip(parent_2_random_index, 0);

        new_population.chip(descendent_index, 0) = cross(parent_1_genes, parent_2_genes);
        descendent_index++;
    }

    if(descendent_index != individuals_number)
        throw logic_error("descendent_index != individuals_number");

    population = new_population;
}


void GeneticAlgorithm::perform_mutation()
{
    /*
    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    for (Index i = 0; i < individuals_number; i++)
        for (Index j = 0; j < genes_number; j++)
            if(get_random_type(0, 1) < mutation_rate)
                population(i,j) = !population(i,j);
    */

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();
    mt19937 gen(rd());

    for (Index i = 0; i < individuals_number; ++i)
    {
        Tensor<bool, 1> individual = population.chip(i, 0);
        Index current_inputs_number = count(individual.data(), individual.data() + genes_number, true);

        vector<Index> to_true_mutations; 
        vector<Index> to_false_mutations;

        for (Index j = 0; j < genes_number; ++j)
            if (get_random_type(0.0, 1.0) < mutation_rate)
                if (individual(j))
                    to_false_mutations.push_back(j);
                else
                    to_true_mutations.push_back(j);

        shuffle(to_true_mutations.begin(), to_true_mutations.end(), gen);
        shuffle(to_false_mutations.begin(), to_false_mutations.end(), gen);

        Index swap_count = std::min(to_true_mutations.size(), to_false_mutations.size());
        for (Index k = 0; k < swap_count; ++k)
        {
            individual(to_true_mutations[k]) = true;
            individual(to_false_mutations[k]) = false;
        }

        for (Index k = swap_count; k < to_true_mutations.size(); ++k)
        {
            if (current_inputs_number < maximum_inputs_number)
            {
                individual(to_true_mutations[k]) = true;
                current_inputs_number++;
            }
        }

        for (Index k = swap_count; k < to_false_mutations.size(); ++k)
        {
            if (current_inputs_number > minimum_inputs_number)
            {
                individual(to_false_mutations[k]) = false;
                current_inputs_number--;
            }
        }

        population.chip(i, 0) = individual;
    }
}


InputsSelectionResults GeneticAlgorithm::perform_input_selection()
{
    const LossIndex* loss_index = training_strategy->get_loss_index();

    Dataset* dataset = loss_index->get_dataset();
    TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    // Selection algorithm

    original_input_raw_variable_indices = dataset->get_raw_variable_indices("Input");
    original_target_raw_variable_indices = dataset->get_raw_variable_indices("Target");
    const vector<Index> time_raw_variable_indices = dataset->get_raw_variable_indices("Time");

    InputsSelectionResults input_selection_results(maximum_epochs_number);

    if(display) cout << "Performing genetic inputs selection...\n" << endl;

    initialize_population();

    // Dataset

    if (dataset->has_nan())
        dataset->scrub_missing_values();

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Optimization algorithm

    Index optimal_individual_index;

    Index optimal_individual_training_index;

    bool stop = false;

    time_t beginning_time;

    time_t current_time;

    type elapsed_time = type(0);

    vector<Index> optimal_inputs_raw_variables_indices;

    time(&beginning_time);

    Index generation_selected = 0;

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        if(display) cout << "Generation: " << epoch + 1 << endl;

        input_selection_results.resize_history(input_selection_results.mean_training_error_history.size() + 1);

        evaluate_population();

        // Optimal individual in population

        optimal_individual_index = minimal_index(selection_errors);

        optimal_individual_training_index = minimal_index(training_errors);

        // Store optimal training and selection error in the history

        input_selection_results.training_error_history(epoch) = training_errors(optimal_individual_training_index);

        input_selection_results.selection_error_history(epoch) = selection_errors(optimal_individual_index);

        // Store mean errors histories

        input_selection_results.mean_selection_error_history(epoch) = mean_selection_error;

        input_selection_results.mean_training_error_history(epoch)= mean_training_error;

        if(selection_errors(optimal_individual_index) < input_selection_results.optimum_selection_error)
        {
            generation_selected = epoch;

            dataset->set_raw_variable_indices(original_input_raw_variable_indices, original_target_raw_variable_indices);

            // Neural network

            input_selection_results.optimal_inputs = population.chip(optimal_individual_index, 0);

            optimal_inputs_raw_variables_indices = get_raw_variable_indices(input_selection_results.optimal_inputs);

            dataset->set_raw_variable_indices(optimal_inputs_raw_variables_indices, original_target_raw_variable_indices);

            input_selection_results.optimal_input_raw_variable_names
                = dataset->get_raw_variable_names("Input");

            input_selection_results.optimal_parameters = parameters(optimal_individual_index);

            // Loss index

            input_selection_results.optimum_training_error = training_errors(optimal_individual_training_index);

            input_selection_results.optimum_selection_error = selection_errors(optimal_individual_index);
        }
        else
        {
            dataset->set_raw_variable_indices(original_input_raw_variable_indices,original_target_raw_variable_indices);
        }

        time(&current_time);

        elapsed_time = type(difftime(current_time, beginning_time));

        if(display)
            cout << endl
                 << "Epoch number: " << epoch << endl
                 << "Generation mean training error: " << training_errors.mean() << endl
                 << "Generation mean selection error: " << input_selection_results.mean_selection_error_history(epoch) << endl
                 << "Generation minimum training error: " << training_errors(optimal_individual_training_index) << endl
                 << "Generation minimum selection error: " << selection_errors(optimal_individual_index) << endl
                 << "Best ever training error: " << input_selection_results.optimum_training_error << endl
                 << "Best ever selection error: " << input_selection_results.optimum_selection_error << endl
                 << "Elapsed time: " << write_time(elapsed_time) << endl
                 << "Best selection error in generation: " << generation_selected << endl;

        // Stopping criteria

        stop = true;

        if (input_selection_results.optimum_selection_error <= selection_error_goal)
        {
            if (display) cout << "Epoch " << epoch << "\nSelection error goal reached: " << input_selection_results.optimum_selection_error << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::SelectionErrorGoal;
        }
        else if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << write_time(elapsed_time) << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
        }
        else if (epoch >= maximum_epochs_number - 1)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch + 1 << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
        }
        else
        {
            stop = false;
        }

        if (stop)
        {
            input_selection_results.elapsed_time = write_time(elapsed_time);
            input_selection_results.resize_history(epoch + 1);
            break;
        }

        perform_fitness_assignment();

        perform_selection();

        perform_crossover();

        if(mutation_rate!=0 && epoch > maximum_epochs_number*0.5 && epoch < maximum_epochs_number*0.8)
            perform_mutation();
    }

    // Set dataset stuff

    const vector<Index> optimal_raw_variable_indices = get_raw_variable_indices(input_selection_results.optimal_inputs);

    dataset->set_raw_variable_indices(optimal_raw_variable_indices, original_target_raw_variable_indices);

    const vector<string> input_variable_scalers = dataset->get_variable_scalers("Input");

    const vector<Descriptives> input_variable_descriptives = dataset->calculate_variable_descriptives("Input");

    const Index optimal_variables_number = dataset->get_variables_number("Input");

    // Set neural network stuff

    if(time_series_dataset)
    {
        if(time_raw_variable_indices.size() == 1)
            dataset->set_raw_variable_role(time_raw_variable_indices[0], "Time");

        const Index past_time_steps = time_series_dataset->get_past_time_steps();
        neural_network->set_input_dimensions({ past_time_steps, optimal_variables_number });
        dataset->set_dimensions("Input", { past_time_steps, optimal_variables_number });

        vector<string> final_feature_names;
        const vector<string> base_names = dataset->get_raw_variable_names("Input");
        const Index time_steps = time_series_dataset->get_past_time_steps();
        final_feature_names.reserve(base_names.size() * time_steps);
        for(const string& base_name : base_names)
        {
            for(Index j = 0; j < time_steps; j++)
            {
                string name = (base_name.empty() ? "variable" : base_name) + "_lag" + to_string(j);
                final_feature_names.push_back(name);
            }
        }
        neural_network->set_feature_names(final_feature_names);
    }
    else
    {
        neural_network->set_input_dimensions({optimal_variables_number});
        dataset->set_dimensions("Input", { optimal_variables_number });

        neural_network->set_feature_names(dataset->get_variable_names("Input"));
    }

    if(neural_network->has("Scaling2d"))
    {
        Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"));
        scaling_layer_2d->set_descriptives(input_variable_descriptives);
        scaling_layer_2d->set_scalers(input_variable_scalers);
    }
    else if(neural_network->has("Scaling3d"))
    {
        Scaling3d* scaling_layer_3d = static_cast<Scaling3d*>(neural_network->get_first("Scaling3d"));
        scaling_layer_3d->set_descriptives(input_variable_descriptives);
        scaling_layer_3d->set_scalers(input_variable_scalers);
    }

    neural_network->set_parameters(input_selection_results.optimal_parameters);

    if(display) input_selection_results.print();

    cout << "Selected generation: " << generation_selected << endl;

    return input_selection_results;
}


vector<Index> GeneticAlgorithm::get_raw_variable_indices(const Tensor<bool, 1>& individual_raw_variables)
{
    vector<Index> indices;
    indices.reserve(individual_raw_variables.size());

    for (Index i = 0; i < individual_raw_variables.size(); ++i)
        if (individual_raw_variables(i))
            indices.push_back(original_input_raw_variable_indices[i]);

    if (indices.empty() && !original_input_raw_variable_indices.empty())
        indices.push_back(original_input_raw_variable_indices[0]);

    return indices;
}


Tensor<string, 2> GeneticAlgorithm::to_string_matrix() const
{
    const Index individuals_number = get_individuals_number();

    Tensor<string, 2> string_matrix(6, 2);

    string_matrix.setValues({
    {"Population size", to_string(individuals_number)},
    {"Elitism size", to_string(elitism_size)},
    {"Mutation rate", to_string(mutation_rate)},
    {"Selection loss goal", to_string(selection_error_goal)},
    {"Maximum Generations number", to_string(maximum_epochs_number)},
    {"Maximum time", to_string(maximum_time)}});

    return string_matrix;
}


void GeneticAlgorithm::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("GeneticAlgorithm");

    add_xml_element(printer, "PopulationSize", to_string(get_individuals_number()));
    add_xml_element(printer, "ElitismSize", to_string(elitism_size));
    add_xml_element(printer, "MutationRate", to_string(mutation_rate));
    add_xml_element(printer, "SelectionErrorGoal", to_string(selection_error_goal));
    add_xml_element(printer, "MinimumInputsNumber", to_string(minimum_inputs_number));
    add_xml_element(printer, "MaximumInputsNumber", to_string(maximum_inputs_number));
    add_xml_element(printer, "MaximumGenerationsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();
}


void GeneticAlgorithm::from_XML(const XMLDocument& document)
{
    const XMLElement* root = document.FirstChildElement("GeneticAlgorithm");

    if(!root)
        throw runtime_error("GeneticAlgorithm element is nullptr.\n");

    set_individuals_number(read_xml_index(root, "PopulationSize"));
    set_mutation_rate(read_xml_type(root, "MutationRate"));
    set_elitism_size(read_xml_index(root, "ElitismSize"));
    set_selection_error_goal(read_xml_type(root, "SelectionErrorGoal"));
    set_minimum_inputs_number(read_xml_index(root, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_xml_index(root, "MaximumInputsNumber"));
    set_maximum_epochs_number(read_xml_index(root, "MaximumGenerationsNumber"));
    set_maximum_time(read_xml_type(root, "MaximumTime"));
}


void GeneticAlgorithm::print() const
{
    cout << "Genetic algorithm" << endl
         << "Individuals number: " << get_individuals_number() << endl
         << "Genes number: " << get_genes_number() << endl;
}


void GeneticAlgorithm::save(const filesystem::path& file_name) const
{
    try
    {
        ofstream file(file_name);

        if (file.is_open())
        {
            XMLPrinter printer;
            to_XML(printer);

            file << printer.CStr();

            file.close();
        }
        else
        {
            throw runtime_error("Cannot open file: " + file_name.string());
        }
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
    }
}


void GeneticAlgorithm::load(const filesystem::path& file_name)
{
    set_default();

    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


REGISTER(InputsSelection, GeneticAlgorithm, "GeneticAlgorithm");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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

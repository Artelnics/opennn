//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <random>

#include "tensors.h"
#include "correlations.h"
#include "genetic_algorithm.h"
#include "tinyxml2.h"

namespace opennn
{

GeneticAlgorithm::GeneticAlgorithm()
    : InputsSelection()
{
    set_default();
}


GeneticAlgorithm::GeneticAlgorithm(TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
//    set(new_training_strategy);

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
    return population.dimension(1);
}


const type& GeneticAlgorithm::get_mutation_rate() const
{
    return mutation_rate;
}


const Index& GeneticAlgorithm::get_elitism_size() const
{
    return elitism_size;
}


const GeneticAlgorithm::InitializationMethod& GeneticAlgorithm::get_initialization_method() const
{
    return initialization_method;
}


Tensor<Index, 1> GeneticAlgorithm::get_original_unused_raw_variables()
{
    return original_unused_raw_variables_indices;
}


void GeneticAlgorithm::set_default()
{
    if (!training_strategy)
        return;

    if (!training_strategy->has_neural_network())
        return;

    const Index genes_number = training_strategy->get_data_set()->get_input_and_unused_variables_number();

    Index individuals_number = 40;

    maximum_epochs_number = 100;

    mutation_rate = type(0.0010);

    // Population stuff

    population.resize(individuals_number, genes_number);

    parameters.resize(individuals_number);

    for(Index i = 0; i < individuals_number; i++) {parameters(i).resize(genes_number);}

    training_errors.resize(individuals_number);

    selection_errors.resize(individuals_number);

    fitness.resize(individuals_number);

    fitness.setConstant(type(-1.0));

    selection.resize(individuals_number);

    // Training operators

    elitism_size = Index(ceil(individuals_number / 4));

    set_initialization_method(GeneticAlgorithm::InitializationMethod::Random);
}


void GeneticAlgorithm::set_population(const Tensor<bool, 2>& new_population)
{
    population = new_population;
}


void GeneticAlgorithm::set_genes_number(const Index& new_genes_number)
{
    genes_number = new_genes_number;
}


void GeneticAlgorithm::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void GeneticAlgorithm::set_training_errors(const Tensor<type, 1>& new_training_errors)
{
    training_errors = new_training_errors;
}


void GeneticAlgorithm::set_selection_errors(const Tensor<type, 1>& new_selection_errors)
{
    selection_errors = new_selection_errors;
}


void GeneticAlgorithm::set_fitness(const Tensor<type, 1>& new_fitness)
{
    fitness = new_fitness;
}


void GeneticAlgorithm::set_individuals_number(const Index& new_individuals_number)
{
    const Index new_genes_number = training_strategy->get_data_set()->get_variables_number(DataSet::VariableUse::Input);

    population.resize(new_individuals_number, new_genes_number);

    parameters.resize(new_individuals_number);

    training_errors.resize(new_individuals_number);

    selection_errors.resize(new_individuals_number);

    fitness.resize(new_individuals_number);

    fitness.setConstant(type(-1.0));

    selection.resize(new_individuals_number);

    if(elitism_size > new_individuals_number) elitism_size = new_individuals_number;
}


void GeneticAlgorithm::set_initialization_method(const GeneticAlgorithm::InitializationMethod& new_initialization_method)
{
    initialization_method = new_initialization_method;
}


void GeneticAlgorithm::set_mutation_rate(const type& new_mutation_rate)
{
    mutation_rate = new_mutation_rate;
}


void GeneticAlgorithm::set_elitism_size(const Index& new_elitism_size)
{
    elitism_size = new_elitism_size;
}


void GeneticAlgorithm::initialize_population()
{
    initialization_method == GeneticAlgorithm::InitializationMethod::Random
        ? initialize_population_random()
        : initialize_population_correlations();
}


void GeneticAlgorithm::initialize_population_random()
{
    DataSet* data_set = training_strategy->get_data_set();

    const Index genes_number = data_set->get_input_and_unused_variables_number();

    const Index individuals_number = get_individuals_number();

    population.resize(individuals_number, genes_number);

    // Original inputs, targets and unused index

    original_input_raw_variables_indices = data_set->get_raw_variable_indices(DataSet::VariableUse::Input);

    original_target_raw_variables_indices = data_set->get_raw_variable_indices(DataSet::VariableUse::Target);

    const Tensor<DataSet::RawVariable, 1> raw_variables = data_set->get_raw_variables();

    Index index = 0;

    Index unused_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
        if(raw_variables(i).use == DataSet::VariableUse::None)
            unused_number++;

    original_unused_raw_variables_indices.resize(unused_number);

    for(Index i = 0; i < raw_variables.size(); i++)
        if(raw_variables(i).use == DataSet::VariableUse::None)
            original_unused_raw_variables_indices(index++) = i;

    const Index raw_variables_number = original_input_raw_variables_indices.size() + original_unused_raw_variables_indices.size();

    const Index random_raw_variables_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);

    type percentage = type(1);

    if(random_raw_variables_number > 10000)
        percentage = type(0.1);
    else if(random_raw_variables_number > 5000)
        percentage = type(0.2);
    else if(random_raw_variables_number > 1000)
        percentage = type(0.4);
    else if(random_raw_variables_number > 500)
       percentage = type(0.6);

    // Original inputs raw_variables

    original_input_raw_variables.resize(raw_variables_number);
    original_input_raw_variables.setConstant(false);

    for(Index i = 0; i < original_input_raw_variables_indices.size(); i++)
        original_input_raw_variables(original_input_raw_variables_indices(i)) = true;

    // Initialization a random population

    population.setConstant(false);

    Tensor<bool, 1> individual_raw_variables(raw_variables_number);

    individual_raw_variables.setConstant(false);

    Tensor<bool, 1> individual_variables(genes_number);

    individual_variables.setConstant(false);

    cout << "Creating initial random population" << endl;

    for(Index i = 0; i < individuals_number; i++)
    {
        random_device rd;

        mt19937 g(rd());

        individual_raw_variables.setConstant(false);

        const int upper_limit = int(ceil(random_raw_variables_number * percentage) - 1);
        const int random_number = rand() % upper_limit + 1;

        for(Index j = 0; j < random_number; j++)
            individual_raw_variables(j) = true;

        shuffle(individual_raw_variables.data(), individual_raw_variables.data() + individual_raw_variables.size(), g);

        individual_variables = get_individual_variables(individual_raw_variables);

        if(is_false(individual_variables))
        {
            Tensor<bool, 1> individual_raw_variables_false = get_individual_raw_variables(individual_variables);

            const Tensor<DataSet::RawVariable, 1> raw_variables = data_set->get_raw_variables();

            for(Index j = 0; j < raw_variables_number; j++)
                if(original_input_raw_variables(j))
                    individual_raw_variables_false(j) = true;

            individual_variables = get_individual_variables(individual_raw_variables_false);
        }

        if(is_false(individual_variables))
            for(Index j = 0; j < individual_variables.size(); j++)
                individual_variables(j) = true;

        population.chip(i, 0) = individual_variables;
    }

    cout << "Initial random population created" << endl;

    cout << "Initial random population: \n" << population << endl;
}


void GeneticAlgorithm::calculate_inputs_activation_probabilities() //outdated
{
    DataSet* data_set = training_strategy->get_data_set();

    const Index raw_variables_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);

    const Tensor<Correlation, 2> correlations_matrix = data_set->calculate_input_target_raw_variable_pearson_correlations();

    const Tensor<type, 1> correlations = get_correlation_values(correlations_matrix).chip(0, 1);

    const Tensor<type, 1> correlations_abs = correlations.abs();

    const Tensor<Index, 1> rank = calculate_rank_greater(correlations_abs);

    Tensor<type, 1> fitness_correlations(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        fitness_correlations(rank(i))=type(i+1);

    Tensor<type, 1> probabilities_vector(raw_variables_number);

    for(Index i = 0; i < raw_variables_number ; i++)
        probabilities_vector[i] = type(2) * type(raw_variables_number - fitness_correlations(i) + 1) / (type(raw_variables_number)*type(raw_variables_number+1));

    inputs_activation_probabilities = probabilities_vector.cumsum(0);
}


void GeneticAlgorithm::initialize_population_correlations() // outdated
{
    DataSet* data_set = training_strategy->get_data_set();

    calculate_inputs_activation_probabilities();

    const Index individuals_number = get_individuals_number();

    const Index genes_number = get_genes_number();

    const Index raw_variables_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);

    Tensor<bool, 1> individual_raw_variables(raw_variables_number);

    Tensor<bool, 1> individual_variables(genes_number);

    random_device rd;

    mt19937 gen(rd());

    uniform_real_distribution<> distribution(0, 1);

    Index raw_variables_active;

    type arrow;

    for(Index i = 0; i < individuals_number; i++)
    {
        individual_raw_variables.setConstant(false);

        individual_variables.setConstant(false);

        raw_variables_active = 1 + rand() % raw_variables_number;

        while(std::count(individual_raw_variables.data(), individual_raw_variables.data() + individual_raw_variables.size(), 1) < raw_variables_active)
        {
            arrow = type(distribution(gen));

            if(arrow < inputs_activation_probabilities(0) && !individual_raw_variables(0))
                individual_raw_variables(0) = true;

            for(Index j = 1; j < raw_variables_number; j++)
            {
                if(arrow >= inputs_activation_probabilities(j - 1)
                && arrow < inputs_activation_probabilities(j)
                && !individual_raw_variables(j))
                    individual_raw_variables(j) = true;
            }
        }

        if(is_false(individual_raw_variables)) individual_raw_variables(rand()%raw_variables_number) = true;

        individual_variables = get_individual_variables(individual_raw_variables);

        for(Index j = 0; j < genes_number; j++)
            population(i, j) = individual_variables(j);
    }

    cout << "Initial population: \n" << population << endl;
}


type GeneticAlgorithm::generate_random_between_0_and_1()
{
    return type(rand()) / type(RAND_MAX);
}


void GeneticAlgorithm::evaluate_population()
{
    // Training strategy

    TrainingResults training_results;

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Data set

    DataSet* data_set = training_strategy->get_data_set();

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Optimization algorithm

    Tensor<bool, 1> individual; //.

    // Model selection

    const Index individuals_number = get_individuals_number();

    Tensor<bool, 1> individual_raw_variables;

    Tensor<Index, 1> individual_raw_variables_indexes;

    Tensor<Index, 1> inputs_number(individuals_number);

    Tensor<string, 1> input_names;

    for(Index i = 0; i < individuals_number; i++)
    {
        individual = population.chip(i, 0);

        cout << endl << "Individual " << i + 1 << endl;

        individual_raw_variables_indexes = get_individual_as_raw_variables_indexes_from_variables(individual);

        inputs_number(i) = individual_raw_variables_indexes.size();

        // Neural network

        data_set->set_input_target_raw_variables_indices(individual_raw_variables_indexes, original_target_raw_variables_indices);

        data_set->scrub_missing_values();

        input_names = data_set->get_variable_names(DataSet::VariableUse::Input);

        neural_network->set_inputs_number(data_set->get_variables_number(DataSet::VariableUse::Input));

        neural_network->set_inputs_names(input_names);

        neural_network->set_parameters_random();

        //Training

        training_results = training_strategy->perform_training();

        parameters(i) = neural_network->get_parameters();

        training_errors(i) = type(training_results.get_training_error());

        selection_errors(i) = type(training_results.get_selection_error());

        if(display)
            cout << "Training error: " << training_results.get_training_error() << endl
                 << "Selection error: " << training_results.get_selection_error() << endl
                 << "Variables number: " << input_names.size() << endl
                 << "Inputs number: " << data_set->get_raw_variables_number(DataSet::VariableUse::Input) << endl;

        data_set->set_input_target_raw_variables_indices(original_input_raw_variables_indices, original_target_raw_variables_indices);
    }

    // Mean generational selection and training error calculation (primitive way)

    type sum_training_errors = type(0);

    type sum_selection_errors = type(0);

    type sum_inputs_number = type(0);

    for(Index i = 0; i < individuals_number; i++)
    {
        sum_training_errors += training_errors(i);

        sum_selection_errors += selection_errors(i);

        sum_inputs_number += type(inputs_number(i));
    }

    mean_training_error = (type(sum_training_errors) / type(individuals_number));

    mean_selection_error = (type(sum_selection_errors) / type(individuals_number));

    mean_inputs_number = (type(sum_inputs_number)/type(individuals_number));
}


void GeneticAlgorithm::perform_fitness_assignment()
{
    const Index individuals_number = get_individuals_number();

    const Tensor<Index, 1> rank = calculate_rank_less(selection_errors);

    for(Index i = 0; i < individuals_number; i++)
        fitness(rank(i)) = type(i+1);
}


Tensor<type, 1> GeneticAlgorithm::calculate_selection_probabilities()
{
    const Index individuals_number = get_individuals_number();

    Tensor<type, 1> selection_probabilities(individuals_number);

    // Calculation of cumulative probabilities

    Index sum_from_1_to_n = 0;

    for(Index i = 0; i < individuals_number; i++)
        sum_from_1_to_n += (i+1);

    Tensor<type, 1> probabilities(individuals_number);

    for(Index i = 0; i < individuals_number; i++)
        probabilities(i) = (type(individuals_number) - type(fitness(i) - 1)) / sum_from_1_to_n;

    selection_probabilities = probabilities.cumsum(0);

    return probabilities;
}


void GeneticAlgorithm::perform_selection()
{
    const Index individuals_number = get_individuals_number();

    selection.setConstant(false);

    const Index selected_individuals_number = Index(type(individuals_number)/type(2));

    const Tensor<type, 1> selection_probabilities = calculate_selection_probabilities();

    if(elitism_size != 0)
        for(Index i = 0; i < individuals_number; i++)
            if(fitness(i) - 1 >= 0 && fitness(i) - 1 < elitism_size)
                selection(i) = true;

    // The next individuals are selected randomly but their probability is set according to their fitness.

    while(std::count(selection.data(), selection.data() + selection.size(), 1) < selected_individuals_number)
    {
        weighted_random(selection_probabilities);
    }
}


Tensor<Index,1> GeneticAlgorithm::get_selected_individuals_indices()
{
    Tensor<Index,1> selection_indexes(std::count(selection.data(), selection.data() + selection.size(), 1));
    Index activated_index_count = 0;

    for(Index i = 0; i < selection.size(); i++)
    {
        if(selection(i))
        {
            selection_indexes(activated_index_count) = i;

            activated_index_count++;
        }
    }

    return selection_indexes;
}


void GeneticAlgorithm::perform_crossover()
{

    DataSet* data_set = training_strategy->get_data_set();

    const Index individuals_number = get_individuals_number();

    const Index genes_number = get_genes_number();

    const Index raw_variables_number = original_input_raw_variables_indices.size() + original_unused_raw_variables_indices.size();

    // Couples generation

    Tensor<bool, 2> new_population(individuals_number, genes_number);

    Tensor<bool, 1> parent_1_variables;

    Tensor<bool, 1> parent_2_variables;

    Tensor<bool,1> descendent_variables;

    Tensor<bool,1 > descendent_genes(genes_number);

    Tensor<bool, 1> descendent_raw_variables(raw_variables_number);

    Tensor<bool, 1> parent_1_raw_variables;

    Tensor<bool, 1> parent_2_raw_variables;

    random_device rd;

    mt19937 g(rd());

    Tensor<Index, 1> parent_1_indices = get_selected_individuals_indices();

    std::shuffle(parent_1_indices.data(), parent_1_indices.data() + parent_1_indices.size(), g);

    Tensor<Index, 1> parent_2_indices = get_selected_individuals_indices();

    Index descendent_index = 0;

    for(Index i = 0; i < parent_1_indices.size(); i++)
    {
        parent_1_variables = population.chip(parent_1_indices(i), 0);

        parent_1_raw_variables = get_individual_raw_variables(parent_1_variables);

        parent_2_variables = population.chip(parent_2_indices(i), 0);

        descendent_raw_variables = get_individual_raw_variables(parent_2_variables);

        for(Index j = 0; j < 2; j++)
        {
            descendent_raw_variables = parent_1_raw_variables;

            for(Index k = 0; k < raw_variables_number; k++)
                if(parent_1_raw_variables(k) != parent_2_raw_variables(k))
                    descendent_raw_variables(k) = calculate_random_bool();

            descendent_genes = get_individual_variables(descendent_raw_variables);

            if(is_false(descendent_genes))
            {
                const Tensor<DataSet::RawVariable, 1> raw_variables = data_set->get_raw_variables();

                Tensor<bool, 1> individual_raw_variables_false = get_individual_raw_variables(descendent_genes);

                for(Index j = 0; j < raw_variables_number; j++)
                    if(original_input_raw_variables(j))
                        individual_raw_variables_false(j) = true;

                descendent_genes = get_individual_variables(individual_raw_variables_false);
            }

            if(is_false(descendent_genes))
                for(Index j = 0; j < descendent_genes.size(); j++)
                    descendent_genes(j) = true;

            new_population.chip(descendent_index, 0) = descendent_genes;
            descendent_index++;
        }
    }

    population = new_population;

    cout << "population:\n" << population << endl;
}


void GeneticAlgorithm::perform_mutation()
{
    const Index individuals_number = get_individuals_number();

    const Index raw_variables_number = original_input_raw_variables_indices.size() + original_unused_raw_variables_indices.size();

    const Index genes_number = get_genes_number();

    Tensor<bool, 1> individual_variables(genes_number);

    Tensor<bool, 1> new_individual_variables(genes_number);

    Tensor<bool, 1> individual_raw_variables(raw_variables_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        individual_variables = population.chip(i, 0);

        individual_raw_variables = get_individual_raw_variables(individual_variables);

        for(Index j = 0; j < raw_variables_number; j++)
        {
            const type random_0_1 = generate_random_between_0_and_1();

            if(random_0_1 < mutation_rate)
                individual_raw_variables(j) = !individual_raw_variables(j);
        }

        new_individual_variables = get_individual_variables(individual_raw_variables);

        if(is_false(new_individual_variables))
        {
            Tensor<bool, 1> individual_raw_variables_false = get_individual_raw_variables(new_individual_variables);

            const Tensor<DataSet::RawVariable, 1> raw_variables = training_strategy->get_data_set()->get_raw_variables();

            for(Index j = 0; j < raw_variables_number; j++)
                if(original_input_raw_variables(j))
                    individual_raw_variables_false(j) = true;

            new_individual_variables = get_individual_variables(individual_raw_variables_false);
        }

        if(is_false(new_individual_variables))
            for(Index j = 0; j < new_individual_variables.size(); j++)
                new_individual_variables(j) = true;

        population.chip(i, 0) = new_individual_variables;
    }
}


InputsSelectionResults GeneticAlgorithm::perform_inputs_selection()
{
    if(display) cout << "Performing genetic inputs selection..." << endl << endl;

    initialize_population();

    // Selection algorithm

    InputsSelectionResults inputs_selection_results(maximum_epochs_number);

    // Training strategy

    training_strategy->set_display(false);

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    // Neural network0

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Optimization algorithm

    Index optimal_individual_index;

    Index optimal_individual_training_index;

    bool stop = false;

    time_t beginning_time;

    time_t current_time;

    type elapsed_time = type(0);

    Tensor<Index, 1> optimal_inputs_raw_variables_indexes;

    std::time(&beginning_time);

    inputs_selection_results.optimum_selection_error = numeric_limits<type>::max();

    training_strategy->get_optimization_algorithm()->set_display(false);

    Index generation_selected = 0;

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        if(display) cout << "Generation: " << epoch + 1 << endl;

        inputs_selection_results.resize_history(inputs_selection_results.mean_training_error_history.size() + 1);

        evaluate_population();

        // Optimal individual in population

        optimal_individual_index = minimal_index(selection_errors);

        optimal_individual_training_index = minimal_index(training_errors);

        // Store optimal training and selection error in the history

        inputs_selection_results.training_error_history(epoch) = training_errors(optimal_individual_training_index);

        inputs_selection_results.selection_error_history(epoch) = selection_errors(optimal_individual_index);

        // Store mean errors histories

        inputs_selection_results.mean_selection_error_history(epoch) = mean_selection_error;

        inputs_selection_results.mean_training_error_history(epoch)= mean_training_error;

        if(selection_errors(optimal_individual_index) < inputs_selection_results.optimum_selection_error)
        {
            generation_selected = epoch;

            data_set->set_input_target_raw_variables_indices(original_input_raw_variables_indices, original_target_raw_variables_indices);

            // Neural network

            inputs_selection_results.optimal_inputs = population.chip(optimal_individual_index, 0);

            optimal_inputs_raw_variables_indexes = get_individual_as_raw_variables_indexes_from_variables(inputs_selection_results.optimal_inputs);

            data_set->set_input_target_raw_variables_indices(optimal_inputs_raw_variables_indexes, original_target_raw_variables_indices);

            inputs_selection_results.optimal_input_raw_variables_names 
                = data_set->get_raw_variable_names(DataSet::VariableUse::Input);

            inputs_selection_results.optimal_parameters = parameters(optimal_individual_index);

            // Loss index

            inputs_selection_results.optimum_training_error = training_errors(optimal_individual_training_index);

            inputs_selection_results.optimum_selection_error = selection_errors(optimal_individual_index);
        }
        else
        {
            data_set->set_input_target_raw_variables_indices(original_input_raw_variables_indices,original_target_raw_variables_indices);
        }

        data_set->set_input_target_raw_variables_indices(original_input_raw_variables_indices, original_target_raw_variables_indices);

        std::time(&current_time);

        elapsed_time = type(difftime(current_time, beginning_time));

        if(display)
            cout << endl
                 << "Epoch number: " << epoch << endl
                 << "Generation mean training error: " << training_errors.mean() << endl
                 << "Generation mean selection error: " << inputs_selection_results.mean_selection_error_history(epoch) << endl
                 << "Mean inputs number  " << mean_inputs_number << endl
                 << "Generation minimum training error: " << training_errors(optimal_individual_training_index) << endl
                 << "Generation minimum selection error: " << selection_errors(optimal_individual_index) << endl
                 << "Best ever training error: " << inputs_selection_results.optimum_training_error << endl
                 << "Best ever selection error: " << inputs_selection_results.optimum_selection_error << endl
                 << "Elapsed time: " << write_time(elapsed_time) << endl
                 << "Best selection error in generation: " << generation_selected << endl;

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            stop = true;

            if(display) cout << "Epoch " << epoch << endl << "Maximum time reached: " << write_time(elapsed_time) << endl;

            inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
        }

        if(epoch >= maximum_epochs_number - 1)
        {
            stop = true;

            if(display) cout << "Epoch " << epoch << endl << "Maximum number of epochs reached: " << epoch << endl;

            inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
        }

        if(stop)
        {
            inputs_selection_results.elapsed_time = write_time(elapsed_time);

            inputs_selection_results.resize_history(epoch + 1);

            break;
        }

        perform_fitness_assignment();

        perform_selection();

        perform_crossover();

        if(mutation_rate!=0 && epoch > maximum_epochs_number*0.5 && epoch < maximum_epochs_number*0.8) 
            perform_mutation();
    }

    // Set data set stuff

    Tensor<Index, 1> optimal_raw_variables = get_individual_as_raw_variables_indexes_from_variables(inputs_selection_results.optimal_inputs);

    data_set->set_input_target_raw_variables_indices(optimal_raw_variables, original_target_raw_variables_indices);

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Input);

    const Tensor<Descriptives, 1> input_variables_descriptives = data_set->calculate_variables_descriptives(DataSet::VariableUse::Input);

    // Set neural network stuff

    neural_network->set_inputs_number( data_set->get_variables_number(DataSet::VariableUse::Input));

    neural_network->set_inputs_names(data_set->get_variable_names(DataSet::VariableUse::Input));

    if(neural_network->has(Layer::Type::Scaling2D))
    {
        ScalingLayer2D* scaling_layer_2d =   neural_network->get_scaling_layer_2d();
        scaling_layer_2d->set_descriptives(input_variables_descriptives);
        scaling_layer_2d->set_scalers(input_variables_scalers);
    }

    neural_network->set_parameters(inputs_selection_results.optimal_parameters);

    if(display) inputs_selection_results.print();

    return inputs_selection_results;
}


void GeneticAlgorithm::check_categorical_raw_variables()
{
    TrainingStrategy* training_strategy = get_training_strategy();

    DataSet* data_set = training_strategy->get_data_set();

    const Index individuals_number = get_individuals_number();

    const Index variables_number = data_set->get_variables_number(DataSet::VariableUse::Input);

    Index raw_variable_index = 0;

    if(data_set->has_categorical_raw_variables())
    {
        for(Index i = 0; i < variables_number; i++)
        {
            const DataSet::RawVariableType column_type = data_set->get_raw_variable_type(raw_variable_index);

            if(column_type != DataSet::RawVariableType::Categorical)
            {
                raw_variable_index++;
                continue;
            }

            const Index categories_number = data_set->get_raw_variables()(raw_variable_index).get_categories_number();

            for(Index j = 0; j < individuals_number; j++)
            {
                const Tensor<bool, 1> individual = population.chip(j, 0);

                if(!(find(individual.data() + i, individual.data() + i + categories_number, 1) == individual.data() + i + categories_number))
                {
                    const Index random_index = rand() % categories_number;

                    for(Index categories_index = 0; categories_index < categories_number; categories_index++)
                        population(j, i + categories_index) = false;

                    population(j, i + random_index) = true;
                }
            }

            i += categories_number - 1;
            raw_variable_index++;
        }
    }
}


Tensor<bool,1 > GeneticAlgorithm::get_individual_raw_variables(Tensor<bool,1>& individual) // upadted
{
    DataSet* data_set = training_strategy->get_data_set();

    const Index raw_variables_number = original_input_raw_variables_indices.size() + original_unused_raw_variables_indices.size();

    Tensor<bool, 1> raw_variables_from_variables(raw_variables_number);

    raw_variables_from_variables.setConstant(false);

    Index genes_count = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(data_set->get_raw_variable_type(i) == DataSet::RawVariableType::Categorical)
        {
            const Index categories_number = data_set->get_raw_variables()(i).get_categories_number();

            if(individual(genes_count))
                raw_variables_from_variables(i) = true;

            genes_count += categories_number;
        }
        else
        {
            raw_variables_from_variables(i) = individual(genes_count);
            genes_count++;
        }
    }

    return raw_variables_from_variables;
}


Tensor<Index, 1> GeneticAlgorithm::get_individual_as_raw_variables_indexes_from_variables(Tensor<bool, 1>& individual) // updated
{
    Tensor<bool, 1> individual_raw_variables = get_individual_raw_variables(individual);

    Tensor<bool, 1> inputs_pre_indexes(individual_raw_variables.size());
    inputs_pre_indexes.setConstant(false);

    Index original_input_index = 0;

    for(Index i = 0; i < original_input_raw_variables.size(); i++)
    {
        if(individual_raw_variables(i) && original_input_raw_variables(i))
        {
            inputs_pre_indexes(i) = true;

            original_input_index = i;
        }
    }

    Index indexes_dimension = 0;

    for(Index i = 0; i < individual_raw_variables.size(); i++)
        if(inputs_pre_indexes(i))
            indexes_dimension++;

    if(is_false(inputs_pre_indexes))
    {
        cout << "/." << endl;
        inputs_pre_indexes(original_input_index) = true;
    }

    Index index = 0;

    Tensor<Index ,1> indexes(indexes_dimension);

    for(Index i = 0; i < individual_raw_variables.size(); i++)
        if(inputs_pre_indexes(i))
            indexes(index) = i;

    return indexes;
}


Tensor<bool, 1> GeneticAlgorithm::get_individual_variables(Tensor<bool, 1>& individual_raw_variables) 
{
    DataSet* data_set = training_strategy->get_data_set();

    const Index genes_number = data_set->get_input_and_unused_variables_number();
    const Index raw_variables_number = individual_raw_variables.size();

    Tensor<bool, 1> individual_raw_variables_to_variables(genes_number);
    individual_raw_variables_to_variables.setConstant(false);

    const Tensor<DataSet::RawVariable, 1> raw_variables = data_set->get_raw_variables();

    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(individual_raw_variables(i))
        {
            if(raw_variables(i).type == DataSet::RawVariableType::Categorical)
            {
                const Index categories_number = data_set->get_raw_variables()(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                    individual_raw_variables_to_variables(variable_index + j) = true;

                variable_index += categories_number;
            }
            else
            {
                individual_raw_variables_to_variables(variable_index) = true;
                variable_index++;
            }
        }
        else
        {
            if(raw_variables(i).type == DataSet::RawVariableType::Categorical)
            {
                const Index categories_number = data_set->get_raw_variables()(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                    individual_raw_variables_to_variables(variable_index + j) = false;

                variable_index += categories_number;
            }
            else
            {
                variable_index++;
            }
        }
    }

    // None variables (no set unused initial raw_variables as inputs)

    Tensor<bool, 1> individual_raw_variables_to_variables_returned(genes_number);
    individual_raw_variables_to_variables_returned.setConstant(false);

    Tensor<bool, 1> original_inputs_variables(genes_number);
    original_inputs_variables.setConstant(false);

    Index unused_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(original_input_raw_variables(i))
        {
            if(raw_variables(i).type == DataSet::RawVariableType::Categorical)
            {
                const Index categories_number = data_set->get_raw_variables()(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                    original_inputs_variables(unused_index + j) = true;

                unused_index += categories_number;
            }
            else
            {
                original_inputs_variables(unused_index) = true;
                unused_index++;
            }
        }
        else
        {
            if(raw_variables(i).type == DataSet::RawVariableType::Categorical)
            {
                const Index categories_number = data_set->get_raw_variables()(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                    original_inputs_variables(unused_index + j) = false;

                unused_index += categories_number;
            }
            else
            {
                unused_index++;
            }
        }
    }

    for(Index i = 0; i < genes_number; i++)
        if(individual_raw_variables_to_variables(i) && original_inputs_variables(i))
            individual_raw_variables_to_variables_returned(i) = true;

    return individual_raw_variables_to_variables_returned;
}


// Tensor<bool, 1> GeneticAlgorithm::get_individual_variables_to_indexes(Tensor<bool, 1>& individual)
// {
//     DataSet* data_set = training_strategy->get_data_set();

//     const Index raw_variables_number = data_set->get_raw_variables_number(VariableUse::Input);

//     Tensor<bool, 1> new_indexes(individual);

//     Index variable_index = 0;

//     if(data_set->has_categorical_raw_variables())
//     {
//         for(Index i = 0; i < raw_variables_number; i++)
//         {
//             if(data_set->get_raw_variable_type(i) == DataSet::RawVariableType::Categorical)
//             {
//                 const Index categories_number = data_set->get_raw_variables()(i).get_categories_number();

//                 if(!(find(individual.data() + variable_index, individual.data() + variable_index + categories_number, 1) == individual.data() + variable_index + categories_number))
//                 {
//                     new_indexes(i) = true;
//                 }
//                 else
//                 {
//                     new_indexes(i) = false;
//                 }
//                 variable_index += categories_number;
//             }
//             else
//             {
//                 new_indexes(i) = individual(variable_index);

//                 variable_index++;
//             }
//         }
//     }

//     return new_indexes;
// }


Tensor<string, 2> GeneticAlgorithm::to_string_matrix() const
{
    const Index individuals_number = get_individuals_number();

    Tensor<string, 1> labels(6);
    Tensor<string, 1> values(6);

    Tensor<string, 2> string_matrix(labels.size(), 2);

    labels(0) = "Population size";
    values(0) = to_string(individuals_number);

    labels(1) = "Elitism size";
    values(1) = to_string(elitism_size);

    labels(2) = "Mutation rate";
    values(2) = to_string(mutation_rate);

    labels(3) = "Selection loss goal";
    values(3) = to_string(selection_error_goal);

    labels(4) = "Maximum Generations number";
    values(4) = to_string(maximum_epochs_number);

    labels(5) = "Maximum time";
    values(5) = to_string(maximum_time);

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

    return string_matrix;
}


Index GeneticAlgorithm::weighted_random(const Tensor<type, 1>& weights) //¿void?
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0, 1);

    type randomNumber = type(dist(gen));

    type sum = type(0);

    for(Index i = 0; i < weights.size(); i++)
    {
       sum += weights(i);

       if(randomNumber <= sum && !selection(i))
       {
           selection(i) = true;
           return i;
       }
    }

    return -1;
}


void GeneticAlgorithm::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("GeneticAlgorithm");

    add_xml_element(printer, "PopulationSize", to_string(get_individuals_number()));
    add_xml_element(printer, "ElitismSize", to_string(elitism_size));
    add_xml_element(printer, "MutationRate", to_string(mutation_rate));
    add_xml_element(printer, "SelectionErrorGoal", to_string(selection_error_goal));
    add_xml_element(printer, "MaximumGenerationsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();
}


void GeneticAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root = document.FirstChildElement("GeneticAlgorithm");

    if(!root)
        throw runtime_error("GeneticAlgorithm element is nullptr.\n");
/*
    set_individuals_number(read_xml_value(root, "PopulationSize", get_individuals_number()));

    set_mutation_rate(read_xml_value(root, "MutationRate", type(0.01)));

    set_elitism_size(read_xml_value(root, "ElitismSize", elitism_size));

    set_selection_error_goal(read_xml_value(root, "SelectionErrorGoal", selection_error_goal));

    set_maximum_epochs_number(read_xml_value(root, "MaximumGenerationsNumber", maximum_epochs_number));

    set_maximum_correlation(read_xml_value(root, "MaximumCorrelation", maximum_correlation));

    set_minimum_correlation(read_xml_value(root, "MinimumCorrelation", minimum_correlation));

    set_maximum_time(read_xml_value(root, "MaximumTime", maximum_time));

    set_display(read_xml_value(root, "Display", true));
*/
}


void GeneticAlgorithm::print() const
{
    cout << "Genetic algorithm" << endl
         << "Individuals number: " << get_individuals_number() << endl
         << "Genes number: " << get_genes_number() << endl;
}


void GeneticAlgorithm::save(const string& file_name) const
{
    try
    {
         std::ofstream file(file_name);

        if (file.is_open())
        {
            tinyxml2::XMLPrinter printer;
            to_XML(printer);

            file << printer.CStr();

            file.close();
        }
        else
        {
            throw runtime_error("Cannot open file: " + file_name);
        }
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
    }
}


void GeneticAlgorithm::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

    from_XML(document);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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

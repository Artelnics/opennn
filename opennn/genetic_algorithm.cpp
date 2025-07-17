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
#include "scaling_layer_2d.h"
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


Index GeneticAlgorithm::get_raw_variables_number() const
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


void GeneticAlgorithm::set_default()
{
    if (!training_strategy || !training_strategy->has_neural_network())
        return;

    const Index individuals_number = 40;        // Modify depending on the Dataset

    const Index raw_variables_number = training_strategy->get_dataset()->get_raw_variables_number("Input");

    population.resize(individuals_number, raw_variables_number);

    maximum_epochs_number = 40;                 // Modify depending on the Dataset

    maximum_time = type(3600);

    mutation_rate = type(0.0010);

    parameters.resize(individuals_number);

    training_errors.resize(individuals_number);

    selection_errors.resize(individuals_number);

    fitness.resize(individuals_number);
    fitness.setConstant(type(-1.0));

    selection.resize(individuals_number);

    elitism_size = Index(ceil(individuals_number / 4));

    initialization_method = GeneticAlgorithm::InitializationMethod::Correlations;
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
        throw runtime_error("Training strategy or data set is null");

    const Index new_raw_variables_number = training_strategy->get_dataset()->get_raw_variables_number("Input");

    population.resize(new_individuals_number, new_raw_variables_number);
    parameters.resize(new_individuals_number);
    training_errors.resize(new_individuals_number);
    selection_errors.resize(new_individuals_number);
    fitness.resize(new_individuals_number);
    fitness.setConstant(type(-1.0));
    selection.resize(new_individuals_number);

    elitism_size = min(elitism_size, new_individuals_number);
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
    Dataset* dataset = training_strategy->get_dataset();

    const Index individuals_number = get_individuals_number();

    const Index original_input_raw_variables_number = original_input_raw_variable_indices.size();

    const Index random_raw_variables_number = dataset->get_raw_variables_number("Input");

    const type percentage = (random_raw_variables_number > 10000) ? type(0.1) :
                            (random_raw_variables_number >  5000) ? type(0.2) :
                            (random_raw_variables_number >  1000) ? type(0.4) :
                            (random_raw_variables_number >   500) ? type(0.6) :
                            type(1);

    original_input_raw_variables.resize(original_input_raw_variables_number, false);

    for(size_t i = 0; i < original_input_raw_variable_indices.size(); i++)
        original_input_raw_variables[original_input_raw_variable_indices[i]] = true;

    population.setConstant(false);

    Tensor<bool, 1> individual_raw_variables(original_input_raw_variables_number);
    individual_raw_variables.setConstant(false);

    cout << "Creating initial random population" << endl;

    const int upper_limit = int(ceil(random_raw_variables_number * percentage) - 1);

    random_device rd;

    mt19937 gen(rd());

    uniform_int_distribution<> dist(1, upper_limit);

    for(Index i = 0; i < individuals_number; i++)
    {
        individual_raw_variables.setConstant(false);

        const int random_number = dist(gen);

        fill_n(individual_raw_variables.data(), random_number, true);

        shuffle(individual_raw_variables.data(), individual_raw_variables.data() + individual_raw_variables.size(), gen);

        if(is_equal(individual_raw_variables, false))
            for(Index j = 0; j < original_input_raw_variables_number; j++)
                if(original_input_raw_variables[j])
                    individual_raw_variables(j) = true;

        if(is_equal(individual_raw_variables, false))
            individual_raw_variables.setConstant(true);

        population.chip(i, 0) = individual_raw_variables;
    }

    cout << "Initial random population created" << endl
         << "Initial random population: \n" << population << endl;
}


void GeneticAlgorithm::initialize_population_correlations()
{
    Dataset* dataset = training_strategy->get_dataset();

    const Index original_input_raw_variables_number = original_input_raw_variable_indices.size();

    const Index raw_variables_number = dataset->get_raw_variables_number("Input");

    fitness_correlations.resize(raw_variables_number);

    const Tensor<Correlation, 2> correlations
        = dataset->calculate_input_target_raw_variable_pearson_correlations();

    const Tensor<type, 2> absolute_correlations = get_correlation_values(correlations).abs();

    Tensor<type, 1> absolute_mean_correlations(absolute_correlations.dimension(0));

    if(absolute_correlations.dimension(1) > 1)
        for(Index i = 0; i < absolute_correlations.dimension(0); i++)
        {
            const Tensor<type, 1> row_correlations = absolute_correlations.chip(i,0);
            absolute_mean_correlations(i) = mean(row_correlations);
        }
    else
        absolute_mean_correlations = absolute_correlations.chip(0,1);

   const Tensor<Index, 1> rank = calculate_rank_less(absolute_mean_correlations);

    for(Index i = 0; i < raw_variables_number; i++)
        fitness_correlations(rank(i)) = type(i+1);

    const Tensor<type,0> fitness_correlations_sum = fitness_correlations.sum();

    const Tensor<type,1> fitness_correlations_cumsum = fitness_correlations.cumsum(0);

    const Index individuals_number = get_individuals_number();

    const Index input_raw_variables_number = dataset->get_raw_variables_number("Input");

    Tensor<bool, 1> individual_raw_variables(input_raw_variables_number);

    original_input_raw_variables.resize(original_input_raw_variables_number, false);

    for(size_t i = 0; i < original_input_raw_variable_indices.size(); i++)
        original_input_raw_variables[original_input_raw_variable_indices[i]] = true;

    population.setConstant(false);

    random_device rd;

    mt19937 gen(rd());

    uniform_real_distribution<> distribution(0, fitness_correlations_sum(0));

    Index raw_variables_active;

    type arrow;

    for(Index i = 0; i < individuals_number; i++)
    {
        individual_raw_variables.setConstant(false);
        
        raw_variables_active = input_raw_variables_number > 100
                                   ? 1 + rand() % 100
                                   : 1 + rand() % input_raw_variables_number;

        while(count(individual_raw_variables.data(), individual_raw_variables.data() + individual_raw_variables.size(), 1) < raw_variables_active)
        {
            arrow = distribution(gen);

            if(arrow < fitness_correlations_cumsum(0) && !individual_raw_variables(0))
                individual_raw_variables(0) = true;

            for(Index j = 1; j < input_raw_variables_number; j++)
                if(arrow >= fitness_correlations_cumsum(j - 1) && arrow < fitness_correlations_cumsum(j) && !individual_raw_variables(j))
                    individual_raw_variables(j) = true;
        }

        if(is_equal(individual_raw_variables, false))
            individual_raw_variables(rand()%input_raw_variables_number) = true;

        for(Index j = 0; j < raw_variables_number; j++)
            population(i, j) = individual_raw_variables(j);
    }
}


void GeneticAlgorithm::evaluate_population()
{
    // Training strategy

    TrainingResults training_results;

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Data set

    Dataset* dataset = training_strategy->get_dataset();

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Model selection

    const Index individuals_number = get_individuals_number();

    Tensor<bool, 1> individual_raw_variables;

    vector<Index> individual_raw_variables_indices;

    Tensor<Index, 1> raw_inputs_number(individuals_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        const Tensor<bool, 1> individual = population.chip(i, 0);

        cout << "\nIndividual " << i + 1 << endl;

        const vector<Index> individual_raw_variables_indices = get_raw_variable_indices(individual);

        raw_inputs_number(i) = individual_raw_variables_indices.size();

        dataset->set_raw_variable_indices(individual_raw_variables_indices,
                                           original_target_raw_variable_indices);

        const Index input_variables_number = dataset->get_variables_number("Input");

        const vector<string> input_names = dataset->get_variable_names("Input");

        neural_network->set_input_dimensions({input_variables_number});

        neural_network->set_input_names(input_names);

        neural_network->set_parameters_random();

        //Training

        training_results = training_strategy->perform_training();

        neural_network->get_parameters(parameters(i));

        training_errors(i) = training_results.get_training_error();

        selection_errors(i) = training_results.get_selection_error();

        if(display)
            cout << "Training error: " << training_results.get_training_error() << endl
                 << "Selection error: " << training_results.get_selection_error() << endl
                 << "Variables number: " << input_names.size() << endl
                 << "Inputs number: " << dataset->get_raw_variables_number("Input") << endl;

        dataset->set_raw_variable_indices(original_input_raw_variable_indices, original_target_raw_variable_indices);
    }

    const Tensor<type, 0> sum_training_errors = training_errors.sum();
    const Tensor<type, 0> sum_selection_errors = selection_errors.sum();

    mean_training_error = type(sum_training_errors(0)) / type(individuals_number);

    mean_selection_error = type(sum_selection_errors(0)) / type(individuals_number);

    const type sum_inputs_number = accumulate(raw_inputs_number.data(), raw_inputs_number.data() + raw_inputs_number.size(), type(0));

    mean_raw_inputs_number = type(sum_inputs_number) / type(individuals_number);
}


void GeneticAlgorithm::perform_fitness_assignment()
{
    const Index individuals_number = get_individuals_number();

    const Tensor<Index, 1> rank = calculate_rank_greater(selection_errors);

    for(Index i = 0; i < individuals_number; i++)
        fitness(rank(i)) = type(i+1);
}

void GeneticAlgorithm::perform_selection()
{
    const Index individuals_number = get_individuals_number();

    const Tensor<type, 0> fitness_sum = fitness.sum();

    selection.setConstant(false);

    const Index selected_individuals_number = Index(type(individuals_number)/type(2));

    if(elitism_size != 0)
        for(Index i = 0; i < individuals_number; i++)
            selection(i) = (fitness(i) - 1 >= 0) && (fitness(i) > (individuals_number - elitism_size));

    // The next individuals are selected randomly but their probability is set according to their fitness.

    while(count(selection.data(), selection.data() + selection.size(), 1) < selected_individuals_number)
    {
        const type random_number = get_random_type(type(0), fitness_sum());

        type sum = type(0);

        for(Index i = 0; i < individuals_number; i++)
        {
            sum += fitness(i);

            if(random_number <= sum )
            {
                selection(i) = true;
                break;
            }
        }
    }
}


vector<Index> GeneticAlgorithm::get_selected_individuals_indices()
{
    vector<Index> selection_indices(count(selection.data(), selection.data() + selection.size(), 1));

    Index count = 0;

    for(Index i = 0; i < selection.size(); i++)
        if(selection(i))
            selection_indices[count++] = i;

    return selection_indices;
}


void GeneticAlgorithm::perform_crossover()
{
    const Index individuals_number = get_individuals_number();

    const Index raw_variables_number = get_raw_variables_number();

    // Couples generation

    Tensor<bool, 2> new_population(individuals_number, raw_variables_number);

    Tensor<bool, 1> descendent_raw_variables(raw_variables_number);

    Tensor<bool, 1> parent_1_raw_variables;

    Tensor<bool, 1> parent_2_raw_variables;

    random_device rd;

    mt19937 g(rd());

    vector<Index> parent_1_indices = get_selected_individuals_indices();

    shuffle(parent_1_indices.begin(), parent_1_indices.end(), g);

    const vector<Index> parent_2_indices = get_selected_individuals_indices();

    Index descendent_index = 0;

    for(size_t i = 0; i < parent_1_indices.size(); i++)
    {
        parent_1_raw_variables = population.chip(parent_1_indices[i], 0);

        parent_2_raw_variables = population.chip(parent_2_indices[i], 0);

        for(Index j = 0; j < 2; j++)
        {
            descendent_raw_variables = parent_1_raw_variables;

            for(Index k = 0; k < raw_variables_number; k++)
                descendent_raw_variables(k) = (rand() % 2) ? parent_1_raw_variables(k) : parent_2_raw_variables(k);


            if(is_equal(descendent_raw_variables, false))
            {
                const Index num_to_activate = max(type(1), type(raw_variables_number / 10));

                vector<Index> indices(raw_variables_number);
                iota(indices.begin(), indices.end(), 0);
                shuffle(indices.begin(), indices.end(), g);

                for (Index i = 0; i < num_to_activate; i++)
                    descendent_raw_variables(indices[i]) = true;
            }

            new_population.chip(descendent_index++, 0) = descendent_raw_variables;
        }
    }

    population = new_population;
}


void GeneticAlgorithm::perform_mutation()
{
    const Index individuals_number = get_individuals_number();

    const Index raw_variables_number = original_input_raw_variable_indices.size();

    for(Index i = 0; i < individuals_number; i++)
    {
        Tensor<bool, 1> individual_raw_variables = population.chip(i, 0);

        for(Index j = 0; j < raw_variables_number; j++)
            individual_raw_variables(j) ^= (get_random_type(0, 1) < mutation_rate);

        if(is_equal(individual_raw_variables, false))
            for(Index j = 0; j < raw_variables_number; j++)
                if(original_input_raw_variables[j])
                    individual_raw_variables[j] = true;

        if(is_equal(individual_raw_variables, false))
            individual_raw_variables.setConstant(true);

        population.chip(i, 0) = individual_raw_variables;
    }
}


InputsSelectionResults GeneticAlgorithm::perform_input_selection()
{
    // Selection algorithm

    original_input_raw_variable_indices = training_strategy->get_dataset()->get_raw_variable_indices("Input");
    original_target_raw_variable_indices = training_strategy->get_dataset()->get_raw_variable_indices("Target");

    InputsSelectionResults input_selection_results(maximum_epochs_number);

    if(display) cout << "Performing genetic inputs selection...\n" << endl;

    initialize_population();

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Data set

    Dataset* dataset = loss_index->get_dataset();

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
                 << "Mean inputs number  " << mean_raw_inputs_number << endl
                 << "Generation minimum training error: " << training_errors(optimal_individual_training_index) << endl
                 << "Generation minimum selection error: " << selection_errors(optimal_individual_index) << endl
                 << "Best ever training error: " << input_selection_results.optimum_training_error << endl
                 << "Best ever selection error: " << input_selection_results.optimum_selection_error << endl
                 << "Elapsed time: " << write_time(elapsed_time) << endl
                 << "Best selection error in generation: " << generation_selected << endl;

        // Stopping criteria

        stop = true;

        if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << write_time(elapsed_time) << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
        }
        else if (epoch >= maximum_epochs_number - 1)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
        }
        else
        {
            stop = false;
        }

        if(stop)
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

    // Set data set stuff

    const vector<Index> optimal_raw_variable_indices = get_raw_variable_indices(input_selection_results.optimal_inputs);

    dataset->set_raw_variable_indices(optimal_raw_variable_indices, original_target_raw_variable_indices);

    dataset->set_dimensions("Input", { Index(optimal_inputs_raw_variables_indices.size()) });

    const vector<Scaler> input_variable_scalers = dataset->get_variable_scalers("Input");

    const vector<Descriptives> input_variable_descriptives = dataset->calculate_variable_descriptives("Input");

    // Set neural network stuff

    neural_network->set_input_dimensions({ dataset->get_variables_number("Input") });

    neural_network->set_input_names(dataset->get_variable_names("Input"));

    if(neural_network->has("Scaling2d"))
    {
        Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"));
        scaling_layer_2d->set_descriptives(input_variable_descriptives);
        scaling_layer_2d->set_scalers(input_variable_scalers);
    }

    neural_network->set_parameters(input_selection_results.optimal_parameters);

    if(display) input_selection_results.print();

    cout << "Selected generation: " << generation_selected << endl;

    return input_selection_results;
}


vector<Index> GeneticAlgorithm::get_raw_variable_indices(const Tensor<bool, 1>& individual_raw_variables) // updated
{
    Tensor<bool, 1> inputs_pre_indices(individual_raw_variables.size());
    inputs_pre_indices.setConstant(false);

    Index original_input_index = 0;

    for(size_t i = 0; i < original_input_raw_variables.size(); i++)
    {
        if(individual_raw_variables(i) && original_input_raw_variables[i])
        {
            inputs_pre_indices(i) = true;

            original_input_index = i;
        }
    }

    const Index indices_dimension = count(inputs_pre_indices.data(),
                                          inputs_pre_indices.data() + inputs_pre_indices.size(),
                                          true);

    if(is_equal(inputs_pre_indices, false))
    {
        cout << "/." << endl;
        inputs_pre_indices(original_input_index) = true;
    }

    Index index = 0;

    vector<Index> indices(indices_dimension);

    for(Index i = 0; i < individual_raw_variables.size(); i++)
        if(inputs_pre_indices(i))
            indices[index++] = i;

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
    set_maximum_epochs_number(read_xml_index(root, "MaximumGenerationsNumber"));
    //set_maximum_correlation(read_xml_type(root, "MaximumCorrelation"));
    //set_minimum_correlation(read_xml_type(root, "MinimumCorrelation"));
    set_maximum_time(read_xml_type(root, "MaximumTime"));
}


void GeneticAlgorithm::print() const
{
    cout << "Genetic algorithm" << endl
         << "Individuals number: " << get_individuals_number() << endl
         << "Raw variables number: " << get_raw_variables_number() << endl;
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

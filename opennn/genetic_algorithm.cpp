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


Index GeneticAlgorithm::get_genes_number() const
{
    return population.size() ? population.dimension(1) : 0;
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

    original_input_raw_variable_indices = dataset->get_raw_variable_indices("Input");
    original_target_raw_variable_indices = dataset->get_raw_variable_indices("Target");

    const Index genes_number = original_input_raw_variable_indices.size();

    population.resize(individuals_number, genes_number);

    minimum_inputs_number = 1;
    maximum_inputs_number = original_input_raw_variable_indices.size();

    maximum_epochs_number = 100;

    maximum_time = type(3600);

    mutation_rate = type(0.001);

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
        throw runtime_error("Training strategy or data set is null");

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

    population.setConstant(false);

    const Tensor<type, 1> correlations = dataset->calculate_input_target_mean_absolute_correlations();

    const Tensor<Index, 1> rank = calculate_rank_less(correlations);

    fitness_correlations.resize(genes_number);

    for (Index i = 0; i < genes_number; i++)
        fitness_correlations(rank(i)) = type(i + 1);

    const Tensor<type, 0> fitness_sum = fitness_correlations.sum();
    const Tensor<type, 1> fitness_cumsum = fitness_correlations.cumsum(0);

    uniform_real_distribution<> distribution(0, fitness_sum(0));

    const Index upper_bound = clamp(maximum_inputs_number, minimum_inputs_number, genes_number);

    uniform_int_distribution<Index> num_inputs_dist(minimum_inputs_number, upper_bound);

    const type* begin = fitness_cumsum.data();
    const type* end   = begin + genes_number;

    for (Index i = 0; i < individuals_number; i++)
    {        
        individual_genes.setConstant(false);

        const Index true_count = get_random_index(minimum_inputs_number, maximum_inputs_number);

        while (count(individual_genes.data(), individual_genes.data() + genes_number, true) < true_count)
        {   
            const type arrow = get_random_type(0, fitness_sum());

            const Index j = static_cast<Index>(std::upper_bound(begin, end, arrow) - begin);

            if (j < genes_number)
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

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Model selection

    const Index individuals_number = get_individuals_number();

    Tensor<Index, 1> raw_inputs_number(individuals_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        const Tensor<bool, 1> individual = population.chip(i, 0);

        if (display) cout << "\nIndividual " << i + 1 << endl;

        const vector<Index> individual_raw_variables_indices = get_raw_variable_indices(individual);

        raw_inputs_number(i) = individual_raw_variables_indices.size();

        dataset->set_raw_variable_indices(individual_raw_variables_indices, original_target_raw_variable_indices);

        const Index input_variables_number = dataset->get_variables_number("Input");

        const vector<string> input_names = dataset->get_variable_names("Input");

        neural_network->set_input_dimensions({input_variables_number});

        neural_network->set_input_names(input_names);

        neural_network->set_parameters_random();

        //Training

        if (!display)
        {
            training_strategy->get_loss_index()->set_display(false);
            training_strategy->get_optimization_algorithm()->set_display(false);
        }

        training_results = training_strategy->train();

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

    const Tensor<type, 0> mean_training_errors = training_errors.mean();
    const Tensor<type, 0> mean_selection_errors = selection_errors.mean();

    mean_training_error = mean_training_errors();
    mean_selection_error = mean_selection_errors();

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

    const Index individuals_to_be_selected = individuals_number/2;

    if(elitism_size != 0)
        for(Index i = 0; i < individuals_number; i++)
            selection(i) = (fitness(i) - 1 >= 0) && (fitness(i) > (individuals_number - elitism_size));

    // The next individuals are selected randomly but their probability is set according to their fitness

    while(get_selected_individuals_number() < individuals_to_be_selected)
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


Index GeneticAlgorithm::get_selected_individuals_number() const
{
    return count(selection.data(), selection.data() + selection.size(), 1);
}


vector<Index> GeneticAlgorithm::get_selected_individuals_indices() const
{
    const Index selected_individuals_number = get_selected_individuals_number();

    vector<Index> selection_indices(selected_individuals_number);

    Index count = 0;

    for(Index i = 0; i < selection.size(); i++)
        if(selection(i))
            selection_indices[count++] = i;

    return selection_indices;
}


void GeneticAlgorithm::perform_crossover()
{
    const Index individuals_number = get_individuals_number();

    const Index genes_number = get_genes_number();

    // Couples generation

    Tensor<bool, 2> new_population(individuals_number, genes_number);

    Tensor<bool, 1> descendent_raw_variables(genes_number);

    Tensor<bool, 1> parent_1_raw_variables;

    Tensor<bool, 1> parent_2_raw_variables;

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

            for(Index k = 0; k < genes_number; k++)
                descendent_raw_variables(k) = (rand() % 2) ? parent_1_raw_variables(k) : parent_2_raw_variables(k);

            if(is_equal(descendent_raw_variables, false))
            {                
                const Index num_to_activate = clamp(genes_number / 10, Index(1), genes_number);

                vector<Index> indices(genes_number);
                iota(indices.begin(), indices.end(), 0);
                shuffle(indices.begin(), indices.end(), g);

                for (Index i = 0; i < num_to_activate; i++)
                    descendent_raw_variables(indices[i]) = true;
            }

            new_population.chip(descendent_index++, 0) = descendent_raw_variables;

            Index active_inputs = count(descendent_raw_variables.data(), descendent_raw_variables.data() + descendent_raw_variables.size(), true);

            while (active_inputs < minimum_inputs_number)
            {
                const Index random_pos = rand() % genes_number;

                if (!descendent_raw_variables(random_pos))
                {
                    descendent_raw_variables(random_pos) = true;
                    active_inputs++;
                }
            }

            while (active_inputs > maximum_inputs_number)
            {
                const Index random_pos = rand() % genes_number;

                if (descendent_raw_variables(random_pos))
                {
                    descendent_raw_variables(random_pos) = false;
                    active_inputs--;
                }
            }
        }
    }

    population = new_population;
}


void GeneticAlgorithm::perform_mutation()
{
    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();
    mt19937 gen(rd());

    for (Index i = 0; i < individuals_number; i++)
    {
        Tensor<bool, 1> individual = population.chip(i, 0);

        for (Index j = 0; j < genes_number; j++)
            if (get_random_type(0, 1) < mutation_rate)
                individual(j) = !individual(j);

        Index active_genes = count(individual.data(), individual.data() + genes_number, true);

        while (active_genes < minimum_inputs_number)
        {
            vector<Index> inactive_indices;
            for (Index j = 0; j < genes_number; ++j)
                if (!individual(j)) inactive_indices.push_back(j);

            if (inactive_indices.empty()) break;

            shuffle(inactive_indices.begin(), inactive_indices.end(), gen);
            individual(inactive_indices[0]) = true;
            active_genes++;
        }

        while (active_genes > maximum_inputs_number)
        {
            vector<Index> active_indices;
            for (Index j = 0; j < genes_number; ++j)
                if (individual(j)) active_indices.push_back(j);

            if (active_indices.empty()) break;

            shuffle(active_indices.begin(), active_indices.end(), gen);
            individual(active_indices[0]) = false;
            active_genes--;
        }

        if (active_genes == 0 && genes_number > 0)
        {
            const Index random_pos = get_random_index(0, genes_number - 1);
            individual(random_pos) = true;
        }

        population.chip(i, 0) = individual;
    }
}


InputsSelectionResults GeneticAlgorithm::perform_input_selection()
{
    const LossIndex* loss_index = training_strategy->get_loss_index();

    Dataset* dataset = loss_index->get_dataset();

    // Selection algorithm

    original_input_raw_variable_indices = dataset->get_raw_variable_indices("Input");
    original_target_raw_variable_indices = dataset->get_raw_variable_indices("Target");

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
                 << "Mean inputs number  " << mean_raw_inputs_number << endl
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

    // Set data set stuff

    const vector<Index> optimal_raw_variable_indices = get_raw_variable_indices(input_selection_results.optimal_inputs);

    dataset->set_raw_variable_indices(optimal_raw_variable_indices, original_target_raw_variable_indices);

    dataset->set_dimensions("Input", { Index(optimal_inputs_raw_variables_indices.size()) });

    const vector<string> input_variable_scalers = dataset->get_variable_scalers("Input");

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

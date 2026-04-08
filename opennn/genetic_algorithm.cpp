//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "time_series_dataset.h"
#include "scaling_layer.h"
#include "training_strategy.h"
#include "genetic_algorithm.h"
#include "random_utilities.h"

namespace opennn
{

GeneticAlgorithm::GeneticAlgorithm(TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
    set_default();
}

void GeneticAlgorithm::set_default()
{
    name = "GeneticAlgorithm";

    if(!training_strategy || !training_strategy->get_neural_network() || !training_strategy->get_dataset())
        return;

    const Dataset* dataset = training_strategy->get_dataset();

    const Index individuals_number = 40;

    original_input_variable_indices = dataset->get_variable_indices("Input");
    original_target_variable_indices = dataset->get_variable_indices("Target");

    const Index genes_number = get_genes_number();

    population.resize(individuals_number, genes_number);

    minimum_inputs_number = 1;
    maximum_inputs_number = genes_number;

    maximum_epochs = 100;

    maximum_time = type(3600);

    mutation_rate = type(0.05);

    individual_parameters.resize(individuals_number);

    training_errors.resize(individuals_number);

    validation_errors.resize(individuals_number);

    fitness.resize(individuals_number);
    fitness.setConstant(type(-1.0));

    selected.resize(individuals_number);

    elitism_size = (individuals_number + 3) / 4;

    initialization_method = "Correlations";
}

void GeneticAlgorithm::set_maximum_inputs_number(const Index new_maximum_inputs_number)
{
    const Dataset* dataset = training_strategy ? training_strategy->get_dataset() : nullptr;
    const Index inputs_number = dataset ? dataset->get_variables_number("Input") : 0;

    maximum_inputs_number = (inputs_number == 0)
        ? new_maximum_inputs_number
        : clamp(new_maximum_inputs_number, Index(1), inputs_number);
}

void GeneticAlgorithm::set_individuals_number(const Index new_individuals_number)
{
    if(!training_strategy || !training_strategy->get_dataset())
        throw runtime_error("Training strategy or dataset is null");

    const Index genes_number = get_genes_number();

    population.resize(new_individuals_number, genes_number);
    individual_parameters.resize(new_individuals_number);
    training_errors.resize(new_individuals_number);
    validation_errors.resize(new_individuals_number);
    fitness.resize(new_individuals_number);
    fitness.setConstant(type(-1.0));
    selected.resize(new_individuals_number);

    elitism_size = min(elitism_size, new_individuals_number);
}

void GeneticAlgorithm::initialize_population()
{
    initialization_method == "Random"
        ? initialize_population_random()
        : initialize_population_correlations();
}

void GeneticAlgorithm::initialize_population_random()
{
    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    population.setConstant(false);

    VectorB individual_genes(genes_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        individual_genes.setConstant(false);

        const Index true_count = random_integer(minimum_inputs_number, maximum_inputs_number);

        individual_genes.head(true_count).setConstant(true);

        shuffle(individual_genes);

        population.row(i) = individual_genes;
    }
}

void GeneticAlgorithm::initialize_population_correlations()
{
    const Dataset* dataset = training_strategy->get_dataset();

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    VectorB individual_genes(genes_number);

    population.setConstant(false);

    const VectorR correlations_rank = dataset->calculate_correlations_rank().cast<type>().array() + 1.0f;

    const type correlations_sum = correlations_rank.sum();

    VectorR correlations_cumsum(genes_number);
    partial_sum(correlations_rank.data(), correlations_rank.data() + genes_number, correlations_cumsum.data());

    const type* begin = correlations_cumsum.data();
    const type* end   = begin + genes_number;

    for(Index i = 0; i < individuals_number; i++)
    {        
        individual_genes.setConstant(false);

        const Index true_count = random_integer(minimum_inputs_number, maximum_inputs_number);

        while (individual_genes.count() < true_count)
        {   
            const type arrow = random_uniform(0, correlations_sum);

            const Index j = min(static_cast<Index>(upper_bound(begin, end, arrow) - begin), genes_number - 1);

            individual_genes(j) = true;
        }

        population.row(i) = individual_genes;
    }
}

void GeneticAlgorithm::evaluate_population()
{
    TrainingResults training_results;

    const Loss* loss = training_strategy->get_loss();
    Dataset* dataset = training_strategy->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index individuals_number = get_individuals_number();

    training_strategy->get_optimization_algorithm()->set_display(false);

    for(Index i = 0; i < individuals_number; i++)
    {
        const VectorB individual = population.row(i);

        if (display) cout << "\nIndividual " << i + 1 << endl;

        const vector<Index> individual_variables_indices = get_true_indices(individual);

        dataset->set_variable_indices(individual_variables_indices, original_target_variable_indices);

        const Index input_features_number = dataset->get_features_number("Input");

        configure_neural_network_inputs(neural_network, dataset, input_features_number);

        neural_network->set_parameters_random();

        training_results = training_strategy->train();

        individual_parameters(i) = neural_network->get_individual_parameters();

        training_errors(i) = training_results.get_training_error();
        validation_errors(i) = training_results.get_validation_error();

        if(display)
            cout << "Training error: " << training_errors(i) << endl
                 << "Validation error: " << validation_errors(i) << endl
                 << "Variables number: " << input_features_number << endl
                 << "Inputs number: " << dataset->get_variables_number("Input") << endl;

        dataset->set_variable_indices(original_input_variable_indices, original_target_variable_indices);
    }

}

void GeneticAlgorithm::assign_fitness()
{
    fitness = calculate_rank_less(validation_errors).cast<type>().array() + type(1.0);
}

void GeneticAlgorithm::perform_selection()
{
    const Index individuals_number = get_individuals_number();

    const type fitness_sum = fitness.sum();

    selected.setConstant(false);

    const Index individuals_to_be_selected = individuals_number/2;

    if(elitism_size != 0)
        selected = (fitness.array() > type(individuals_number - elitism_size)).matrix();

    VectorR fitness_cumsum(individuals_number);
    partial_sum(fitness.data(), fitness.data() + individuals_number, fitness_cumsum.data());

    Index selected_count = selected.count();

    while (selected_count < individuals_to_be_selected)
    {
        const type arrow = random_uniform(type(0), fitness_sum);

        const Index i = min(static_cast<Index>(upper_bound(fitness_cumsum.data(), fitness_cumsum.data() + individuals_number, arrow) - fitness_cumsum.data()), individuals_number - 1);

        if(!selected(i))
        {
            selected(i) = true;
            selected_count++;
        }
    }
}

vector<Index> GeneticAlgorithm::get_selected_indices() const
{
    vector<Index> selection_indices;
    selection_indices.reserve(selected.count());

    for(Index i = 0; i < selected.size(); i++)
        if(selected(i))
            selection_indices.push_back(i);

    return selection_indices;
}

VectorB GeneticAlgorithm::crossover(const VectorB& parent_1, const VectorB& parent_2)
{
    const Index genes_number = get_genes_number();

    VectorB descendent(genes_number);
    descendent.setConstant(false);

    vector<Index> intersection, difference;

    for(Index i = 0; i < genes_number; ++i)
        if (parent_1(i) && parent_2(i)) 
            intersection.push_back(i);
        else if (parent_1(i) != parent_2(i))
            difference.push_back(i);

    for(const Index idx : intersection)
        descendent(idx) = true;

    const Index current_size = intersection.size();

    if (current_size > maximum_inputs_number) 
    {
        shuffle_vector(intersection);

        descendent.setConstant(false);
        for(Index i = 0; i < maximum_inputs_number; ++i)
            descendent(intersection[i]) = true;

        return descendent;
    }

    const Index target_size = random_integer(max(minimum_inputs_number, current_size), maximum_inputs_number);

    shuffle_vector(difference);
    const Index genes_to_add = target_size - current_size;
    
    for(Index i = 0; i < genes_to_add && i < difference.size(); ++i)
        descendent(difference[i]) = true;
    
    const Index final_count = descendent.count();

    if (final_count < minimum_inputs_number) 
    {
        vector<Index> never_true_indices;
        for(Index i = 0; i < genes_number; ++i)
            if(!parent_1(i) && !parent_2(i))
                never_true_indices.push_back(i);

        shuffle_vector(never_true_indices);

        const Index genes_needed = minimum_inputs_number - final_count;

        for(Index i = 0; i < genes_needed && i < never_true_indices.size(); ++i)
            descendent(never_true_indices[i]) = true;
    }

    return descendent;
}

void GeneticAlgorithm::perform_crossover()
{
    const Index individuals_number = get_individuals_number();

    const vector<Index> selected_individual_indices = get_selected_indices();

    MatrixB new_population(individuals_number, get_genes_number());

    for(Index i = 0; i < individuals_number; i++)
    {
        const Index p1 = get_random_element(selected_individual_indices);
        Index p2 = get_random_element(selected_individual_indices);

        while (selected_individual_indices.size() > 1 && p1 == p2)
            p2 = get_random_element(selected_individual_indices);

        new_population.row(i) = crossover(population.row(p1), population.row(p2));
    }

    population.swap(new_population);
}

void GeneticAlgorithm::perform_mutation()
{
    if(mutation_rate == 0) return;

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    for(Index i = 0; i < individuals_number; ++i)
    {
        VectorB individual = population.row(i);
        Index current_inputs_number = individual.count();

        vector<Index> to_true_mutations; 
        vector<Index> to_false_mutations;

        for(Index j = 0; j < genes_number; ++j)
            if (random_uniform(0.0, 1.0) < mutation_rate)
                (individual(j) ? to_false_mutations : to_true_mutations).push_back(j);

        shuffle_vector(to_true_mutations);
        shuffle_vector(to_false_mutations);

        const Index swap_count = std::min(to_true_mutations.size(), to_false_mutations.size());

        for(Index k = 0; k < swap_count; ++k)
        {
            individual(to_true_mutations[k]) = true;
            individual(to_false_mutations[k]) = false;
        }

        for(size_t k = swap_count; k < to_true_mutations.size(); ++k)
        {
            if (current_inputs_number < maximum_inputs_number)
            {
                individual(to_true_mutations[k]) = true;
                current_inputs_number++;
            }
        }

        for(size_t k = swap_count; k < to_false_mutations.size(); ++k)
        {
            if (current_inputs_number > minimum_inputs_number)
            {
                individual(to_false_mutations[k]) = false;
                current_inputs_number--;
            }
        }

        population.row(i) = individual;
    }
}

InputsSelectionResults GeneticAlgorithm::perform_input_selection()
{
    const Loss* loss = training_strategy->get_loss();

    Dataset* dataset = loss->get_dataset();

    // Validation algorithm

    original_input_variable_indices = dataset->get_variable_indices("Input");
    original_target_variable_indices = dataset->get_variable_indices("Target");
    const vector<Index> time_variable_indices = dataset->get_variable_indices("Time");

    InputsSelectionResults input_selection_results(maximum_epochs);

    if(display) cout << "Performing genetic input selection...\n" << endl;

    initialize_population();

    // Dataset

    if (dataset->has_nan())
        dataset->scrub_missing_values();

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    Index optimal_individual_index;
    time_t beginning_time, current_time;
    type elapsed_time = type(0);
    vector<Index> best_input_indices;
    Index best_generation = 0;

    time(&beginning_time);

    for(Index epoch = 0; epoch < maximum_epochs; epoch++)
    {
        if(display) cout << "Generation: " << epoch + 1 << endl;

        input_selection_results.resize_history(input_selection_results.mean_training_error_history.size() + 1);

        evaluate_population();

        // Optimal individual in population

        optimal_individual_index = minimal_index(validation_errors);

        const type optimal_training_error = training_errors(optimal_individual_index);
        const type optimal_validation_error = validation_errors(optimal_individual_index);

        input_selection_results.training_error_history(epoch) = optimal_training_error;
        input_selection_results.validation_error_history(epoch) = optimal_validation_error;
        input_selection_results.mean_validation_error_history(epoch) = validation_errors.mean();
        input_selection_results.mean_training_error_history(epoch) = training_errors.mean();

        dataset->set_variable_indices(original_input_variable_indices, original_target_variable_indices);

        if(optimal_validation_error < input_selection_results.optimum_validation_error)
        {
            best_generation = epoch;

            input_selection_results.optimal_inputs = population.row(optimal_individual_index);

            best_input_indices = get_variable_indices(input_selection_results.optimal_inputs);

            dataset->set_variable_indices(best_input_indices, original_target_variable_indices);

            input_selection_results.optimal_input_variable_names
                = dataset->get_variable_names("Input");

            dataset->set_variable_indices(original_input_variable_indices, original_target_variable_indices);

            input_selection_results.optimal_parameters = individual_parameters(optimal_individual_index);
            input_selection_results.optimum_training_error = optimal_training_error;
            input_selection_results.optimum_validation_error = optimal_validation_error;
        }

        time(&current_time);

        elapsed_time = type(difftime(current_time, beginning_time));

        if(display)
            cout << endl
                 << "Epoch number: " << epoch << endl
                 << "Generation mean training error: " << training_errors.mean() << endl
                 << "Generation mean selection error: " << input_selection_results.mean_validation_error_history(epoch) << endl
                 << "Generation minimum training error: " << optimal_training_error << endl
                 << "Generation minimum selection error: " << optimal_validation_error << endl
                 << "Best ever training error: " << input_selection_results.optimum_training_error << endl
                 << "Best ever selection error: " << input_selection_results.optimum_validation_error << endl
                 << "Elapsed time: " << write_time(elapsed_time) << endl
                 << "Best selection error in generation: " << best_generation << endl;

        // Stopping criteria

        bool stop = false;

        if (input_selection_results.optimum_validation_error <= validation_error_goal)
        {
            if (display) cout << "Epoch " << epoch << "\nSelection error goal reached: " << input_selection_results.optimum_validation_error << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::SelectionErrorGoal;
            stop = true;
        }
        else if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << write_time(elapsed_time) << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
            stop = true;
        }
        else if (epoch >= maximum_epochs - 1)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch + 1 << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
            stop = true;
        }

        if (stop)
        {
            input_selection_results.elapsed_time = write_time(elapsed_time);
            input_selection_results.resize_history(epoch + 1);
            break;
        }

        assign_fitness();

        perform_selection();

        perform_crossover();

        perform_mutation();
    }

    // Set dataset stuff

    const vector<Index> optimal_variable_indices = get_variable_indices(input_selection_results.optimal_inputs);

    dataset->set_variable_indices(optimal_variable_indices, original_target_variable_indices);

    const vector<string> input_variable_scalers = dataset->get_feature_scalers("Input");

    const vector<Descriptives> input_variable_descriptives = dataset->calculate_feature_descriptives("Input");

    const Index optimal_variables_number = dataset->get_features_number("Input");

    // Set neural network stuff

    const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    if(time_series_dataset && time_variable_indices.size() == 1)
        dataset->set_variable_role(time_variable_indices[0], "Time");

    configure_neural_network_inputs(neural_network, dataset, optimal_variables_number);

    if(neural_network->has("Scaling2d"))
    {
        Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(neural_network->get_first("Scaling2d"));
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }
    else if(neural_network->has("Scaling3d"))
    {
        Scaling<3>* scaling_layer = static_cast<Scaling<3>*>(neural_network->get_first("Scaling3d"));
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }

    neural_network->set_individual_parameters(input_selection_results.optimal_parameters);

    if(display)
    {
        input_selection_results.print();
        cout << "Selected generation: " << best_generation << endl;
    }

    return input_selection_results;
}

void GeneticAlgorithm::configure_neural_network_inputs(NeuralNetwork* neural_network, Dataset* dataset, Index input_features_number)
{
    const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    if(time_series_dataset)
    {
        const Index past_time_steps = time_series_dataset->get_past_time_steps();
        neural_network->set_input_shape({ past_time_steps, input_features_number });
        dataset->set_shape("Input", { past_time_steps, input_features_number });

        vector<string> final_feature_names;
        const vector<string> base_names = dataset->get_variable_names("Input");
        final_feature_names.reserve(base_names.size() * past_time_steps);

        for(const string& base_name : base_names)
            for(Index j = 0; j < past_time_steps; j++)
                final_feature_names.push_back((base_name.empty() ? "variable" : base_name) + "_lag" + to_string(j));

        neural_network->set_input_names(final_feature_names);
    }
    else
    {
        neural_network->set_input_shape({input_features_number});
        dataset->set_shape("Input", { input_features_number });
        neural_network->set_input_names(dataset->get_feature_names("Input"));
    }
}


void GeneticAlgorithm::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("GeneticAlgorithm");

    write_xml_properties(printer, {
        {"PopulationSize", to_string(get_individuals_number())},
        {"ElitismSize", to_string(elitism_size)},
        {"MutationRate", to_string(mutation_rate)},
        {"SelectionErrorGoal", to_string(validation_error_goal)},
        {"MinimumInputsNumber", to_string(minimum_inputs_number)},
        {"MaximumInputsNumber", to_string(maximum_inputs_number)},
        {"MaximumGenerationsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)}});

    printer.CloseElement();
}

void GeneticAlgorithm::from_XML(const XMLDocument& document)
{
    const XMLElement* root = get_xml_root(document, "GeneticAlgorithm");
    set_individuals_number(read_xml_index(root, "PopulationSize"));
    set_mutation_rate(read_xml_type(root, "MutationRate"));
    set_elitism_size(read_xml_index(root, "ElitismSize"));
    set_validation_error_goal(read_xml_type(root, "SelectionErrorGoal"));
    set_minimum_inputs_number(read_xml_index(root, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_xml_index(root, "MaximumInputsNumber"));
    set_maximum_epochs(read_xml_index(root, "MaximumGenerationsNumber"));
    set_maximum_time(read_xml_type(root, "MaximumTime"));
}

REGISTER(InputsSelection, GeneticAlgorithm, "GeneticAlgorithm");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

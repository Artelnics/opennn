//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "tabular_dataset.h"
#include "time_series_dataset.h"
#include "scaling_layer.h"
#include "training_strategy.h"
#include "genetic_algorithm.h"
#include "random_utilities.h"
#include "cross_validation.h"

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

    if (!training_strategy || !training_strategy->get_neural_network() || !training_strategy->get_dataset())
        return;

    const Dataset* dataset = training_strategy->get_dataset();

    const Index individuals_number = 40;

    original_input_indices = dataset->get_variable_indices("Input");
    original_target_indices = dataset->get_variable_indices("Target");

    const Index genes_number = get_genes_number();

    population.resize(individuals_number, genes_number);

    minimum_inputs_number = 1;
    maximum_inputs_number = genes_number;

    maximum_epochs = 100;

    maximum_time = 3600.0f;

    mutation_rate = 0.05f;

    individual_parameters.resize(individuals_number);

    training_errors.resize(individuals_number);

    validation_errors.resize(individuals_number);

    fitness = VectorR::Constant(individuals_number, -1.0f);

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
    throw_if(!training_strategy || !training_strategy->get_dataset(),
             "training strategy or dataset is not set.");

    const Index genes_number = get_genes_number();

    population.resize(new_individuals_number, genes_number);
    individual_parameters.resize(new_individuals_number);
    training_errors.resize(new_individuals_number);
    validation_errors.resize(new_individuals_number);
    fitness = VectorR::Constant(new_individuals_number, -1.0f);
    selected.resize(new_individuals_number);

    elitism_size = min(elitism_size, new_individuals_number);
}

void GeneticAlgorithm::initialize_population()
{
    population.resize(get_individuals_number(), get_genes_number());

    if (initialization_method == "Random")
        initialize_population_random();
    else
        initialize_population_correlations();
}

void GeneticAlgorithm::initialize_population_random()
{
    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    VectorB individual_genes(genes_number);

    for (Index i = 0; i < individuals_number; ++i)
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

    const auto* correlations_dataset = dynamic_cast<const TabularDataset*>(dataset);
    throw_if(!correlations_dataset, "Expected TabularDataset.");

    const VectorI correlations_argsort = correlations_dataset->calculate_correlations_rank();

    VectorR gene_weight(genes_number);
    for (Index p = 0; p < genes_number; ++p)
        gene_weight(correlations_argsort(p)) = float(p + 1);

    const float correlations_sum = gene_weight.sum();

    VectorR correlations_cumsum(genes_number);
    partial_sum(gene_weight.data(), gene_weight.data() + genes_number, correlations_cumsum.data());

    const float* begin = correlations_cumsum.data();
    const float* end   = begin + genes_number;

    for (Index i = 0; i < individuals_number; ++i)
    {
        individual_genes.setConstant(false);

        const Index true_count = random_integer(minimum_inputs_number, maximum_inputs_number);

        while (individual_genes.count() < true_count)
        {
            const float arrow = random_uniform(0, correlations_sum);

            const Index j = min(static_cast<Index>(upper_bound(begin, end, arrow) - begin), genes_number - 1);

            individual_genes(j) = true;
        }

        population.row(i) = individual_genes;
    }
}

vector<Index> GeneticAlgorithm::genes_to_variable_indices(const VectorB& genes) const
{
    vector<Index> variable_indices;
    variable_indices.reserve(genes.count());

    for (Index i = 0; i < genes.size(); ++i)
        if (genes(i))
            variable_indices.push_back(original_input_indices[i]);

    return variable_indices;
}

void GeneticAlgorithm::evaluate_population()
{
    Loss* loss = training_strategy->get_loss();
    Dataset* dataset = training_strategy->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index individuals_number = get_individuals_number();

    training_strategy->get_optimization_algorithm()->set_display(false);

    const vector<vector<Index>> fold_partition =
        folds_number > 1 ? build_fold_partition(training_strategy, folds_number) : vector<vector<Index>>{};

    for (Index i = 0; i < individuals_number; ++i)
    {
        if (display) cout << "\nIndividual " << i + 1 << "\n";

        const vector<Index> individual_variables_indices = genes_to_variable_indices(population.row(i));

        dataset->set_variable_indices(individual_variables_indices, original_target_indices);

        const Index input_features_number = dataset->get_features_number("Input");

        configure_neural_network_inputs(neural_network, dataset, input_features_number);

        if (folds_number > 1)
        {
            const FoldEvaluation evaluation = evaluate_folds(training_strategy, fold_partition);
            training_errors(i) = evaluation.training_error;
            validation_errors(i) = evaluation.validation_error;
            individual_parameters(i) = VectorR();
        }
        else
        {
            neural_network->set_parameters_random();
            const TrainingResult training_results = training_strategy->train();

            individual_parameters(i) = VectorMap(neural_network->get_parameters_data(),
                                                 neural_network->get_parameters_size());

            training_errors(i) = training_results.get_training_error();
            validation_errors(i) = training_results.get_validation_error();
        }

        if (!isfinite(training_errors(i)))   training_errors(i)   = numeric_limits<float>::max();
        if (!isfinite(validation_errors(i))) validation_errors(i) = numeric_limits<float>::max();

        if (display)
            cout << "Training error: " << training_errors(i) << "\n"
                 << "Validation error: " << validation_errors(i) << "\n"
                 << "Variables number: " << input_features_number << "\n"
                 << "Inputs number: " << dataset->get_variables_number("Input") << "\n";

        dataset->set_variable_indices(original_input_indices, original_target_indices);
    }

}

void GeneticAlgorithm::assign_fitness()
{
    const Index individuals_number = validation_errors.size();

    const VectorI order = calculate_rank(validation_errors);

    fitness.resize(individuals_number);

    for (Index k = 0; k < individuals_number; ++k)
        fitness(order(k)) = float(individuals_number - k);
}

void GeneticAlgorithm::perform_selection()
{
    const Index individuals_number = get_individuals_number();

    const float fitness_sum = fitness.sum();

    selected.setConstant(false);

    const Index individuals_to_be_selected = individuals_number/2;

    if (elitism_size != 0)
        selected = (fitness.array() > float(individuals_number - elitism_size)).matrix();

    VectorR fitness_cumsum(individuals_number);
    partial_sum(fitness.data(), fitness.data() + individuals_number, fitness_cumsum.data());

    Index selected_count = selected.count();

    while (selected_count < individuals_to_be_selected)
    {
        const float arrow = random_uniform(0.0f, fitness_sum);

        const Index i = min(static_cast<Index>(upper_bound(fitness_cumsum.data(), fitness_cumsum.data() + individuals_number, arrow) - fitness_cumsum.data()), individuals_number - 1);

        if (!selected(i))
        {
            selected(i) = true;
            ++selected_count;
        }
    }
}

vector<Index> GeneticAlgorithm::get_selected_indices() const
{
    vector<Index> selection_indices;
    selection_indices.reserve(selected.count());

    for (Index i = 0; i < selected.size(); ++i)
        if (selected(i))
            selection_indices.push_back(i);

    return selection_indices;
}

VectorB GeneticAlgorithm::crossover(const VectorB& parent_1, const VectorB& parent_2)
{
    const Index genes_number = get_genes_number();

    VectorB descendent(genes_number);
    descendent.setConstant(false);

    vector<Index> intersection, difference;
    intersection.reserve(genes_number);
    difference.reserve(genes_number);

    for (Index i = 0; i < genes_number; ++i)
    {
        const bool parent_1_gene = parent_1(i);
        const bool parent_2_gene = parent_2(i);

        if (parent_1_gene && parent_2_gene)
        {
            intersection.push_back(i);
            descendent(i) = true;
        }
        else if (parent_1_gene != parent_2_gene)
            difference.push_back(i);
    }

    const Index current_size = intersection.size();

    if (current_size > maximum_inputs_number)
    {
        shuffle_vector(intersection);

        descendent.setConstant(false);
        for (Index i = 0; i < maximum_inputs_number; ++i)
            descendent(intersection[i]) = true;

        return descendent;
    }

    const Index parents_max_size = max<Index>(parent_1.count(), parent_2.count());
    const Index size_lower = max(minimum_inputs_number, current_size);
    const Index size_upper = min(maximum_inputs_number, max<Index>(size_lower, parents_max_size));
    const Index target_size = random_integer(size_lower, size_upper);

    shuffle_vector(difference);
    const Index genes_to_add = target_size - current_size;

    const size_t add_count = genes_to_add > 0
        ? min(static_cast<size_t>(genes_to_add), difference.size())
        : size_t(0);
    for (size_t i = 0; i < add_count; ++i)
        descendent(difference[i]) = true;

    if (const Index final_count = descendent.count(); final_count < minimum_inputs_number)
    {
        vector<Index> never_true_indices;
        never_true_indices.reserve(genes_number);
        for (Index i = 0; i < genes_number; ++i)
            if (!parent_1(i) && !parent_2(i))
                never_true_indices.push_back(i);

        shuffle_vector(never_true_indices);

        const Index genes_needed = minimum_inputs_number - final_count;

        const size_t fill_count = genes_needed > 0
            ? min(static_cast<size_t>(genes_needed), never_true_indices.size())
            : size_t(0);
        for (size_t i = 0; i < fill_count; ++i)
            descendent(never_true_indices[i]) = true;
    }

    return descendent;
}

void GeneticAlgorithm::perform_crossover()
{
    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    const vector<Index> selected_individual_indices = get_selected_indices();

    MatrixB new_population(individuals_number, genes_number);

    if (elitism_size > 0)
    {
        const Index elite_count = min(elitism_size, individuals_number);

        vector<pair<float, Index>> fitness_indexed(individuals_number);
        for (Index i = 0; i < individuals_number; ++i)
            fitness_indexed[i] = {fitness(i), i};

        partial_sort(fitness_indexed.begin(),
                     fitness_indexed.begin() + elite_count,
                     fitness_indexed.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

        for (Index i = 0; i < elite_count; ++i)
            new_population.row(i) = population.row(fitness_indexed[i].second);
    }

    for (Index i = elitism_size; i < individuals_number; ++i)
    {
        const Index parent_index_1 = get_random_element(selected_individual_indices);
        Index parent_index_2 = get_random_element(selected_individual_indices);

        while (selected_individual_indices.size() > 1 && parent_index_1 == parent_index_2)
            parent_index_2 = get_random_element(selected_individual_indices);

        new_population.row(i) = crossover(population.row(parent_index_1), population.row(parent_index_2));
    }

    population.swap(new_population);
}

void GeneticAlgorithm::perform_mutation()
{
    if (mutation_rate == 0) return;

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    for (Index i = 0; i < individuals_number; ++i)
    {
        VectorB individual = population.row(i);

        vector<Index> active, inactive;
        active.reserve(genes_number);
        inactive.reserve(genes_number);
        for (Index j = 0; j < genes_number; ++j)
            (individual(j) ? active : inactive).push_back(j);

        const Index size = Index(active.size());

        vector<Index> to_remove;
        for (const Index a : active)
            if (random_uniform(0.0, 1.0) < mutation_rate) to_remove.push_back(a);

        Index additions = 0;
        for (Index t = 0; t < size; ++t)
            if (random_uniform(0.0, 1.0) < mutation_rate) ++additions;

        shuffle_vector(to_remove);
        shuffle_vector(inactive);

        Index current = size;
        for (const Index r : to_remove)
        {
            if (current <= minimum_inputs_number) break;
            individual(r) = false;
            --current;
        }
        for (Index a = 0; a < additions && a < Index(inactive.size()); ++a)
        {
            if (current >= maximum_inputs_number) break;
            individual(inactive[a]) = true;
            ++current;
        }

        population.row(i) = individual;
    }
}

InputsSelectionResult GeneticAlgorithm::perform_input_selection()
{
    Loss* loss = training_strategy->get_loss();

    Dataset* dataset = loss->get_dataset();


    original_input_indices = dataset->get_variable_indices("Input");
    original_target_indices = dataset->get_variable_indices("Target");
    const vector<Index> time_variable_indices = dataset->get_variable_indices("Time");

    throw_if(folds_number <= 1 && !dataset->has_validation(),
             "dataset has no validation samples. "
             "The genetic algorithm uses validation error to rank individuals.");

    InputsSelectionResult input_selection_results(maximum_epochs);

    if (display) cout << "Performing genetic input selection...\n" << "\n";

    initialize_population();


    if (dataset->has_nan())
        dataset->scrub_missing_values();


    NeuralNetwork* neural_network = loss->get_neural_network();

    time_t beginning_time, current_time;
    float elapsed_time = 0.0f;
    Index best_generation = 0;

    time(&beginning_time);

    for (Index epoch = 0; epoch < maximum_epochs; ++epoch)
    {
        if (display) cout << "Generation: " << epoch + 1 << "\n";

        input_selection_results.resize_history(input_selection_results.mean_training_error_history.size() + 1);

        evaluate_population();


        const Index optimal_individual_index = minimal_index(validation_errors);

        const float optimal_training_error = training_errors(optimal_individual_index);
        const float optimal_validation_error = validation_errors(optimal_individual_index);

        input_selection_results.training_error_history(epoch) = optimal_training_error;
        input_selection_results.validation_error_history(epoch) = optimal_validation_error;
        input_selection_results.mean_validation_error_history(epoch) = validation_errors.mean();
        input_selection_results.mean_training_error_history(epoch) = training_errors.mean();

        dataset->set_variable_indices(original_input_indices, original_target_indices);

        if (optimal_validation_error < input_selection_results.optimum_validation_error)
        {
            best_generation = epoch;

            input_selection_results.optimal_inputs = population.row(optimal_individual_index);

            const vector<Index> best_input_indices = genes_to_variable_indices(input_selection_results.optimal_inputs);

            dataset->set_variable_indices(best_input_indices, original_target_indices);

            input_selection_results.optimal_input_variable_names
                = dataset->get_variable_names("Input");

            dataset->set_variable_indices(original_input_indices, original_target_indices);

            input_selection_results.optimal_parameters = individual_parameters(optimal_individual_index);
            input_selection_results.optimum_training_error = optimal_training_error;
            input_selection_results.optimum_validation_error = optimal_validation_error;
        }

        time(&current_time);

        elapsed_time = float(difftime(current_time, beginning_time));

        if (display)
            cout << "\n"
                 << "Epoch number: " << epoch << "\n"
                 << "Generation mean training error: " << training_errors.mean() << "\n"
                 << "Generation mean validation error: " << input_selection_results.mean_validation_error_history(epoch) << "\n"
                 << "Generation minimum training error: " << optimal_training_error << "\n"
                 << "Generation minimum validation error: " << optimal_validation_error << "\n"
                 << "Best ever training error: " << input_selection_results.optimum_training_error << "\n"
                 << "Best ever validation error: " << input_selection_results.optimum_validation_error << "\n"
                 << "Elapsed time: " << get_time(elapsed_time) << "\n"
                 << "Best generation by validation error: " << best_generation << "\n";


        if (input_selection_results.optimum_validation_error <= validation_error_goal)
        {
            if (display) cout << "Epoch " << epoch << "\nValidation error goal reached: " << input_selection_results.optimum_validation_error << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::ValidationErrorGoal;
        }
        else if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << get_time(elapsed_time) << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
        }
        else if (epoch >= maximum_epochs - 1)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch + 1 << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
        }

        if (input_selection_results.stopping_condition)
        {
            input_selection_results.elapsed_time = get_time(elapsed_time);
            input_selection_results.resize_history(epoch + 1);
            break;
        }

        assign_fitness();

        perform_selection();

        perform_crossover();

        perform_mutation();
    }


    const vector<Index> optimal_variable_indices = genes_to_variable_indices(input_selection_results.optimal_inputs);

    dataset->set_variable_indices(optimal_variable_indices, original_target_indices);

    auto* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
    const vector<string> input_variable_scalers = tabular_dataset ? tabular_dataset->get_feature_scalers("Input") : vector<string>{};

    const vector<Descriptives> input_variable_descriptives = tabular_dataset ? tabular_dataset->calculate_feature_descriptives("Input") : vector<Descriptives>{};

    const Index optimal_variables_number = dataset->get_features_number("Input");


    const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    if (time_series_dataset && time_variable_indices.size() == 1)
        dataset->set_variable_role(time_variable_indices[0], "Time");

    configure_neural_network_inputs(neural_network, dataset, optimal_variables_number);

    if (auto* scaling_layer = dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling)))
    {
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }

    if (input_selection_results.optimal_parameters.size() == neural_network->get_parameters_size())
    {
        neural_network->set_parameters(input_selection_results.optimal_parameters);
    }
    else if (folds_number > 1)
    {
        if (display) cout << "Refitting the final model on all development samples.\n";
        refit_final_model_on_development(training_strategy, folds_number);
    }
    else
    {
        if (display) cout << "Refitting the final model on the selected inputs.\n";
        neural_network->set_parameters_random();
        training_strategy->train();
    }

    if (display)
    {
        input_selection_results.print();
        cout << "Selected generation: " << best_generation << "\n";
    }

    return input_selection_results;
}

void GeneticAlgorithm::to_JSON(JsonWriter& printer) const
{
    printer.open_element("GeneticAlgorithm");

    write_json(printer, {
        {"PopulationSize", to_string(get_individuals_number())},
        {"ElitismSize", to_string(elitism_size)},
        {"MutationRate", to_string(mutation_rate)},
        {"ValidationErrorGoal", to_string(validation_error_goal)},
        {"MinimumInputsNumber", to_string(minimum_inputs_number)},
        {"MaximumInputsNumber", to_string(maximum_inputs_number)},
        {"MaximumGenerationsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)},
        {"FoldsNumber", to_string(folds_number)}});

    printer.close_element();
}

void GeneticAlgorithm::from_JSON(const JsonDocument& document)
{
    const Json* root = get_json_root(document, "GeneticAlgorithm");
    set_individuals_number(read_json_index(root, "PopulationSize"));
    set_mutation_rate(read_json_float(root, "MutationRate"));
    set_elitism_size(read_json_index(root, "ElitismSize"));
    set_validation_error_goal(read_json_float(root,
        root->has("ValidationErrorGoal") ? "ValidationErrorGoal" : "SelectionErrorGoal"));
    set_minimum_inputs_number(read_json_index(root, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_json_index(root, "MaximumInputsNumber"));
    set_maximum_epochs(read_json_index(root, "MaximumGenerationsNumber"));
    set_maximum_time(read_json_float(root, "MaximumTime"));

    if (root->has("FoldsNumber"))
        set_folds_number(read_json_index(root, "FoldsNumber"));
}

REGISTER(InputsSelection, GeneticAlgorithm, "GeneticAlgorithm");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/genetic_algorithm.h"

using namespace opennn;


TEST(GeneticAlgorithmTest, DefaultConstructor)
{
    GeneticAlgorithm genetic_algorithm;

    EXPECT_EQ(genetic_algorithm.has_training_strategy(), false);
    EXPECT_EQ(genetic_algorithm.get_population().dimension(0), 0);
}


TEST(GeneticAlgorithmTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;
    
    GeneticAlgorithm genetic_algorithm(&training_strategy);

    EXPECT_EQ(genetic_algorithm.has_training_strategy(), true);
}


TEST(GeneticAlgorithmTest, InitializePopulationRandom)
{
    const Index individuals_number = 5000;
    const Index inputs_number = 20;
    const Index targets_number = 1;
    const Index samples_number = 10;

    const Index min_features_to_select = 5;
    const Index max_features_to_select = 10;

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_display(false);

    ApproximationNetwork neural_network({ inputs_number }, { 1 }, { targets_number });
    TrainingStrategy training_strategy(&neural_network, &dataset);
    GeneticAlgorithm genetic_algorithm(&training_strategy);
    
    genetic_algorithm.set_individuals_number(individuals_number);
    genetic_algorithm.set_minimum_inputs_number(min_features_to_select);
    genetic_algorithm.set_maximum_inputs_number(max_features_to_select);

    genetic_algorithm.initialize_population_random();
    
    Tensor<bool, 2> population = genetic_algorithm.get_population();

    EXPECT_EQ(population.dimension(0), individuals_number);
    EXPECT_EQ(population.dimension(1), inputs_number);

    Tensor<double, 1> counts(inputs_number);
    counts.setZero();

    for (Index i = 0; i < individuals_number; i++) {
        Index active_genes_in_individual = 0;
        for (Index j = 0; j < inputs_number; j++) {
            if (population(i, j)) {
                counts(j)++;
                active_genes_in_individual++;
            }
        }
        EXPECT_GE(active_genes_in_individual, min_features_to_select);
        EXPECT_LE(active_genes_in_individual, max_features_to_select);
    }

    double average_features = (min_features_to_select + max_features_to_select) / 2.0;
    double total_expected_features = individuals_number * average_features;
    double expected_count_per_gene = total_expected_features / inputs_number;

    double tolerance = 0.20;
    for (Index j = 0; j < inputs_number; j++) 
        EXPECT_NEAR(counts(j), expected_count_per_gene, expected_count_per_gene * tolerance);
}


TEST(GeneticAlgorithmTest, InitializePopulationCorrelations)
{
    const Index individuals_number = 10000;
    const Index inputs_number = 3;
    const Index targets_number = 1;
    const Index total_variables = inputs_number + targets_number;
    const Index samples_number = 100;

    const Index min_features_to_select = 2;
    const Index max_features_to_select = 3;

    Dataset dataset(samples_number, { inputs_number }, { targets_number });
    dataset.set_display(false);

    Tensor<type, 2> full_data(samples_number, total_variables);

    for (Index i = 0; i < samples_number; ++i) {
        type value = static_cast<type>(i);
        full_data(i, 0) = static_cast<type>(rand()) / RAND_MAX;
        full_data(i, 1) = value;                              
        full_data(i, 2) = value / 10.0 + (static_cast<type>(rand()) / RAND_MAX) * (samples_number / 10.0);
        full_data(i, 3) = value;                             
    }

    dataset.set_data(full_data);
    dataset.set_raw_variable_use(0, "Input");
    dataset.set_raw_variable_use(1, "Input");
    dataset.set_raw_variable_use(2, "Input");
    dataset.set_raw_variable_use(3, "Target");

    ApproximationNetwork neural_network({ inputs_number }, { 1 }, { targets_number });
    TrainingStrategy training_strategy(&neural_network, &dataset);
    GeneticAlgorithm genetic_algorithm(&training_strategy);

    genetic_algorithm.set_individuals_number(individuals_number);

    genetic_algorithm.set_minimum_inputs_number(min_features_to_select);
    genetic_algorithm.set_maximum_inputs_number(max_features_to_select);

    genetic_algorithm.initialize_population_correlations();

    Tensor<bool, 2> population = genetic_algorithm.get_population();

    EXPECT_EQ(population.dimension(0), individuals_number);
    EXPECT_EQ(population.dimension(1), inputs_number);

    Tensor<double, 1> counts(inputs_number);
    counts.setZero();
    bool found_min = false;
    bool found_max = false;

    for (Index i = 0; i < individuals_number; i++) {
        Index active_genes_in_individual = 0;
        for (Index j = 0; j < inputs_number; j++) {
            if (population(i, j)) {
                counts(j)++;
                active_genes_in_individual++;
            }
        }

        EXPECT_GE(active_genes_in_individual, min_features_to_select);
        EXPECT_LE(active_genes_in_individual, max_features_to_select);

        if (active_genes_in_individual == min_features_to_select) found_min = true;
        if (active_genes_in_individual == max_features_to_select) found_max = true;
    }

    EXPECT_TRUE(found_min);
    EXPECT_TRUE(found_max);

    EXPECT_GT(counts(1), counts(2));
    EXPECT_GT(counts(2), counts(0));
}


TEST(GeneticAlgorithmTest, Selection)
{
    const Index individuals_number = 4;
    const Index inputs_number = 3;
    Index elitism_size = 0;

    Dataset dataset(10, { inputs_number }, { 1 });
    ApproximationNetwork neural_network({ inputs_number }, { 1 }, { 1 });
    TrainingStrategy training_strategy(&neural_network, &dataset);
   
    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_individuals_number(individuals_number);
    genetic_algorithm.initialize_population_random();

    Tensor<type, 1> simulated_fitness(individuals_number);
    
    simulated_fitness.setValues({
        0.5,  // Low Fitness
        2.5,  // Medium Fitness
        10.0, // Very high Fitness
        7.5   // High Fitness
        });
    
    genetic_algorithm.set_fitness(simulated_fitness);
    
    genetic_algorithm.set_elitism_size(elitism_size);
    
    const int num_trials = 1000; 
    Tensor<int, 1> selection_counts(individuals_number);
    selection_counts.setZero();

    for (int i = 0; i < num_trials; ++i) {
        genetic_algorithm.perform_selection();

        Tensor<bool, 1> current_selection = genetic_algorithm.get_selection();

        ASSERT_EQ(current_selection.dimension(0), individuals_number);

        for (Index j = 0; j < individuals_number; ++j) {
            if (current_selection(j)) {
                selection_counts(j)++;
            }
        }
    }

    EXPECT_GT(selection_counts(2), selection_counts(3));
    EXPECT_GT(selection_counts(3), selection_counts(1));
    EXPECT_GT(selection_counts(1), selection_counts(0));

    elitism_size = 1;
    genetic_algorithm.set_elitism_size(elitism_size);

    genetic_algorithm.perform_selection();

    Tensor<bool, 1> selection_with_elitism = genetic_algorithm.get_selection();

    EXPECT_TRUE(selection_with_elitism(2));
}


TEST(GeneticAlgorithmTest, Crossover)
{
    const Index individuals_number = 4;
    const Index inputs_number = 8;
    Index elitism_size = 0;
    
    Dataset dataset(10, {inputs_number}, {1});
    ApproximationNetwork neural_network({inputs_number}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_individuals_number(individuals_number);
    genetic_algorithm.set_elitism_size(elitism_size);
    
    Tensor<bool, 2> initial_population(individuals_number, inputs_number);
    initial_population.setValues({ {true, true, true, true, true, true, true, true},
                                   {false, false, false, false, false, false, false, false},
                                   {false, false, false, false, false, false, false, false},
                                   {false, false, false, false, false, false, false, false} });
    genetic_algorithm.set_population(initial_population);
    
    Tensor<bool, 1> forced_selection(individuals_number);
    forced_selection.setValues({false, true, false, true}); 
    genetic_algorithm.set_selection(forced_selection); 

    genetic_algorithm.perform_crossover();

    Tensor<bool, 2> new_population = genetic_algorithm.get_population();
    const Tensor<bool, 1> child = new_population.chip(0, 0);
    const Tensor<bool, 1> parent1 = initial_population.chip(0, 0);
    const Tensor<bool, 1> parent2 = initial_population.chip(1, 0);

    bool is_valid_crossover = true;
    bool is_clone_p1 = true;
    bool is_clone_p2 = true;

    for (Index j = 0; j < inputs_number; ++j) {
        if (child(j) != parent1(j) && child(j) != parent2(j)) {
            is_valid_crossover = false;
            break;
        }

        if (child(j) != parent1(j)) is_clone_p1 = false;
        if (child(j) != parent2(j)) is_clone_p2 = false;
    }

    bool is_mixed = is_valid_crossover && !is_clone_p1 && !is_clone_p2;

    EXPECT_TRUE(is_mixed);
}


TEST(GeneticAlgorithmTest, Mutation)
{
    const Index individuals_number = 10;
    const Index inputs_number = 100;

    Dataset dataset(10, { inputs_number }, { 1 });
    ApproximationNetwork neural_network({ inputs_number }, { 1 }, { 1 });
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GeneticAlgorithm genetic_algorithm(&training_strategy);
    genetic_algorithm.set_individuals_number(individuals_number);

    Tensor<bool, 2> population(individuals_number, inputs_number);
    population.setRandom();
    Tensor<bool, 2> original_population = population;
    genetic_algorithm.set_population(original_population);

    genetic_algorithm.set_mutation_rate(0.0);
    genetic_algorithm.perform_mutation();
    Tensor<bool, 2> mutated_population_zero_rate = genetic_algorithm.get_population();

    bool are_equal = true;
    for (Index i = 0; i < individuals_number; ++i) 
    {
        for (Index j = 0; j < inputs_number; ++j) 
        {
            if (original_population(i, j) != mutated_population_zero_rate(i, j)) 
            {
                are_equal = false;
                break;
            }
        }
        if (!are_equal) 
            break;
    }
    EXPECT_TRUE(are_equal);

    genetic_algorithm.set_population(original_population);
    genetic_algorithm.set_mutation_rate(0.5);
    genetic_algorithm.perform_mutation();
    Tensor<bool, 2> mutated_population_high_rate = genetic_algorithm.get_population();

    Index mutated_genes = 0;
    for (Index i = 0; i < individuals_number; i++)
        for (Index j = 0; j < inputs_number; j++)
            if (original_population(i, j) != mutated_population_high_rate(i, j))
                mutated_genes++;

    const double total_genes = individuals_number * inputs_number;
    const double expected_mutations = total_genes * 0.5;

    EXPECT_NEAR(mutated_genes, expected_mutations, total_genes * 0.1);
}


TEST(GeneticAlgorithmTest, InputSelection_StopsByErrorGoal)
{
    const Index inputs_number = 3;
    Dataset dataset(20, { inputs_number }, { 1 });

    Tensor<type, 2> data(20, inputs_number + 1);
    for (Index i = 0; i < 20; i++) {
        data(i, 0) = type(i) / 20.0;
        data(i, 1) = type(10.0);
        data(i, 2) = type(10.0);
        data(i, 2) = type(i) / 20.0;
    }
    dataset.set_data(data);
    dataset.set_raw_variable_use(0, "Input");
    dataset.set_raw_variable_use(1, "Input");
    dataset.set_raw_variable_use(2, "None");
    dataset.set_raw_variable_use(3, "Target");

    ApproximationNetwork neural_network(dataset.get_input_dimensions(), {6}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);
    GeneticAlgorithm genetic_algorithm(&training_strategy);

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(6);
    genetic_algorithm.set_maximum_epochs_number(10);
    genetic_algorithm.set_selection_error_goal(1.0);

    InputsSelectionResults input_selection_results = genetic_algorithm.perform_input_selection();

    EXPECT_EQ(input_selection_results.stopping_condition, InputsSelection::StoppingCondition::SelectionErrorGoal);
    ASSERT_GT(input_selection_results.selection_error_history.dimension(0), 0);
    EXPECT_LE(input_selection_results.selection_error_history(input_selection_results.selection_error_history.dimension(0) - 1), 0.1);
}


TEST(GeneticAlgorithmTest, InputSelection_StopsByMaxEpochs)
{
    const Index inputs_number = 2;
    Dataset dataset(20, { inputs_number }, { 1 });

    Tensor<type, 2> data(20, inputs_number + 1);
    for (Index i = 0; i < 20; i++) {
        data(i, 0) = type(i) / 20.0;
        data(i, 1) = type(10.0);
        data(i, 2) = type(i) / 20.0;
    }
    dataset.set_data(data);
    dataset.set_raw_variable_use(0, "Input");
    dataset.set_raw_variable_use(1, "Input");
    dataset.set_raw_variable_use(2, "Target");

    ApproximationNetwork neural_network(dataset.get_input_dimensions(), {6}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);
    GeneticAlgorithm genetic_algorithm(&training_strategy);

    genetic_algorithm.set_display(false);
    genetic_algorithm.set_individuals_number(6);
    genetic_algorithm.set_maximum_epochs_number(1);
    genetic_algorithm.set_selection_error_goal(0.0);

    InputsSelectionResults input_selection_results = genetic_algorithm.perform_input_selection();

    EXPECT_EQ(input_selection_results.stopping_condition, InputsSelection::StoppingCondition::MaximumEpochs);
    EXPECT_EQ(input_selection_results.get_epochs_number(), 1);
}
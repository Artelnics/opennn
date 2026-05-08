//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file genetic_algorithm.h
 * @brief Declares the GeneticAlgorithm input-selection method.
 */

#pragma once

#include "inputs_selection.h"

namespace opennn
{

class NeuralNetwork;
class Dataset;

/**
 * @class GeneticAlgorithm
 * @brief Genetic-algorithm based input feature selection.
 *
 * Encodes each candidate input subset as a binary chromosome (one gene per
 * input variable), evolves a population of chromosomes through selection,
 * crossover and mutation, and returns the subset that achieved the lowest
 * validation error.
 */
class GeneticAlgorithm final : public InputsSelection
{

public:

    /**
     * @brief Constructs the algorithm.
     * @param training_strategy Training strategy used to evaluate candidates.
     */
    GeneticAlgorithm(TrainingStrategy* training_strategy = nullptr);
    /** @brief Number of individuals in the current population. */
    Index get_individuals_number() const { return population.rows(); }

    /** @brief Number of genes per chromosome (= number of original inputs). */
    Index get_genes_number() const { return original_input_variable_indices.size(); }

    /** @brief Resets all hyperparameters to their default values. */
    void set_default();

    /** @brief Minimum number of selected inputs allowed in any individual. */
    Index get_minimum_inputs_number() const override { return minimum_inputs_number; }
    /** @brief Maximum number of selected inputs allowed in any individual. */
    Index get_maximum_inputs_number() const override { return maximum_inputs_number; }

    /**
     * @brief Sets the minimum number of selected inputs.
     * @param new_minimum New lower bound on the number of selected inputs.
     */
    void set_minimum_inputs_number(const Index new_minimum) { minimum_inputs_number = new_minimum; }
    /**
     * @brief Sets the maximum number of selected inputs.
     *
     * Receives the new upper bound on the number of selected inputs.
     */
    void set_maximum_inputs_number(const Index);

    /**
     * @brief Sets the population size.
     * @param new_individuals_number Number of individuals to maintain.
     */
    void set_individuals_number(const Index new_individuals_number = 4);

    /**
     * @brief Sets the population initialization strategy.
     * @param method Strategy name ("Random" or "Correlations").
     */
    void set_initialization_method(string method) { initialization_method = move(method); }

    /**
     * @brief Sets the mutation rate.
     * @param rate Per-gene flip probability, clamped to [0, 1].
     */
    void set_mutation_rate(const float rate) { mutation_rate = clamp(rate, 0.0f, 1.0f); }

    /**
     * @brief Sets the elitism size.
     * @param size Number of best individuals carried unchanged to the next
     *             generation, clamped to [0, individuals_number].
     */
    void set_elitism_size(const Index size) { elitism_size = clamp<Index>(size, 0, get_individuals_number()); }

    /**
     * @brief Runs the genetic algorithm.
     * @return Best-of-run input subset and supporting statistics.
     */
    InputsSelectionResults perform_input_selection() override;

    /**
     * @brief Loads hyperparameters from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&) override;

    /**
     * @brief Writes hyperparameters to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const override;

private:

    /** @brief Dispatches to the configured initialization method. */
    void initialize_population();
    /** @brief Random initialization of the population chromosomes. */
    void initialize_population_random();
    /** @brief Initialization biased by feature-target correlation. */
    void initialize_population_correlations();
    /** @brief Trains the network for each individual and stores its errors. */
    void evaluate_population();
    /** @brief Computes a fitness score per individual from its errors. */
    void assign_fitness();
    /** @brief Selects parents using fitness-proportional sampling. */
    void perform_selection();
    /**
     * @brief Single-point crossover between two parent chromosomes.
     * @param parent_a First parent.
     * @param parent_b Second parent.
     * @return Child chromosome combining genes from both parents.
     */
    VectorB crossover(const VectorB& parent_a, const VectorB& parent_b);
    /** @brief Generates the next generation via crossover. */
    void perform_crossover();
    /** @brief Applies mutation to non-elite individuals. */
    void perform_mutation();
    /**
     * @brief Returns the indices of the currently selected individuals.
     * @return Indices into @ref population.
     */
    vector<Index> get_selected_indices() const;
    /**
     * @brief Configures the network's input layer for the i-th individual.
     * @param network Network whose input layer is reconfigured.
     * @param dataset Dataset whose input variables are filtered by the chromosome.
     * @param individual_index Row of @ref population to use.
     */
    void configure_neural_network_inputs(NeuralNetwork* network,
                                         Dataset* dataset,
                                         Index individual_index);

    /** @brief Cached per-individual parameters (used to restore the best individual). */
    Tensor<VectorR, 1> individual_parameters;

    /** @brief Indices of the input variables in the original dataset. */
    vector<Index> original_input_variable_indices;
    /** @brief Indices of the target variables in the original dataset. */
    vector<Index> original_target_variable_indices;

    /** @brief One row per individual; one column per gene (input variable). */
    MatrixB population;

    /** @brief Final training error for each individual after evaluate_population(). */
    VectorR training_errors;

    /** @brief Final validation error for each individual after evaluate_population(). */
    VectorR validation_errors;

    /** @brief Per-individual fitness score derived from validation errors. */
    VectorR fitness;

    /** @brief Mask of currently selected parents. */
    VectorB selected;

    /** @brief Lower bound on the number of selected inputs in any individual. */
    Index minimum_inputs_number = 1;
    /** @brief Upper bound on the number of selected inputs in any individual. */
    Index maximum_inputs_number;

    /** @brief Per-gene flip probability used by perform_mutation(). */
    float mutation_rate;

    /** @brief Number of best individuals carried unchanged to the next generation. */
    Index elitism_size;

    /** @brief Population initialization strategy name. */
    string initialization_method;
};

}

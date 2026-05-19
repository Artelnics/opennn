//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "inputs_selection.h"

namespace opennn
{

/// @brief Selects the optimal subset of input features using an evolutionary genetic algorithm.
class GeneticAlgorithm final : public InputsSelection
{

public:

    /// @brief Constructs the algorithm bound to an optional training strategy.
    GeneticAlgorithm(TrainingStrategy* = nullptr);
    Index get_individuals_number() const { return population.rows(); }

    Index get_genes_number() const { return original_input_variable_indices.size(); }

    /// @brief Restores default population size, mutation rate, elitism and other parameters.
    void set_default();

    Index get_minimum_inputs_number() const override { return minimum_inputs_number; }
    Index get_maximum_inputs_number() const override { return maximum_inputs_number; }

    void set_minimum_inputs_number(const Index new_minimum) { minimum_inputs_number = new_minimum; }
    /// @brief Sets the upper bound on the number of selected inputs.
    void set_maximum_inputs_number(const Index);

    /// @brief Sets the size of the population evolved by the algorithm.
    void set_individuals_number(const Index new_individuals_number = 4);

    void set_initialization_method(string method) { initialization_method = move(method); }

    void set_mutation_rate(const float rate) { mutation_rate = clamp(rate, 0.0f, 1.0f); }

    void set_elitism_size(const Index size) { elitism_size = clamp<Index>(size, 0, get_individuals_number()); }

    /// @brief Runs the genetic algorithm until the stopping criterion is met.
    /// @return Selection results including the best individual's inputs and error history.
    InputsSelectionResults perform_input_selection() override;

    /// @brief Loads algorithm configuration from a JSON document.
    void from_JSON(const JsonDocument&) override;

    /// @brief Writes algorithm configuration to a JSON writer.
    void to_JSON(JsonWriter&) const override;

private:

    void initialize_population();
    void initialize_population_random();
    void initialize_population_correlations();
    void evaluate_population();
    void assign_fitness();
    void perform_selection();
    VectorB crossover(const VectorB&, const VectorB&);
    void perform_crossover();
    void perform_mutation();
    vector<Index> get_selected_indices() const;
    void configure_neural_network_inputs(NeuralNetwork*, Dataset*, Index);

    Tensor<VectorR, 1> individual_parameters;

    vector<Index> original_input_variable_indices;
    vector<Index> original_target_variable_indices;

    MatrixB population;

    VectorR training_errors;

    VectorR validation_errors;

    VectorR fitness;

    VectorB selected;

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number;

    float mutation_rate;

    Index elitism_size;

    string initialization_method;
};

}

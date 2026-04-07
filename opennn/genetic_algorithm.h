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

class GeneticAlgorithm final : public InputsSelection
{

public:

    GeneticAlgorithm(TrainingStrategy* = nullptr);

    //enum class InitializationMethod{Random,Correlations};

    const MatrixB& get_population() const { return population; }

    const VectorR& get_training_errors() const { return training_errors; }

    const VectorR& get_validation_errors() const { return validation_errors; }

    const VectorR& get_fitness() const { return fitness; }

    const VectorB& get_selection() const { return selection; }

    Index get_individuals_number() const { return population.rows(); }

    Index get_genes_number() const { return original_input_variable_indices.size(); }

    Index get_minimum_inputs_number() const override { return minimum_inputs_number; }
    Index get_maximum_inputs_number() const override { return maximum_inputs_number; }

    const string& get_initialization_method() const { return initialization_method; }

    void set_default();

    void set_minimum_inputs_number(const Index n) { minimum_inputs_number = n; }
    void set_maximum_inputs_number(const Index);

    void set_population(const MatrixB& p) { population = p; }

    void set_individuals_number(const Index new_individuals_number = 4);

    void set_initialization_method(const string& m) { initialization_method = m; }

    void set_mutation_rate(const type r) { mutation_rate = clamp(r, type(0), type(1)); }

    void set_elitism_size(const Index n) { elitism_size = clamp<Index>(n, 0, get_individuals_number()); }

    void set_maximum_epochs(const Index n) { maximum_epochs = n; }

    void set_fitness(const VectorR& f) { fitness = f; }
    void set_selection(const VectorB& s) { selection = s; }

    InputsSelectionResults perform_input_selection() override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

    void initialize_population();
    void initialize_population_random();
    void initialize_population_correlations();
    void evaluate_population();
    void perform_fitness_assignment();
    void perform_selection();
    VectorB cross(const VectorB&, const VectorB&);
    void perform_crossover();
    void perform_mutation();
    vector<Index> get_selected_individual_indices() const;
    vector<Index> get_variable_indices(const VectorB&);

    Tensor<VectorR, 1> parameters;

    vector<Index> original_input_variable_indices;
    vector<Index> original_target_variable_indices;
    
    MatrixB population;

    VectorR training_errors;

    VectorR validation_errors;

    VectorR fitness;

    VectorB selection;

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number;

    type mutation_rate;

    Index elitism_size;

    string initialization_method;
};

}

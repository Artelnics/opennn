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

    Index get_individuals_number() const { return population.rows(); }

    Index get_genes_number() const { return original_input_variable_indices.size(); }

    void set_default();

    Index get_minimum_inputs_number() const override { return minimum_inputs_number; }
    Index get_maximum_inputs_number() const override { return maximum_inputs_number; }

    void set_minimum_inputs_number(const Index n) { minimum_inputs_number = n; }
    void set_maximum_inputs_number(const Index);

    void set_individuals_number(const Index new_individuals_number = 4);

    void set_initialization_method(const string& m) { initialization_method = m; }

    void set_mutation_rate(const type r) { mutation_rate = clamp(r, type(0), type(1)); }

    void set_elitism_size(const Index n) { elitism_size = clamp<Index>(n, 0, get_individuals_number()); }

    InputsSelectionResults perform_input_selection() override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

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

    type mutation_rate;

    Index elitism_size;

    string initialization_method;
};

}

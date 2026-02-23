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

    GeneticAlgorithm(const TrainingStrategy* = nullptr);

    //enum class InitializationMethod{Random,Correlations};

    const MatrixB& get_population() const;

    const VectorR& get_training_errors() const;

    const VectorR& get_validation_errors() const;

    const VectorR& get_fitness() const;

    const VectorB& get_selection() const;

    Index get_individuals_number() const;

    Index get_genes_number() const;

    Index get_minimum_inputs_number() const override;
    Index get_maximum_inputs_number() const override;

    type get_mutation_rate() const;

    Index get_elitism_size() const;

    const string& get_initialization_method() const;

    void set_default();

    void set_minimum_inputs_number(const Index);
    void set_maximum_inputs_number(const Index);

    void set_population(const MatrixB&);

    void set_individuals_number(const Index new_individuals_number = 4);

    void set_initialization_method(const string&);

    void set_mutation_rate(const type);

    void set_elitism_size(const Index);

    void set_maximum_epochs(const Index);

    void set_fitness(const VectorR&); // Used in testing
    void set_selection(const VectorB&); // Used in testing

    void initialize_population();

    void initialize_population_random();
    void initialize_population_correlations();

    void evaluate_population();

    void perform_fitness_assignment();

    void perform_selection();

    VectorB cross(const VectorB&, const VectorB&);

    void perform_crossover();

    void perform_mutation();

    Index get_selected_individuals_number() const;

    vector<Index> get_selected_individual_indices() const;

    vector<Index> get_variable_indices(const VectorB&);

    InputsSelectionResults perform_input_selection() override;

    Tensor<string, 2> to_string_matrix() const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    void print() const override;
    
    void save(const filesystem::path&) const;

    void load(const filesystem::path&);

private:

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

    type mean_training_error;

    type mean_validation_error;
    
    MatrixB optimal_individuals_history;

    type mutation_rate;

    Index elitism_size;

    string initialization_method;
};

}

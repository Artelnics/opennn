//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include "inputs_selection.h"

namespace opennn
{

class GeneticAlgorithm final : public InputsSelection
{

public:

    GeneticAlgorithm(const TrainingStrategy* = nullptr);

    //enum class InitializationMethod{Random,Correlations};

    const Tensor<bool, 2>& get_population() const;

    const Tensor<type, 1>& get_training_errors() const;

    const Tensor<type, 1>& get_selection_errors() const;

    const Tensor<type, 1>& get_fitness() const;

    const Tensor<bool, 1>& get_selection() const;

    Index get_individuals_number() const;

    Index get_genes_number() const;

    const Index& get_minimum_inputs_number() const override;
    const Index& get_maximum_inputs_number() const override;

    const type& get_mutation_rate() const;

    const Index& get_elitism_size() const;

    const string& get_initialization_method() const;

    void set_default();

    void set_minimum_inputs_number(const Index&);
    void set_maximum_inputs_number(const Index&);

    void set_population(const Tensor<bool, 2>&);

    void set_individuals_number(const Index& new_individuals_number = 4);

    void set_initialization_method(const string&);

    void set_mutation_rate(const type&);

    void set_elitism_size(const Index&);

    void set_maximum_epochs_number(const Index&);

    void set_fitness(const Tensor<type, 1>&); // Used in testing
    void set_selection(const Tensor<bool, 1>&); // Used in testing

    void initialize_population();

    void initialize_population_random();
    void initialize_population_correlations();

    void evaluate_population();

    void perform_fitness_assignment();

    void perform_selection();

    void perform_crossover();

    void perform_mutation();

    Index get_selected_individuals_number() const;

    vector<Index> get_selected_individuals_indices() const;

    vector<Index> get_raw_variable_indices(const Tensor<bool, 1>&);

    InputsSelectionResults perform_input_selection() override;

    Tensor<string, 2> to_string_matrix() const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    void print() const override;
    
    void save(const filesystem::path&) const;

    void load(const filesystem::path&);

private:

    Tensor<Tensor<type, 1>, 1> parameters;

    vector<Index> original_input_raw_variable_indices;
    vector<Index> original_target_raw_variable_indices;
    
    Tensor<type, 1> fitness_correlations;

    Tensor<bool, 2> population;

    Tensor<type, 1> training_errors;

    Tensor<type, 1> selection_errors;

    Tensor<type, 1> fitness;

    Tensor<bool, 1> selection;

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number;

    type mean_training_error;

    type mean_selection_error;

    type mean_raw_inputs_number;
    
    Tensor<bool, 2> optimal_individuals_history;

    type mutation_rate;

    Index elitism_size;

    string initialization_method;

    random_device rd;
};

}

#endif

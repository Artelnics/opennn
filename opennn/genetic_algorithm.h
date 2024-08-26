//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

// OpenNN includes

#include "training_strategy.h"
#include "inputs_selection.h"
#include "config.h"

namespace opennn
{

class GeneticAlgorithm : public InputsSelection
{

public:

    // Constructors

    explicit GeneticAlgorithm();

    explicit GeneticAlgorithm(TrainingStrategy*);

    enum class InitializationMethod{Random,Correlations};

    // Get

    const Tensor<bool, 2>& get_population() const;

    const Tensor<type, 1>& get_training_errors() const;

    const Tensor<type, 1>& get_selection_errors() const;

    const Tensor<type, 1>& get_fitness() const;

    const Tensor<bool, 1>& get_selection() const;

    Index get_individuals_number() const;

    Index get_genes_number() const;

    const type& get_mutation_rate() const;

    const Index& get_elitism_size() const;

    const InitializationMethod& get_initialization_method() const;

    // Set

    virtual void set_default() final;

    void set_population(const Tensor<bool, 2>&);

    void set_individuals_number(const Index& new_individuals_number=4);

    void set_genes_number(const Index&);  

    void set_initialization_method(const GeneticAlgorithm::InitializationMethod&);

    void set_training_errors(const Tensor<type, 1>&);

    void set_selection_errors(const Tensor<type, 1>&);

    void set_fitness(const Tensor<type, 1>&);

    void set_mutation_rate(const type&);

    void set_elitism_size(const Index&);

    void set_maximum_epochs_number(const Index&);

//    void set_initial_raw_variables_indices(const Tensor<Index ,1>&);

    // GENETIC OPERATORS METHODS

    // Population

    void initialize_population();

    void initialize_population_random();

    void calculate_inputs_activation_probabilities();
    
    void initialize_population_correlations();

    type generate_random_between_0_and_1();

    void evaluate_population();

    void perform_fitness_assignment();

    Tensor<type, 1> calculate_selection_probabilities();

    // Selection

    void perform_selection();

    Index weighted_random(const Tensor<type, 1>&);

    // Crossover

    void perform_crossover();

    // Mutation

    void perform_mutation();

    // Check 

    void check_categorical_raw_variables();

//    Tensor<bool, 1> get_individual_variables_to_indexes (Tensor<bool, 1>&);

    Tensor<bool, 1> get_individual_raw_variables(Tensor<bool, 1>&);

    Tensor<bool, 1> get_individual_variables(Tensor<bool,1>&);

    Tensor<Index, 1> get_selected_individuals_indices ();

    Tensor<Index, 1> get_individual_as_raw_variables_indexes_from_variables( Tensor<bool, 1>&);

    void set_unused_raw_variables(Tensor<Index, 1>&);

    Tensor<Index, 1> get_original_unused_raw_variables();

    InputsSelectionResults perform_inputs_selection ()  final;

    // Serialization method

    Tensor<string, 2> to_string_matrix() const;

    void from_XML(const tinyxml2::XMLDocument&);

    void to_XML(tinyxml2::XMLPrinter&) const;

    void print() const;
    
    void save(const string&) const;

    void load(const string&);

    Tensor<Tensor<type, 1>, 1> parameters;

private:
    
    Tensor<Index, 1> initial_raw_variables_indices;
    Tensor<bool, 1> original_input_raw_variables;

    Tensor<Index, 1> original_unused_raw_variables_indices;
    Tensor<bool, 1> original_unused_raw_variables;
    
    Tensor<type, 1> inputs_activation_probabilities;

    Tensor<bool, 2> population;

    Tensor<type, 1> training_errors;

    Tensor<type, 1> selection_errors;

    Tensor<type, 1> fitness;

    Tensor<bool, 1> selection;

    type mean_training_error;

    type mean_selection_error;

    type mean_inputs_number;
    
    Tensor<bool, 2> optimal_individuals_history;

    Tensor<Index, 1> original_input_raw_variables_indices;

    Tensor<Index, 1> original_target_raw_variables_indices;

    Index genes_number;

    type mutation_rate;

    Index elitism_size;

    InitializationMethod initialization_method;
};

}

#endif

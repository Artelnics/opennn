//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

// System includes

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

// OpenNN includes

#include "training_strategy.h"
#include "tensor_utilities.h"
#include "inputs_selection.h"
#include "config.h"

namespace OpenNN
{
/// This concrete class represents a genetic algorithm, inspired by the process of natural selection[1] such as mutation,
/// crossover and selection.

///
/// This algorithm are commonly used in optimization and search problems. if the dataset has many inputs,
/// but we do not know how they affect the target,
/// then this algorithm provides the best possible combination of variables to optimize the problem.
///
/// \cite 1 Neural Designer "Genetic Algorithms for Feature Selection."
/// \ref https://www.neuraldesigner.com/blog/genetic_algorithms_for_feature_selection

class GeneticAlgorithm : public InputsSelection
{

public:

    // Constructors

    explicit GeneticAlgorithm();

    explicit GeneticAlgorithm(TrainingStrategy*);

    // Destructor

    virtual ~GeneticAlgorithm();

    // Get methods

    const Tensor<bool, 2>& get_population() const;

    const Tensor<type, 1>& get_fitness() const;
    const Tensor<bool, 1>& get_selection() const;

    Index get_individuals_number() const;
    Index get_genes_number() const;

    const type& get_mutation_rate() const;

    const Index& get_elitism_size() const;

    const type& get_selective_pressure() const;

    const bool& get_reserve_generation_mean() const;

    const bool& get_reserve_generation_minimum_selection() const;

    const bool& get_reserve_generation_optimum_loss() const;

    // Set methods

    void set_default();

    void set_population(const Tensor<bool, 2>&);
    void set_individuals_number(const Index&);

    void set_training_errors(const Tensor<type, 1>&);
    void set_selection_errors(const Tensor<type, 1>&);

    void set_fitness(const Tensor<type, 1>&);

    void set_mutation_rate(const type&);

    void set_elitism_size(const Index&);

    void set_selective_pressure(const type&);

    void set_reserve_generation_mean(const bool&);

    void set_reserve_generation_minimum_selection(const bool&);

    void set_reserve_generation_optimum_loss(const bool&);

    // GENETIC METHODS

    // Population methods

    void initialize_population();

    void evaluate_population();

    void perform_fitness_assignment();

    // Selection methods

    void perform_selection();

    // Crossover methods

    void perform_crossover();

    // Mutation methods

    void perform_mutation();

    // Inputs selection methods

    InputsSelectionResults perform_inputs_selection();

    // Serialization methods

    Tensor<string, 2> to_string_matrix() const;

    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

    void print() const;
    
    void save(const string&) const;
    void load(const string&);

private:

    // Population stuff

    Tensor<Index, 1> original_input_columns_indices;
    Tensor<Index, 1> original_target_columns_indices;

    /// Population matrix.

    Tensor<bool, 2> population;

    /// Fitness of population.

    Tensor<type, 1> fitness;

    Tensor<bool, 1> selection;


    /// Performance of population.

    Tensor<Tensor<type, 1>, 1> parameters;

    Tensor<type, 1> training_errors;
    Tensor<type, 1> selection_errors;


    /// Mutation rate.
    /// The mutation rate value must be between 0 and 1.
    /// This is a parameter of the mutation operator.

    type mutation_rate;

    /// Elitism size.
    /// It represents the number of individuals which will always be selected for recombination.
    /// This is a parameter of the selection operator.

    Index elitism_size;

    /// Linear ranking allows values for the selective pressure greater than 0.
    /// This is a parameter of the fitness assignment operator.

    type selective_pressure;

    // Inputs selection results

    /// True if the mean of selection error are to be reserved in each generation.

    bool reserve_generation_mean;

    /// True if the minimum of selection error are to be reserved in each generation.

    bool reserve_generation_minimum_selection;

    /// True if the optimum of loss are to be reserved in each generation.

    bool reserve_generation_optimum_loss;
};

}

#endif

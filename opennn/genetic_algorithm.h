/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G E N E T I C   A L G O R I T H M   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __GENETICALGORITHM_H__
#define __GENETICALGORITHM_H__

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "training_strategy.h"

#include "inputs_selection_algorithm.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a genetic algorithm for the inputs selection of a neural network.
///

class GeneticAlgorithm : public InputsSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit GeneticAlgorithm();

    // TRAINING STRATEGY CONSTRUCTOR

    explicit GeneticAlgorithm(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit GeneticAlgorithm(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit GeneticAlgorithm(const string&);

    // DESTRUCTOR

    virtual ~GeneticAlgorithm();

    // ENUMERATIONS

    /// Enumeration of available methods for the initialization of the population.

    enum InitializationMethod{Random, Weigthed};

    /// Enumeration of available methods for the crossover of the population.

    enum CrossoverMethod{OnePoint, TwoPoint, Uniform};

    /// Enumeration of available methods for the fitness assignement of the population.

    enum FitnessAssignment{ObjectiveBased, RankBased};

    // STRUCTURES

    ///
    /// This structure contains the training results for the genetic algorithm method.
    ///

    struct GeneticAlgorithmResults : public InputsSelectionAlgorithm::InputsSelectionResults
    {
        /// Default constructor.

        explicit GeneticAlgorithmResults() : InputsSelectionAlgorithm::InputsSelectionResults()
        {
        }

        /// Destructor.

        virtual ~GeneticAlgorithmResults()
        {
        }

        string object_to_string() const;

        /// Values of the minimum loss in each generation.

        Vector<double> generation_optimum_loss_history;

        /// Values of the minimum selection error in each generation.

        Vector<double> generation_minimum_selection_history;

        /// Mean of the selection error in each generation.

        Vector<double> generation_mean_history;

        /// Standard deviation of the selection error in each generation.

        Vector<double> generation_standard_deviation_history;
    };

    // METHODS

    // Get methods

    const Vector< Vector<bool> >& get_population() const;

    const Matrix<double>& get_loss() const;

    const Vector<double>& get_fitness() const;

    const InitializationMethod& get_initialization_method() const;

    const CrossoverMethod& get_crossover_method() const;

    const FitnessAssignment& get_fitness_assignment_method() const;

    const size_t& get_population_size() const;

    const double& get_mutation_rate() const;

    const size_t& get_elitism_size() const;

    const size_t& get_crossover_first_point() const;

    const size_t& get_crossover_second_point() const;

    const double& get_selective_pressure() const;

    const double& get_incest_prevention_distance() const;

    const bool& get_reserve_generation_mean() const;

    const bool& get_reserve_generation_standard_deviation() const;

    const bool& get_reserve_generation_minimum_selection() const;

    const bool& get_reserve_generation_optimum_loss() const;

    string write_initialization_method() const;

    string write_crossover_method() const;

    string write_fitness_assignment_method() const;

    // Set methods

    void set_default();

    void set_population(const Vector< Vector<bool> >&);

    void set_loss(const Matrix<double>&);

    void set_fitness(const Vector<double>&);

    void set_inicialization_method(const InitializationMethod&);
    void set_fitness_assignment_method(const FitnessAssignment&);
    void set_crossover_method(const CrossoverMethod&);

    void set_inicialization_method(const string&);
    void set_fitness_assignment_method(const string&);
    void set_crossover_method(const string&);

    void set_population_size(const size_t&);

    void set_mutation_rate(const double&);

    void set_elitism_size(const size_t&);

    void set_crossover_first_point(const size_t&);

    void set_crossover_second_point(const size_t&);

    void set_selective_pressure(const double&);

    void set_incest_prevention_distance(const double&);

    void set_reserve_generation_mean(const bool&);

    void set_reserve_generation_standard_deviation(const bool&);

    void set_reserve_generation_minimum_selection(const bool&);

    void set_reserve_generation_optimum_loss(const bool&);

    // GENETIC METHODS

    // Population methods

    void initialize_population();

    void initialize_random_population();

    void initialize_weighted_population();

    void evaluate_population();

    void calculate_fitness();

    void calculate_objetive_fitness();

    void calculate_rank_fitness();

    void evolve_population();

    // Selection methods

    void perform_selection();

    // Crossover methods

    void perform_crossover();

    void perform_1point_crossover();

    void perform_2point_crossover();

    void perform_uniform_crossover();

    // Mutation methods

    void perform_mutation();

    // Order selection methods

    size_t get_optimal_individual_index() const;

    GeneticAlgorithmResults* perform_inputs_selection();

    // Serialization methods

    Matrix<string> to_string_matrix() const;

    tinyxml2::XMLDocument* to_XML() const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;
    // void read_XML(   );

    void save(const string&) const;
    void load(const string&);

private:

    // MEMBERS

    // Population stuff

    /// Population matrix.

    Vector< Vector<bool> > population;

    /// Performance of population.

    Matrix<double> loss;

    /// Fitness of population.

    Vector<double> fitness;

    // Training operators

    /// Initialization method used in the algorithm.

    InitializationMethod initialization_method;

    /// Crossover method used in the algorithm.

    CrossoverMethod crossover_method;

    /// Fitness assignment method used in the algorithm.

    FitnessAssignment fitness_assignment_method;

    /// Initial uses of the variables in the data set.

    Vector<Variables::Use> original_uses;

    /// Size of the population.

    size_t population_size;

    /// Incest prevention distance
    /// Distance between two individuals to prevent the crossover.

    double incest_prevention_distance;

    /// Mutation rate.
    /// The mutation rate value must be between 0 and 1.
    /// This is a parameter of the mutation operator.

    double mutation_rate;

    /// Elitism size.
    /// It represents the number of individuals which will always be selected for recombination.
    /// This is a parameter of the selection operator.

    size_t elitism_size;

    /// First point used in the OnePoint and TwoPoint crossover method.
    /// If it is 0 the algorithm selects a random point for each pair of offsprings.

    size_t crossover_first_point;

    /// Second point used in the TwoPoint crossover method.
    /// If it is 0 the algorithm selects a random point for each pair of offsprings.

    size_t crossover_second_point;

    /// Linear ranking allows values for the selective pressure greater than 0.
    /// This is a parameter of the fitness assignment operator.

    double selective_pressure;

    // Inputs selection results

    /// True if the mean of selection error are to be reserved in each generation.

    bool reserve_generation_mean;

    /// True if the standard deviation of selection error are to be reserved in each generation.

    bool reserve_generation_standard_deviation;

    /// True if the minimum of selection error are to be reserved in each generation.

    bool reserve_generation_minimum_selection;

    /// True if the optimum of loss are to be reserved in each generation.

    bool reserve_generation_optimum_loss;

};

}

#endif

/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E V O L U T I O N A R Y   A L G O R I T H M   C L A S S   H E A D E R                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __EVOLUTIONARYALGORITHM_H__
#define __EVOLUTIONARYALGORITHM_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <time.h>

// OpenNN includes

#include "training_algorithm.h"
#include "loss_index.h"

namespace OpenNN
{

///
/// This concrete class represents an evolutionary training algorithm for a loss functional of a neural network.
///

class EvolutionaryAlgorithm : public TrainingAlgorithm
{

public:

   // ENUMERATIONS

   /// Enumeration of the available training operators for fitness assignment.

   enum FitnessAssignmentMethod{LinearRanking};

   /// Enumeration of the available training operators for selection. 

   enum SelectionMethod{RouletteWheel};

   /// Enumeration of the available training operators for recombination.

   enum RecombinationMethod{Line, Intermediate};

   /// Enumeration of the available training operators for mutation.

   enum MutationMethod{Normal, Uniform};


   // DEFAULT CONSTRUCTOR

   explicit EvolutionaryAlgorithm(void);

    // GENERAL CONSTRUCTOR

   explicit EvolutionaryAlgorithm(LossIndex*);

   // XML CONSTRUCTOR

   explicit EvolutionaryAlgorithm(const tinyxml2::XMLDocument&);


   // DESTRUCTOR

   virtual ~EvolutionaryAlgorithm(void);

      // STRUCTURES

   ///
   /// This structure contains the training results for the evolutionary algorithm. 
   ///

   struct EvolutionaryAlgorithmResults : public TrainingAlgorithm::TrainingAlgorithmResults
   {  
       /// Default constructor.

       EvolutionaryAlgorithmResults(void)
       {
           evolutionary_algorithm_pointer = NULL;
       }

       /// Evolutionary algorithm constructor.

       EvolutionaryAlgorithmResults(EvolutionaryAlgorithm* new_evolutionary_algorithm_pointer)
       {
           evolutionary_algorithm_pointer = new_evolutionary_algorithm_pointer;
       }

       /// Destructor.

       virtual ~EvolutionaryAlgorithmResults(void)
       {
       }

       /// Pointer to the evolutionary algorithm object for which the training results are to be stored.

      EvolutionaryAlgorithm* evolutionary_algorithm_pointer;

      // Training history

      /// History of the population matrix over the generations. 

      Vector< Matrix<double> > population_history;

      /// History of the best individual parameters over the generations. 

      Vector< Vector<double> > best_individual_history;

      /// History of the mean norm of the individuals over the generations. 

      Vector<double> mean_norm_history;

      /// History of the standard deviation of the individuals norm over the generations. 

      Vector<double> standard_deviation_norm_history;

      /// History of the norm of the best individual over the generations. 

      Vector<double> best_norm_history;

      /// History of the population loss over the generations. 

      Vector< Vector<double> > loss_history;

      /// History of the mean loss of the individuals over the generations. 

      Vector<double> mean_loss_history;

      /// History of the standard deviation of the population loss over the generations. 

      Vector<double> standard_deviation_loss_history;

      /// History of the loss of the best individual over each generations. 

      Vector<double> best_loss_history;

      /// History of the selection loss of the best individual over each generations.

      Vector<double> selection_loss_history;

      /// History of the elapsed time over the generations.

      Vector<double> elapsed_time_history;

      // Final values

      /// Final mean norm of the population. 

      double final_mean_norm;

      /// Final standard deviation of the population norm. 

      double final_standard_deviation_norm;

      /// Final norm of the best individual ever. 

      double final_best_norm;

      /// Final mean population loss. 

      double final_mean_loss;

      /// Final standard deviation of the population loss. 

      double final_standard_deviation_loss;

      /// Performance of the best individual ever. 

      double final_best_loss;

      /// Selection loss after training.

      double final_selection_loss;

      /// Total elapsed time in the training process.

      double elapsed_time;

      /// Number of generations needed by the evolutionary algorithm.

      size_t generations_number;

      void resize_training_history(const size_t&);
      std::string to_string(void) const;
      Matrix<std::string> write_final_results(const size_t& precision = 3) const;
   };


   // METHODS

   // Get methods

   // Training parameters

   const double& get_warning_parameters_norm(void) const;

   const double& get_error_parameters_norm(void) const;

   // Stopping criteria

   const double& get_best_loss_goal(void) const;
   const size_t& get_maximum_selection_loss_decreases(void) const;

   const size_t& get_maximum_generations_number(void) const;
   const double& get_maximum_time(void) const;

   // Reserve training history

   const bool& get_reserve_selection_loss_history(void) const;

   const bool& get_reserve_elapsed_time_history(void) const;

   // Population methods

   size_t get_population_size(void) const;

   const Matrix<double>& get_population(void) const;

   // Training operators

   const FitnessAssignmentMethod& get_fitness_assignment_method(void) const;
   std::string write_fitness_assignment_method(void) const;

   const SelectionMethod& get_selection_method(void) const;
   std::string write_selection_method(void) const;

   const RecombinationMethod& get_recombination_method(void) const;
   std::string write_recombination_method(void) const;

   const MutationMethod& get_mutation_method(void) const;
   std::string write_mutation_method(void) const;

   // Population values

   const Vector<double>& get_loss(void) const;
   const Vector<double>& get_fitness(void) const;
   const Vector<bool>& get_selection(void) const;

   const size_t& get_elitism_size(void) const;
   const double& get_selective_pressure(void) const;

   const double& get_recombination_size(void) const;
   const double& get_mutation_rate(void) const;
   const double& get_mutation_range(void) const;    
   const double& get_mean_loss_goal(void) const;
   const double& get_standard_deviation_loss_goal(void) const;

   const bool& get_reserve_population_history(void) const;
   const bool& get_reserve_best_individual_history(void) const;
   const bool& get_reserve_mean_norm_history(void) const;
   const bool& get_reserve_standard_deviation_norm_history(void) const;
   const bool& get_reserve_best_norm_history(void) const;

   const bool& get_reserve_mean_loss_history(void) const;
   const bool& get_reserve_standard_deviation_loss_history(void) const;
   const bool& get_reserve_best_loss_history(void) const;

   // Set methods

   void set(void);
   void set(LossIndex*);

   void set_default(void);

   void set_fitness_assignment_method(const FitnessAssignmentMethod&);
   void set_fitness_assignment_method(const std::string&);

   void set_selection_method(const SelectionMethod&);
   void set_selection_method(const std::string&);

   void set_recombination_method(const RecombinationMethod&);
   void set_recombination_method(const std::string&);

   void set_mutation_method(const MutationMethod&);
   void set_mutation_method(const std::string&);

   void set_population_size(const size_t&);

   void set_population(const Matrix<double>&);

   void set_loss(const Vector<double>&);
   void set_fitness(const Vector<double>&);
   void set_selection(const Vector<bool>&);

   void set_elitism_size(const size_t&);
   void set_selective_pressure(const double&);
   void set_recombination_size(const double&);

   void set_mutation_rate(const double&);
   void set_mutation_range(const double&);

   void set_maximum_generations_number(const size_t&);
   void set_mean_loss_goal(const double&);
   void set_standard_deviation_loss_goal(const double&);

   void set_reserve_population_history(const bool&);

   void set_reserve_best_individual_history(const bool&);

   void set_reserve_mean_norm_history(const bool&);
   void set_reserve_standard_deviation_norm_history(const bool&);
   void set_reserve_best_norm_history(const bool&);

   void set_reserve_mean_loss_history(const bool&);
   void set_reserve_standard_deviation_loss_history(const bool&);
   void set_reserve_best_loss_history(const bool&);

   void set_reserve_all_training_history(const bool&);

   // Training parameters

   void set_warning_parameters_norm(const double&);

   void set_error_parameters_norm(const double&);

   // Stopping criteria

   void set_best_loss_goal(const double&);
   void set_maximum_selection_loss_decreases(const size_t&);

   void set_maximum_time(const double&);

   // Reserve training history

   void set_reserve_selection_loss_history(const bool&);

   void set_reserve_elapsed_time_history(const bool&);

   // Utilities

   void set_display_period(const size_t&);

   // Population methods

   Vector<double> get_individual(const size_t&) const;
   void set_individual(const size_t&, const Vector<double>&);

   size_t calculate_best_individual_index(void) const;

   double calculate_mean_loss(void) const;
   double calculate_standard_deviation_loss(void) const;

   // Initialization methods

   void initialize_population(const double&);

   void randomize_population_uniform(void);
   void randomize_population_uniform(const double&, const double&);
   void randomize_population_uniform(const Vector<double>&, const Vector<double>&);

   void randomize_population_normal(void);
   void randomize_population_normal(const double&, const double&);
   void randomize_population_normal(const Vector<double>&, const Vector<double>&);
    
   // Population norm methods

   Vector<double> calculate_population_norm(void) const;

   // Population loss methods

   void perform_fitness_assignment(void);
   void perform_selection(void);
   void perform_recombination(void);
   void perform_mutation(void);

   void evolve_population(void);

   void evaluate_population(void);

   // Fitness assignment methods

   void perform_linear_ranking_fitness_assignment(void);

   // Selection methods

   void perform_roulette_wheel_selection(void);

   // Recombination methods

   void perform_intermediate_recombination(void);
   void perform_line_recombination(void);

   // Mutation methods

   void perform_normal_mutation(void);
   void perform_uniform_mutation(void);

   // Training methods

   EvolutionaryAlgorithmResults* perform_training(void);

   std::string write_training_algorithm_type(void) const;

   // Serialization methods

   Matrix<std::string> to_string_matrix(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

   void initialize_random(void);

private:

   // MEMBERS

   // Population stuff

   /// Population matrix.

   Matrix<double> population;

   /// Performance of population.

   Vector<double> loss;

   /// Fitness of population.

   Vector<double> fitness;

   /// Selected individuals in population.

   Vector<bool> selection;

   // Training operators

   /// Fitness assignment training operators enumeration.

   FitnessAssignmentMethod fitness_assignment_method;

   /// Selection training operators enumeration.

   SelectionMethod selection_method;

   /// Recombination training operators enumeration.

   RecombinationMethod recombination_method;

   /// Mutation training operators enumeration.

   MutationMethod mutation_method;
   
   /// Elitism size.
   /// It represents the number of individuals which will always be selected for recombination.
   /// This is a parameter of the selection operator.

   size_t elitism_size;

   /// Selective pressure. 
   /// Linear ranking allows values for the selective pressure greater than 0.
   /// This is a parameter of the selection operator.

   double selective_pressure;

   /// Recombination size. 
   /// The recombination size value must be equal or greater than 0.
   /// This is a parameter of the recombination operator.

   double recombination_size;

   /// Mutation rate.
   /// The mutation rate value must be between 0 and 1.
   /// This is a parameter of the mutation operator.

   double mutation_rate;

   /// Mutation range.
   /// The mutation range value must be 0 or a positive number.
   /// This is a parameter of the mutation operator.

   double mutation_range;

   /// Value for the parameters norm at which a warning message is written to the screen. 

   double warning_parameters_norm;

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   double error_parameters_norm;


   // STOPPING CRITERIA

   /// Target value for the mean loss of the population.
   /// It is used as a stopping criterion.

   double mean_loss_goal;

   /// Target value for the standard deviation of the population loss.
   /// It is used as a stopping criterion.

   double standard_deviation_loss_goal;

   /// Best goal value for the loss. It is used as a stopping criterion.

   double best_loss_goal;

   /// Maximum number of generations to perform_training.

   size_t maximum_generations_number;

   /// Number of generations where the selection loss increases.
   /// This is an early stopping method for improving selection.

   size_t maximum_selection_loss_decreases;

   /// Maximum training time. It is used as a stopping criterion.

   double maximum_time;

   // Training history

   /// True if the population history, which is a vector of matrices, is to be reserved, false otherwise.
   /// Reserving the population history can be compuationally expensive if the number of parameters,
   /// the population size and the number of generations are big numbers.

   bool reserve_population_history;

   /// True if the history of the best individual ever is to be reserved, and false otherwise.
   /// The best individual history is a vector of vectors.

   bool reserve_best_individual_history;

   /// True if the mean norm history vector is to be reserved, false otherwise.

   bool reserve_mean_norm_history;

   /// True if the standard deviation of norm history vector is to be reserved, false otherwise.

   bool reserve_standard_deviation_norm_history;

   /// True if the best norm history vector is to be reserved, false otherwise.

   bool reserve_best_norm_history;

   /// True if the mean loss history vector is to be reserved, false otherwise.

   bool reserve_mean_loss_history;

   /// True if the standard deviation of loss history vector is to be reserved, false otherwise.

   bool reserve_standard_deviation_loss_history;

   /// True if the best loss history vector is to be reserved, false otherwise.

   bool reserve_best_loss_history;

   /// True if the elapsed time history vector is to be reserved, false otherwise.

   bool reserve_elapsed_time_history;

   /// True if the selection loss history vector is to be reserved, false otherwise.

   bool reserve_selection_loss_history;
};

}

#endif


// OpenNN: An Open Source Neural Networks C++ OpenNN.
// Copyright (c) 2005-2016 Roberto Lopez.
//
// This OpenNN is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This OpenNN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this OpenNN; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


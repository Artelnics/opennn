/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E V O L U T I O N A R Y   A L G O R I T H M   T E S T   C L A S S   H E A D E R                            */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __EVOLUTIONARYALGORITHMTEST_H__
#define __EVOLUTIONARYALGORITHMTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class EvolutionaryAlgorithmTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit EvolutionaryAlgorithmTest();


   // DESTRUCTOR

   virtual ~EvolutionaryAlgorithmTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();

   // Get methods

   void test_get_population_size();

   void test_get_population();

   void test_get_loss();
   void test_get_fitness();
   void test_get_selection();

   void test_get_selective_pressure();
   void test_get_recombination_size();
   void test_get_mutation_rate();
   void test_get_mutation_range();

   void test_get_maximum_generations_number();

   void test_get_reserve_population_history();
   void test_get_reserve_mean_norm_history();
   void test_get_reserve_standard_deviation_norm_history();
   void test_get_reserve_best_norm_history();
   void test_get_reserve_mean_loss_history();
   void test_get_reserve_standard_deviation_loss_history();
   void test_get_reserve_best_loss_history();

   void test_get_fitness_assignment_method();
   void test_get_selection_method();
   void test_get_recombination_method();
   void test_get_mutation_method();

   // Set methods

   void test_set();
   void test_set_default();

   void test_set_population_size();

   void test_set_population();

   void test_set_loss();
   void test_set_fitness();
   void test_set_selection();

   void test_set_selective_pressure();
   void test_set_recombination_size();

   void test_set_mutation_rate();
   void test_set_mutation_range();

   void test_set_maximum_generations_number();
   void test_set_mean_loss_goal();
   void test_set_standard_deviation_loss_goal();

   void test_set_fitness_assignment_method();
   void test_set_selection_method();
   void test_set_recombination_method();
   void test_set_mutation_method();

   void test_set_reserve_population_history();
   void test_set_reserve_mean_norm_history();
   void test_set_reserve_standard_deviation_norm_history();
   void test_set_reserve_best_norm_history();
   void test_set_reserve_mean_loss_history();
   void test_set_reserve_standard_deviation_loss_history();
   void test_set_reserve_best_loss_history();

   void test_set_reserve_all_training_history();

   // Population methods

   void test_get_individual();
   void test_set_individual();

   void test_randomize_population_uniform();
   void test_randomize_population_normal();

   void test_calculate_population_norm();

   // Population evaluation methods

   void test_evaluate_population();

   // Fitness assignment methods

   void test_perform_linear_ranking_fitness_assignment();

   // Selection methods

   void test_perform_elite_selection();

   void test_perform_roulette_wheel_selection();

   // Recombination methods

   void test_perform_intermediate_recombination();
   void test_perform_line_recombination();

   // Mutation methods

   void test_perform_normal_mutation();
   void test_perform_uniform_mutation();

   // Training methods

   void test_perform_training();

   // Training history methods

   void test_get_training_history_XML();

   // Serialization methods

   void test_to_XML();
   void test_from_XML();

   // Unit testing methods

   void run_test_case();

};

#endif


// OpenNN: An Open Source Neural Networks C++ OpenNN.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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

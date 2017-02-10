/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __TRAININGSTRATEGY_H__
#define __TRAININGSTRATEGY_H__

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

#ifdef __OPENNN_MPI__
#include <mpi.h>
#endif
// OpenNN includes

#include "loss_index.h"

#include "training_algorithm.h"

#include "random_search.h"
#include "evolutionary_algorithm.h"

#include "gradient_descent.h"
#include "conjugate_gradient.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"

#include "newton_method.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This class represents the concept of training strategy for a neural network. 
/// A training strategy is composed of three training algorithms:
/// <ul>
/// <li> Initialization training algorithm.
/// <li> Main training algorithm.
/// <li> Refinement trainining algorithm.
/// </ul> 
   
class TrainingStrategy
{

public:

   // DEFAULT CONSTRUCTOR

   explicit TrainingStrategy(void);

   // GENERAL CONSTRUCTOR

   explicit TrainingStrategy(LossIndex*);

   // XML CONSTRUCTOR

   explicit TrainingStrategy(const tinyxml2::XMLDocument&);

   // FILE CONSTRUCTOR

   explicit TrainingStrategy(const std::string&);

   // DESTRUCTOR

   virtual ~TrainingStrategy(void);

   // ENUMERATIONS

    /// Enumeration of all the available types of training algorithms.

    enum InitializationType
    {
       NO_INITIALIZATION,
       RANDOM_SEARCH,
       EVOLUTIONARY_ALGORITHM,
       USER_INITIALIZATION
    };

    /// Enumeration of all the available types of training algorithms.

    enum MainType
    {
       NO_MAIN,
       GRADIENT_DESCENT,
       CONJUGATE_GRADIENT,
       NEWTON_METHOD,
       QUASI_NEWTON_METHOD,
       LEVENBERG_MARQUARDT_ALGORITHM,
       USER_MAIN
    };

    /// Enumeration of all the available types of training algorithms.

    enum RefinementType
    {
       NO_REFINEMENT,
       //NEWTON_METHOD,
       USER_REFINEMENT
    };


   // STRUCTURES 

   /// This structure stores the results from the training strategy.
   /// They are composed of the initialization, refinement and training algorithms results. 

   struct Results
   {
        /// Default constructor.

        explicit Results(void);

        /// Destructor.

        virtual ~Results(void);

        void save(const std::string&) const;

        /// Pointer to a structure with the results from the random search training algorithm.

        RandomSearch::RandomSearchResults* random_search_results_pointer;

        /// Pointer to a structure with the results from the evolutionary training algorithm.

        EvolutionaryAlgorithm::EvolutionaryAlgorithmResults* evolutionary_algorithm_results_pointer;

        /// Pointer to a structure with the results from the gradient descent training algorithm.

        GradientDescent::GradientDescentResults* gradient_descent_results_pointer;

        /// Pointer to a structure with the results from the conjugate gradient training algorithm.

        ConjugateGradient::ConjugateGradientResults* conjugate_gradient_results_pointer;

        /// Pointer to a structure with the results from the quasi-Newton method training algorithm.

        QuasiNewtonMethod::QuasiNewtonMethodResults* quasi_Newton_method_results_pointer;

        /// Pointer to a structure with the results from the Levenberg-Marquardt training algorithm.

        LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults* Levenberg_Marquardt_algorithm_results_pointer;

        /// Pointer to a structure with results from the Newton method training algorithm.

        NewtonMethod::NewtonMethodResults* Newton_method_results_pointer;

  };

   // METHODS

   // Checking methods

   void check_loss_index(void) const;
   void check_training_algorithms(void) const;

   // Initialization methods

   void initialize_random(void);

   // Get methods

   LossIndex* get_loss_index_pointer(void) const;

   bool has_loss_index(void) const;

   RandomSearch* get_random_search_pointer(void) const;
   EvolutionaryAlgorithm* get_evolutionary_algorithm_pointer(void) const;

   GradientDescent* get_gradient_descent_pointer(void) const;
   ConjugateGradient* get_conjugate_gradient_pointer(void) const;
   QuasiNewtonMethod* get_quasi_Newton_method_pointer(void) const;
   LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer(void) const;

   NewtonMethod* get_Newton_method_pointer(void) const;

   const InitializationType& get_initialization_type(void) const;
   const MainType& get_main_type(void) const;
   const RefinementType& get_refinement_type(void) const;

   std::string write_initialization_type(void) const;
   std::string write_main_type(void) const;
   std::string write_refinement_type(void) const;

   std::string write_initialization_type_text(void) const;
   std::string write_main_type_text(void) const;
   std::string write_refinement_type_text(void) const;

   const bool& get_display(void) const;

   // Set methods

   void set(void);
   void set(LossIndex*);
   virtual void set_default(void);

#ifdef __OPENNN_MPI__
   void set_MPI(LossIndex*, const TrainingStrategy*);
#endif

   void set_loss_index_pointer(LossIndex*);

   void set_initialization_type(const InitializationType&);
   void set_main_type(const MainType&);
   void set_refinement_type(const RefinementType&);

   void set_initialization_type(const std::string&);
   void set_main_type(const std::string&);
   void set_refinement_type(const std::string&);

   void set_display(const bool&);

   // Pointer methods

   void destruct_initialization(void);
   void destruct_main(void);
   void destruct_refinement(void);

   // Training methods

   // This method trains a neural network which has a loss functional associated. 

   void initialize_layers_autoencoding(void);

   Results perform_training(void);

   // Serialization methods

   std::string to_string(void) const;

   void print(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

   void save(const std::string&) const;
   void load(const std::string&);

protected:

   /// Pointer to an external loss functional object.

   LossIndex* loss_index_pointer;

   /// Pointer to a random search object to be used for initialization in the training strategy.

    RandomSearch* random_search_pointer;

    /// Pointer to a evolutionary training object to be used for initialization in the training strategy.

    EvolutionaryAlgorithm* evolutionary_algorithm_pointer;

    /// Pointer to a gradient descent object to be used as a main training algorithm.

    GradientDescent* gradient_descent_pointer;

    /// Pointer to a conjugate gradient object to be used as a main training algorithm.

    ConjugateGradient* conjugate_gradient_pointer;

    /// Pointer to a quasi-Newton method object to be used as a main training algorithm.

    QuasiNewtonMethod* quasi_Newton_method_pointer;

    /// Pointer to a Levenberg-Marquardt algorithm object to be used as a main training algorithm.

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm_pointer;

    /// Pointer to a Newton method object to be used for refinement in the training strategy.

    NewtonMethod* Newton_method_pointer;

   /// Type of initialization training algorithm. 

   InitializationType initialization_type;

   /// Type of main training algorithm. 

   MainType main_type;

   /// Type of refinement training algorithm. 

   RefinementType refinement_type;

   /// Display messages to screen.

   bool display;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


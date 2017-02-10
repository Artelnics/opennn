/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E V O L U T I O N A R Y   A L G O R I T H M   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "evolutionary_algorithm.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a evolutionary training algorithm not associated to any loss functional.
/// It also initializes the class members to their default values.

EvolutionaryAlgorithm::EvolutionaryAlgorithm(void) : TrainingAlgorithm()
{
   set_default();
}


// PERFORMANCE FUNCTIONAL CONSTRUCTOR 

/// Loss index constructor. 
/// It creates a evolutionary training algorithm associated to a loss functional.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss functional object.

EvolutionaryAlgorithm::EvolutionaryAlgorithm(LossIndex* new_loss_index_pointer)
: TrainingAlgorithm(new_loss_index_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a evolutionary training algorithm not associated to any loss functional.
/// It also loads the class members from a XML document.
/// @param evolutionary_algorithm_document TinyXML document with the evolutionary algorithm object members.

EvolutionaryAlgorithm::EvolutionaryAlgorithm(const tinyxml2::XMLDocument& evolutionary_algorithm_document)
 : TrainingAlgorithm(evolutionary_algorithm_document)
{
}


// DESTRUCTOR

/// Destructor.

EvolutionaryAlgorithm::~EvolutionaryAlgorithm(void)
{

}


// METHODS

// const FitnessAssignmentMethod& get_fitness_assignment_method(void) const method

/// Returns the fitness assignment method used for training.
 
const EvolutionaryAlgorithm::FitnessAssignmentMethod& EvolutionaryAlgorithm::get_fitness_assignment_method(void) const
{
   return(fitness_assignment_method);
}


// std::string write_fitness_assignment_method(void) const method

/// Returns a string with the name of the method used for fitness assignment.  

std::string EvolutionaryAlgorithm::write_fitness_assignment_method(void) const
{
   switch(fitness_assignment_method)
   {
      case LinearRanking:
      {
         return("LinearRanking");
	  }
      break;

	  default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "std::string write_fitness_assignment_method(void) const method.\n"
                << "Unknown fitness assignment method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
      break;
   }
}


// const SelectionMethod& get_selection_method(void) const method

/// Returns the selection method used for training.

const EvolutionaryAlgorithm::SelectionMethod& EvolutionaryAlgorithm::get_selection_method(void) const
{
   return(selection_method);
}


// std::string write_selection_method(void) const method

/// Returns a string with the name of the method used for selection.  

std::string EvolutionaryAlgorithm::write_selection_method(void) const
{
   switch(selection_method)
   {
      case RouletteWheel:
      {
         return("RouletteWheel");
	  }
      break;

	  default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "std::string write_selection_method(void) const method.\n"
                << "Unknown selection method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
      break;
   }
}


// const RecombinationMethod& get_recombination_method(void) const method

/// Returns the recombination method used for training.

const EvolutionaryAlgorithm::RecombinationMethod& EvolutionaryAlgorithm::get_recombination_method(void) const
{
   return(recombination_method);
}


// std::string write_recombination_method(void) const method

/// Returns a string with the name of the method used for recombination.  

std::string EvolutionaryAlgorithm::write_recombination_method(void) const
{
   switch(recombination_method)
   {
      case Line:
      {
         return("Line");
	  }
      break;

      case Intermediate:
      {
         return("Intermediate");
	  }
      break;

	  default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "std::string write_recombination_method(void) const method.\n"
                << "Unknown recombination method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
      break;
   }
}


// const MutationMethod get_mutation_method(void) const method

/// Returns the mutation method used for training.

const EvolutionaryAlgorithm::MutationMethod& EvolutionaryAlgorithm::get_mutation_method(void) const
{
   return(mutation_method);
}


// std::string write_mutation_method(void) const method

/// Returns a string with the name of the method used for mutation.  

std::string EvolutionaryAlgorithm::write_mutation_method(void) const
{
   switch(mutation_method)
   {
      case Normal:
      {
         return("Normal");
	  }
      break;

      case Uniform:
      {
         return("Uniform");
	  }
      break;

	  default:
      {
         std::ostringstream buffer;
 
         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "std::string get_mutation_method_name(void) const method.\n"
                << "Unknown mutation method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
      break;
   }
}


// size_t get_population_size(void) const method

/// Returns the number of individuals in the population.

size_t EvolutionaryAlgorithm::get_population_size(void) const
{
   return(population.get_rows_number());
}


// const Matrix<double>& get_population(void) const method

/// Returns the population matrix.

const Matrix<double>& EvolutionaryAlgorithm::get_population(void) const
{
   return(population);
}


// const Vector<double>& get_loss(void) const method

/// Returns the actual loss value of all individuals in the population.

const Vector<double>& EvolutionaryAlgorithm::get_loss(void) const
{
   return(loss);
}


// const Vector<double>& get_fitness(void) const method

/// Returns the actual fitness value of all individuals in the population.

const Vector<double>& EvolutionaryAlgorithm::get_fitness(void) const
{
   return(fitness);
}


// const Vector<bool>& get_selection(void) const method

/// Returns the actual selection value of all individuals in the population.

const Vector<bool>& EvolutionaryAlgorithm::get_selection(void) const
{
   return(selection);
}


// const double& get_warning_parameters_norm(void) const method

/// Returns the minimum value for the norm of the parameters vector at wich a warning message is 
/// written to the screen. 

const double& EvolutionaryAlgorithm::get_warning_parameters_norm(void) const
{
   return(warning_parameters_norm);       
}


// const double& get_error_parameters_norm(void) const method

/// Returns the value for the norm of the parameters vector at wich an error message is 
/// written to the screen and the program exits. 

const double& EvolutionaryAlgorithm::get_error_parameters_norm(void) const
{
   return(error_parameters_norm);
}


// const double& get_best_loss_goal(void) const method

/// Returns the goal value for the loss. 
/// This is used as a stopping criterion when training a multilayer perceptron

const double& EvolutionaryAlgorithm::get_best_loss_goal(void) const
{
   return(best_loss_goal);
}


// const size_t& get_maximum_selection_loss_decreases(void) const method

/// Returns the maximum number of selection failures during the training process. 

const size_t& EvolutionaryAlgorithm::get_maximum_selection_loss_decreases(void) const
{
   return(maximum_selection_loss_decreases);
}


// const double& get_maximum_time(void) const method

/// Returns the maximum training time.  

const double& EvolutionaryAlgorithm::get_maximum_time(void) const
{
   return(maximum_time);
}


// const bool& get_reserve_elapsed_time_history(void) const method

/// Returns true if the elapsed time history vector is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_elapsed_time_history(void) const
{
   return(reserve_elapsed_time_history);     
}


// const bool& get_reserve_selection_loss_history(void) const method

/// Returns true if the selection loss history vector is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_selection_loss_history(void) const
{
   return(reserve_selection_loss_history);
}


// const bool& get_reserve_population_history(void) const method

/// Returns true if the population history vector of matrices is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_population_history(void) const
{
   return(reserve_population_history);
}


// const bool& get_reserve_best_individual_history(void) const method

/// Returns true if the best individual history vector of vectors is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_best_individual_history(void) const
{
   return(reserve_best_individual_history);
}


// const bool& get_reserve_mean_norm_history(void) const method

/// Returns true if the mean population norm history vector is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_mean_norm_history(void) const
{
   return(reserve_mean_norm_history);
}


// const bool& get_reserve_standard_deviation_norm_history(void) const method

/// Returns true if the standard deviation of the population norm history vector is to be reserved,
/// and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_standard_deviation_norm_history(void) const
{
   return(reserve_standard_deviation_norm_history);
}


// const bool& get_reserve_best_norm_history(void) const method

/// Returns true if the norm of the best individual in the population history vector is to be 
/// reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_best_norm_history(void) const
{
   return(reserve_best_norm_history);
}


// const bool& get_reserve_mean_loss_history(void) const method

/// Returns true if the mean loss history vector is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_mean_loss_history(void) const
{
   return(reserve_mean_loss_history);
}


// const bool& get_reserve_standard_deviation_loss_history(void) const method

/// Returns true if the standard deviation of the loss history vector is to be reserved,
/// and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_standard_deviation_loss_history(void) const
{
   return(reserve_standard_deviation_loss_history);
}


// const bool& get_reserve_best_loss_history(void) const method

/// Returns true if the best loss history vector is to be reserved, and false otherwise.

const bool& EvolutionaryAlgorithm::get_reserve_best_loss_history(void) const
{
   return(reserve_best_loss_history);
}


// void set(void) method

/// Sets the loss functional pointer of this object to NULL. 
/// It also sets the rest of members to their default values. 

void EvolutionaryAlgorithm::set(void)
{
   loss_index_pointer = NULL;

   set_default();
}


// void set(LossIndex*) method

/// Sets a new loss functional pointer to the evolutionary algorithm object. 
/// It also sets the rest of members to their default values. 

void EvolutionaryAlgorithm::set(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   set_default();
}


// void set_default(void) method

/// Sets the members of the evolutionary algorithm object to their default values.
/// Training operators:
/// <ul>
/// <li> Fitness assignment method: Linear ranking.
/// <li> Selection method: Roulette wheel.
/// <li> Recombination method: Intermediate.
/// <li> Mutation method: Normal.
/// </ul>
/// Training parameters:
/// <ul>
/// <li> Population size: 10*parameters_number or 0.
/// <li> Perform elitism: false.
/// <li> Selective pressure: 1.5.
/// <li> Recombination size: 0.25.
/// <li> Mutation rate: = 1/parameters_number or 0.
/// <li> Mutation range: = 0.1
/// </ul>
/// Stopping criteria:
/// <ul> 
/// <li> Performance goal: -1.0e99.
/// <li> Mean loss goal: -1.0e99.
/// <li> Standard deviation of loss goal: -1.0e99.
/// <li> Maximum training time: 1.0e6.
/// <li> Maximum number of generations: 100. 
/// </ul> 
/// Training history:
/// <ul> 
/// <li> Population = false.
/// <li> Mean norm = false.
/// <li> Standard deviation norm = false.
/// <li> Best norm = false.
/// <li> Mean loss = false.
/// <li> Standard deviation loss = false.
/// <li> Best loss = false.
/// </ul> 
/// User stuff: 
/// <ul>
/// <li> Display: true. 
/// <li> Display period: 1. 
/// </ul>

void EvolutionaryAlgorithm::set_default(void)
{
   // Fitness assignment method

   fitness_assignment_method = LinearRanking;

   // Selection method

   selection_method = RouletteWheel;

   // Recombination method

   recombination_method = Intermediate;

   // Mutation method

   mutation_method = Normal;

   // Training parameters

   elitism_size = 2;

   selective_pressure = 1.5;

   recombination_size = 0.25;

   mutation_rate = 0.1;

   mutation_range = 0.1;

   // Stopping criteria

   mean_loss_goal = -1.0e99;
   standard_deviation_loss_goal = 0.0;
   best_loss_goal = -1.0e99;

   maximum_time = 1.0e6;

   maximum_generations_number = 1000;

   // Training history

   reserve_population_history = false;

   reserve_best_individual_history = false;

   reserve_mean_norm_history = false;
   reserve_standard_deviation_norm_history = false;
   reserve_best_norm_history = false;

   reserve_mean_loss_history = false;
   reserve_standard_deviation_loss_history = false;
   reserve_best_loss_history = false;

   reserve_elapsed_time_history = false;

   // User stuff

   display_period = 5;
}


// void set_population_size(size_t) method

/// Sets a new population with a new number of individuals.  
/// The new population size must be an even number equal or greater than four. 
///
/// @param new_population_size Number of individuals in the population. This must be an even number equal or 
/// greater than four. 

void EvolutionaryAlgorithm::set_population_size(const size_t& new_population_size)
{
    if(new_population_size == 0)
    {
        population.set();

        loss.set();

        fitness.set();

        selection.set();
    }
    else
    {
       // Control sentence (if debug)

       #ifdef __OPENNN_DEBUG__

       check();

       #endif

       const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

       // Control sentence (if debug)

       #ifdef __OPENNN_DEBUG__

       if(!neural_network_pointer)
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                 << "void set_population_size(size_t) method.\n"
                 << "Neural network pointer is NULL.\n";

          throw std::logic_error(buffer.str());
       }

       #endif

       const size_t parameters_number = neural_network_pointer->count_parameters_number();

       if(new_population_size < 4)
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                 << "void set_population_size(size_t) method.\n"
                 << "New population size must be equal or greater than 4.\n";

          throw std::logic_error(buffer.str());
       }
       else if(new_population_size%2 != 0)
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                 << "void set_population_size(size_t) method.\n"
                 << "New population size is not divisible by 2.\n";

          throw std::logic_error(buffer.str());
       }
       else
       {
          // Set population matrix

          population.set(new_population_size, parameters_number);

          randomize_population_normal();

          // Set loss vector

          loss.resize(new_population_size);

          // Set fitness vector

          fitness.resize(new_population_size);

          // Set selection vector

          selection.resize(new_population_size);
       }
    }
}


// void set_fitness_assignment_method(const std::string&) method

/// Sets a new method for fitness assignment from a string containing the name.
/// Possible values are:
/// <ul>
/// <li> "LinearRanking"
/// </ul>
/// @param new_fitness_assignment_method_name String with name of method for fitness assignment.   

void EvolutionaryAlgorithm::set_fitness_assignment_method(const std::string& new_fitness_assignment_method_name)
{
   if(new_fitness_assignment_method_name == "LinearRanking")
   {
      fitness_assignment_method = LinearRanking;
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_fitness_assignment_method(const std::string&) method.\n"
			 << "Unknown fitness assignment method: " << new_fitness_assignment_method_name << ".\n";
   
      throw std::logic_error(buffer.str());	  
   }
}


// void set_selection_method(const std::string&) method

/// Sets a new method for selection from a string containing the name.
/// Possible values are:
/// <ul>
/// <li> "LinearRanking"
/// </ul>
/// @param new_selection_method_name String with name of method for selection.   

void EvolutionaryAlgorithm::set_selection_method(const std::string& new_selection_method_name)
{
   if(new_selection_method_name == "RouletteWheel")
   {
      selection_method = RouletteWheel;
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_selection_method(const std::string&) method.\n"
			 << "Unknown selection method: " << new_selection_method_name << ".\n";
   
      throw std::logic_error(buffer.str());	  
   }
}


// void set_recombination_method(const std::string&) method

/// Sets a new method for recombination from a string containing the name.
/// Possible values are:
/// <ul>
/// <li> "Line"
/// <li> "Intermediate"
/// </ul>
/// @param new_recombination_method_name String with name of method for recombination.   

void EvolutionaryAlgorithm::set_recombination_method(const std::string& new_recombination_method_name)
{
   if(new_recombination_method_name == "Line")
   {
      recombination_method = Line;
   }
   else if(new_recombination_method_name == "Intermediate")
   {
      recombination_method = Intermediate;
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_recombination_method(const std::string&) method.\n"
			 << "Unknown recombination method: " << new_recombination_method_name << ".\n";
   
      throw std::logic_error(buffer.str());	  
   }
}


// void set_mutation_method(const std::string&) method

/// Sets a new method for mutation from a string containing the name.
/// Possible values are:
/// <ul>
/// <li> "Normal"
/// <li> "Uniform"
/// </ul>
/// @param new_mutation_method_name String with name of method for mutation.   

void EvolutionaryAlgorithm::set_mutation_method(const std::string& new_mutation_method_name)
{
   if(new_mutation_method_name == "Normal")
   {
      mutation_method = Normal;
   }
   else if(new_mutation_method_name == "Uniform")
   {
      mutation_method = Uniform;
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_mutation_method(const std::string&) method.\n"
			 << "Unknown mutationg method: " << new_mutation_method_name << ".\n";
   
      throw std::logic_error(buffer.str());	  
   }
}


// void set_population(const Matrix<double>&) method

/// Sets a new population.
///
/// @param new_population Population Matrix.

void EvolutionaryAlgorithm::set_population(const Matrix<double>& new_population)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

    const size_t population_size = get_population_size();

    if(population_size == 0 && new_population.empty())
    {
        return;
    }

    check();

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();   

   if(new_population.get_rows_number() != population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_population(const Matrix<double>&) method.\n"
             << "New population size is not equal to population size.\n";

      throw std::logic_error(buffer.str());	  
   }
   else if(new_population.get_columns_number() != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_population(const Matrix<double>&) method.\n"
             << "New number of parameters is not equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set population

   population = new_population;
}


// void set_loss(const Vector<double>&) method

/// Sets a new population loss vector.
///
/// @param new_loss Population loss values.

void EvolutionaryAlgorithm::set_loss(const Vector<double>& new_loss)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t population_size = get_population_size();

   if(new_loss.size() != population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_loss(const Vector<double>&) method.\n"
             << "Size is not equal to population size.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set loss

   loss = new_loss;
}


// void set_fitness(const Vector<double>&) method

/// Sets a new population fitness vector.
///
/// @param new_fitness Population fitness values.

void EvolutionaryAlgorithm::set_fitness(const Vector<double>& new_fitness)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t population_size = get_population_size();

   if(new_fitness.size() != population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_fitness(Vector<double>) method.\n"
             << "Size is not equal to population size.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set fitness

   fitness = new_fitness;
}


// void set_selection(const Vector<bool>&) method

/// Sets a new population selection vector.
///
/// @param new_selection Population selection values.

void EvolutionaryAlgorithm::set_selection(const Vector<bool>& new_selection)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t population_size = get_population_size();

   if(new_selection.size() != population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_selection(Vector<double>) method.\n"
             << "Size is not equal to population size.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set selection

   selection = new_selection;
}


// void set_reserve_population_history(bool) method

/// Makes the population history vector of matrices to be reseved or not in memory.
/// @param new_reserve_population_history True if the population history vector of matrices is to be reserved, false 
/// otherwise.

void EvolutionaryAlgorithm::set_reserve_population_history(const bool& new_reserve_population_history)
{
   reserve_population_history = new_reserve_population_history;
}


// void set_reserve_best_individual_history(bool) method

/// Makes the best individual history vector of vectors to be reseved or not in memory.
/// @param new_reserve_best_individual_history True if the best individual history vector of vectors is to be reserved, 
/// false otherwise.

void EvolutionaryAlgorithm::set_reserve_best_individual_history(const bool& new_reserve_best_individual_history)
{
   reserve_best_individual_history = new_reserve_best_individual_history;
}


// void set_reserve_mean_norm_history(bool) method

/// Makes the mean norm history vector to be reseved or not in memory.
///
/// @param new_reserve_mean_norm_history True if the mean norm history vector is to be reserved, false otherwise.

void EvolutionaryAlgorithm::set_reserve_mean_norm_history(const bool& new_reserve_mean_norm_history)
{
   reserve_mean_norm_history = new_reserve_mean_norm_history;
}


// void set_reserve_standard_deviation_norm_history(bool) method

/// Makes the standard deviation norm history vector to be reseved or not in memory.
///
/// @param new_reserve_standard_deviation_norm_history True if the standard deviation norm history vector is to be 
/// reserved, false otherwise.

void EvolutionaryAlgorithm::
set_reserve_standard_deviation_norm_history(const bool& new_reserve_standard_deviation_norm_history)
{
   reserve_standard_deviation_norm_history = new_reserve_standard_deviation_norm_history;
}


// void set_reserve_best_norm_history(bool) method

/// Makes the best norm history vector to be reseved or not in memory.
///
/// @param new_reserve_best_norm_history True if the best norm history vector is to be reserved, false otherwise.

void EvolutionaryAlgorithm::set_reserve_best_norm_history(const bool& new_reserve_best_norm_history)
{
   reserve_best_norm_history = new_reserve_best_norm_history;
}


// void set_reserve_mean_loss_history(bool) method

/// Makes the mean loss history vector to be reseved or not in memory.
///
/// @param new_reserve_mean_loss_history True if the mean loss history vector is to be reserved, false 
/// otherwise.

void EvolutionaryAlgorithm::set_reserve_mean_loss_history(const bool& new_reserve_mean_loss_history) 
{
   reserve_mean_loss_history = new_reserve_mean_loss_history;
}


// void set_reserve_standard_deviation_loss_history(bool) method

/// Makes the standard deviation loss history vector to be reseved or not in memory.
///
/// @param new_reserve_standard_deviation_loss_history True if the standard deviation loss history vector 
/// is to be reserved, false otherwise.

void EvolutionaryAlgorithm
::set_reserve_standard_deviation_loss_history(const bool& new_reserve_standard_deviation_loss_history)
{
   reserve_standard_deviation_loss_history = new_reserve_standard_deviation_loss_history;
}


// void set_reserve_best_loss_history(bool) method

/// Makes the best loss history vector to be reseved or not in memory.
///
/// @param new_reserve_best_loss_history True if the best loss history vector is to be reserved, 
/// false otherwise.

void EvolutionaryAlgorithm::set_reserve_best_loss_history(const bool& new_reserve_best_loss_history)
{
   reserve_best_loss_history = new_reserve_best_loss_history;
}


// void set_reserve_all_training_history(bool) method

/// Makes the training history of all variables to reseved or not in memory.
///
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved, 
/// false otherwise.

void EvolutionaryAlgorithm::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
   // Multilayer perceptron

   reserve_population_history = new_reserve_all_training_history;

   reserve_best_individual_history = new_reserve_all_training_history;

   reserve_mean_norm_history = new_reserve_all_training_history;
   reserve_standard_deviation_norm_history = new_reserve_all_training_history;
   reserve_best_norm_history = new_reserve_all_training_history;

   // Objective functional

   reserve_mean_loss_history = new_reserve_all_training_history;
   reserve_standard_deviation_loss_history = new_reserve_all_training_history;
   reserve_best_loss_history = new_reserve_all_training_history;

   // Training algorithm 

   reserve_elapsed_time_history = new_reserve_all_training_history;
}


// Vector<double> get_individual(const size_t&) const method

/// Returns the Vector of parameters corresponding to the individual i in the population.
///
/// @param i Index of individual in the population.

Vector<double> EvolutionaryAlgorithm::get_individual(const size_t& i) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t population_size = get_population_size();

   if(i >= population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "Vector<double> get_individual(const size_t&) const method.\n"
             << "Index must be less than population size.\n";

      throw std::logic_error(buffer.str());	  
   }
  
   #endif

   // Get individual

   const Vector<double> individual = population.arrange_row(i);

   return(individual);
}


// set_individual(const size_t&, Vector<double>) method

/// Sets a new Vector of parameters to the individual i in the population. 
///
/// @param i Index of individual in the population.
/// @param individual Vector of parameters to be assigned to individual i.

void EvolutionaryAlgorithm::set_individual(const size_t& i, const Vector<double>& individual)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 
 
   const size_t size = individual.size();

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   const size_t population_size = get_population_size();

   if(i >= population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "set_individual(const size_t&, Vector<double>) method.\n"
             << "Index must be less than population size.\n";

      throw std::logic_error(buffer.str());	  
   }
   else if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "set_individual(const size_t&, Vector<double>) method.\n"
             << "Size must be equal to number of parameters.\n";

      throw std::logic_error(buffer.str());	  
   }
  
   #endif

   // Get individual

   population.set_row(i, individual);
}


// size_t calculate_best_individual_index(void) const method

/// Returns the index of the individual with greatest fitness.

size_t EvolutionaryAlgorithm::calculate_best_individual_index(void) const
{
    return(fitness.calculate_maximal_index());
}


// void set_warning_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which a warning message is written to the 
/// screen. 
/// @param new_warning_parameters_norm Warning norm of parameters vector value. 

void EvolutionaryAlgorithm::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_parameters_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_warning_parameters_norm(const double&) method.\n"
             << "Warning parameters norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set warning parameters norm

   warning_parameters_norm = new_warning_parameters_norm;     
}


// void set_error_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which an error message is written to the 
/// screen and the program exits. 
/// @param new_error_parameters_norm Error norm of parameters vector value. 

void EvolutionaryAlgorithm::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_parameters_norm < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_error_parameters_norm(const double&) method.\n"
             << "Error parameters norm must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set error parameters norm

   error_parameters_norm = new_error_parameters_norm;
}


// void set_best_loss_goal(const double&) method

/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a multilayer perceptron.
/// @param new_best_loss_goal Goal value for the loss.

void EvolutionaryAlgorithm::set_best_loss_goal(const double& new_best_loss_goal)
{
   best_loss_goal = new_best_loss_goal;
}


// void set_maximum_selection_loss_decreases(const size_t&) method

/// Sets a new maximum number of selection failures. 
/// @param new_maximum_selection_loss_decreases Maximum number of iterations in which the selection evalutation decreases. 

void EvolutionaryAlgorithm::set_maximum_selection_loss_decreases(const size_t& new_maximum_selection_loss_decreases)
{
   maximum_selection_loss_decreases = new_maximum_selection_loss_decreases;
}


// void set_maximum_time(const double&) method

/// Sets a new maximum training time.  
/// @param new_maximum_time Maximum training time.

void EvolutionaryAlgorithm::set_maximum_time(const double& new_maximum_time)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_time < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_maximum_time(const double&) method.\n"
             << "Maximum time must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }
   
   #endif

   // Set maximum time

   maximum_time = new_maximum_time;
}


// void set_reserve_elapsed_time_history(const bool&) method

/// Makes the elapsed time over the iterations to be reseved or not in memory. This is a vector.
/// @param new_reserve_elapsed_time_history True if the elapsed time history vector is to be reserved, false 
/// otherwise.

void EvolutionaryAlgorithm::set_reserve_elapsed_time_history(const bool& new_reserve_elapsed_time_history)
{
   reserve_elapsed_time_history = new_reserve_elapsed_time_history;     
}


// void set_reserve_selection_loss_history(const bool&) method

/// Makes the selection loss history to be reserved or not in memory.
/// This is a vector. 
/// @param new_reserve_selection_loss_history True if the selection loss history is to be reserved, false otherwise.

void EvolutionaryAlgorithm::set_reserve_selection_loss_history(const bool& new_reserve_selection_loss_history)  
{
   reserve_selection_loss_history = new_reserve_selection_loss_history;
}


// void set_display_period(const size_t&) method

/// Sets a new number of iterations between the training showing progress. 
/// @param new_display_period
/// Number of iterations between the training showing progress. 

void EvolutionaryAlgorithm::set_display_period(const size_t& new_display_period)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 
     
   if(new_display_period <= 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void set_display_period(const size_t&) method.\n"
             << "First training rate must be greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   display_period = new_display_period;
}



// Vector<double> calculate_population_norm(void) const method

/// Returns a vector containing the norm of each individual in the population.

Vector<double> EvolutionaryAlgorithm::calculate_population_norm(void) const
{
   const size_t population_size = get_population_size();

   Vector<double> population_norm(population_size);

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Vector<double> individual(parameters_number);

   for(size_t i = 0; i < population_size; i++)
   {
      individual = get_individual(i);
           
      population_norm[i] = individual.calculate_norm();     
   }               
   
   return(population_norm);            
}


// double calculate_mean_loss(void) const method

/// Returns the mean value of the individuals loss. 

double EvolutionaryAlgorithm::calculate_mean_loss(void) const
{
   return(loss.calculate_mean());
}


// double calculate_standard_deviation_loss(void) const method

/// Returns the standard deviation value of the individuals loss. 

double EvolutionaryAlgorithm::calculate_standard_deviation_loss(void) const
{
   return(loss.calculate_standard_deviation());
}


// Training parameters

// const bool& get_elitism_size(void) const method

/// Returns the number of individuals which will always be selected for recombination.

const size_t& EvolutionaryAlgorithm::get_elitism_size(void) const
{
   return(elitism_size);
}


// const double& get_selective_pressure(void) const method 

/// Returns the selective pressure value.

const double& EvolutionaryAlgorithm::get_selective_pressure(void) const
{
   return(selective_pressure);
}


// const double& get_recombination_size(void) const method

/// Returns the recombination size value.

const double& EvolutionaryAlgorithm::get_recombination_size(void) const
{
   return(recombination_size);
}


// const double& get_mutation_rate(void) const method

/// Returns the mutation rate value.

const double& EvolutionaryAlgorithm::get_mutation_rate(void) const
{
   return(mutation_rate);
}


// const double& get_mutation_range(void) const method

/// Returns the mutation range value.

const double& EvolutionaryAlgorithm::get_mutation_range(void) const
{
   return(mutation_range);
}


// const double& get_mean_loss_goal(void) const method

/// Returns the mean loss value of the population at which training will stop.

const double& EvolutionaryAlgorithm::get_mean_loss_goal(void) const
{
   return(mean_loss_goal);
}


// const double& get_standard_deviation_loss_goal(void) const method

/// Returns the standard deviation of the loss at which training will stop.

const double& EvolutionaryAlgorithm::get_standard_deviation_loss_goal(void) const
{
   return(standard_deviation_loss_goal);
}


// const size_t& get_maximum_generations_number(void) const method

/// Returns the maximum number of generations to train. 

const size_t& EvolutionaryAlgorithm::get_maximum_generations_number(void) const
{
   return(maximum_generations_number);
}


// void set_elitism_size(const size_t&) method

/// Sets a new elitism size to the evolutionary algorithm.
/// The elitism size is the number of individuals which will always be selected for recombination.
/// @param new_elitism_size Elitism size to be set.

void EvolutionaryAlgorithm::set_elitism_size(const size_t& new_elitism_size)
{
    const size_t half_population_size = get_population_size()/2;

    if(new_elitism_size > half_population_size)
    {
//       buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
//              << "void set_elitism_size(const size_t&) method.\n"
//              << "Eltism size (" << new_elitism_size << ") must be less or equal than half the population size (" << half_population_size << ").\n";
//
//       throw std::logic_error(buffer.str());

        elitism_size = 0;
    }

   elitism_size = new_elitism_size;
}


// void set_selective_pressure(const double&) method

/// Sets a new value for the selective pressure parameter.
/// Linear ranking allows values for the selective pressure greater than 0.
/// @param new_selective_pressure Selective pressure value.

void EvolutionaryAlgorithm::set_selective_pressure(const double& new_selective_pressure)
{
    if(new_selective_pressure <= 0.0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
              << "void set_selective_pressure(const double&) method. "
              << "Selective pressure must be greater than 0.\n";

       throw std::logic_error(buffer.str());
    }

    // Set selective pressure

    selective_pressure = new_selective_pressure;
}


// void set_recombination_size(const double&) method

/// Sets a new value for the recombination size parameter.
/// The recombination size value must be equal or greater than 0.
///
/// @param new_recombination_size Recombination size value. This must be equal or greater than 0.

void EvolutionaryAlgorithm::set_recombination_size(const double& new_recombination_size)
{
   if(new_recombination_size < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_recombination_size(const double&) method.\n"
             << "Recombination size must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Set recombination size

   recombination_size = new_recombination_size;
}


// void set_mutation_rate(const double&) method

/// Sets a new value for the mutation rate parameter.
/// The mutation rate value must be between 0 and 1.
///
/// @param new_mutation_rate Mutation rate value. This value must lie in the interval [0,1]. 

void EvolutionaryAlgorithm::set_mutation_rate(const double& new_mutation_rate)
{
   // Control sentence

   if(new_mutation_rate < 0.0 || new_mutation_rate > 1.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_mutation_rate(const double&) method.\n"
             << "Mutation rate must be a value between 0 and 1.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Set mutation rate

   mutation_rate = new_mutation_rate;
}


// void set_mutation_range(const double&) method

/// Sets a new value for the mutation range parameter.
/// The mutation range value must be 0 or a positive number. 
///
/// @param new_mutation_range Mutation range value. This must be equal or greater than 0.

void EvolutionaryAlgorithm::set_mutation_range(const double& new_mutation_range)
{
   // Control sentence

   if(new_mutation_range < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_mutation_range(const double&) method.\n"
             << "Mutation range must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Set mutation range

   mutation_range = new_mutation_range;
}


// void set_maximum_generations_number(size_t) method

/// Sets a new value for the maximum number of generations to perform_training.
/// The maximum number of generations value must be a positive number. 
/// @param new_maximum_generations_number Maximum number of generations value.

void EvolutionaryAlgorithm::set_maximum_generations_number(const size_t& new_maximum_generations_number)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_generations_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_maximum_generations_number(size_t) method.\n"
             << "Maximum number of generations must be greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set maximum number of generations

   maximum_generations_number = new_maximum_generations_number;
}



// void set_mean_loss_goal(const double&) method

/// Sets a new value for the mean loss goal stopping criterion.
/// @param new_mean_loss_goal Goal value for the mean loss of the population. 

void EvolutionaryAlgorithm::set_mean_loss_goal(const double& new_mean_loss_goal)
{
   mean_loss_goal = new_mean_loss_goal;
}


// void set_standard_deviation_loss_goal(const double&) method

/// Sets a new value for the standard deviation loss goal stopping criterion.
/// @param new_standard_deviation_loss_goal Goal for the standard deviation loss of the population. 

void EvolutionaryAlgorithm::set_standard_deviation_loss_goal(const double& new_standard_deviation_loss_goal)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_standard_deviation_loss_goal < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void set_standard_deviation_loss_goal(const double&) method.\n"
             << "Standard deviation of loss goal must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set standard deviation of loss goal

   standard_deviation_loss_goal = new_standard_deviation_loss_goal;

}

// void set_fitness_assignment_method(FitnessAssignmentMethod) method

/// Sets a new fitness assignment method to be used for training.
/// @param new_fitness_assignment_method Fitness assignment method chosen for training.

void EvolutionaryAlgorithm::set_fitness_assignment_method
(const EvolutionaryAlgorithm::FitnessAssignmentMethod& new_fitness_assignment_method)
{
   fitness_assignment_method = new_fitness_assignment_method;
}


// void set_selection_method(SelectionMethod) method

/// Sets a new selection method to be used for training.
///
/// @param new_selection_method Selection method chosen for training.

void EvolutionaryAlgorithm::
set_selection_method(const EvolutionaryAlgorithm::SelectionMethod& new_selection_method)
{
   selection_method = new_selection_method;
}


// void set_recombination_method(RecombinationMethod) method

/// Sets a new recombination method to be used for training.
///
/// @param new_recombination_method Recombination method chosen for training. 

void EvolutionaryAlgorithm
::set_recombination_method(const EvolutionaryAlgorithm::RecombinationMethod& new_recombination_method)
{
   recombination_method = new_recombination_method;
}


// void set_mutation_method(MutationMethod) method

/// Sets a new mutation method to be used for training.
///
/// @param new_mutation_method Mutation method chosen for training. 

void EvolutionaryAlgorithm::set_mutation_method(const EvolutionaryAlgorithm::MutationMethod& new_mutation_method)
{
   mutation_method = new_mutation_method;
}	


// void initialize_population(const double&) method

/// Initializes the population matrix with a given value.
/// @param new_value Initialization value. 

void EvolutionaryAlgorithm::initialize_population(const double& new_value)
{
   population.initialize(new_value);
}



// void randomize_population_uniform(void) method

/// Initializes the parameters of all the individuals in the population at random, with values 
/// comprised between -1 and 1.

void EvolutionaryAlgorithm::randomize_population_uniform(void)
{
   population.randomize_uniform();
}


// void randomize_population_uniform(const double&, const double&) method

/// Initializes the parameters of all the individuals in the population at random, with values 
/// comprised between a minimum and a maximum value.
///
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void EvolutionaryAlgorithm::randomize_population_uniform(const double& minimum, const double& maximum)
{
   population.randomize_uniform(minimum, maximum);
}


// void randomize_population_uniform(Vector<double>, Vector<double>) method

/// Initializes the parameters of all the individuals in the population at random, with values 
/// comprised between different minimum and maximum values for each variable.
///
/// @param minimum Vector of minimum initialization values.
/// @param maximum Vector of maximum initialization values.

void EvolutionaryAlgorithm::randomize_population_uniform(const Vector<double>& minimum, const Vector<double>& maximum)
{
   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();   

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t minimum_size = minimum.size();
   const size_t maximum_size = maximum.size();

   if(minimum_size != parameters_number || maximum_size != parameters_number)   
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void randomize_population_uniform(Vector<double>, Vector<double>).\n"
             << "Minimum value and maximum value sizes must be equal to number of parameters.\n";
 
      throw std::logic_error(buffer.str());	  
   }

   #endif

   Vector<double> individual(parameters_number);

   const size_t population_size = get_population_size();

   for(size_t i = 0; i < population_size; i++)
   {
      individual.randomize_uniform(minimum, maximum);

      set_individual(i, individual);
   }
}


// void randomize_population_normal(void) method

/// Initializes the parameters of all the individuals in the population with random values chosen
/// from a normal distribution with mean 0 and standard deviation 1.

void EvolutionaryAlgorithm::randomize_population_normal(void)
{
   population.randomize_normal();
}


// void randomize_population_normal(const double&, const double&) method

/// Initializes the parameters of all the individuals in the population with random values chosen
/// from a normal distribution with a given mean and a given standard deviation.
///
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void EvolutionaryAlgorithm::randomize_population_normal(const double& mean, const double& standard_deviation)
{
   population.randomize_normal(mean, standard_deviation);
}


// void randomize_population_normal(Vector<double>, Vector<double>) method

/// Initializes the parameters of all the individuals in the population with random values chosen
/// from normal distributions with different mean and standard deviation for each free parameter.
///
/// @param mean Vector of mean values.
/// @param standard_deviation Vector of standard deviation values.

void EvolutionaryAlgorithm::
randomize_population_normal(const Vector<double>& mean, const Vector<double>& standard_deviation)
{
   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();   

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t mean_size = mean.size();
   const size_t standard_deviation_size = standard_deviation.size();

   if(mean_size != parameters_number || standard_deviation_size != parameters_number)   
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void randomize_population_normal(Vector<double>, Vector<double>).\n"
             << "Mean and standard deviation sizes must be equal to number of parameters.\n";
 
      throw std::logic_error(buffer.str());	  
   }

   #endif

   Vector<double> individual(parameters_number);

   const size_t population_size = get_population_size();

   for(size_t i = 0; i < population_size; i++)
   {
      individual.randomize_normal(mean, standard_deviation);

      set_individual(i, individual);
   }
}


// void perform_fitness_assignment(void) method

/// Assigns a fitness value to all the individuals in the population according to the finess assignment operator.

void EvolutionaryAlgorithm::perform_fitness_assignment(void)
{
   switch(fitness_assignment_method)
   {
      case LinearRanking:
      { 
         perform_linear_ranking_fitness_assignment();
      }

      break;

	  default:
	  {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void perform_fitness_assignment(void).\n"
                << "Unknown fitness assignment method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
	  break;
   }
}


// void perform_selection(void) method

/// Selects for recombination some individuals from the population according to the selection operator.

void EvolutionaryAlgorithm::perform_selection(void)
{
   switch(selection_method)
   {
      case RouletteWheel:
      {
         perform_roulette_wheel_selection();
      }
      break;

	  default:
	  {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void perform_selection(void).\n"
                << "Unknown selection method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
	  break;
   }

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t selected_individuals_number = selection.count_occurrences(true);

   const size_t population_size = get_population_size();

   if(selected_individuals_number != population_size/2)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void perform_selection(void).\n"
             << "Number of selected individuals is not equal to half of the population size.\n";

      throw std::logic_error(buffer.str());
   }

   #endif
}


// void perform_recombination(void) method

/// Recombinates the selected individuals according to the recombination operator.

void EvolutionaryAlgorithm::perform_recombination(void)
{
   switch(recombination_method)
   {
      case Intermediate:
      {
         perform_intermediate_recombination();
      }
      break;

      case Line:
      {
         perform_line_recombination();
      } 
      break;

	  default:
	  {
         std::ostringstream buffer;
 
         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void perform_recombination(void).\n"
                << "Unknown recombination method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
	  break;
   }
}


// void perform_mutation(void) method

/// Mutates the population matrix according to the mutation operator.

void EvolutionaryAlgorithm::perform_mutation(void)
{
   switch(mutation_method)
   {
      case Normal:
      {
         perform_normal_mutation();
      }
      break;

      case Uniform:
      {
         perform_uniform_mutation();
      }
      break;

	  default:
	  {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void perform_mutationg(void).\n"
                << "Unknown mutation method.\n";
 
         throw std::logic_error(buffer.str());	     
	  }
	  break;
   }
}


// void evolve_population(void) method

/// Generates a new population matrix by applying fitness assignment, selection, recombination and mutation.

void EvolutionaryAlgorithm::evolve_population(void)
{
   // Fitness assignment
  
   perform_fitness_assignment();

   // Selection

   perform_selection();

   // Recombination

   perform_recombination();

   // Mutation

   perform_mutation();
}


// void evaluate_population(void) method

/// Evaluates the loss functional of all individuals in the population.
/// Results are stored in the loss vector.

void EvolutionaryAlgorithm::evaluate_population(void)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(loss_index_pointer == NULL)   
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void evaluate_population(void).\n"
             << "Loss index pointer is NULL.\n";
 
      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Neural network

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(!neural_network_pointer)   
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void evaluate_population(void).\n"
             << "Neural network pointer is NULL.\n";
 
      throw std::logic_error(buffer.str());	  
   }

   #endif

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Vector<double> individual(parameters_number);

   // Evaluate loss functional for all individuals

   const size_t population_size = get_population_size();

   for(size_t i = 0; i < population_size; i++)
   {
      individual = get_individual(i);

      loss[i] = loss_index_pointer->calculate_loss(individual);
      
      if(!(loss[i] > -1.0e99 && loss[i] < 1.0e99))
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void evaluate_population(void) method.\n"
                << "Performance of individual " << i << " is not a real number.\n";

         throw std::logic_error(buffer.str());	     
      }                
   }
}


// void perform_linear_ranking_fitness_assignment(void) method

/// Ranks all individuals in the population by their loss,
/// so that the least fit individual has rank 1 and the fittest individual has rank [population size].
/// It then assigns them a fitness value linearly proportional to their rank.
/// The smallest fitness corresponds to the smallest loss,
/// and the greatest fitness to the greatest loss.
/// Results are stored in the fitness vector.

void EvolutionaryAlgorithm::perform_linear_ranking_fitness_assignment(void)
{
   // Sorted loss vector

   const Vector<size_t> rank = loss.calculate_greater_rank();

   // Perform linear ranking fitness assignment
   // Cannot do that without loop because of different types of fitness and rank vectors

   const size_t population_size = get_population_size();

   for(size_t i = 0; i < population_size; i++)
   {
      fitness[i] = selective_pressure*rank[i];
   }
}


// void perform_roulette_wheel_selection(void) method

/// This metod performs selection with roulette wheel selection.
/// It selects half of the individuals from the population.
/// Results are stored in the selection vector. 

void EvolutionaryAlgorithm::perform_roulette_wheel_selection(void)
{
   const size_t population_size = get_population_size();

   // Set selection vector to false 

   selection.initialize(false);

   const Vector<size_t> elite_individuals = fitness.calculate_maximal_indices(elitism_size);

   for(size_t i = 0; i < elitism_size; i++)
   {
       const size_t elite_individual_index = elite_individuals[i];

       selection[elite_individual_index] = true;
   }

   const size_t selection_target = population_size/2 - elitism_size;

   if(selection_target <= 0)
   {
       return;
   }

   // Cumulative fitness vector

   const Vector<double> cumulative_fitness = fitness.calculate_cumulative();

   const double fitness_sum = fitness.calculate_sum();

   // Select individuals until the desired number of selections is obtained

   size_t selection_count = 0;

   double pointer;

   while(selection_count != selection_target)
   {
      // Random number between 0 and total cumulative fitness

      pointer = calculate_random_uniform(0.0, fitness_sum);

      // Perform selection

      if(pointer < cumulative_fitness[0])
      {
         if(!selection[0])
         {
            selection[0] = true;
            selection_count++;
         }
      }
      else
      {
          for(size_t i = 1; i < population_size; i++)
          {
             if(pointer < cumulative_fitness[i] && pointer >= cumulative_fitness[i-1])
             {
                if(!selection[i])
                {
                   selection[i] = true;
                   selection_count++;
                }
             }
          }
      }
   }

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(selection.count_occurrences(true) != population_size/2)
   {
       std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void perform_roulette_wheel_selection(void) method.\n"
             << "Selection count (" << selection.count_occurrences(true) << ") is not equal to half population size (" << population_size/2 << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif
}


// void perform_intermediate_recombination(void) method

/// Performs intermediate recombination between pairs of selected individuals to generate a new population.
/// Each selected individual is to be recombined with two other selected individuals chosen at random. 
/// Results are stored in the population matrix.

void EvolutionaryAlgorithm::perform_intermediate_recombination(void)
{
   const size_t population_size = get_population_size();

    #ifdef __OPENNN_DEBUG__

    if(selection.count_occurrences(true) != population_size/2)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
              << "void perform_intermediate_recombination(void) method.\n"
              << "Selection count (" << selection.count_occurrences(true) << ") is not equal to half population size (" << population_size/2 << ").\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();
     
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Matrix<double> new_population(population_size, parameters_number);

   Vector<double> parent_1(parameters_number);
   Vector<double> parent_2(parameters_number);

   Vector<double> offspring(parameters_number);

   Matrix<size_t> recombination(population_size, 2);

   // Start recombination   

   size_t new_population_size_count = 0;

   bool parent_2_candidate;

   size_t parent_2_candidate_index;

   double scaling_factor;

   for(size_t i = 0; i < population_size; i++)
   {
      if(selection[i])
      {
         // Set parent 1

         parent_1 = get_individual(i);

         // Generate 2 offspring with parent 1

         for(size_t j = 0; j < 2; j++)
         {
            // Choose parent 2 at random among selected individuals   

            parent_2_candidate = false;

            do{
               // Integer random number beteen 0 and population size

               parent_2_candidate_index = (size_t)calculate_random_uniform(0.0, (double)population_size);

               // Check if candidate for parent 2 is ok

               if(selection[parent_2_candidate_index] && parent_2_candidate_index != i)
               {
                  parent_2_candidate = true;

                  recombination(new_population_size_count,0) = i;

                  recombination(new_population_size_count,1) = parent_2_candidate_index;

                  parent_2 = get_individual(parent_2_candidate_index);

                  // Perform inediate recombination between parent 1 and parent 2

                  for(size_t j = 0; j < parameters_number; j++)
                  {
                     // Choose the scaling factor to be a random number between
                     // -recombination_size and 1+recombination_size for each
                     // variable anew.

                     scaling_factor = calculate_random_uniform(-recombination_size, 1.0 + recombination_size);

                     offspring[j] = scaling_factor*parent_1[j] + (1.0 - scaling_factor)*parent_2[j];
                  }

                  // Add offspring to new_population matrix

                  new_population.set_row(new_population_size_count, offspring);   
                  
                  new_population_size_count++;
               }
            }while(parent_2_candidate != true);
         }
      }
   }

   // Count number of new individuals control sentence

   #ifdef __OPENNN_DEBUG__

   if(new_population_size_count != population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void perform_intermediate_recombination(void) method.\n"
             << "Count new population size is not equal to population size.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Set new population

   population = new_population;
}


// void perform_line_recombination(void) method

/// Performs line recombination between pairs of selected individuals to generate a new population.
/// Each selected individual is to be recombined with two other selected individuals chosen at random. 
/// Results are stored in the population matrix.

void EvolutionaryAlgorithm::perform_line_recombination(void)
{    
   const size_t population_size = get_population_size();

    #ifdef __OPENNN_DEBUG__

    if(selection.count_occurrences(true) != population_size/2)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
              << "void perform_line_recombination(void) method.\n"
              << "Selection count (" << selection.count_occurrences(true) << ") is not equal to half population size (" << population_size/2 << ").\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();
     
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Matrix<double> new_population(population_size, parameters_number);

   Vector<double> parent_1(parameters_number);
   Vector<double> parent_2(parameters_number);

   Vector<double> offspring(parameters_number);
   Vector<double> parent_1_term(parameters_number);
   Vector<double> parent_2_term(parameters_number);

   Matrix<size_t> recombination(population_size, 2);

   // Start recombination   

   size_t new_population_size_count = 0;

   bool parent_2_candidate;

   size_t parent_2_candidate_index;

   double scaling_factor;

   for(size_t i = 0; i < population_size; i++)
   {
      if(selection[i])
      {
         // Set parent 1

         parent_1 = get_individual(i);

         // Generate 2 offspring with parent 1

         for(size_t j = 0; j < 2; j++)
         {
            // Choose parent 2 at random among selected individuals   

            parent_2_candidate = false;

            do
            {
               // Integer random number beteen 0 and population size

               parent_2_candidate_index = (size_t)calculate_random_uniform(0.0, (double)population_size);

               // Check if candidate for parent 2 is ok

               if(selection[parent_2_candidate_index] && parent_2_candidate_index != i)
               {
                  parent_2_candidate = true;

                  recombination(new_population_size_count,0) = i;
                  recombination(new_population_size_count,1) = parent_2_candidate_index;

                  parent_2 = get_individual(parent_2_candidate_index);

                  // Perform inediate recombination between parent 1 and parent 2

                  // Choose the scaling factor to be a random number between
                  // -recombination_size and 1+recombination_size for all
                  // variables.

                  scaling_factor = calculate_random_uniform(-recombination_size , 1.0+recombination_size);

                  parent_1_term = parent_1*scaling_factor;
                  parent_2_term = parent_2*(1.0 - scaling_factor); 

                  offspring = parent_1_term + parent_2_term;

                  // Add offspring to new_population matrix

                  new_population.set_row(new_population_size_count, offspring);   

                  new_population_size_count++;
               }
            }while(!parent_2_candidate);
         }
      }
   }

   // Count new population size control sentence

   if(new_population_size_count != population_size)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
             << "void perform_line_recombination(void) method.\n"
             << "Count new population size is not equal to population size.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Set new population

   population = new_population;
}


// void perform_normal_mutation(void) method

/// Performs normal mutation to all individuals in order to generate a new population.
/// Results are stored in the population matrix.

void EvolutionaryAlgorithm::perform_normal_mutation(void)
{
   const size_t population_size = get_population_size();

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();
     
   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   Vector<double> individual(parameters_number);

   double pointer;

   for(size_t i = 0; i < population_size; i++)
   {
      individual = get_individual(i);

      for(size_t j = 0; j < parameters_number; j++)
      {
         // Random number between 0 and 1

         pointer = calculate_random_uniform(0.0, 1.0);

         if(pointer < mutation_rate)
         {
            individual[j] += calculate_random_normal(0.0, mutation_range);
         }
      }

      set_individual(i, individual);
   }
}  


// void perform_uniform_mutation(void) method

/// Performs uniform mutation to all individuals in order to generate a new population.
/// Results are stored in the population matrix.

void EvolutionaryAlgorithm::perform_uniform_mutation(void)
{
   const size_t population_size = get_population_size();

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();
     
   const size_t parameters_number = neural_network_pointer->count_parameters_number();
   
   Vector<double> individual(parameters_number, 0.0);

   double pointer;

   for(size_t i = 0; i < population_size; i++)
   {
      individual = get_individual(i);

      for(size_t j = 0; j < parameters_number; j++)
      {
         // random number between 0 and 1

          pointer = calculate_random_uniform(0.0, 1.0);

         if(pointer < mutation_rate)
         {
            individual[j] += calculate_random_uniform(-mutation_range, mutation_range);
         }
      }

      set_individual(i, individual);
   }
}


// std::string to_string(void) const method

/// Returns a string representation of the current evolutionary algorithm resutls structure. 

std::string EvolutionaryAlgorithm::EvolutionaryAlgorithmResults::to_string(void) const
{
   std::ostringstream buffer;

   // Population history

   if(!population_history.empty())
   {
	   if(!population_history[0].empty())
	   {
          buffer << "% Population history:\n"
                 << population_history << "\n"; 
	   }
   }

   // Best individual history

   if(!best_individual_history.empty())
   {
      if(!population_history[0].empty())
	  {
          buffer << "% Best individual history:\n"
                 << best_individual_history << "\n"; 
	  }
   }

   // Mean norm history   

   if(!mean_norm_history.empty())
   {
       buffer << "% Mean norm history:\n"
              << mean_norm_history << "\n"; 
   }

   // Standard deviation norm history

   if(!standard_deviation_norm_history.empty())
   {
       buffer << "% Standard deviation norm history:\n"
              << standard_deviation_norm_history << "\n"; 
   }

   // Best norm history 

   if(!best_norm_history.empty())
   {
       buffer << "% Best norm history:\n"
              << best_norm_history << "\n"; 
   }

   // loss history

   if(!loss_history.empty())
   {
       buffer << "% loss history:\n"
              << loss_history << "\n"; 
   }

   // Mean loss history

   if(!mean_loss_history.empty())
   {
       buffer << "% Mean loss history:\n"
              << mean_loss_history << "\n"; 
   }

   // Standard deviation loss history

   if(!standard_deviation_loss_history.empty())
   {
       buffer << "% Standard deviation loss history:\n"
              << standard_deviation_loss_history << "\n"; 
   }

   // Best loss history

   if(!best_loss_history.empty())
   {
       buffer << "% Best loss history:\n"
              << best_loss_history << "\n"; 
   }

   // Selection loss history

   if(!selection_loss_history.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_loss_history << "\n"; 
   }

   // Elapsed time history   

   if(!elapsed_time_history.empty())
   {
       buffer << "% Elapsed time history:\n"
              << elapsed_time_history << "\n"; 
   }

   return(buffer.str());
}


// void resize_training_history(const size_t&) method

/// Resizes all the training history variables. 
/// @param new_size Size of training history variables. 

void EvolutionaryAlgorithm::EvolutionaryAlgorithmResults::resize_training_history(const size_t& new_size) 
{
//    evolutionary_algorithm_pointer->set_reserve_population_history(false);
//    evolutionary_algorithm_pointer->set_reserve_best_individual_history(false);
//    evolutionary_algorithm_pointer->set_reserve_elapsed_time_history(false);

    if(evolutionary_algorithm_pointer->get_reserve_population_history())
    {
         population_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_best_individual_history())
    {
        best_individual_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_mean_norm_history())
    {
        mean_norm_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_standard_deviation_norm_history())
    {
        standard_deviation_norm_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_best_norm_history())
    {
        best_norm_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_mean_loss_history())
    {
        mean_loss_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_standard_deviation_loss_history())
    {
        standard_deviation_loss_history.resize(new_size);
    }
// Bug?
    if(evolutionary_algorithm_pointer->get_reserve_best_loss_history())
    {
        best_loss_history.resize(new_size);
    }
//
    if(evolutionary_algorithm_pointer->get_reserve_selection_loss_history())
    {
        selection_loss_history.resize(new_size);
    }

    if(evolutionary_algorithm_pointer->get_reserve_elapsed_time_history())
    {
        elapsed_time_history.resize(new_size);
    }
}


// Matrix<std::string> write_final_results(const size_t& precision) const method

Matrix<std::string> EvolutionaryAlgorithm::EvolutionaryAlgorithmResults::write_final_results(const size_t& precision) const
{
   std::ostringstream buffer;

   Vector<std::string> names;
   Vector<std::string> values;

   // Final mean norm of the population.

   names.push_back("Final mean norm");

   buffer.str("");
   buffer << std::setprecision(precision) << final_mean_norm;

   values.push_back(buffer.str());

   // Final standard deviation of the population norm.

   names.push_back("Final standard deviation norm");

   buffer.str("");
   buffer << std::setprecision(precision) << final_standard_deviation_norm;

   values.push_back(buffer.str());

   // Final norm of the best individual ever.

   names.push_back("Final best norm");

   buffer.str("");
   buffer << std::setprecision(precision) << final_best_norm;

   values.push_back(buffer.str());

   // Final mean population loss.

   names.push_back("Final mean loss");

   buffer.str("");
   buffer << std::setprecision(precision) << final_mean_loss;

   values.push_back(buffer.str());

   // Final standard deviation of the population loss.

   names.push_back("Final standard deviation loss");

   buffer.str("");
   buffer << std::setprecision(precision) << final_standard_deviation_loss;

   values.push_back(buffer.str());

   // Performance of the best individual ever

   names.push_back("Final best loss");

   buffer.str("");
   buffer << std::setprecision(precision) << final_best_loss;

   values.push_back(buffer.str());

   // Final selection loss

   const LossIndex* loss_index_pointer = evolutionary_algorithm_pointer->get_loss_index_pointer();

   if(loss_index_pointer->has_selection())
   {
       names.push_back("Final selection loss");

       buffer.str("");
       buffer << std::setprecision(precision) << final_selection_loss;

       values.push_back(buffer.str());
    }

   // Generations number

   names.push_back("Generations number");

   buffer.str("");
   buffer << std::setprecision(precision) << generations_number;

   values.push_back(buffer.str());

   // Total elapsed time in the training process.

   names.push_back("Elapsed time");

   buffer.str("");
   buffer << std::setprecision(precision) << elapsed_time;

   values.push_back(buffer.str());

   // Matrix

   const size_t rows_number = names.size();
   const size_t columns_number = 2;

   Matrix<std::string> final_results(rows_number, columns_number);

   final_results.set_column(0, names);
   final_results.set_column(1, values);

   return(final_results);
}


// EvolutionaryAlgorithmResults* perform_training(void) method

/// Trains a neural network with an associated loss function according to the evolutionary algorithm.
/// Training occurs according to the training operators and their related parameters.

EvolutionaryAlgorithm::EvolutionaryAlgorithmResults* EvolutionaryAlgorithm::perform_training(void)
{
    #ifdef __OPENNN_DEBUG__

    check();

    const size_t population_size = get_population_size();

    if(population_size == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
               << "EvolutionaryAlgorithmResults* perform_training(void) method.\n"
               << "Population size is zero.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

   if(display)
   {
      std::cout << "Training with the evolutionary algorithm...\n";
   }

   EvolutionaryAlgorithmResults* results_pointer = new EvolutionaryAlgorithmResults(this);

   results_pointer->resize_training_history(1+maximum_generations_number);

   size_t selection_failures = 0;

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time;

   bool stop_training = false;

   // Loss index

   Vector<double> population_norm;

   double mean_norm;
   double standard_deviation_norm;

   Vector<double> best_individual;
   size_t best_individual_index;
   double best_norm = 0.0;

   double best_loss_ever = 1.0e99;

   double best_generation_loss = 1.0e99;

   double selection_loss = 0.0; 
//   double old_selection_loss = 0.0;

    // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   // Main loop

   for(size_t generation = 0; generation <= maximum_generations_number; generation++)
   {
      // Population stuff

      if(reserve_population_history)
      {
         results_pointer->population_history[generation] = population;
      }

      population_norm = calculate_population_norm();

      // Mean norm 

      mean_norm = population_norm.calculate_mean();

      if(reserve_mean_norm_history)
      {
         results_pointer->mean_norm_history[generation] = mean_norm;
      }

      // Standard deviation of norm

      standard_deviation_norm = population_norm.calculate_standard_deviation();

      if(reserve_standard_deviation_norm_history)
      {
         results_pointer->standard_deviation_norm_history[generation] = standard_deviation_norm;
      }
    
      // Population evaluation

      evaluate_population();

      best_generation_loss = loss.calculate_minimum();

     if(best_generation_loss < best_loss_ever)
     {
         best_individual_index = loss.calculate_minimal_index();

         best_individual = get_individual(best_individual_index);

        neural_network_pointer->set_parameters(best_individual);

        best_norm = best_individual.calculate_norm();

        best_loss_ever = best_generation_loss;

        //old_selection_loss = selection_loss;

        selection_loss = loss_index_pointer->calculate_selection_loss();
     }

      // Best individual 

      if(reserve_best_individual_history)
      {
         results_pointer->best_individual_history[generation] = best_individual;
      }

      // Best individual norm

      if(reserve_best_norm_history)
      {
         results_pointer->best_norm_history[generation] = best_norm;
      }

      // Mean loss

      const double mean_loss = loss.calculate_mean();

      if(reserve_mean_loss_history)
      {
         results_pointer->mean_loss_history[generation] = mean_loss;
      }

      // Standard deviation of loss

      const double standard_deviation_loss = loss.calculate_standard_deviation();

      if(reserve_standard_deviation_loss_history)
      {
         results_pointer->standard_deviation_loss_history[generation] = standard_deviation_loss;
      }

      // Best loss

      if(reserve_best_loss_history)
      {
         results_pointer->best_loss_history[generation] = best_loss_ever;
      }

      // selection loss

      if(reserve_selection_loss_history)
      {
         results_pointer->selection_loss_history[generation] = selection_loss;
      }

      // Elapsed time

      time(&current_time);
      elapsed_time = difftime(current_time, beginning_time);

      if(reserve_elapsed_time_history)
      {
         results_pointer->elapsed_time_history[generation] = elapsed_time;
      }

      // Training history neural network

      if(reserve_population_history)
      {
         results_pointer->population_history[generation] = population;
      }

      if(reserve_best_individual_history)
      {
         results_pointer->best_individual_history[generation] = best_individual;
      }

      if(reserve_mean_norm_history)
      {
         results_pointer->mean_norm_history[generation] = mean_norm;
      }

      if(reserve_standard_deviation_norm_history)
      {
         results_pointer->standard_deviation_norm_history[generation] = standard_deviation_norm;
      }

      if(reserve_best_norm_history)
      {
         results_pointer->best_norm_history[generation] = best_norm;
      }

      // Training history training algorithm

      if(reserve_mean_loss_history)
      {
         results_pointer->mean_loss_history[generation] = mean_loss;
      }

      if(reserve_standard_deviation_loss_history)
      {
         results_pointer->standard_deviation_loss_history[generation] = standard_deviation_loss;
      }

      if(reserve_best_loss_history)
      {
         results_pointer->best_loss_history[generation] = best_loss_ever;
      }

      if(reserve_elapsed_time_history)
      {
         results_pointer->elapsed_time_history[generation] = elapsed_time;
      }

      // Stopping criteria

      if(best_loss_ever <= best_loss_goal)
      {
         if(display)
         {
            std::cout << "Generation " << generation << ": Performance goal reached.\n"
                      << loss_index_pointer->write_information();
         }

		 stop_training = true;
      }

      if(mean_loss <= mean_loss_goal)
      {
         if(display)
         {
            std::cout << "Generation " << generation << ": Mean loss goal reached.\n";
         }
         
		 stop_training = true;
      }

      if(standard_deviation_loss <= standard_deviation_loss_goal)
      {
         if(display)
         {
            std::cout << "Generation " << generation << ": Standard deviation of loss goal reached.\n";
         }
         
		 stop_training = true;
      }

      else if(selection_failures > maximum_selection_loss_decreases)
      {
         if(display)
         {
            std::cout << "Generation " << generation << ": Maximum selection loss increases reached.\n";
            std::cout << "Selection loss increases: "<< selection_failures << std::endl;
         }

         stop_training = true;
      }

      else if(elapsed_time >= maximum_time)
      {
         if(display)
         {
            std::cout << "Generation " << generation << ": Maximum training time reached.\n";
         }

		 stop_training = true;
      }

      else if(generation == maximum_generations_number)
      {
         if(display)
         {
            std::cout << "Generation " << generation << ": Maximum number of generations reached.\n";
         }

         stop_training = true;
      }

      if(generation != 0 && generation % save_period == 0)
      {
            neural_network_pointer->save(neural_network_file_name);
      }

      if(stop_training)
      {
          if(display)
          {
             std::cout << "Mean norm: " << mean_norm << "\n"
                       << "Standard deviation of norm: " << standard_deviation_norm << "\n"
                       << "Best norm: " << best_norm << "\n"
                       << "Mean loss: " << mean_loss << "\n"
                       << "Standard deviation of loss: " << standard_deviation_loss << "\n"
                       << "Best loss: " << best_loss_ever << "\n"
                       << loss_index_pointer->write_information()
                       << "Elapsed time: " << elapsed_time << ";\n";
          }

          results_pointer->resize_training_history(1+generation);

         results_pointer->final_mean_norm = mean_norm;
         results_pointer->final_standard_deviation_norm = standard_deviation_norm;
         results_pointer->final_best_norm = best_norm;
         results_pointer->final_mean_loss = mean_loss;
         results_pointer->final_standard_deviation_loss = standard_deviation_loss;
         results_pointer->final_best_loss = best_loss_ever;
         results_pointer->final_selection_loss = selection_loss;
         results_pointer->elapsed_time = elapsed_time;
         results_pointer->generations_number = generation;

         break;
	  }
      else if(display && generation % display_period == 0)
      {
         std::cout << "Generation " << generation << ";\n"
                   << "Mean norm: " << mean_norm << "\n" 
                   << "Standard deviation of norm: " << standard_deviation_norm << "\n"
                   << "Best norm: " << best_norm << "\n"
                   << "Mean loss: " << mean_loss << "\n"
                   << "Standard deviation of loss: " << standard_deviation_loss << "\n"
                   << "Best loss: " << best_loss_ever << "\n"
                   << loss_index_pointer->write_information()
                   << "Elapsed time: " << elapsed_time << ";\n";
      }

      // Update stuff

 //     old_selection_loss = selection_loss;

      selection.initialize(false);

      evolve_population();

   }

   return(results_pointer);
}


// std::string write_training_algorithm_type(void) const method

std::string EvolutionaryAlgorithm::write_training_algorithm_type(void) const
{
   return("EVOLUTIONARY_ALGORITHM");
}


// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> EvolutionaryAlgorithm::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

    // Population size

    labels.push_back("Population size");

    buffer.str("");
    buffer << get_population_size();

    values.push_back(buffer.str());

    // Fitness assignment method

    labels.push_back("Fitness assignment method");
    values.push_back(write_fitness_assignment_method());

    // Selection method

    labels.push_back("Selection method");
    values.push_back(write_selection_method());

    // Recombination method

    labels.push_back("Recombination");
    values.push_back(write_recombination_method());

    // Mutation method

    labels.push_back("Mutation method");
    values.push_back(write_mutation_method());

    // Elitism size

    labels.push_back("Elitism size");

    buffer.str("");
    buffer << elitism_size;

    values.push_back(buffer.str());

    // Selective pressure

    labels.push_back("Selective pressure");

    buffer.str("");
    buffer << selective_pressure;

    values.push_back(buffer.str());

    // Recombination size

    labels.push_back("Recombination size");

    buffer.str("");
    buffer << recombination_size;

    values.push_back(buffer.str());

    // Mutation rate

    labels.push_back("Mutation rate");

    buffer.str("");
    buffer << mutation_rate;

    values.push_back(buffer.str());

    // Mutation range

    labels.push_back("Mutation range");

    buffer.str("");
    buffer << mutation_range;

    values.push_back(buffer.str());

   // Best loss goal

   labels.push_back("Best loss goal");

   buffer.str("");
   buffer << best_loss_goal;

   values.push_back(buffer.str());

   // Maximum selection failures

   labels.push_back("Maximum selection failures");

   buffer.str("");
   buffer << maximum_selection_loss_decreases;

   values.push_back(buffer.str());

   // Maximum generations number

   labels.push_back("Maximum generations number");

   buffer.str("");
   buffer << maximum_generations_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Reserve selection loss history

   labels.push_back("Reserve selection loss history");

   buffer.str("");
   buffer << reserve_selection_loss_history;

   values.push_back(buffer.str());

   // Reserve elapsed time history

   labels.push_back("Reserve elapsed time history");

   buffer.str("");
   buffer << reserve_elapsed_time_history;

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the members of the evolutionary algorithm object.
///
/// Training operators:
/// <ul>
/// <li> Fitness assignment method.
/// <li> Selection method.
/// <li> Recombination method.
/// <li> Mutation method.
/// </ul>
///
/// Training parameters:
/// <ul>
/// <li> Population size.
/// <li> Selective pressure.
/// <li> Recombination size.
/// <li> Mutation rate.
/// <li> Mutation range.
/// </ul>
///
/// Stopping criteria:
/// <ul> 
/// <li> Performance goal.
/// <li> Mean loss goal.
/// <li> Standard deviation of loss goal.
/// <li> Maximum time.
/// <li> Maximum number of generations. 
/// </ul> 
///  
/// User stuff: 
/// <ul>
/// <li> Display. 
/// <li> Display period. 
/// <li> Reserve elapsed time.
/// <li> Reserve mean norm history.
/// <li> Reserve standard deviation of norm history.
/// <li> Reserve best norm history.
/// <li> Reserve mean loss history.
/// <li> Reserve standard deviation of loss history.
/// <li> Reserve best loss history.
/// </ul>
///
/// Population matrix. 

tinyxml2::XMLDocument* EvolutionaryAlgorithm::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Evolutionary algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("EvolutionaryAlgorithm");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Population

   element = document->NewElement("Population");
   root_element->LinkEndChild(element);

   const std::string population_string = population.to_string();

   text = document->NewText(population_string.c_str());
   element->LinkEndChild(text);

   // Fitness assignment method
   
   element = document->NewElement("FitnessAssignmentMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_fitness_assignment_method().c_str());
   element->LinkEndChild(text);

   // Selection method

   element = document->NewElement("SelectionMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_selection_method().c_str());
   element->LinkEndChild(text);

   // Recombination method

   element = document->NewElement("RecombinationMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_recombination_method().c_str());
   element->LinkEndChild(text);

   // Mutation method

   element = document->NewElement("MutationMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_mutation_method().c_str());
   element->LinkEndChild(text);

   // Elitism

   element = document->NewElement("ElitismSize");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << elitism_size;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Selective pressure. 

   element = document->NewElement("SelectivePressure");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << selective_pressure;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Recombination size. 

   element = document->NewElement("RecombinationSize");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << recombination_size;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Mutation rate.

   element = document->NewElement("MutationRate");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << mutation_rate;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Mutation range

   element = document->NewElement("MutationRange");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << mutation_range;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
    
   // Mean loss goal

   element = document->NewElement("MeanPerformanceGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << mean_loss_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Standard deviation loss goal

   element = document->NewElement("StandardDeviationPerformanceGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << standard_deviation_loss_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Best loss goal

   element = document->NewElement("BestPerformanceGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << best_loss_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum selection loss decreases

   element = document->NewElement("MaximumSelectionLossDecreases");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_loss_decreases;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum generations number

   element = document->NewElement("MaximumGenerationsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_generations_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Maximum time

   element = document->NewElement("MaximumTime");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_time;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve mean norm history

   element = document->NewElement("ReserveMeanNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_mean_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve standard deviation norm history

   element = document->NewElement("ReserveStandardDeviationNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_standard_deviation_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve best norm history

   element = document->NewElement("ReserveBestNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_best_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve mean loss history

   element = document->NewElement("ReserveMeanPerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_mean_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve standard deviation loss history

   element = document->NewElement("ReserveStandardDeviationPerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_standard_deviation_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve best loss history

   element = document->NewElement("ReserveBestPerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_best_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   // Reserve selection loss history

   element = document->NewElement("ReserveSelectionLossHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the evolutionary algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void EvolutionaryAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("EvolutionaryAlgorithm");

    // Population

    file_stream.OpenElement("Population");

    file_stream.PushText(population.to_string().c_str());

    file_stream.CloseElement();

    // Fitness assignment method

    file_stream.OpenElement("FitnessAssignmentMethod");

    file_stream.PushText(write_fitness_assignment_method().c_str());

    file_stream.CloseElement();

    // Selection method

    file_stream.OpenElement("SelectionMethod");

    file_stream.PushText(write_selection_method().c_str());

    file_stream.CloseElement();

    // Recombination method

    file_stream.OpenElement("RecombinationMethod");

    file_stream.PushText(write_recombination_method().c_str());

    file_stream.CloseElement();

    // Mutation method

    file_stream.OpenElement("MutationMethod");

    file_stream.PushText(write_recombination_method().c_str());

    file_stream.CloseElement();

    // Elitism

    file_stream.OpenElement("ElitismSize");

    buffer.str("");
    buffer << elitism_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Selective pressure

    file_stream.OpenElement("SelectivePressure");

    buffer.str("");
    buffer << selective_pressure;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Recombination size

    file_stream.OpenElement("RecombinationSize");

    buffer.str("");
    buffer << recombination_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Mutation rate

    file_stream.OpenElement("MutationRate");

    buffer.str("");
    buffer << mutation_rate;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Mutation range

    file_stream.OpenElement("MutationRange");

    buffer.str("");
    buffer << mutation_range;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Mean loss goal

    file_stream.OpenElement("MeanPerformanceGoal");

    buffer.str("");
    buffer << mean_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Standard deviation loss goal

    file_stream.OpenElement("StandardDeviationPerformanceGoal");

    buffer.str("");
    buffer << standard_deviation_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Best loss goal

    file_stream.OpenElement("BestPerformanceGoal");

    buffer.str("");
    buffer << best_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection loss decreases

    file_stream.OpenElement("MaximumSelectionLossDecreases");

    buffer.str("");
    buffer << maximum_selection_loss_decreases;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum generations number

    file_stream.OpenElement("MaximumGenerationsNumber");

    buffer.str("");
    buffer << maximum_generations_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve mean norm history

    file_stream.OpenElement("ReserveMeanNormHistory");

    buffer.str("");
    buffer << reserve_mean_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve standard deviation norm history

    file_stream.OpenElement("ReserveStandardDeviationNormHistory");

    buffer.str("");
    buffer << reserve_standard_deviation_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve best norm history

    file_stream.OpenElement("ReserveBestNormHistory");

    buffer.str("");
    buffer << reserve_best_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve mean loss history

    file_stream.OpenElement("ReserveMeanPerformanceHistory");

    buffer.str("");
    buffer << reserve_mean_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve standard deviation loss history

    file_stream.OpenElement("ReserveStandardDeviationPerformanceHistory");

    buffer.str("");
    buffer << reserve_standard_deviation_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve best loss history

    file_stream.OpenElement("ReserveBestPerformanceHistory");

    buffer.str("");
    buffer << reserve_best_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection loss history

    file_stream.OpenElement("ReserveSelectionLossHistory");

    buffer.str("");
    buffer << reserve_selection_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}

// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a evolutionary algorithm object from a XML document.
/// Please mind about the file format, wich is specified in the User's Guide. 
/// @param document TinyXML document with the evolutionary algorithm object members.

void EvolutionaryAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
   const tinyxml2::XMLElement* root_element = document.FirstChildElement("EvolutionaryAlgorithm");

   if(!root_element)
   {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Evolutionary algorithm element is NULL.\n";

       throw std::logic_error(buffer.str());
   }

   set_default();

   // Population
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Population");

       if(element)
       {
          const char* population_text = element->GetText();

          if(population_text)
          {
             Matrix<double> new_population;
             new_population.parse(population_text);

             const size_t new_population_size = new_population.get_rows_number();

             set_population_size(new_population_size);

             set_population(new_population);
          }
       }
   }


   // Fitness assignment method
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("FitnessAssignmentMethod");

       if(element)
       {
          const std::string new_fitness_assignment_method = element->GetText();

          try
          {
             set_fitness_assignment_method(new_fitness_assignment_method);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Selection method
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionMethod");

       if(element)
       {
          const std::string new_selection_method = element->GetText();

          try
          {
             set_selection_method(new_selection_method);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Recombination method
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("RecombinationMethod");

       if(element)
       {
          const std::string new_recombination_method = element->GetText();

          try
          {
             set_recombination_method(new_recombination_method);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Mutation method
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MutationMethod");

       if(element)
       {
          const std::string new_mutation_method = element->GetText();

          try
          {
             set_mutation_method(new_mutation_method);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Elitism size
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ElitismSize");

       if(element)
       {
          const size_t new_elitism_size = atoi(element->GetText());

          try
          {
             set_elitism_size(new_elitism_size);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Selective pressure
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectivePressure");

       if(element)
       {
          const double new_selective_pressure = atof(element->GetText());

          try
          {
             set_selective_pressure(new_selective_pressure);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Recombination size
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("RecombinationSize");

       if(element)
       {
          const double new_recombination_size = atof(element->GetText());

          try
          {
             set_recombination_size(new_recombination_size);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Mutation rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MutationRate");

       if(element)
       {
          const double new_mutation_rate = atof(element->GetText());

          try
          {
             set_mutation_rate(new_mutation_rate);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Mutation range
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MutationRange");

       if(element)
       {
          const double new_mutation_range = atof(element->GetText());

          try
          {
             set_mutation_range(new_mutation_range);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Mean loss goal
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MeanPerformanceGoal");

       if(element)
       {
          const double new_mean_loss_goal = atof(element->GetText());

          try
          {
             set_mean_loss_goal(new_mean_loss_goal);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Standard deviation loss goal
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("StandardDeviationPerformanceGoal");

       if(element)
       {
          const double new_standard_deviation_loss_goal = atof(element->GetText());

          try
          {
             set_standard_deviation_loss_goal(new_standard_deviation_loss_goal);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Best loss goal
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("BestPerformanceGoal");

       if(element)
       {
          const double new_best_loss_goal = atof(element->GetText());

          try
          {
             set_best_loss_goal(new_best_loss_goal);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Maximum selection loss decreases
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionLossDecreases");

       if(element)
       {
          const size_t new_maximum_selection_loss_decreases = atoi(element->GetText());

          try
          {
             set_maximum_selection_loss_decreases(new_maximum_selection_loss_decreases);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Maximum generations number
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumGenerationsNumber");

       if(element)
       {
          const size_t new_maximum_generations_number = atoi(element->GetText());

          try
          {
             set_maximum_generations_number(new_maximum_generations_number);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Maximum time
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

       if(element)
       {
          const double new_maximum_time = atof(element->GetText());

          try
          {
             set_maximum_time(new_maximum_time);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Reserve mean norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMeanNormHistory");

       if(element)
       {
          const bool new_mean_norm_history = (atoi(element->GetText()) != 0);
          set_reserve_mean_norm_history(new_mean_norm_history);
       }
   }

   // Reserve standard deviation norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveStandardDeviationNormHistory");

       if(element)
       {
          const bool new_standard_deviation_norm_history = (atoi(element->GetText()) != 0);
          set_reserve_standard_deviation_norm_history(new_standard_deviation_norm_history);
       }
   }

   // Reserve best norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveBestNormHistory");

       if(element)
       {
          const bool new_best_norm_history = (atoi(element->GetText()) != 0);
          set_reserve_best_norm_history(new_best_norm_history);
       }
   }

   // Reserve mean loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMeanPerformanceHistory");

       if(element)
       {
          const bool new_mean_loss_history = (atoi(element->GetText()) != 0);
          set_reserve_mean_loss_history(new_mean_loss_history);
       }
   }

   // Reserve standard deviation loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveStandardDeviationPerformanceHistory");

       if(element)
       {
          const bool new_standard_deviation_loss_history = (atoi(element->GetText()) != 0);
          set_reserve_standard_deviation_loss_history(new_standard_deviation_loss_history);
       }
   }

   // Reserve best loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveBestPerformanceHistory");

       if(element)
       {
          const bool new_best_loss_history = (atoi(element->GetText()) != 0);
          set_reserve_best_loss_history(new_best_loss_history);
       }
   }

   // Reserve selection loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

       if(element)
       {
          const bool new_selection_loss_history = (atoi(element->GetText()) != 0);
          set_reserve_selection_loss_history(new_selection_loss_history);
       }
   }
}


// void initialize_random(void) method

void EvolutionaryAlgorithm::initialize_random(void)
{
    // Fitness assingment method

    fitness_assignment_method = LinearRanking;

    // Selection method

    selection_method = RouletteWheel;

    // Recombination method

    switch(rand()%2)
   {
      case 0:
      {
         recombination_method = Line;
      }
      break;

      case 1:
      {
         recombination_method = Intermediate;
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void initialize_random(void) method.\n"
                << "Unknown recombination method.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

    // Mutation method

    switch(rand()%2)
   {
      case 0:
      {
         mutation_method = Normal;
      }
      break;

      case 1:
      {
         mutation_method = Uniform;
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: EvolutionaryAlgorithm class.\n"
                << "void initialize_random(void) method.\n"
                << "Unknown mutation method.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

    const size_t new_population_size = (size_t)calculate_random_uniform(1.0, 11.0)*4;

    set_population_size(new_population_size);
}

}


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

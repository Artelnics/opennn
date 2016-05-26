/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   S T R A T E G Y   C L A S S                                                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "training_strategy.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a training strategy object not associated to any performance functional object.  
/// It also constructs the main training algorithm object. 

TrainingStrategy::TrainingStrategy(void)
 : performance_functional_pointer(NULL)
 , random_search_pointer(NULL)
 , evolutionary_algorithm_pointer(NULL)
 , gradient_descent_pointer(NULL)
 , conjugate_gradient_pointer(NULL)
 , quasi_Newton_method_pointer(NULL)
 , Levenberg_Marquardt_algorithm_pointer(NULL)
 , Newton_method_pointer(NULL)
{
    set_initialization_type(NO_INITIALIZATION);
    set_main_type(QUASI_NEWTON_METHOD);
    set_refinement_type(NO_REFINEMENT);

    set_default();
}


// PERFORMANCE FUNCTIONAL CONSTRUCTOR

/// Performance functional constructor. 
/// It creates a training strategy object associated to a performance functional object.
/// It also constructs the main training algorithm object. 
/// @param new_performance_functional_pointer Pointer to a performance functional object.

TrainingStrategy::TrainingStrategy(PerformanceFunctional* new_performance_functional_pointer)
 : performance_functional_pointer(new_performance_functional_pointer)
 , random_search_pointer(NULL)
 , evolutionary_algorithm_pointer(NULL)
 , gradient_descent_pointer(NULL)
 , conjugate_gradient_pointer(NULL)
 , quasi_Newton_method_pointer(NULL)
 , Levenberg_Marquardt_algorithm_pointer(NULL)
 , Newton_method_pointer(NULL)
{
    set_initialization_type(NO_INITIALIZATION);
    set_main_type(QUASI_NEWTON_METHOD);
    set_refinement_type(NO_REFINEMENT);

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a training strategy object not associated to any performance functional object.
/// It also loads the members of this object from a XML document. 
/// @param document Document of the TinyXML library.

TrainingStrategy::TrainingStrategy(const tinyxml2::XMLDocument& document)
 : performance_functional_pointer(NULL)
 , random_search_pointer(NULL)
 , evolutionary_algorithm_pointer(NULL)
 , gradient_descent_pointer(NULL)
 , conjugate_gradient_pointer(NULL)
 , quasi_Newton_method_pointer(NULL)
 , Levenberg_Marquardt_algorithm_pointer(NULL)
 , Newton_method_pointer(NULL)
{
    set_initialization_type(NO_INITIALIZATION);
    set_main_type(QUASI_NEWTON_METHOD);
    set_refinement_type(NO_REFINEMENT);

   set_default();

   from_XML(document);
}


// FILE CONSTRUCTOR

/// File constructor. 
/// It creates a training strategy object associated to a performance functional object.
/// It also loads the members of this object from a XML file. 
/// @param file_name Name of training strategy XML file.

TrainingStrategy::TrainingStrategy(const std::string& file_name)
 : performance_functional_pointer(NULL)
 , random_search_pointer(NULL)
 , evolutionary_algorithm_pointer(NULL)
 , gradient_descent_pointer(NULL)
 , conjugate_gradient_pointer(NULL)
 , quasi_Newton_method_pointer(NULL)
 , Levenberg_Marquardt_algorithm_pointer(NULL)
 , Newton_method_pointer(NULL)
{
    set_initialization_type(NO_INITIALIZATION);
    set_main_type(QUASI_NEWTON_METHOD);
    set_refinement_type(NO_REFINEMENT);

   set_default();

   load(file_name);
}


// DESTRUCTOR 

/// Destructor.
/// This destructor deletes the initialization, main and refinement training algorithm objects.

TrainingStrategy::~TrainingStrategy(void)
{
    delete random_search_pointer;
    delete evolutionary_algorithm_pointer;

    delete gradient_descent_pointer;
    delete conjugate_gradient_pointer;
    delete quasi_Newton_method_pointer;
    delete Levenberg_Marquardt_algorithm_pointer;

    delete Newton_method_pointer;
}


// METHODS


// void check_performance_functional(void) const method

/// Throws an exception if the training strategy has not a performance functional associated.

void TrainingStrategy::check_performance_functional(void) const
{
    if(!performance_functional_pointer)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: TrainingStrategy class.\n"
              << "void check_performance_functional(void) const.\n"
              << "Pointer to performance functional is NULL.\n";

       throw std::logic_error(buffer.str());
    }
}


// void check_training_algorithms(void) const method

/// Throws an exception if the training strategy does not have any
/// initialization, main or refinement algorithms.

void TrainingStrategy::check_training_algorithms(void) const
{
    if(initialization_type == NO_INITIALIZATION
    && main_type == NO_MAIN
    && refinement_type == NO_REFINEMENT)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "void check_training_algorithms(void) const method.\n"
               << "None initialization, main or refinement terms are used.\n";

        throw std::logic_error(buffer.str());
    }
}


// void initialize_random(void) method

/// Initializes the initialization, main and refinement algorithms at random.
/// @todo

void TrainingStrategy::initialize_random(void)
{
    // Initialization training algorithm

    switch(rand()%2)
    {
      case 0:
      {
      }
      break;

      case 1:
      {
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void initialize_random(void) method.\n"
                << "Unknown initialization training algorithm.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

    // Main training algorithm

    // Refinement training algorithm

}


// PerformanceFunctional* get_performance_functional_pointer(void) const method

/// Returns a pointer to the performance functional object to which the training strategy is associated.

PerformanceFunctional* TrainingStrategy::get_performance_functional_pointer(void) const
{
    if(!performance_functional_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "PerformanceFunctional* get_performance_functional_pointer(void) const method.\n"
               << "Performance functional pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

   return(performance_functional_pointer);
}


// bool has_performance_functional(void) const method

/// Returns true if this training strategy has a performance functional associated,
/// and false otherwise.

bool TrainingStrategy::has_performance_functional(void) const
{
    if(performance_functional_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// RandomSearch* get_random_search_pointer(void) const method

/// Returns a pointer to the random search initialization algorithm.
/// It also throws an exception if that pointer is NULL.

RandomSearch* TrainingStrategy::get_random_search_pointer(void) const
{
    if(!random_search_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "RandomSearch* get_random_search_pointer(void) const method.\n"
               << "Random search pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(random_search_pointer);
}


// EvolutionaryAlgorithm* get_evolutionary_algorithm_pointer(void) const method

/// Returns a pointer to the evolutionary algorithm initialization algorithm.
/// It also throws an exception if that pointer is NULL.

EvolutionaryAlgorithm* TrainingStrategy::get_evolutionary_algorithm_pointer(void) const
{
    if(!evolutionary_algorithm_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "EvolutionaryAlgorithm* get_evolutionary_algorithm_pointer(void) const method.\n"
               << "Evolutionary algorithm pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(evolutionary_algorithm_pointer);
}


// GradientDescent* get_gradient_descent_pointer(void) const method

/// Returns a pointer to the gradient descent main algorithm.
/// It also throws an exception if that pointer is NULL.

GradientDescent* TrainingStrategy::get_gradient_descent_pointer(void) const
{
    if(!gradient_descent_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "GradientDescent* get_gradient_descent_pointer(void) const method.\n"
               << "Gradient descent pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(gradient_descent_pointer);
}


// ConjugateGradient* get_conjugate_gradient_pointer(void) const method

/// Returns a pointer to the conjugate gradient main algorithm.
/// It also throws an exception if that pointer is NULL.

ConjugateGradient* TrainingStrategy::get_conjugate_gradient_pointer(void) const
{
    if(!conjugate_gradient_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "ConjugateGradient* get_conjugate_gradient_pointer(void) const method.\n"
               << "Conjugate gradient pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(conjugate_gradient_pointer);
}


// QuasiNewtonMethod* get_quasi_Newton_method_pointer(void) const method

/// Returns a pointer to the Newton method main algorithm.
/// It also throws an exception if that pointer is NULL.

QuasiNewtonMethod* TrainingStrategy::get_quasi_Newton_method_pointer(void) const
{
    if(!quasi_Newton_method_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "QuasiNetwtonMethod* get_quasi_Newton_method_pointer(void) const method.\n"
               << "Quasi-Newton method pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(quasi_Newton_method_pointer);
}


// LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer(void) const method

/// Returns a pointer to the Levenberg-Marquardt main algorithm.
/// It also throws an exception if that pointer is NULL.

LevenbergMarquardtAlgorithm* TrainingStrategy::get_Levenberg_Marquardt_algorithm_pointer(void) const
{
    if(!Levenberg_Marquardt_algorithm_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer(void) const method.\n"
               << "Levenberg-Marquardt algorithm pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(Levenberg_Marquardt_algorithm_pointer);
}


// NewtonMethod* get_Newton_method_pointer(void) const method

/// Returns a pointer to the Newton method refinement algorithm.
/// It also throws an exception if that pointer is NULL.

NewtonMethod* TrainingStrategy::get_Newton_method_pointer(void) const
{
    if(!Newton_method_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "NewtonMethod* get_Newton_method_pointer(void) const method.\n"
               << "Newton method pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    return(Newton_method_pointer);
}


// const TrainingAlgorithmType& get_initialization_type(void) const method

/// Returns the type of the initialization training algorithm composing this training strategy object.

const TrainingStrategy::InitializationType& TrainingStrategy::get_initialization_type(void) const
{
   return(initialization_type);
}


// const MainType& get_main_type(void) const method

/// Returns the type of the main training algorithm composing this training strategy object.

const TrainingStrategy::MainType& TrainingStrategy::get_main_type(void) const
{
   return(main_type);
}


// const RefinementType& get_refinement_type(void) const method

/// Returns the type of the refinement training algorithm composing this training strategy object.

const TrainingStrategy::RefinementType& TrainingStrategy::get_refinement_type(void) const
{
   return(refinement_type);
}


// std::string TrainingStrategy::write_initialization_type(void) const

/// Returns a string with the type of the initialization training algorithm composing this training strategy object.

std::string TrainingStrategy::write_initialization_type(void) const
{
   if(initialization_type == NO_INITIALIZATION)
   {
      return("NO_INITIALIZATION");
   }
   else if(initialization_type == RANDOM_SEARCH)
   {
      return("RANDOM_SEARCH");
   }
   else if(initialization_type == EVOLUTIONARY_ALGORITHM)
   {
      return("EVOLUTIONARY_ALGORITHM");
   }
   else if(initialization_type == USER_INITIALIZATION)
   {
      return("USER_INITIALIZATION");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "std::string write_initialization_type(void) const method.\n"
             << "Unknown training algorithm type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string TrainingStrategy::write_main_type(void) const

/// Returns a string with the type of the main training algorithm composing this training strategy object.

std::string TrainingStrategy::write_main_type(void) const
{
   if(main_type == NO_MAIN)
   {
      return("NO_MAIN");
   }
   else if(main_type == GRADIENT_DESCENT)
   {
      return("GRADIENT_DESCENT");
   }
   else if(main_type == CONJUGATE_GRADIENT)
   {
      return("CONJUGATE_GRADIENT");
   }
   else if(main_type == QUASI_NEWTON_METHOD)
   {
      return("QUASI_NEWTON_METHOD");
   }
   else if(main_type == LEVENBERG_MARQUARDT_ALGORITHM)
   {
      return("LEVENBERG_MARQUARDT_ALGORITHM");
   }
   else if(main_type == USER_MAIN)
   {
      return("USER_MAIN");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "std::string write_main_type(void) const method.\n"
             << "Unknown main type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string TrainingStrategy::write_refinement_type(void) const

/// Returns a string with the type of the refinement training algorithm composing this training strategy object.

std::string TrainingStrategy::write_refinement_type(void) const
{
   if(refinement_type == NO_REFINEMENT)
   {
      return("NO_REFINEMENT");
   }
   else if(refinement_type == NEWTON_METHOD)
   {
      return("NEWTON_METHOD");
   }
   else if(refinement_type == USER_REFINEMENT)
   {
      return("USER_REFINEMENT");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "std::string write_refinement_type(void) const method.\n"
             << "Unknown refinement type.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// std::string TrainingStrategy::write_initialization_type_text(void) const

/// Returns a string with the initialization type in text format.

std::string TrainingStrategy::write_initialization_type_text(void) const
{
   if(initialization_type == NO_INITIALIZATION)
   {
      return("none");
   }
   else if(initialization_type == RANDOM_SEARCH)
   {
      return("random search");
   }
   else if(initialization_type == EVOLUTIONARY_ALGORITHM)
   {
      return("evolutionary algorithm");
   }
   else if(initialization_type == USER_INITIALIZATION)
   {
      return("user defined");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "std::string write_initialization_type_text(void) const method.\n"
             << "Unknown training algorithm type.\n";

      throw std::logic_error(buffer.str());
   }
}


// std::string TrainingStrategy::write_main_type_text(void) const

/// Returns a string with the main type in text format.

std::string TrainingStrategy::write_main_type_text(void) const
{
   if(main_type == NO_MAIN)
   {
      return("none");
   }
   else if(main_type == GRADIENT_DESCENT)
   {
      return("gradient descent");
   }
   else if(main_type == CONJUGATE_GRADIENT)
   {
      return("conjugate gradient");
   }
   else if(main_type == QUASI_NEWTON_METHOD)
   {
      return("quasi-Newton method");
   }
   else if(main_type == LEVENBERG_MARQUARDT_ALGORITHM)
   {
      return("Levenberg-Marquardt algorithm");
   }
   else if(main_type == USER_MAIN)
   {
      return("user defined");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "std::string write_main_type_text(void) const method.\n"
             << "Unknown main type.\n";

      throw std::logic_error(buffer.str());
   }
}


// std::string TrainingStrategy::write_refinement_type(void) const

/// Returns a string with the refinement type in text format.

std::string TrainingStrategy::write_refinement_type_text(void) const
{
   if(refinement_type == NO_REFINEMENT)
   {
      return("none");
   }
   else if(refinement_type == NEWTON_METHOD)
   {
      return("Newton method");
   }
   else if(refinement_type == USER_REFINEMENT)
   {
      return("user defined");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "std::string write_refinement_type_text(void) const method.\n"
             << "Unknown refinement type.\n";

      throw std::logic_error(buffer.str());
   }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& TrainingStrategy::get_display(void) const
{
   return(display);
}


// void set(void) method

/// Sets the performance functional pointer to NULL.
/// It also destructs the initialization, main and refinement training algorithms. 
/// Finally, it sets the rest of members to their default values. 

void TrainingStrategy::set(void)
{
   performance_functional_pointer = NULL;

   set_initialization_type(NO_INITIALIZATION);
   set_main_type(QUASI_NEWTON_METHOD);
   set_refinement_type(NO_REFINEMENT);

   set_default();
}


// void set(PerformanceFunctional*) method

/// Sets a new performance functional pointer.
/// It also destructs the initialization, main and refinement training algorithms. 
/// Finally, it sets the rest of members to their default values. 
/// @param new_performance_functional_pointer Pointer to a performance functional object. 

void TrainingStrategy::set(PerformanceFunctional* new_performance_functional_pointer)
{
   performance_functional_pointer = new_performance_functional_pointer;

   set_initialization_type(NO_INITIALIZATION);
   set_main_type(QUASI_NEWTON_METHOD);
   set_refinement_type(NO_REFINEMENT);

   set_default();
}


// void set_initialization_type(const InitializationType&) method

/// Sets a new type of initialization training algorithm.
/// @param new_initialization_type Type of initialization training algorithm.

void TrainingStrategy::set_initialization_type(const InitializationType& new_initialization_type)
{
    destruct_initialization();

   initialization_type = new_initialization_type;

   switch(initialization_type)
   {
       case NO_INITIALIZATION:
       {
          // do nothing
       }
       break;

      case RANDOM_SEARCH:
      {
         random_search_pointer = new RandomSearch(performance_functional_pointer);
      }
      break;

      case EVOLUTIONARY_ALGORITHM:
      {
         evolutionary_algorithm_pointer = new EvolutionaryAlgorithm(performance_functional_pointer);
      }
      break;

      case USER_INITIALIZATION:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void set_initialization_type(const InitializationType&) method.\n"
                << "Unknown initialization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
}


// void set_main_type(const MainType&) method

/// Sets a new type of main training algorithm.
/// @param new_main_type Type of main training algorithm.

void TrainingStrategy::set_main_type(const MainType& new_main_type)
{
   destruct_main();

   main_type = new_main_type;

   switch(main_type)
   {
      case NO_MAIN:
      {
         // do nothing
      }
      break;

      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer = new GradientDescent(performance_functional_pointer);
      }
      break;

      case CONJUGATE_GRADIENT:
      {
         conjugate_gradient_pointer = new ConjugateGradient(performance_functional_pointer);
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
         quasi_Newton_method_pointer = new QuasiNewtonMethod(performance_functional_pointer);
      }
      break;

      case NEWTON_METHOD:
      {
         Newton_method_pointer = new NewtonMethod(performance_functional_pointer);
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
         Levenberg_Marquardt_algorithm_pointer = new LevenbergMarquardtAlgorithm(performance_functional_pointer);
      }
      break;

      case USER_MAIN:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void set_initialization_type(const MainType&) method.\n"
                << "Unknown main type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
}


// void set_refinement_type(const RefinementType&) method

/// Sets a new type of refinement algorithm into this training strategy.
/// Note that it destructs the current refinement algorithm object and constructs a new one.
/// @param new_refinement_type Type of refinement training algorithm.

void TrainingStrategy::set_refinement_type(const RefinementType& new_refinement_type)
{
   destruct_refinement();

   refinement_type = new_refinement_type;

   switch(refinement_type)
   {
      case NO_REFINEMENT:
      {
         // do nothing
      }
      break;

      case NEWTON_METHOD:
      {
         Newton_method_pointer = new NewtonMethod(performance_functional_pointer);
      }
      break;

      case USER_REFINEMENT:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void set_refinement_type(const RefinementType&) method.\n"
                << "Unknown refinement type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
}


// void set_initialization_type(const std::string&) method

/// Sets a new initialization training algorithm from a string.
/// @param new_initialization_type String with the initialization type.

void TrainingStrategy::set_initialization_type(const std::string& new_initialization_type)
{
   if(new_initialization_type == "NO_INITIALIZATION")
   {
      set_initialization_type(NO_INITIALIZATION);
   }
   else if(new_initialization_type == "RANDOM_SEARCH")
   {
      set_initialization_type(RANDOM_SEARCH);
   }
   else if(new_initialization_type == "EVOLUTIONARY_ALGORITHM")
   {
      set_initialization_type(EVOLUTIONARY_ALGORITHM);
   }
   else if(new_initialization_type == "USER_INITIALIZATION")
   {
      set_initialization_type(USER_INITIALIZATION);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "void set_initialization_type(const std::string&) method.\n"
             << "Unknown initialization type: " << new_initialization_type << ".\n";

      throw std::logic_error(buffer.str());
   }   
}


// void set_main_type(const std::string&) method

/// Sets a new main training algorithm from a string containing the type.
/// @param new_main_type String with the type of main training algorithm.

void TrainingStrategy::set_main_type(const std::string& new_main_type)
{
   if(new_main_type == "NO_MAIN")
   {
      set_main_type(NO_MAIN);
   }
   else if(new_main_type == "GRADIENT_DESCENT")
   {
      set_main_type(GRADIENT_DESCENT);
   }
   else if(new_main_type == "CONJUGATE_GRADIENT")
   {
      set_main_type(CONJUGATE_GRADIENT);
   }
   else if(new_main_type == "QUASI_NEWTON_METHOD")
   {
      set_main_type(QUASI_NEWTON_METHOD);
   }
   else if(new_main_type == "NEWTON_METHOD")
   {
      set_main_type(NEWTON_METHOD);
   }
   else if(new_main_type == "LEVENBERG_MARQUARDT_ALGORITHM")
   {
      set_main_type(LEVENBERG_MARQUARDT_ALGORITHM);
   }
   else if(new_main_type == "USER_MAIN")
   {
      set_main_type(USER_MAIN);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "void set_main_type(const std::string&) method.\n"
             << "Unknown main type: " << new_main_type << ".\n";

      throw std::logic_error(buffer.str());
   }   
}


// void set_refinement_type(const std::string&) method

/// Sets a new refinement algorithm from a string.
/// It destructs the previous refinement algorithm object and constructs a new object.
/// @param new_refinement_type String with the refinement training algorithm type.

void TrainingStrategy::set_refinement_type(const std::string& new_refinement_type)
{
   if(new_refinement_type == "NO_REFINEMENT")
   {
      set_refinement_type(NO_REFINEMENT);
   }
//   else if(new_refinement_type == "NEWTON_METHOD")
//   {
//      set_refinement_type(NEWTON_METHOD);
//   }
   else if(new_refinement_type == "USER_REFINEMENT")
   {
      set_refinement_type(USER_REFINEMENT);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "void set_refinement_type(const std::string&) method.\n"
             << "Unknown refinement type: " << new_refinement_type << ".\n";

      throw std::logic_error(buffer.str());
   }   
}


// void set_performance_functional_pointer(PerformanceFunctional*) method

/// Sets a pointer to a performance functional object to be associated to the training strategy.
/// @param new_performance_functional_pointer Pointer to a performance functional object.

void TrainingStrategy::set_performance_functional_pointer(PerformanceFunctional* new_performance_functional_pointer)
{
   performance_functional_pointer = new_performance_functional_pointer;

   // Initialization

   switch(initialization_type)
   {
       case NO_INITIALIZATION:
       {
          // do nothing
       }
       break;

      case RANDOM_SEARCH:
      {
         random_search_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case EVOLUTIONARY_ALGORITHM:
      {
         evolutionary_algorithm_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case USER_INITIALIZATION:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void set_performance_functional_pointer(PerformanceFunctional*) method.\n"
                << "Unknown initialization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Main

   switch(main_type)
   {
      case NO_MAIN:
      {
         // do nothing
      }
      break;

      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case CONJUGATE_GRADIENT:
      {
         conjugate_gradient_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
         quasi_Newton_method_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
         Levenberg_Marquardt_algorithm_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case USER_MAIN:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void set_performance_functional_pointer(PerformanceFunctional*) method.\n"
                << "Unknown main type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Refinement

   switch(refinement_type)
   {
      case NO_REFINEMENT:
      {
         // do nothing
      }
      break;

      case NEWTON_METHOD:
      {
           Newton_method_pointer->set_performance_functional_pointer(new_performance_functional_pointer);
      }
      break;

      case USER_REFINEMENT:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "void set_performance_functional_pointer(PerformanceFunctional) method.\n"
                << "Unknown refinement type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TrainingStrategy::set_display(const bool& new_display)
{
   display = new_display;

   switch(initialization_type)
   {
       case NO_INITIALIZATION:
       {
          // do nothing
       }
       break;

      case RANDOM_SEARCH:
      {
         random_search_pointer->set_display(display);
      }
      break;

      case EVOLUTIONARY_ALGORITHM:
      {
           evolutionary_algorithm_pointer->set_display(display);
      }
      break;

      case USER_INITIALIZATION:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "Results set_display(void) method.\n"
                << "Unknown initialization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Main

   switch(main_type)
   {
      case NO_MAIN:
      {
         // do nothing
      }
      break;

      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer->set_display(display);
      }
      break;

      case CONJUGATE_GRADIENT:
      {
           conjugate_gradient_pointer->set_display(display);
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           quasi_Newton_method_pointer->set_display(display);
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           Levenberg_Marquardt_algorithm_pointer->set_display(display);
      }
      break;

      case USER_MAIN:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "Results set_display(void) method.\n"
                << "Unknown main type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Refinement

   switch(refinement_type)
   {
      case NO_REFINEMENT:
      {
         // do nothing
      }
      break;

      case NEWTON_METHOD:
      {
           Newton_method_pointer->set_display(display);
      }
      break;

      case USER_REFINEMENT:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "Results set_display(void) method.\n"
                << "Unknown refinement type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
}


// void set_default(void) method 

/// Sets the members of the training strategy object to their default values:
/// <ul>
/// <li> Display: true.
/// </ul> 

void TrainingStrategy::set_default(void)
{


   display = true;
}



// void destruct_initialization(void) method

/// This method deletes the initialization training algorithm object which composes this training strategy object. 

void TrainingStrategy::destruct_initialization(void)
{
    delete random_search_pointer;
    delete evolutionary_algorithm_pointer;

    random_search_pointer = NULL;
    evolutionary_algorithm_pointer = NULL;

   initialization_type = NO_INITIALIZATION;
}


// void destruct_main(void) method

/// This method deletes the main training algorithm object which composes this training strategy object. 

void TrainingStrategy::destruct_main(void)
{
    delete gradient_descent_pointer;
    delete conjugate_gradient_pointer;
    delete quasi_Newton_method_pointer;
    delete Levenberg_Marquardt_algorithm_pointer;

    gradient_descent_pointer = NULL;
    conjugate_gradient_pointer = NULL;
    quasi_Newton_method_pointer = NULL;
    Levenberg_Marquardt_algorithm_pointer = NULL;

   main_type = NO_MAIN;
}


// void destruct_refinement(void) method

/// This method deletes the refinement training algorithm object which composes this training strategy object. 

void TrainingStrategy::destruct_refinement(void)
{
   delete Newton_method_pointer;

   Newton_method_pointer = NULL;

   refinement_type = NO_REFINEMENT;
}


// void initialize_layers_autoencoding(void) method

void TrainingStrategy::initialize_layers_autoencoding(void)
{
    // Data set

    DataSet* data_set_pointer = performance_functional_pointer->get_data_set_pointer();

    Instances* instances_pointer = data_set_pointer->get_instances_pointer();

    const size_t training_instances_number = instances_pointer->count_training_instances_number();

    // Neural network

    NeuralNetwork* neural_network_pointer = performance_functional_pointer->get_neural_network_pointer();

    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Vector<size_t> architecture = multilayer_perceptron_pointer->arrange_architecture();

    // Autoencoding

    Matrix<double> input_data;

    DataSet data_set;

    NeuralNetwork neural_network;

    size_t inputs_number;
    size_t layer_size;

    Vector<double> parameters;

    PerformanceFunctional performance_functional(&neural_network, &data_set);

    QuasiNewtonMethod quasi_Newton_method(&performance_functional);
    quasi_Newton_method.set_performance_goal(1.0e-3);
    quasi_Newton_method.set_gradient_norm_goal(1.0e-3);

    quasi_Newton_method.set_display_period(1000);

    std::cout << "Layers number: " << architecture.size() - 1 << std::endl;

    for(size_t i = 1; i < architecture.size()-1; i++)
    {
        std::cout << "Layer: " << i-1 << std::endl;
        std::cout << "Size: " << architecture[i-1] << std::endl;

        // Neural network

        inputs_number = architecture[i-1];
        layer_size = architecture[i];

        neural_network.set(inputs_number, layer_size, inputs_number);

        // Data set

        input_data.set(training_instances_number, inputs_number);
        input_data.randomize_normal();

        data_set.set(training_instances_number, inputs_number, inputs_number);
        data_set.set_data(input_data.assemble_columns(input_data));

        Vector<Variables::Use> inputs(inputs_number, Variables::Input);
        Vector<Variables::Use> targets(inputs_number, Variables::Target);

        data_set.get_variables_pointer()->set_uses(inputs.assemble(targets));

        data_set.get_instances_pointer()->set_training();

        // Training strategy

        quasi_Newton_method.perform_training();

        // Set parameters

        parameters = neural_network.get_multilayer_perceptron_pointer()->get_layer(0).arrange_parameters();

        multilayer_perceptron_pointer->set_layer_parameters(i-1, parameters);
    }
}


// Results perform_training(void) method

/// This is the most important method of this class. 
/// It optimizes the performance functional of a neural network.
/// The most general training strategy consists of three steps: initialization, main and refinement training processes. 
/// This method also returns a structure with the results from training. 

TrainingStrategy::Results TrainingStrategy::perform_training(void)
{
   #ifdef __OPENNN_DEBUG__ 

    check_performance_functional();

    check_training_algorithms();

   #endif

//   initialize_layers_autoencoding();

   Results training_strategy_results;

   // Initialization

   switch(initialization_type)
   {
       case NO_INITIALIZATION:
       {
          // do nothing
       }
       break;

      case RANDOM_SEARCH:
      {
         random_search_pointer->set_display(display);

         training_strategy_results.random_search_results_pointer
         = random_search_pointer->perform_training();
      }
      break;

      case EVOLUTIONARY_ALGORITHM:
      {
           evolutionary_algorithm_pointer->set_display(display);

           training_strategy_results.evolutionary_algorithm_results_pointer
           = evolutionary_algorithm_pointer->perform_training();
      }
      break;

      case USER_INITIALIZATION:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "Results perform_training(void) method.\n"
                << "Unknown initialization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Main

   switch(main_type)
   {
      case NO_MAIN:
      {
         // do nothing
      }
      break;

      case GRADIENT_DESCENT:
      {
         gradient_descent_pointer->set_display(display);

         training_strategy_results.gradient_descent_results_pointer
         = gradient_descent_pointer->perform_training();

      }
      break;

      case CONJUGATE_GRADIENT:
      {
           conjugate_gradient_pointer->set_display(display);

           training_strategy_results.conjugate_gradient_results_pointer
           = conjugate_gradient_pointer->perform_training();
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           quasi_Newton_method_pointer->set_display(display);

           training_strategy_results.quasi_Newton_method_results_pointer
           = quasi_Newton_method_pointer->perform_training();
      }
      break;

      case NEWTON_METHOD:
      {
           Newton_method_pointer->set_display(display);

           training_strategy_results.Newton_method_results_pointer
           = Newton_method_pointer->perform_training();
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           Levenberg_Marquardt_algorithm_pointer->set_display(display);

           training_strategy_results.Levenberg_Marquardt_algorithm_results_pointer
           = Levenberg_Marquardt_algorithm_pointer->perform_training();
      }
      break;

      case USER_MAIN:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "Results perform_training(void) method.\n"
                << "Unknown main type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Refinement

   switch(refinement_type)
   {
      case NO_REFINEMENT:
      {
         // do nothing
      }
      break;

//      case NEWTON_METHOD:
//      {
//           Newton_method_pointer->set_display(display);

//           training_strategy_results.Newton_method_results_pointer
//           = Newton_method_pointer->perform_training();
//      }
//      break;

      case USER_REFINEMENT:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "Results perform_training(void) method.\n"
                << "Unknown refinement type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   return(training_strategy_results);
}


// std::string to_string(void) const method

/// Returns a string representation of the training strategy.

std::string TrainingStrategy::to_string(void) const
{
   std::ostringstream buffer;

   buffer << "Training strategy\n";

   // Initialization

   buffer << "Initialization type: " << write_initialization_type() << "\n";

   switch(initialization_type)
   {
       case NO_INITIALIZATION:
       {
          // do nothing
       }
       break;

      case RANDOM_SEARCH:
      {
         buffer << random_search_pointer->to_string();

      }
      break;

      case EVOLUTIONARY_ALGORITHM:
      {
           buffer << evolutionary_algorithm_pointer->to_string();
      }
      break;

      case USER_INITIALIZATION:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "std::string to_string(void) const method.\n"
                << "Unknown initialization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Main

   buffer << "Main type: " << write_main_type() << "\n";

   switch(main_type)
   {
      case NO_MAIN:
      {
         // do nothing
      }
      break;

      case GRADIENT_DESCENT:
      {
         buffer << gradient_descent_pointer->to_string();

      }
      break;

      case CONJUGATE_GRADIENT:
      {
           buffer << conjugate_gradient_pointer->to_string();
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           buffer << quasi_Newton_method_pointer->to_string();
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           buffer << Levenberg_Marquardt_algorithm_pointer->to_string();
      }
      break;

      case USER_MAIN:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "std::string to_string(void) const method.\n"
                << "Unknown main type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Refinement

   buffer << "Refinement type: " << write_refinement_type() << "\n";

   switch(refinement_type)
   {
      case NO_REFINEMENT:
      {
         // do nothing
      }
      break;

      case NEWTON_METHOD:
      {
           buffer << Newton_method_pointer->to_string();
      }
      break;

      case USER_REFINEMENT:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "std::string to_string(void) const method.\n"
                << "Unknown refinement type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   return(buffer.str());
}


// void print(void) const method

/// Prints to the screen the string representation of the training strategy object.

void TrainingStrategy::print(void) const
{
   std::cout << to_string();
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Returns a default string representation in XML-type format of the training algorithm object.
/// This containts the training operators, the training parameters, stopping criteria and other stuff.

tinyxml2::XMLDocument* TrainingStrategy::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Training strategy

   tinyxml2::XMLElement* training_strategy_element = document->NewElement("TrainingStrategy");

   document->InsertFirstChild(training_strategy_element);

//   tinyxml2::XMLElement* element = NULL;
//   tinyxml2::XMLText* text = NULL;

   // Initialization
/*
   switch(initialization_type)
   {
       case NO_INITIALIZATION:
       {
           tinyxml2::XMLElement* initialization_element = document->NewElement("Initialization");
           training_strategy_element->LinkEndChild(initialization_element);

           initialization_element->SetAttribute("Type", "NO_INITIALIZATION");
       }
       break;

      case RANDOM_SEARCH:
      {
           tinyxml2::XMLElement* initialization_element = document->NewElement("Initialization");
           training_strategy_element->LinkEndChild(initialization_element);

           initialization_element->SetAttribute("Type", "RANDOM_SEARCH");

           const tinyxml2::XMLDocument* random_search_document = random_search_pointer->to_XML();

           const tinyxml2::XMLElement* random_search_element = random_search_document->FirstChildElement("RandomSearch");

           DeepClone(initialization_element, random_search_element, document, NULL);

           delete random_search_document;
      }
      break;

      case EVOLUTIONARY_ALGORITHM:
      {
           tinyxml2::XMLElement* initialization_element = document->NewElement("Initialization");
           training_strategy_element->LinkEndChild(initialization_element);

           initialization_element->SetAttribute("Type", "EVOLUTIONARY_ALGORITHM");

           const tinyxml2::XMLDocument* evolutionary_algorithm_document = evolutionary_algorithm_pointer->to_XML();

           const tinyxml2::XMLElement* evolutionary_algorithm_element = evolutionary_algorithm_document->FirstChildElement("EvolutionaryAlgorithm");

           DeepClone(initialization_element, evolutionary_algorithm_element, document, NULL);

           delete evolutionary_algorithm_document;
      }
      break;

      case USER_INITIALIZATION:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
                << "Unknown initialization type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
*/
    // Main

   switch(main_type)
   {
      case NO_MAIN:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "NO_MAIN");
      }
      break;

      case GRADIENT_DESCENT:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "GRADIENT_DESCENT");

           const tinyxml2::XMLDocument* gradient_descent_document = gradient_descent_pointer->to_XML();

           const tinyxml2::XMLElement* gradient_descent_element = gradient_descent_document->FirstChildElement("GradientDescent");

           DeepClone(main_element, gradient_descent_element, document, NULL);

           delete gradient_descent_document;
      }
      break;

      case CONJUGATE_GRADIENT:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "CONJUGATE_GRADIENT");

           const tinyxml2::XMLDocument* conjugate_gradient_document = conjugate_gradient_pointer->to_XML();

           const tinyxml2::XMLElement* conjugate_gradient_element = conjugate_gradient_document->FirstChildElement("ConjugateGradient");

           DeepClone(main_element, conjugate_gradient_element, document, NULL);

           delete conjugate_gradient_document;
      }
      break;

      case QUASI_NEWTON_METHOD:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "QUASI_NEWTON_METHOD");

           const tinyxml2::XMLDocument* quasi_Newton_method_document = quasi_Newton_method_pointer->to_XML();

           const tinyxml2::XMLElement* quasi_Newton_method_element = quasi_Newton_method_document->FirstChildElement("QuasiNewtonMethod");

           DeepClone(main_element, quasi_Newton_method_element, document, NULL);

           delete quasi_Newton_method_document;
      }
      break;

      case LEVENBERG_MARQUARDT_ALGORITHM:
      {
           tinyxml2::XMLElement* main_element = document->NewElement("Main");
           training_strategy_element->LinkEndChild(main_element);

           main_element->SetAttribute("Type", "LEVENBERG_MARQUARDT_ALGORITHM");

           const tinyxml2::XMLDocument* Levenberg_Marquardt_algorithm_document = Levenberg_Marquardt_algorithm_pointer->to_XML();

           const tinyxml2::XMLElement* Levenberg_Marquardt_algorithm_element = Levenberg_Marquardt_algorithm_document->FirstChildElement("LevenbergMarquardtAlgorithm");

           DeepClone(main_element, Levenberg_Marquardt_algorithm_element, document, NULL);

           delete Levenberg_Marquardt_algorithm_document;
      }
      break;

      case USER_MAIN:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
                << "Unknown main type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   switch(refinement_type)
   {
      case NO_REFINEMENT:
      {
         // do nothing
      }
      break;

      case NEWTON_METHOD:
      {
           tinyxml2::XMLElement* refinement_element = document->NewElement("Refinement");
           training_strategy_element->LinkEndChild(refinement_element);

           refinement_element->SetAttribute("Type", "NEWTON_METHOD");

           const tinyxml2::XMLDocument* Newton_method_document = Newton_method_pointer->to_XML();

           const tinyxml2::XMLElement* Newton_method_element = Newton_method_document->FirstChildElement("NewtonMethod");

           DeepClone(refinement_element, Newton_method_element, document, NULL);

           delete Newton_method_document;
      }
      break;

      case USER_REFINEMENT:
      {
         // do nothing
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingStrategy class.\n"
                << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
                << "Unknown refinement type.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }

   // Display
//   {
//      element = document->NewElement("Display");
//      training_strategy_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void TrainingStrategy::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("TrainingStrategy");


    switch(main_type)
    {
       case NO_MAIN:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "NO_MAIN");

            file_stream.CloseElement();
       }
       break;

       case GRADIENT_DESCENT:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "GRADIENT_DESCENT");

            gradient_descent_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case CONJUGATE_GRADIENT:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "CONJUGATE_GRADIENT");

            conjugate_gradient_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case QUASI_NEWTON_METHOD:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "QUASI_NEWTON_METHOD");

            quasi_Newton_method_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case LEVENBERG_MARQUARDT_ALGORITHM:
       {
            file_stream.OpenElement("Main");

            file_stream.PushAttribute("Type", "LEVENBERG_MARQUARDT_ALGORITHM");

            Levenberg_Marquardt_algorithm_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case USER_MAIN:
       {
          // do nothing
       }
       break;

       default:
       {
          std::ostringstream buffer;

          file_stream.CloseElement();

          buffer << "OpenNN Exception: TrainingStrategy class.\n"
                 << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
                 << "Unknown main type.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }

    switch(refinement_type)
    {
       case NO_REFINEMENT:
       {
          // do nothing
       }
       break;

       case NEWTON_METHOD:
       {
            file_stream.OpenElement("Refinement");

            file_stream.PushAttribute("Type", "NEWTON_METHOD");

            Newton_method_pointer->write_XML(file_stream);

            file_stream.CloseElement();
       }
       break;

       case USER_REFINEMENT:
       {
          // do nothing
       }
       break;

       default:
       {
          std::ostringstream buffer;

          file_stream.CloseElement();

          buffer << "OpenNN Exception: TrainingStrategy class.\n"
                 << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
                 << "Unknown refinement type.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads the members of this training strategy object from a XML document.
/// @param document XML document of the TinyXML library.

void TrainingStrategy::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TrainingStrategy");

   if(!root_element)
   {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: TrainingStrategy class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Training strategy element is NULL.\n";

       throw std::logic_error(buffer.str());
   }

   // Initialization
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Initialization");

       if(element)
       {
          const std::string new_initialization_type = element->Attribute("Type");

          set_initialization_type(new_initialization_type);

          switch(initialization_type)
          {
              case NO_INITIALIZATION:
              {
                 // do nothing
              }
              break;

             case RANDOM_SEARCH:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("RandomSearch");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  random_search_pointer->from_XML(new_document);
             }
             break;

             case EVOLUTIONARY_ALGORITHM:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("EvolutionaryAlgorithm");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  evolutionary_algorithm_pointer->from_XML(new_document);
             }
             break;

             case USER_INITIALIZATION:
             {
                // do nothing
             }
             break;

             default:
             {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: TrainingStrategy class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown initialization type.\n";

                throw std::logic_error(buffer.str());
             }
             break;
          }// end switch
       }
   }

   // Main
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Main");

       if(element)
       {
          const std::string new_main_type = element->Attribute("Type");

          set_main_type(new_main_type);

          switch(main_type)
          {
             case NO_MAIN:
             {
                // do nothing
             }
             break;

             case GRADIENT_DESCENT:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("GradientDescent");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  gradient_descent_pointer->from_XML(new_document);
             }
             break;

             case CONJUGATE_GRADIENT:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("ConjugateGradient");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  conjugate_gradient_pointer->from_XML(new_document);
             }
             break;

             case QUASI_NEWTON_METHOD:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("QuasiNewtonMethod");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  quasi_Newton_method_pointer->from_XML(new_document);
             }
             break;

             case LEVENBERG_MARQUARDT_ALGORITHM:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("LevenbergMarquardtAlgorithm");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  Levenberg_Marquardt_algorithm_pointer->from_XML(new_document);
             }
             break;

             case USER_MAIN:
             {
                // do nothing
             }
             break;

             default:
             {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: TrainingStrategy class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown main type.\n";

                throw std::logic_error(buffer.str());
             }
             break;
          }
       }
   }

   // Refinement
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Refinement");

       if(element)
       {
          const std::string new_refinement_type = element->Attribute("Type");

          set_refinement_type(new_refinement_type);

          switch(refinement_type)
          {
             case NO_REFINEMENT:
             {
                // do nothing
             }
             break;

             case NEWTON_METHOD:
             {
                  tinyxml2::XMLDocument new_document;

                  tinyxml2::XMLElement* element_clone = new_document.NewElement("NewtonMethod");
                  new_document.InsertFirstChild(element_clone);

                  DeepClone(element_clone, element, &new_document, NULL);

                  Newton_method_pointer->from_XML(new_document);
             }
             break;

             case USER_REFINEMENT:
             {
                // do nothing
             }
             break;

             default:
             {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: TrainingStrategy class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown refinement type.\n";

                throw std::logic_error(buffer.str());
             }
             break;
          }
       }
   }

   // Display 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

       if(element)
       {
          const std::string new_display = element->GetText();

          try
          {
             set_display(new_display != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

}


// void save(const std::string&) const method

/// Saves to a XML-type file the members of the training algorithm object.
/// @param file_name Name of training algorithm XML-type file. 

void TrainingStrategy::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();   

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a gradient descent object from a XML-type file.
/// Please mind about the file format, wich is specified in the User's Guide. 
/// @param file_name Name of training algorithm XML-type file. 

void TrainingStrategy::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;
   
   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingStrategy class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}


// Results constructor

TrainingStrategy::Results::Results(void)
{
    random_search_results_pointer = NULL;

    evolutionary_algorithm_results_pointer = NULL;

    gradient_descent_results_pointer = NULL;

    conjugate_gradient_results_pointer = NULL;

    quasi_Newton_method_results_pointer = NULL;

    Levenberg_Marquardt_algorithm_results_pointer = NULL;

    Newton_method_results_pointer = NULL;
}


// Results destructor

TrainingStrategy::Results::~Results(void)
{
//    delete random_search_results_pointer;

//    delete evolutionary_algorithm_results_pointer;

//    delete gradient_descent_results_pointer;

//    delete conjugate_gradient_results_pointer;

//    delete quasi_Newton_method_results_pointer;

//    delete Levenberg_Marquardt_algorithm_results_pointer;

//    delete Newton_method_results_pointer;

}


// void Results::save(const std::string&) const method

/// Saves the results structure to a data file.
/// @param file_name Name of training strategy results data file. 

void TrainingStrategy::Results::save(const std::string& file_name) const
{
   std::ofstream file(file_name.c_str());

   if(random_search_results_pointer)
   {
      file << random_search_results_pointer->to_string();
   }

   if(evolutionary_algorithm_results_pointer)
   {
      file << evolutionary_algorithm_results_pointer->to_string();
   }

   if(gradient_descent_results_pointer)
   {
      file << gradient_descent_results_pointer->to_string();
   }

   if(conjugate_gradient_results_pointer)
   {
      file << conjugate_gradient_results_pointer->to_string();
   }

   if(quasi_Newton_method_results_pointer)
   {
      file << quasi_Newton_method_results_pointer->to_string();
   }

   if(Levenberg_Marquardt_algorithm_results_pointer)
   {
      file << Levenberg_Marquardt_algorithm_results_pointer->to_string();
   }

   if(Newton_method_results_pointer)
   {
      file << Newton_method_results_pointer->to_string();
   }

   file.close();
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

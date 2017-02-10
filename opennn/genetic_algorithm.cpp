/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G E N E T I C   A L G O R I T H M   C L A S S                                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "genetic_algorithm.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

GeneticAlgorithm::GeneticAlgorithm(void)
    : InputsSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

GeneticAlgorithm::GeneticAlgorithm(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML order selection file.

GeneticAlgorithm::GeneticAlgorithm(const std::string& file_name)
    : InputsSelectionAlgorithm(file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/// @param genetic_algorithm_document Pointer to a TinyXML document containing the genetic algorithm data.

GeneticAlgorithm::GeneticAlgorithm(const tinyxml2::XMLDocument& genetic_algorithm_document)
    : InputsSelectionAlgorithm(genetic_algorithm_document)
{
    from_XML(genetic_algorithm_document);
}


// DESTRUCTOR

/// Destructor.

GeneticAlgorithm::~GeneticAlgorithm(void)
{
}

// METHODS

// const Vector< Vector<bool> >& get_population(void) const method

/// Returns the population matrix.

const Vector< Vector<bool> >& GeneticAlgorithm::get_population(void) const
{
    return(population);
}

// const Matrix<double>& get_loss(void) const method

/// Returns the training and selection losss of the population.

const Matrix<double>& GeneticAlgorithm::get_loss(void) const
{
    return(loss);
}

// const Vector<double>& get_fitness(void) const method

/// Returns the fitness of the population.

const Vector<double>& GeneticAlgorithm::get_fitness(void) const
{
    return(fitness);
}

// const InitializationMethod& get_initialization_method(void) const method

/// Returns the method for the initialization of the population.

const GeneticAlgorithm::InitializationMethod& GeneticAlgorithm::get_initialization_method(void) const
{
    return(initialization_method);
}

// const CrossoverMethod& get_crossover_method(void) const method

/// Returns the method for the crossover of the population.

const GeneticAlgorithm::CrossoverMethod& GeneticAlgorithm::get_crossover_method(void) const
{
    return(crossover_method);
}

// const FitnessAssignment& get_fitness_assignment_method(void) const method

/// Returns the method for the fitness assignment of the population.

const GeneticAlgorithm::FitnessAssignment& GeneticAlgorithm::get_fitness_assignment_method(void) const
{
    return(fitness_assignment_method);
}


// const size_t& get_population_size(void) const method

/// Returns the size of the population.

const size_t& GeneticAlgorithm::get_population_size(void) const
{
    return(population_size);
}

// const double& get_mutation_rate(void) const method

/// Returns the rate used in the mutation.

const double& GeneticAlgorithm::get_mutation_rate(void) const
{
    return(mutation_rate);
}

// const size_t& get_elitism_size(void) const method

/// Returns the size of the elite in the selection.

const size_t& GeneticAlgorithm::get_elitism_size(void) const
{
    return(elitism_size);
}

// const size_t& get_crossover_first_point(void) const method

/// Returns the first point used for the crossover.

const size_t& GeneticAlgorithm::get_crossover_first_point(void) const
{
    return(crossover_first_point);
}

// const size_t& get_crossover_second_point(void) const method

/// Returns the second point used for the crossover.

const size_t& GeneticAlgorithm::get_crossover_second_point(void) const
{
    return(crossover_second_point);
}

// const double& get_selective_pressure(void) const method

/// Returns the selective pressure used for the fitness assignment.

const double& GeneticAlgorithm::get_selective_pressure(void) const
{
    return(selective_pressure);
}

// const double& get_incest_prevention_distance(void) const method

/// Returns the incest prevention distance used for the crossover.

const double& GeneticAlgorithm::get_incest_prevention_distance(void) const
{
    return(incest_prevention_distance);
}

// const bool& get_reserve_generation_mean(void) const method

/// Returns true if the generation mean of the selection losss are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_mean(void) const
{
    return(reserve_generation_mean);
}

// const bool& get_reserve_generation_standard_deviation(void) const method

/// Returns true if the generation standard deviation of the selection losss are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_standard_deviation(void) const
{
    return(reserve_generation_standard_deviation);
}

// const bool& get_reserve_generation_minimum_selection(void) const method

/// Returns true if the generation minimum selection loss are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_minimum_selection(void) const
{
    return(reserve_generation_minimum_selection);
}

// const bool& get_reserve_generation_optimum_loss(void) const method

/// Returns true if the generation optimum loss error are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_optimum_loss(void) const
{
    return(reserve_generation_optimum_loss);
}

// std::string write_initialization_method(void) const method

/// Return a string with the initialization method of genetic algorithm.

std::string GeneticAlgorithm::write_initialization_method(void) const
{
    switch (initialization_method)
    {
    case Random:
    {
        return ("Random");
    }
    case Weigthed:
    {
        return ("Weigthed");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "std::string write_initialization_method(void) const method.\n"
               << "Unknown initialization method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}

// std::string write_crossover_method(void) const method

/// Return a string with the crossover method of genetic algorithm.

std::string GeneticAlgorithm::write_crossover_method(void) const
{
    switch (crossover_method)
    {
    case OnePoint:
    {
        return ("OnePoint");
    }
    case TwoPoint:
    {
        return ("TwoPoint");
    }
    case Uniform:
    {
        return ("Uniform");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "std::string write_crossover_method(void) const method.\n"
               << "Unknown crossover method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}

// std::string write_fitness_assignment_method(void) const method

/// Return a string with the fitness assignment method of genetic algorithm.

std::string GeneticAlgorithm::write_fitness_assignment_method(void) const
{
    switch (fitness_assignment_method)
    {
    case ObjectiveBased:
    {
        return ("ObjectiveBased");
    }
    case RankBased:
    {
        return ("RankBased");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "std::string write_fitness_assignment_method(void) const method.\n"
               << "Unknown fitness assignment method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}


// void set_default(void) method

/// Sets the members of the genetic algorithm object to their default values.

void GeneticAlgorithm::set_default(void)
{
    size_t inputs_number ;

    if(training_strategy_pointer == NULL
    || !training_strategy_pointer->has_loss_index())
    {
        maximum_iterations_number = 100;

        mutation_rate = 0.5;

        population_size = 10;

    }
    else
    {
        inputs_number = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer()->get_inputs_number();
        maximum_iterations_number = (size_t)std::max(100.,inputs_number*5.);

        mutation_rate = 1./inputs_number;

        population_size = 10*inputs_number;

    }

    // Population stuff

    population.set();

    loss.set();

    fitness.set();

    // Training operators

    initialization_method = Random;

    crossover_method = Uniform;

    fitness_assignment_method = RankBased;

    elitism_size = 2;

    crossover_first_point = 0;

    crossover_second_point = 0;

    selective_pressure = 1.5;

    incest_prevention_distance = 0;

    // inputs selection results

    reserve_generation_mean = true;

    reserve_generation_standard_deviation = false;

    reserve_generation_minimum_selection = true;

    reserve_generation_optimum_loss = true;

}

// void set_population(const Vector< Vector<bool> >&) method

/// Sets a new popualtion.
/// @param new_population New population matrix.

void GeneticAlgorithm::set_population(const Vector< Vector<bool> >& new_population)
{
#ifdef __OPENNN_DEBUG__

    // Training algorithm stuff

    std::ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to training strategy is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to loss functional is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to neural network is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    if(new_population[0].size() != inputs_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population(const Matrix<double>&) method.\n"
               << "Population columns ("<<new_population[0].size()<< ") must be equal to inputs number("<<inputs_number<<").\n";

        throw std::logic_error(buffer.str());
    }

    if(new_population.size() != population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population(const Matrix<double>&) method.\n"
               << "Population rows ("<<new_population.size()<< ") must be equal to population size("<<population_size<<").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    population.set(new_population);
}

// void set_loss(const Matrix<double>&) method

/// Sets a new training losses of the population.
/// @param new_loss New training losses.

void GeneticAlgorithm::set_loss(const Matrix<double>& new_loss)
{
#ifdef __OPENNN_DEBUG__

    if(new_loss.get_columns_number() != 2)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_loss(const Matrix<double>&&) method.\n"
               << "Performance columns must be equal to 2.\n";

        throw std::logic_error(buffer.str());
    }

    if(new_loss.get_rows_number() != population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_loss(const Matrix<double>&&) method.\n"
               << "Performance rows ("<<new_loss.get_rows_number()<< ") must be equal to population size("<<population_size<<").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    loss.set(new_loss);
}

// void set_fitness(const Vector<double>&) method

/// Sets a new fitness for the population.
/// @param new_fitness New fitness values.

void GeneticAlgorithm::set_fitness(const Vector<double>& new_fitness)
{
#ifdef __OPENNN_DEBUG__

    if(new_fitness.size() != population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_fitness(const Matrix<double>&&) method.\n"
               << "Fitness size("<<new_fitness.size()<< ") must be equal to population size("<<population_size<<").\n";

        throw std::logic_error(buffer.str());
    }

    for (size_t i = 0; i < new_fitness.size(); i++)
    {
        if(new_fitness[i] < 0)
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
                   << "void set_fitness(const Matrix<double>&&) method.\n"
                   << "Fitness must be greater than 0.\n";

            throw std::logic_error(buffer.str());
        }
    }

#endif

    fitness.set(new_fitness);
}

// void set_inicialization_method(const InitializationMethod&) method

/// Sets a new method to initiate the population in the algorithm.
/// @param new_initialization_method Method to initialize the population(Random or Weighted).

void GeneticAlgorithm::set_inicialization_method(const InitializationMethod& new_initialization_method)
{
    initialization_method = new_initialization_method;
}

// void set_fitness_assignment_method(const FitnessAssignment&) method

/// Sets a new method to assign the fitness in the algorithm.
/// @param new_fitness_assignment_method Method to assign the fitness(RankBased or ObjectiveBased).

void GeneticAlgorithm::set_fitness_assignment_method(const FitnessAssignment& new_fitness_assignment_method)
{
    fitness_assignment_method = new_fitness_assignment_method;
}

// void set_crossover_method(const CrossoverMethod&) method

/// Sets a new method to perform the crossover in the algorithm.
/// @param new_crossover_method Method to perform the crossover of the selected population(Uniform, OnePoint or TwoPoint).

void GeneticAlgorithm::set_crossover_method(const CrossoverMethod& new_crossover_method)
{
    crossover_method = new_crossover_method;
}

// void set_inicialization_method(const std::string&) method

/// Sets a new initialization method from a string.
/// @param new_initialization_method String with the crossover method.

void GeneticAlgorithm::set_inicialization_method(const std::string& new_initialization_method)
{
    if(new_initialization_method == "Random")
    {
        initialization_method = Random;

    }
    else if(new_initialization_method == "Weigthed")
    {
        initialization_method = Weigthed;

    }
    else{
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_inicialization_method(const std::string&) method.\n"
               << "Unknown initialization method.\n";

        throw std::logic_error(buffer.str());

    }
}

// void set_fitness_assignment_method(const std::string&) method

/// Sets a new fitness assignment method from a string.
/// @param new_fitness_assignment_method String with the fitness assignment method.

void GeneticAlgorithm::set_fitness_assignment_method(const std::string& new_fitness_assignment_method)
{
    if(new_fitness_assignment_method == "RankBased")
    {
        fitness_assignment_method = RankBased;

    }
    else if(new_fitness_assignment_method == "ObjectiveBased")
    {
        fitness_assignment_method = ObjectiveBased;

    }
    else{
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_fitness_assignment_method(const std::string&) method.\n"
               << "Unknown fitness assignment method.\n";

        throw std::logic_error(buffer.str());

    }
}

// void set_crossover_method(const std::string&) method

/// Sets a new crossover method from a string.
/// @param new_crossover_method String with the crossover method.

void GeneticAlgorithm::set_crossover_method(const std::string& new_crossover_method)
{
    if(new_crossover_method == "Uniform")
    {
        crossover_method = Uniform;

    }
    else if(new_crossover_method == "OnePoint")
    {
        crossover_method = OnePoint;

    }
    else if(new_crossover_method == "TwoPoint")
    {
        crossover_method = TwoPoint;

    }
    else{
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_method(const std::string&) method.\n"
               << "Unknown crossover method.\n";

        throw std::logic_error(buffer.str());

    }
}

// void set_population_size(const size_t&) method

/// Sets a new population size. It must be greater than 4.
/// @param new_population_size Size of the population.

void GeneticAlgorithm::set_population_size(const size_t& new_population_size)
{
#ifdef __OPENNN_DEBUG__

    if(new_population_size < 4)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population_size(const size_t&) method.\n"
               << "Population size must be greater than 4.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    population_size = new_population_size;
}

// void set_mutation_rate(const double&) method

/// Sets a new rate used in the mutation.
/// It is a number between 0 and 1.
/// @param new_mutation_rate Rate used for the mutation.

void GeneticAlgorithm::set_mutation_rate(const double& new_mutation_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_mutation_rate < 0 || new_mutation_rate > 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_mutation_rate(const double&) method.\n"
               << "Mutation rate must be between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    mutation_rate = new_mutation_rate;
}

// void set_elitism_size(const size_t&) method

/// Sets the number of individuals with the greatest fitness selected.
/// @param new_elitism_size Size of the elitism.

void GeneticAlgorithm::set_elitism_size(const size_t& new_elitism_size)
{
#ifdef __OPENNN_DEBUG__

    if(new_elitism_size > population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_elitism_size(const size_t&) method.\n"
               << "Elitism size("<< new_elitism_size<<") must be lower than the population size("<<population_size<<").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    elitism_size = new_elitism_size;
}

// void set_crossover_first_point(const size_t&) method

/// Sets the point of for the OnePoint and TwoPoint crossover.
/// If it is set to 0, the algorithm will select it randomly for each crossover.
/// @param new_crossover_first_point Point for the OnePoint and first point for TwoPoint crossover.

void GeneticAlgorithm::set_crossover_first_point(const size_t& new_crossover_first_point)
{
#ifdef __OPENNN_DEBUG__

    if(new_crossover_first_point > population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_first_point(const size_t&) method.\n"
               << "First crossover point("<< new_crossover_first_point<<") must be lower than the population size("<<population_size<<").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    crossover_first_point = new_crossover_first_point;
}

// void set_crossover_second_point(const size_t&) method

/// Sets the point of for the TwoPoint crossover.
/// If it is set to 0, the algorithm will select it randomly for each crossover.
/// @param new_crossover_second_point Second point for the TwoPoint crossover.

void GeneticAlgorithm::set_crossover_second_point(const size_t& new_crossover_second_point)
{
#ifdef __OPENNN_DEBUG__

    if(new_crossover_second_point > population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_second_point(const size_t&) method.\n"
               << "Second crossover point("<< new_crossover_second_point<<") must be lower than the population size("<<population_size<<").\n";

        throw std::logic_error(buffer.str());
    }

    if(new_crossover_second_point <= crossover_first_point && new_crossover_second_point != 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_second_point(const size_t&) method.\n"
               << "Second crossover point("<< new_crossover_second_point<<") must be greater than the first point("<<crossover_first_point<<").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    crossover_second_point = new_crossover_second_point;
}

// void set_selective_pressure(const double&) method

/// Sets a new value for the selective pressure parameter.
/// Linear ranking allows values for the selective pressure greater than 0.
/// @param new_selective_pressure Selective pressure value.

void GeneticAlgorithm::set_selective_pressure(const double& new_selective_pressure)
{
#ifdef __OPENNN_DEBUG__

    if(new_selective_pressure <= 0.0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_selective_pressure(const double&) method. "
               << "Selective pressure must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    selective_pressure = new_selective_pressure;
}

// void set_incest_prevention_distance(const double&) method

/// Sets a new value for the incest prevention distance used in the crossover.
/// @param new_incest_prevention_distance Incest prevention distance value.

void GeneticAlgorithm::set_incest_prevention_distance(const double& new_incest_prevention_distance)
{
    incest_prevention_distance = new_incest_prevention_distance;
}

// void set_reserve_generation_mean(const bool&) method

/// Sets the reserve flag for the generation mean history.
/// @param new_reserve_generation_mean Flag value.

void GeneticAlgorithm::set_reserve_generation_mean(const bool& new_reserve_generation_mean)
{
    reserve_generation_mean = new_reserve_generation_mean;
}

// void set_reserve_generation_standard_deviation(const bool&) method

/// Sets the reserve flag for the generation standard deviation history.
/// @param new_reserve_generation_standard_deviation Flag value.

void GeneticAlgorithm::set_reserve_generation_standard_deviation(const bool& new_reserve_generation_standard_deviation)
{
    reserve_generation_standard_deviation = new_reserve_generation_standard_deviation;
}

// void set_reserve_generation_minimum_selection(const bool&) method

/// Sets the reserve flag for the generation minimum selection loss history.
/// @param new_reserve_generation_minimum_selection Flag value.

void GeneticAlgorithm::set_reserve_generation_minimum_selection(const bool& new_reserve_generation_minimum_selection)
{
    reserve_generation_minimum_selection = new_reserve_generation_minimum_selection;
}

// void set_reserve_generation_optimum_loss(const bool&) method

/// Sets the reserve flag for the optimum loss error history.
/// @param new_reserve_generation_optimum_loss Flag value.

void GeneticAlgorithm::set_reserve_generation_optimum_loss(const bool& new_reserve_generation_optimum_loss)
{
    reserve_generation_optimum_loss = new_reserve_generation_optimum_loss;
}

// GENETIC METHODS

// Population methods

// void initialize_population(void) method

/// Initialize the population depending on the intialization method.

void GeneticAlgorithm::initialize_population(void)
{

#ifdef __OPENNN_DEBUG__

    if(population_size == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void initialize_population(void) method.\n"
               << "Population size must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    population.set(population_size);

    switch (initialization_method)
    {
    case Random:
    {
        initialize_random_population();
        break;
    }
    case Weigthed:
    {
        initialize_weighted_population();
        break;
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void initialize_population(void) method.\n"
               << "Unknown initialization method.\n";

        throw std::logic_error(buffer.str());
    }
    }
}

// void initialize_random_population(void) method

/// Initialize the population with the random intialization method.

void GeneticAlgorithm::initialize_random_population(void)
{
    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    Vector<bool> inputs(inputs_number,true);

    size_t zero_ocurrences;

    size_t random;

    size_t random_loops = 0;

    for (size_t i = 0; i < population_size; i++)
    {
        zero_ocurrences = 0;

        for (size_t j = 0; j < inputs_number; j++)
        {
            random = rand()%2;

            if(random == 0)
            {
                inputs[j] = false;

                zero_ocurrences++;
            }
            else
            {
                inputs[j] = true;
            }
        }

        if(zero_ocurrences == inputs_number)
        {
            inputs[rand()%inputs_number] = true;
        }

        if(population.contains(inputs) && random_loops <= 5)
        {
            random_loops++;

            i--;

        }
        else
        {
            population[i] = inputs;

            random_loops = 0;
        }
    }
}

// void initialize_weighted_population(void) method

/// Initialize the population with the weighted intialization method.

void GeneticAlgorithm::initialize_weighted_population(void)
{
    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    Vector<bool> inputs(inputs_number, false);

    Vector<double> final_correlations = calculate_final_correlations();

    double sum;
    Vector<double> correlations_sum;

    size_t zero_ocurrences;

    double random;

    size_t random_loops = 0;

    for(size_t i = 0; i < final_correlations.size(); i++)
    {
        if(final_correlations[i] < 1.0/population_size)
        {
            final_correlations[i] = 1.0/population_size;
        }
    }

    sum = final_correlations.calculate_sum();
    correlations_sum = final_correlations.calculate_cumulative();

    for (size_t i = 0; i < population_size; i++)
    {

        zero_ocurrences = 0;

        inputs.set(inputs_number,false);

        for (size_t j = 0; j < inputs_number; j++)
        {
            random = calculate_random_uniform(0.,sum);

            for (size_t k = 0; k < correlations_sum.size(); k++)
            {
                if(k == 0 && random < correlations_sum[0])
                {
                    inputs[k] = true;

                    k = correlations_sum.size();
                }
                else if(random < correlations_sum[k] && random >= correlations_sum[k-1])
                {
                    inputs[k] = true;

                    k = correlations_sum.size();
                }
            }

            if(inputs[j] == false)
            {
                zero_ocurrences++;
            }
        }

        if(zero_ocurrences == inputs_number)
        {
            inputs[rand()%inputs_number] = true;
        }

        if(population.contains(inputs) && random_loops <= 5)
        {
            random_loops++;

            i--;

        }
        else
        {
            population[i] = inputs;

            random_loops = 0;
        }
    }
}

// void evaluate_population(void) method

/// Evaluate a population.
/// Perform the trainings of the neural networks and calculate their fitness.

void GeneticAlgorithm::evaluate_population(void)
{
#ifdef __OPENNN_DEBUG__

    check();

    if(population.size() == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void evaluate_population(void) method.\n"
               << "Population size must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();
    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();
    Variables* variables_pointer = data_set_pointer->get_variables_pointer();
    Vector<Variables::Use> current_uses(original_uses);

    Vector<bool> current_inputs;
    size_t current_inputs_number;

    size_t index;
    Vector<double> errors(2);

    loss.set(population_size,2);

    for (size_t i = 0; i < population_size; i++)
    {
#ifdef __OPENNN_MPI__
        // Send population

        const int inputs_number = (int)population[i].size();
        Vector<int> individual_int(inputs_number);

        for(int j = 0; j < inputs_number; j++)
        {
            individual_int[j] = (int)population[i][j];
        }

        MPI_Bcast(individual_int.data(), inputs_number, MPI_INT, 0, MPI_COMM_WORLD);

        for(int j = 0; j < inputs_number; j++)
        {
            population[i][j] = (individual_int[j] == 1);
        }

#endif
        current_inputs = population[i];
        current_inputs_number = current_inputs.count_occurrences(true);

        for (size_t j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);

            if(current_inputs[j] == false)
            {
                current_uses[index] = Variables::Unused;
            }
            else
            {
                current_uses[index] = Variables::Input;
            }
        }

        variables_pointer->set_uses(current_uses);

        set_neural_inputs(current_inputs);

        errors = perform_model_evaluation(population[i]);

        loss.set_row(i, errors);

//        std::cout << current_inputs << std::endl;
    }

    calculate_fitness();
}

// void calculate_fitness(void) method

/// Calculate the fitness with the errors depending on the fitness assignment method.

void GeneticAlgorithm::calculate_fitness(void)
{
#ifdef __OPENNN_DEBUG__

    if(loss.get_rows_number() == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void calculate_fitness(void) method.\n"
               << "Performance size must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    switch (fitness_assignment_method)
    {
    case ObjectiveBased:
    {
        calculate_objetive_fitness();
        break;
    }
    case RankBased:
    {
        calculate_rank_fitness();
        break;
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void calculate_fitness(void) method.\n"
               << "Unknown fitness assignment method.\n";

        throw std::logic_error(buffer.str());
    }
    }

}

// void calculate_objetive_fitness(void) method

/// Calculate the fitness with objective based fitness assignment method.

void GeneticAlgorithm::calculate_objetive_fitness(void)
{

    fitness.set(loss.get_rows_number(),0.);

    for (size_t i = 0; i < fitness.size(); i++)
    {
        fitness[i] = 1/(1 + loss(i,1));
    }
}

// void calculate_objetive_fitness(void) method

/// Calculate the fitness with rank based fitness assignment method.

void GeneticAlgorithm::calculate_rank_fitness(void)
{
    const Vector<size_t> rank = loss.arrange_column(1).calculate_greater_rank();

    fitness.set(loss.get_rows_number(),0.);

    for(size_t i = 0; i < population_size; i++)
    {
        fitness[i] = selective_pressure*rank[i];
    }
}

// void evolve_population(void) method

/// Evolve the population to a new generation.
/// Perform the selection, crossover and mutation of the current generation.

void GeneticAlgorithm::evolve_population(void)
{
    size_t zero_ocurrences;

    perform_selection();

    perform_crossover();

    perform_mutation();

    for(size_t i = 0; i < population.size(); i++)
    {
        zero_ocurrences = 0;
        for(size_t j = 0; j < population[i].size(); j++)
        {
            if(population[i][j] == false)
            {
                zero_ocurrences++;
            }
        }

        if(zero_ocurrences == population[i].size())
        {
            population[i][rand()%population[i].size()] = true;
        }
    }

}

// Selection methods

// void perform_selection(void) method

/// Selects for crossover some individuals from the population.

void GeneticAlgorithm::perform_selection(void)
{
#ifdef __OPENNN_DEBUG__

    if(population.size() == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_selection(void) method.\n"
               << "Population size must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(fitness.empty())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_selection(void) method.\n"
               << "No fitness found.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    const size_t selected_population_size = (size_t)(population_size/2);

    Vector< Vector<bool> > population_copy;

    Vector<bool> selected_population(population.size(),false);

    size_t selected_index;

    Vector<double> fitness_sum = fitness.calculate_cumulative();

    double sum = fitness.calculate_sum();

    double random;

    size_t random_loops = 0;

    Vector<double> fitness_copy(fitness);

    population_copy.set();

    if(elitism_size >= 1)
    {
        selected_index = get_optimal_individual_index();

        selected_population[selected_index] = true;

        population_copy.push_back(population[selected_index]);

        fitness_copy[selected_index] = -1;
    }

    while (population_copy.size() < elitism_size && population_copy.size() < selected_population_size)
    {
        selected_index = fitness_copy.calculate_maximal_index();

        if(!population_copy.contains(population[selected_index]))
        {
            selected_population[selected_index] = true;

            population_copy.push_back(population[selected_index]);
        }

        fitness_copy[selected_index] = -1;
    }

    while (population_copy.size() != selected_population_size)
    {
        random = calculate_random_uniform(0.,sum);

        for (size_t k = 0; k < fitness_sum.size(); k++)
        {
            if(k == 0 && random < fitness_sum[0])
            {
                selected_index = k;
                k = fitness_sum.size();
            }
            else if(random < fitness_sum[k] && random >= fitness_sum[k-1])
            {
                selected_index = k;
                k = fitness_sum.size();
            }
        }

        if((selected_population[selected_index] == false) || random_loops == 5)
        {
            selected_population[selected_index] = true;

            population_copy.push_back(population[selected_index]);

            random_loops = 0;
        }
        else
        {
            random_loops++;
        }
    }

    population.set(population_copy);
}

// Crossover methods

// void perform_crossover(void) method

/// Perform the crossover depending on the crossover method.

void GeneticAlgorithm::perform_crossover(void)
{
#ifdef __OPENNN_DEBUG__

    if(population.size() <= 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_crossover(void) method.\n"
               << "Selected population size must be greater than 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    switch (crossover_method)
    {
    case OnePoint:
    {
        perform_1point_crossover();
        break;
    }
    case TwoPoint:
    {
        perform_2point_crossover();
        break;
    }
    case Uniform:
    {
        perform_uniform_crossover();
        break;
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_crossover(void) method.\n"
               << "Unknown crossover method.\n";

        throw std::logic_error(buffer.str());
    }
    }
}

// void perform_1point_crossover(void) method

/// Perform the OnePoint crossover method.

void GeneticAlgorithm::perform_1point_crossover(void)
{
    const size_t inputs_number = population[0].size();
    const size_t selected_population = population.size();

    size_t parent1_index;
    Vector<bool> parent1(inputs_number);
    size_t parent2_index;
    Vector<bool> parent2(inputs_number);

    Vector<bool> offspring1(inputs_number);
    Vector<bool> offspring2(inputs_number);

    size_t random_loops = 0;

    size_t first_point = crossover_first_point;

    Vector< Vector<bool> > new_population;

    while(new_population.size() < population_size)
    {
        parent1_index = rand()%selected_population;
        parent2_index = rand()%selected_population;

        random_loops = 0;
        while(population[parent1_index].calculate_distance(population[parent2_index]) <= incest_prevention_distance)
        {
            parent2_index = rand()%selected_population;
            random_loops++;

            if(random_loops == 5 && parent1_index != selected_population-1)
            {
                parent2_index = parent1_index+1;
                break;
            }
            else if(random_loops == 5)
            {
                parent2_index = parent1_index-1;
                break;
            }
        }

        parent1 = population[parent1_index];
        parent2 = population[parent2_index];

        if(crossover_first_point == 0)
        {
            first_point = 1 + rand()%(inputs_number-1);
        }

        for (size_t i = 0; i < inputs_number; i++)
        {
            if(i < first_point)
            {
                offspring1[i] = parent1[i];
                offspring2[i] = parent2[i];
            }
            else
            {
                offspring1[i] = parent2[i];
                offspring2[i] = parent1[i];
            }
        }

        new_population.push_back(offspring1);
        if(new_population.size() != population_size)
        {
            new_population.push_back(offspring2);
        }
    }

    set_population(new_population);
}

// void perform_2point_crossover(void) method

/// Perform the TwoPoint crossover method.

void GeneticAlgorithm::perform_2point_crossover(void)
{
    const size_t inputs_number = population[0].size();
    const size_t selected_population = population.size();

    size_t parent1_index;
    Vector<bool> parent1(inputs_number);
    size_t parent2_index;
    Vector<bool> parent2(inputs_number);

    Vector<bool> offspring1(inputs_number);
    Vector<bool> offspring2(inputs_number);

    size_t random_loops = 0;

    size_t first_point = crossover_first_point;
    size_t second_point = crossover_second_point;

    Vector< Vector<bool> > new_population;

    while(new_population.size() < population_size)
    {
        parent1_index = rand()%selected_population;
        parent2_index = rand()%selected_population;

        random_loops = 0;
        while(population[parent1_index].calculate_distance(population[parent2_index]) <= incest_prevention_distance)
        {
            parent2_index = rand()%selected_population;
            random_loops++;

            if(random_loops == 5 && parent1_index != selected_population-1)
            {
                parent2_index = parent1_index+1;
                break;
            }
            else if(random_loops == 5)
            {
                parent2_index = parent1_index-1;
                break;
            }
        }

        parent1 = population[parent1_index];
        parent2 = population[parent2_index];

        if(crossover_first_point == 0)
        {
            first_point = 1 + rand()%(inputs_number-2);
        }

        if(crossover_second_point == 0)
        {
            second_point = first_point + rand()%(inputs_number-1-first_point);
        }

        for (size_t i = 0; i < inputs_number; i++)
        {
            if(i < first_point)
            {
                offspring1[i] = parent1[i];
                offspring2[i] = parent2[i];
            }
            else if(i < second_point)
            {
                offspring1[i] = parent2[i];
                offspring2[i] = parent1[i];
            }
            else
            {
                offspring1[i] = parent1[i];
                offspring2[i] = parent2[i];
            }
        }

        new_population.push_back(offspring1);
        if(new_population.size() != population_size)
        {
            new_population.push_back(offspring2);
        }
    }

    set_population(new_population);
}


// void perform_uniform_crossover(void) method

/// Perform the uniform crossover method.

void GeneticAlgorithm::perform_uniform_crossover(void)
{
    const size_t inputs_number = population[0].size();
    const size_t selected_population = population.size();

    size_t parent1_index;
    Vector<bool> parent1(inputs_number);
    size_t parent2_index;
    Vector<bool> parent2(inputs_number);

    Vector<bool> offspring1(inputs_number);
    Vector<bool> offspring2(inputs_number);

    double random_uniform;
    size_t random_loops = 0;

    Vector< Vector<bool> > new_population;

    while(new_population.size() < population_size)
    {
        parent1_index = rand()%selected_population;
        parent2_index = rand()%selected_population;

        random_loops = 0;
        while(population[parent1_index].calculate_distance(population[parent2_index]) <= incest_prevention_distance)
        {
            parent2_index = rand()%selected_population;
            random_loops++;

            if(random_loops == 5 && parent1_index != selected_population-1)
            {
                parent2_index = parent1_index+1;
                break;
            }
            else if(random_loops == 5)
            {
                parent2_index = parent1_index-1;
                break;
            }
        }

        parent1 = population[parent1_index];
        parent2 = population[parent2_index];

        for (size_t i = 0; i < inputs_number; i++)
        {
            random_uniform = calculate_random_uniform(0.,1.);

            if(random_uniform > 0.5)
            {
                offspring1[i] = parent1[i];
                offspring2[i] = parent2[i];
            }
            else
            {
                offspring1[i] = parent2[i];
                offspring2[i] = parent1[i];
            }
        }

        new_population.push_back(offspring1);
        if(new_population.size() != population_size)
        {
            new_population.push_back(offspring2);
        }
    }

    set_population(new_population);
}

// Mutation methods

// void perform_mutation(void) method

/// Perform the mutation of the individuals generated in the crossover.

void GeneticAlgorithm::perform_mutation(void)
{
#ifdef __OPENNN_DEBUG__

    if(population.size() != population_size)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_mutation(void) method.\n"
               << "Population size must be equal to "<< population_size <<".\n";

        throw std::logic_error(buffer.str());
    }

#endif

    const size_t selected_population_size = population_size - elitism_size;

    double random;

    for (size_t i = selected_population_size; i < population.size(); i++)
    {
        for (size_t j = 0; j < population[i].size(); j++)
        {
            random = calculate_random_uniform(0.,1.);

            if(random <= mutation_rate)
            {
                population[i][j] = !population[i][j];
            }
       }
    }
}

// size_t get_optimal_individual_index(void) const method

/// Return the index of the optimal individual of the population considering the tolerance.

size_t GeneticAlgorithm::get_optimal_individual_index(void) const
{
    size_t index = 0;

    Vector<bool> optimal_inputs = population[0];

    double optimum_error = loss(0,1);

    Vector<bool> current_inputs;

    double current_error;

    for (size_t i = 1; i < population_size; i++)
    {
        current_inputs = population[i];
        current_error = loss(i,1);

        if((fabs(optimum_error-current_error) < tolerance &&
             current_inputs.count_occurrences(true) > optimal_inputs.count_occurrences(true)) ||
            (fabs(optimum_error-current_error) >= tolerance &&
             current_error < optimum_error)   )
        {
            optimal_inputs = current_inputs;
            optimum_error = current_error;

            index = i;
        }
    }

    return(index);
}

// GeneticAlgorithmResults* perform_inputs_selection(void) method

/// Perform the inputs selection with the genetic method.

GeneticAlgorithm::GeneticAlgorithmResults* GeneticAlgorithm::perform_inputs_selection(void)
{

#ifdef __OPENNN_DEBUG__

    check();

#endif

    GeneticAlgorithmResults* results = new GeneticAlgorithmResults();

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Variables* variables = data_set_pointer->get_variables_pointer();

    Vector< Statistics<double> > original_statistics;
    ScalingLayer::ScalingMethod original_scaling_method;

    bool has_scaling_layer = neural_network_pointer->has_scaling_layer();

    if(has_scaling_layer)
    {
        original_statistics = neural_network_pointer->get_scaling_layer_pointer()->get_statistics();
        original_scaling_method = neural_network_pointer->get_scaling_layer_pointer()->get_scaling_method();
    }

    size_t current_minimal_selection_error_index;
    double current_minimum_loss_error;
    double current_minimum_selection_error;
    Vector<bool> current_inputs;
    Vector<Variables::Use> current_uses;
    double current_mean;
    double current_standard_deviation;

    double previous_minimum_selection_error = 1e10;

    double optimum_selection_error = 1e10;
    double optimum_loss_error;
    Vector<bool> optimal_inputs;
    Vector<double> optimal_parameters;

    bool end = false;

    size_t iterations = 0;

    size_t index = 0;

    time_t beginning_time, current_time;
    double elapsed_time;

    if(display)
    {
        std::cout << "Performing genetic inputs selection..." << std::endl;
    }

    original_uses = variables->arrange_uses();

    current_uses = original_uses;

    optimal_inputs.set(original_uses.count_occurrences(Variables::Input),0);

    time(&beginning_time);

    initialize_population();

    while(!end)
    {
        if(iterations != 0)
        {
            evolve_population();
        }

        evaluate_population();

        current_minimal_selection_error_index = get_optimal_individual_index();

        current_mean = loss.arrange_column(1).calculate_mean();

        current_standard_deviation = loss.arrange_column(1).calculate_standard_deviation();

        current_inputs = population[current_minimal_selection_error_index];

        current_minimum_selection_error = loss(current_minimal_selection_error_index,1);

        current_minimum_loss_error = loss(current_minimal_selection_error_index,0);

        if((fabs(optimum_selection_error - current_minimum_selection_error) >= tolerance &&
             optimum_selection_error > current_minimum_selection_error) ||
                (fabs(optimum_selection_error - current_minimum_selection_error) < tolerance &&
                 optimal_inputs.count_occurrences(true) < current_inputs.count_occurrences(true)))
        {
            optimal_inputs = current_inputs;
            optimum_loss_error = current_minimum_loss_error;
            optimum_selection_error = current_minimum_selection_error;
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        previous_minimum_selection_error = current_minimum_selection_error;

        iterations++;

        if(reserve_generation_mean)
        {
            results->generation_mean_history.push_back(current_mean);
        }

        if(reserve_generation_standard_deviation)
        {
            results->generation_standard_deviation_history.push_back(current_standard_deviation);
        }

        if(reserve_generation_minimum_selection)
        {
            results->generation_minimum_selection_history.push_back(current_minimum_selection_error);
        }

        if(reserve_generation_optimum_loss)
        {
            results->generation_optimum_loss_history.push_back(current_minimum_loss_error);
        }

        // STOPPING CRITERIA

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum time reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::MaximumTime;
        }
        else if(current_minimum_selection_error <= selection_loss_goal)
        {
            end = true;

            if(display)
            {
                std::cout << "selection loss reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::SelectionLossGoal;
        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                std::cout << "Maximum number of iterations reached." << std::endl;
            }

            results->stopping_condition = InputsSelectionAlgorithm::MaximumIterations;
        }

        for (size_t j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);

            if(current_inputs[j] == false)
            {
                current_uses[index] = Variables::Unused;
            }
            else
            {
                current_uses[index] = Variables::Input;
            }
        }

        variables->set_uses(current_uses);

        set_neural_inputs(current_inputs);

        if(display)
        {
            std::cout << "Generation: " << iterations << std::endl;
            std::cout << "Generation optimal inputs: " << variables->arrange_inputs_name().to_string() << " " << std::endl;
            std::cout << "Generation optimal number of inputs: " << current_inputs.count_occurrences(true) << std::endl;
            std::cout << "Generation optimum selection loss: " << current_minimum_selection_error << std::endl;
            std::cout << "Corresponding training loss: " << current_minimum_loss_error << std::endl;
            std::cout << "Generation selection mean = " << loss.arrange_column(1).calculate_mean() << std::endl;
            std::cout << "Generation selection standard deviation = " << loss.arrange_column(1).calculate_standard_deviation() << std::endl;
            std::cout << "Elapsed time: " << elapsed_time << std::endl;

            std::cout << std::endl;
        }
    }

    results->inputs_data.set(inputs_history);

    if(reserve_loss_data)
    {
        results->loss_data.set(loss_history);
    }

    if(reserve_selection_loss_data)
    {
        results->selection_loss_data.set(selection_loss_history);
    }

    if(reserve_parameters_data)
    {
        results->parameters_data.set(parameters_history);
    }

    optimal_parameters = get_parameters_inputs(optimal_inputs);
    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }

    results->optimal_inputs = optimal_inputs;
    results->final_selection_loss = optimum_selection_error;
    results->final_loss = perform_model_evaluation(optimal_inputs)[0];
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    Vector< Statistics<double> > statistics;
    size_t original_index;
    Vector<Variables::Use> optimal_uses = original_uses;

    for (size_t i = 0; i < optimal_inputs.size(); i++)
    {
        original_index = get_input_index(original_uses, i);
        if(optimal_inputs[i] == 1)
        {
            optimal_uses[original_index] = Variables::Input;
            if(has_scaling_layer)
                statistics.push_back(original_statistics[i]);
        }
        else
        {
            optimal_uses[original_index] = Variables::Unused;
        }
    }

    variables->set_uses(optimal_uses);

    set_neural_inputs(optimal_inputs);
    neural_network_pointer->set_parameters(optimal_parameters);

    if(neural_network_pointer->has_inputs())
    {
        neural_network_pointer->get_inputs_pointer()->set_names(variables->arrange_inputs_name());
    }

    if(has_scaling_layer)
    {
        ScalingLayer scaling_layer(statistics);
        scaling_layer.set_scaling_method(original_scaling_method);
        neural_network_pointer->set_scaling_layer(scaling_layer);
    }

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        if(neural_network_pointer->has_inputs())
        {
            std::cout << "Optimal inputs: " << neural_network_pointer->get_inputs_pointer()->arrange_names().to_string() << std::endl;
        }

        std::cout << "Optimal number of inputs: " << optimal_inputs.count_occurrences(true) << std::endl;
        std::cout << "Optimum training loss: " << optimum_loss_error << std::endl;
        std::cout << "Optimum selection loss: " << optimum_selection_error << std::endl;
        std::cout << "Elapsed time: " << elapsed_time << std::endl;
    }

    return results;
}

// Matrix<std::string> to_string_matrix(void) const method

/// Writes as matrix of strings the most representative atributes.

Matrix<std::string> GeneticAlgorithm::to_string_matrix(void) const
{
    std::ostringstream buffer;

    Vector<std::string> labels;
    Vector<std::string> values;

   // Trials number

   labels.push_back("Trials number");

   buffer.str("");
   buffer << trials_number;

   values.push_back(buffer.str());

   // Tolerance

   labels.push_back("Tolerance");

   buffer.str("");
   buffer << tolerance;

   values.push_back(buffer.str());

   // Population size

   labels.push_back("Population size");

   buffer.str("");
   buffer << population_size;

   values.push_back(buffer.str());

   // Initialization method

   labels.push_back("Initialization method");

   const std::string initialization_method = write_initialization_method();

   values.push_back(initialization_method);

   // Fitness assignment method

   labels.push_back("Fitness assignment method");

   const std::string fitnes_assignment_method = write_fitness_assignment_method();

   values.push_back(fitnes_assignment_method);

   // Crossover method

   labels.push_back("Crossover method");

   const std::string crossover_method = write_crossover_method();

   values.push_back(crossover_method);

   // Elitism size

   labels.push_back("Elitism size");

   buffer.str("");
   buffer << elitism_size;

   values.push_back(buffer.str());

   // Crossover first point

   labels.push_back("Crossover first point");

   buffer.str("");
   buffer << crossover_first_point;

   values.push_back(buffer.str());

   // Crossover second point

   labels.push_back("Crossover second point");

   buffer.str("");
   buffer << crossover_second_point;

   values.push_back(buffer.str());

   // Selective pressure

   labels.push_back("Selective pressure");

   buffer.str("");
   buffer << selective_pressure;

   values.push_back(buffer.str());

   // Mutation rate

   labels.push_back("Mutation rate");

   buffer.str("");
   buffer << mutation_rate;

   values.push_back(buffer.str());

   // Selection loss goal

   labels.push_back("Selection loss goal");

   buffer.str("");
   buffer << selection_loss_goal;

   values.push_back(buffer.str());

   // Maximum Generations number

   labels.push_back("Maximum Generations number");

   buffer.str("");
   buffer << maximum_iterations_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Plot training loss history

   labels.push_back("Plot training loss history");

   buffer.str("");

   if(reserve_loss_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot selection loss history

   labels.push_back("Plot selection loss history");

   buffer.str("");

   if(reserve_selection_loss_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot generation mean history

   labels.push_back("Plot generation mean history");

   buffer.str("");

   if(reserve_generation_mean)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot generation standard deviation history

   labels.push_back("Plot generation standard deviation history");

   buffer.str("");

   if(reserve_generation_standard_deviation)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<std::string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels);
   string_matrix.set_column(1, values);

    return(string_matrix);
}

// tinyxml2::XMLDocument* to_XML(void) const method

/// Prints to the screen the genetic algorithm parameters, the stopping criteria
/// and other user stuff concerning the genetic algorithm object.

tinyxml2::XMLDocument* GeneticAlgorithm::to_XML(void) const
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Order Selection algorithm

    tinyxml2::XMLElement* root_element = document->NewElement("GeneticAlgorithm");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = NULL;
    tinyxml2::XMLText* text = NULL;

    // Regression
//    {
//        element = document->NewElement("Approximation");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << approximation;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Trials number
    {
        element = document->NewElement("TrialsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << trials_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Tolerance
    {
        element = document->NewElement("Tolerance");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << tolerance;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Population size
    {
        element = document->NewElement("PopulationSize");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << population_size;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Initialization method
    {
        element = document->NewElement("InitializationMethod");
        root_element->LinkEndChild(element);

        text = document->NewText(write_initialization_method().c_str());
        element->LinkEndChild(text);
    }

    // Fitness assignment method
    {
        element = document->NewElement("FitnessAssignmentMethod");
        root_element->LinkEndChild(element);

        text = document->NewText(write_fitness_assignment_method().c_str());
        element->LinkEndChild(text);
    }

    // Crossover method
    {
        element = document->NewElement("CrossoverMethod");
        root_element->LinkEndChild(element);

        text = document->NewText(write_crossover_method().c_str());
        element->LinkEndChild(text);
    }

    // Elitism size
    {
        element = document->NewElement("ElitismSize");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << elitism_size;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Crossover first point
    {
        element = document->NewElement("CrossoverFirstPoint");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << crossover_first_point;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Crossover second point
    {
        element = document->NewElement("CrossoverSecondPoint");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << crossover_second_point;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Selective pressure
    {
        element = document->NewElement("SelectivePressure");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << selective_pressure;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Mutation rate
    {
        element = document->NewElement("MutationRate");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << mutation_rate;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // selection loss goal
    {
        element = document->NewElement("SelectionLossGoal");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << selection_loss_goal;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Maximum iterations
    {
        element = document->NewElement("MaximumGenerationsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_iterations_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Maximum time
    {
        element = document->NewElement("MaximumTime");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_time;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Reserve generation optimum loss
    {
        element = document->NewElement("ReserveGenerationOptimumPerformance");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << reserve_generation_optimum_loss;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Reserve generation minimum selection
    {
        element = document->NewElement("ReserveGenerationMinimumSelection");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << reserve_generation_minimum_selection;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Reserve generation mean
    {
        element = document->NewElement("ReserveGenerationMean");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << reserve_generation_mean;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Reserve generation standard deviation
    {
        element = document->NewElement("ReserveGenerationStandardDeviation");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << reserve_generation_standard_deviation;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Performance calculation method
//    {
//        element = document->NewElement("PerformanceCalculationMethod");
//        root_element->LinkEndChild(element);

//        text = document->NewText(write_loss_calculation_method().c_str());
//        element->LinkEndChild(text);
//    }

    // Incest prevention distance
//    {
//        element = document->NewElement("IncestPreventionDistance");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << incest_prevention_distance;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Reserve parameters data
//    {
//        element = document->NewElement("ReserveParametersData");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << reserve_parameters_data;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Reserve loss data
//    {
//        element = document->NewElement("ReservePerformanceHistory");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << reserve_loss_data;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Reserve selection loss data
//    {
//        element = document->NewElement("ReserveSelectionLossHistory");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << reserve_selection_loss_data;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Reserve minimal parameters
//    {
//        element = document->NewElement("ReserveMinimalParameters");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << reserve_minimal_parameters;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Display
//    {
//        element = document->NewElement("Display");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << display;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Maximum correlation
//    {
//        element = document->NewElement("MaximumCorrelation");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << maximum_correlation;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Minimum correlation
//    {
//        element = document->NewElement("MinimumCorrelation");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << minimum_correlation;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the genetic algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GeneticAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("GeneticAlgorithm");

    // Trials number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Tolerance

    file_stream.OpenElement("Tolerance");

    buffer.str("");
    buffer << tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Population size

    file_stream.OpenElement("PopulationSize");

    buffer.str("");
    buffer << population_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Initialization method

    file_stream.OpenElement("InitializationMethod");

    file_stream.PushText(write_initialization_method().c_str());

    file_stream.CloseElement();

    // Fitness assignment method

    file_stream.OpenElement("FitnessAssignmentMethod");

    file_stream.PushText(write_fitness_assignment_method().c_str());

    file_stream.CloseElement();

    // Crossover method

    file_stream.OpenElement("CrossoverMethod");

    file_stream.PushText(write_crossover_method().c_str());

    file_stream.CloseElement();

    // Elitism size

    file_stream.OpenElement("ElitismSize");

    buffer.str("");
    buffer << elitism_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Crossover first point

    file_stream.OpenElement("CrossoverFirstPoint");

    buffer.str("");
    buffer << crossover_first_point;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Crossover second point

    file_stream.OpenElement("CrossoverSecondPoint");

    buffer.str("");
    buffer << crossover_second_point;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Selective pressure

    file_stream.OpenElement("SelectivePressure");

    buffer.str("");
    buffer << selective_pressure;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Mutation rate

    file_stream.OpenElement("MutationRate");

    buffer.str("");
    buffer << mutation_rate;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // selection loss goal

    file_stream.OpenElement("SelectionLossGoal");

    buffer.str("");
    buffer << selection_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations

    file_stream.OpenElement("MaximumGenerationsNumber");

    buffer.str("");
    buffer << maximum_iterations_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve generation optimum loss

    file_stream.OpenElement("ReserveGenerationOptimumPerformance");

    buffer.str("");
    buffer << reserve_generation_optimum_loss;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve generation minimum selection

    file_stream.OpenElement("ReserveGenerationMinimumSelection");

    buffer.str("");
    buffer << reserve_generation_minimum_selection;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve generation mean

    file_stream.OpenElement("ReserveGenerationMean");

    buffer.str("");
    buffer << reserve_generation_mean;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve generation standard deviation

    file_stream.OpenElement("ReserveGenerationStandardDeviation");

    buffer.str("");
    buffer << reserve_generation_standard_deviation;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this genetic algorithm object.
/// @param document TinyXML document containing the member data.

void GeneticAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GeneticAlgorithm");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GeneticAlgorithm element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Regression
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Approximation");

        if(element)
        {
            const std::string new_approximation = element->GetText();

            try
            {
                set_approximation(new_approximation != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Trials number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
            const size_t new_trials_number = atoi(element->GetText());

            try
            {
                set_trials_number(new_trials_number);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Performance calculation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PerformanceCalculationMethod");

        if(element)
        {
            const std::string new_loss_calculation_method = element->GetText();

            try
            {
                set_loss_calculation_method(new_loss_calculation_method);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Initialization method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InitializationMethod");

        if(element)
        {
            const std::string new_initialization_method = element->GetText();

            try
            {
                set_inicialization_method(new_initialization_method);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Crossover method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CrossoverMethod");

        if(element)
        {
            const std::string new_crossover_method = element->GetText();

            try
            {
                set_crossover_method(new_crossover_method);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
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

    // Population size
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PopulationSize");

        if(element)
        {
            const size_t new_population_size = atoi(element->GetText());

            try
            {
                set_population_size(new_population_size);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Incest prevention distance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("IncestPreventionDistance");

        if(element)
        {
            const double new_incest_prevention_rate = atof(element->GetText());

            try
            {
                set_incest_prevention_distance(new_incest_prevention_rate);
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

    // Crossover first point
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CrossoverFirstPoint");

        if(element)
        {
            const size_t new_crossover_first_point = atoi(element->GetText());

            try
            {
                set_crossover_first_point(new_crossover_first_point);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Crossover second point
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CrossoverSecondPoint");

        if(element)
        {
            const size_t new_crossover_second_point = atoi(element->GetText());

            try
            {
                set_crossover_second_point(new_crossover_second_point);
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

    // Reserve generation mean
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationMean");

        if(element)
        {
            const std::string new_reserve_generation_mean = element->GetText();

            try
            {
                set_reserve_generation_mean(new_reserve_generation_mean != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve generation standard deviation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationStandardDeviation");

        if(element)
        {
            const std::string new_reserve_generation_standard_deviation = element->GetText();

            try
            {
                set_reserve_generation_standard_deviation(new_reserve_generation_standard_deviation != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve generation minimum selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationMinimumSelection");

        if(element)
        {
            const std::string new_reserve_generation_minimum_selection = element->GetText();

            try
            {
                set_reserve_generation_minimum_selection(new_reserve_generation_minimum_selection != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve generation optimum loss
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationOptimumPerformance");

        if(element)
        {
            const std::string new_reserve_generation_optimum_loss = element->GetText();

            try
            {
                set_reserve_generation_optimum_loss(new_reserve_generation_optimum_loss != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve parameters data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersData");

        if(element)
        {
            const std::string new_reserve_parameters_data = element->GetText();

            try
            {
                set_reserve_parameters_data(new_reserve_parameters_data != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

        if(element)
        {
            const std::string new_reserve_loss_data = element->GetText();

            try
            {
                set_reserve_loss_data(new_reserve_loss_data != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve selection loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

        if(element)
        {
            const std::string new_reserve_selection_loss_data = element->GetText();

            try
            {
                set_reserve_selection_loss_data(new_reserve_selection_loss_data != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Reserve minimal parameters
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMinimalParameters");

        if(element)
        {
            const std::string new_reserve_minimal_parameters = element->GetText();

            try
            {
                set_reserve_minimal_parameters(new_reserve_minimal_parameters != "0");
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
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

    // selection loss goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionLossGoal");

        if(element)
        {
            const double new_selection_loss_goal = atof(element->GetText());

            try
            {
                set_selection_loss_goal(new_selection_loss_goal);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Maximum iterations number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumGenerationsNumber");

        if(element)
        {
            const size_t new_maximum_iterations_number = atoi(element->GetText());

            try
            {
                set_maximum_iterations_number(new_maximum_iterations_number);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Maximum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumCorrelation");

        if(element)
        {
            const double new_maximum_correlation = atof(element->GetText());

            try
            {
                set_maximum_correlation(new_maximum_correlation);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

    // Minimum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumCorrelation");

        if(element)
        {
            const double new_minimum_correlation = atof(element->GetText());

            try
            {
                set_minimum_correlation(new_minimum_correlation);
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
            const double new_maximum_time = atoi(element->GetText());

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

    // Tolerance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Tolerance");

        if(element)
        {
            const double new_tolerance = atof(element->GetText());

            try
            {
                set_tolerance(new_tolerance);
            }
            catch(const std::logic_error& e)
            {
                std::cout << e.what() << std::endl;
            }
        }
    }

}

// void save(const std::string&) const method

/// Saves to a XML-type file the members of the genetic algorithm object.
/// @param file_name Name of genetic algorithm XML-type file.

void GeneticAlgorithm::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a genetic algorithm object from a XML-type file.
/// @param file_name Name of genetic algorithm XML-type file.

void GeneticAlgorithm::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}

// std::string to_string(void) const method

/// Returns a string representation of the current genetic algorithm results structure.

std::string GeneticAlgorithm::GeneticAlgorithmResults::to_string(void) const
{
    std::ostringstream buffer;

    // Parameters history

    if(!parameters_data.empty())
    {
        buffer << "% Parameters history:\n"
               << parameters_data.to_row_matrix() << "\n";
    }

    // Performance history

    if(!loss_data.empty())
    {
        buffer << "% Performance history:\n"
               << loss_data.to_row_matrix() << "\n";
    }

    // selection loss history

    if(!selection_loss_data.empty())
    {
        buffer << "% Selection loss history:\n"
               << selection_loss_data.to_row_matrix() << "\n";
    }

    // Generation optimum loss history

    if(!generation_optimum_loss_history.empty())
    {
        buffer << "% Generation optimum loss history:\n"
               << generation_optimum_loss_history.to_string() << "\n";
    }

    // Generation minimum selection history

    if(!generation_minimum_selection_history.empty())
    {
        buffer << "% Generation minimum selection history:\n"
               << generation_minimum_selection_history.to_string() << "\n";
    }

    // Generation mean history

    if(!generation_mean_history.empty())
    {
        buffer << "% Generation mean history:\n"
               << generation_mean_history.to_string() << "\n";
    }

    // Generation standard deviation history

    if(!generation_standard_deviation_history.empty())
    {
        buffer << "% Generation standard deviation history:\n"
               << generation_standard_deviation_history.to_string() << "\n";
    }

    // Minimal parameters

    if(!minimal_parameters.empty())
    {
        buffer << "% Minimal parameters:\n"
               << minimal_parameters << "\n";
    }

    // Stopping condition

    buffer << "% Stopping condition\n"
           << write_stopping_condition() << "\n";

    // Optimum selection loss

    if(final_selection_loss != 0)
    {
        buffer << "% Optimum selection loss:\n"
               << final_selection_loss << "\n";
    }

    // Final loss

    if(final_loss != 0)
    {
        buffer << "% Final loss:\n"
               << final_loss << "\n";
    }

    // Optimal input

    if(!optimal_inputs.empty())
    {
        buffer << "% Optimal input:\n"
               << optimal_inputs << "\n";
    }

    // Iterations number


    buffer << "% Number of iterations:\n"
           << iterations_number << "\n";


    // Elapsed time

    buffer << "% Elapsed time:\n"
           << elapsed_time << "\n";



    return(buffer.str());
}
}



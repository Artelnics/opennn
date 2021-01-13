//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "genetic_algorithm.h"

namespace OpenNN
{

/// Default constructor.

GeneticAlgorithm::GeneticAlgorithm()
    : InputsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

GeneticAlgorithm::GeneticAlgorithm(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// Destructor.

GeneticAlgorithm::~GeneticAlgorithm()
{
}


/// Returns the population matrix.

const Tensor<bool, 2>& GeneticAlgorithm::get_population() const
{
    return population;
}


/// Returns the training and selection losses of the population.

const Tensor<type, 2>& GeneticAlgorithm::get_loss() const
{
    return loss;
}


/// Returns the fitness of the population.

const Tensor<type, 1>& GeneticAlgorithm::get_fitness() const
{
    return fitness;
}


/// Returns the method for the initialization of the population.

const GeneticAlgorithm::InitializationMethod& GeneticAlgorithm::get_initialization_method() const
{
    return initialization_method;
}


/// Returns the method for the crossover of the population.

const GeneticAlgorithm::CrossoverMethod& GeneticAlgorithm::get_crossover_method() const
{
    return crossover_method;
}


/// Returns the method for the fitness assignment of the population.

const GeneticAlgorithm::FitnessAssignment& GeneticAlgorithm::get_fitness_assignment_method() const
{
    return fitness_assignment_method;
}


/// Returns the size of the population.

const Index& GeneticAlgorithm::get_population_size() const
{
    return population_size;
}


/// Returns the rate used in the mutation.

const type& GeneticAlgorithm::get_mutation_rate() const
{
    return mutation_rate;
}


/// Returns the size of the elite in the selection.

const Index& GeneticAlgorithm::get_elitism_size() const
{
    return elitism_size;
}


/// Returns the first point used for the crossover.

const Index& GeneticAlgorithm::get_crossover_first_point() const
{
    return crossover_first_point;
}


/// Returns the second point used for the crossover.

const Index& GeneticAlgorithm::get_crossover_second_point() const
{
    return crossover_second_point;
}


/// Returns the selective pressure used for the fitness assignment.

const type& GeneticAlgorithm::get_selective_pressure() const
{
    return selective_pressure;
}


/// Returns the incest prevention distance used for the crossover.

const type& GeneticAlgorithm::get_incest_prevention_distance() const
{
    return incest_prevention_distance;
}


/// Returns true if the generation mean of the selection losses are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_mean() const
{
    return reserve_generation_mean;
}


/// Returns true if the generation standard deviation of the selection losses are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_standard_deviation() const
{
    return reserve_generation_standard_deviation;
}


/// Returns true if the generation minimum selection error are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_minimum_selection() const
{
    return reserve_generation_minimum_selection;
}


/// Returns true if the generation optimum loss error are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_optimum_loss() const
{
    return reserve_generation_optimum_loss;
}


/// Return a string with the initialization method of genetic algorithm.

string GeneticAlgorithm::write_initialization_method() const
{
    switch(initialization_method)
    {
    case Random:
    {
        return "Random";
    }
    case Weigthed:
    {
        return "Weigthed";
    }
    }

    return string();
}


/// Return a string with the crossover method of genetic algorithm.

string GeneticAlgorithm::write_crossover_method() const
{
    switch(crossover_method)
    {
    case OnePoint:
    {
        return "OnePoint";
    }
    case TwoPoint:
    {
        return "TwoPoint";
    }
    case Uniform:
    {
        return "Uniform";
    }
    }

    return string();
}


/// Return a string with the fitness assignment method of genetic algorithm.

string GeneticAlgorithm::write_fitness_assignment_method() const
{
    switch(fitness_assignment_method)
    {
    case ObjectiveBased:
    {
        return "ObjectiveBased";
    }
    case RankBased:
    {
        return "RankBased";
    }
    }

    return string();
}


/// Sets the members of the genetic algorithm object to their default values.

void GeneticAlgorithm::set_default()
{
    Index inputs_number;

    if(training_strategy_pointer == nullptr
            || !training_strategy_pointer->has_neural_network())
    {
        maximum_epochs_number = 100;

        mutation_rate = 0.5;

        population_size = 10;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();
        maximum_epochs_number = static_cast<Index>(max(100.,inputs_number*5.));

        mutation_rate = static_cast<type>(1.0/inputs_number);

        population_size = 10 * inputs_number;

    }

    // Population stuff

    population.resize(0, 0);//set();

    loss.resize(0, 0);//set();

    fitness.resize(0);//set();

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

    //trials number

    trials_number = 1;

}


/// Sets a new popualtion.
/// @param new_population New population matrix.

void GeneticAlgorithm::set_population(const Tensor<bool, 2>& new_population)
{
#ifdef __OPENNN_DEBUG__

    // Optimization algorithm

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(new_population.dimension(0)  != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population(const Tensor<type, 2>&) method.\n"
               << "Population rows("<<new_population.dimension(0)
               << ") must be equal to population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    population.setZero();

    population = new_population;
}


/// Sets a new training losses of the population.
/// @param new_loss New training losses.

void GeneticAlgorithm::set_loss(const Tensor<type, 2>& new_loss)
{
#ifdef __OPENNN_DEBUG__

    if(new_loss.dimension(1) != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_loss(const Tensor<type, 2>&&) method.\n"
               << "Performance columns must be equal to 2.\n";

        throw logic_error(buffer.str());
    }

    if(new_loss.dimension(0) != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_loss(const Tensor<type, 2>&&) method.\n"
               << "Performance rows("<<new_loss.dimension(0)
               << ") must be equal to population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    copy(new_loss.data(), new_loss.data() + new_loss.size(), loss.data());
}


/// Sets a new fitness for the population.
/// @param new_fitness New fitness values.

void GeneticAlgorithm::set_fitness(const Tensor<type, 1>& new_fitness)
{
#ifdef __OPENNN_DEBUG__

    if(new_fitness.size() != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_fitness(const Tensor<type, 2>&&) method.\n"
               << "Fitness size("<<new_fitness.size()
               << ") must be equal to population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

    for(Index i = 0; i < new_fitness.size(); i++)
    {
        if(new_fitness[i] < 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
                   << "void set_fitness(const Tensor<type, 2>&&) method.\n"
                   << "Fitness must be greater than 0.\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    copy(new_fitness.data(), new_fitness.data() + new_fitness.size(), fitness.data());
}


/// Sets a new method to initiate the population in the algorithm.
/// @param new_initialization_method Method to initialize the population(Random or Weighted).

void GeneticAlgorithm::set_inicialization_method(const InitializationMethod& new_initialization_method)
{
    initialization_method = new_initialization_method;
}


/// Sets a new method to assign the fitness in the algorithm.
/// @param new_fitness_assignment_method Method to assign the fitness(RankBased or ObjectiveBased).

void GeneticAlgorithm::set_fitness_assignment_method(const FitnessAssignment& new_fitness_assignment_method)
{
    fitness_assignment_method = new_fitness_assignment_method;
}


/// Sets a new method to perform the crossover in the algorithm.
/// @param new_crossover_method Method to perform the crossover of the selected population
/// (Uniform, OnePoint or TwoPoint).

void GeneticAlgorithm::set_crossover_method(const CrossoverMethod& new_crossover_method)
{
    crossover_method = new_crossover_method;
}


/// Sets a new initialization method from a string.
/// @param new_initialization_method String with the crossover method.

void GeneticAlgorithm::set_inicialization_method(const string& new_initialization_method)
{
    if(new_initialization_method == "Random")
    {
        initialization_method = Random;

    }
    else if(new_initialization_method == "Weigthed")
    {
        initialization_method = Weigthed;

    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_inicialization_method(const string&) method.\n"
               << "Unknown initialization method.\n";

        throw logic_error(buffer.str());

    }
}


/// Sets a new fitness assignment method from a string.
/// @param new_fitness_assignment_method String with the fitness assignment method.

void GeneticAlgorithm::set_fitness_assignment_method(const string& new_fitness_assignment_method)
{
    if(new_fitness_assignment_method == "RankBased")
    {
        fitness_assignment_method = RankBased;

    }
    else if(new_fitness_assignment_method == "ObjectiveBased")
    {
        fitness_assignment_method = ObjectiveBased;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_fitness_assignment_method(const string&) method.\n"
               << "Unknown fitness assignment method.\n";

        throw logic_error(buffer.str());

    }
}


/// Sets a new crossover method from a string.
/// @param new_crossover_method String with the crossover method.

void GeneticAlgorithm::set_crossover_method(const string& new_crossover_method)
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
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_method(const string&) method.\n"
               << "Unknown crossover method.\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new population size. It must be greater than 4.
/// @param new_population_size Size of the population.

void GeneticAlgorithm::set_population_size(const Index& new_population_size)
{
#ifdef __OPENNN_DEBUG__

    if(new_population_size < 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population_size(const Index&) method.\n"
               << "Population size must be greater than 4.\n";

        throw logic_error(buffer.str());
    }

#endif

    population_size = new_population_size;
}


/// Sets a new rate used in the mutation.
/// It is a number between 0 and 1.
/// @param new_mutation_rate Rate used for the mutation.

void GeneticAlgorithm::set_mutation_rate(const type& new_mutation_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_mutation_rate < 0 || new_mutation_rate > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_mutation_rate(const type&) method.\n"
               << "Mutation rate must be between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    mutation_rate = new_mutation_rate;
}


/// Sets the number of individuals with the greatest fitness selected.
/// @param new_elitism_size Size of the elitism.

void GeneticAlgorithm::set_elitism_size(const Index& new_elitism_size)
{
#ifdef __OPENNN_DEBUG__

    if(new_elitism_size > population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_elitism_size(const Index&) method.\n"
               << "Elitism size("<< new_elitism_size
               <<") must be lower than the population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    elitism_size = new_elitism_size;
}


/// Sets the point of for the OnePoint and TwoPoint crossover.
/// If it is set to 0, the algorithm will select it randomly for each crossover.
/// @param new_crossover_first_point Point for the OnePoint and first point for TwoPoint crossover.

void GeneticAlgorithm::set_crossover_first_point(const Index& new_crossover_first_point)
{
#ifdef __OPENNN_DEBUG__

    if(new_crossover_first_point > population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_first_point(const Index&) method.\n"
               << "First crossover point("<< new_crossover_first_point
               <<") must be lower than the population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    crossover_first_point = new_crossover_first_point;
}


/// Sets the point of for the TwoPoint crossover.
/// If it is set to 0, the algorithm will select it randomly for each crossover.
/// @param new_crossover_second_point Second point for the TwoPoint crossover.

void GeneticAlgorithm::set_crossover_second_point(const Index& new_crossover_second_point)
{
#ifdef __OPENNN_DEBUG__

    if(new_crossover_second_point > population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_second_point(const Index&) method.\n"
               << "Second crossover point("<< new_crossover_second_point
               <<") must be lower than the population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

    if(new_crossover_second_point <= crossover_first_point && new_crossover_second_point != 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_second_point(const Index&) method.\n"
               << "Second crossover point("<< new_crossover_second_point
               <<") must be greater than the first point("<<crossover_first_point<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    crossover_second_point = new_crossover_second_point;
}


/// Sets a new value for the selective pressure parameter.
/// Linear ranking allows values for the selective pressure greater than 0.
/// @param new_selective_pressure Selective pressure value.

void GeneticAlgorithm::set_selective_pressure(const type& new_selective_pressure)
{
#ifdef __OPENNN_DEBUG__

    if(new_selective_pressure <= static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_selective_pressure(const type&) method. "
               << "Selective pressure must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selective_pressure = new_selective_pressure;
}


/// Sets a new value for the incest prevention distance used in the crossover.
/// @param new_incest_prevention_distance Incest prevention distance value.

void GeneticAlgorithm::set_incest_prevention_distance(const type& new_incest_prevention_distance)
{
    incest_prevention_distance = new_incest_prevention_distance;
}


/// Sets the reserve flag for the generation mean history.
/// @param new_reserve_generation_mean Flag value.

void GeneticAlgorithm::set_reserve_generation_mean(const bool& new_reserve_generation_mean)
{
    reserve_generation_mean = new_reserve_generation_mean;
}


/// Sets the reserve flag for the generation standard deviation history.
/// @param new_reserve_generation_standard_deviation Flag value.

void GeneticAlgorithm::set_reserve_generation_standard_deviation(const bool& new_reserve_generation_standard_deviation)
{
    reserve_generation_standard_deviation = new_reserve_generation_standard_deviation;
}


/// Sets the reserve flag for the generation minimum selection error history.
/// @param new_reserve_generation_minimum_selection Flag value.

void GeneticAlgorithm::set_reserve_generation_minimum_selection(const bool& new_reserve_generation_minimum_selection)
{
    reserve_generation_minimum_selection = new_reserve_generation_minimum_selection;
}


/// Sets the reserve flag for the optimum loss error history.
/// @param new_reserve_generation_optimum_loss Flag value.

void GeneticAlgorithm::set_reserve_generation_optimum_loss(const bool& new_reserve_generation_optimum_loss)
{
    reserve_generation_optimum_loss = new_reserve_generation_optimum_loss;
}

// GENETIC METHODS

// Population methods


/// Initialize the population depending on the intialization method.

void GeneticAlgorithm::initialize_population()
{

#ifdef __OPENNN_DEBUG__

    if(population_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void initialize_population() method.\n"
               << "Population size must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();

    population.resize(population_size, inputs_number);
    population.setZero();

    switch(initialization_method)
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
    }
}


/// Initialize the population with the random intialization method.

void GeneticAlgorithm::initialize_random_population()
{
    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    Tensor<bool, 1> inputs(inputs_number);

    inputs.setConstant(true);

    Index zero_ocurrences;

    Index random;

    Index random_loops = 0;

    for(Index i = 0; i < population_size; i++)
    {
        zero_ocurrences = 0;

        for(Index j = 0; j < inputs_number; j++)
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
            inputs(static_cast<Index>(rand())%inputs_number) = true;
        }

        bool contains = true;

        for(Index k = 0; k < population.dimension(1); k++)
        {
            for(Index l = 0; l < inputs.size(); l++)
            {
                if(population(l,k) != inputs(l)) contains = false;

                contains = true;
            }
        }
        if(contains && random_loops <= 5)
        {
            random_loops++;

            i--;
        }
        else
        {
            for(Index k = 0; k < inputs.size(); k++)
            {
                population(i,k) = inputs(k);
            }
            random_loops = 0;
        }        
    }
}


/// Initialize the population with the weighted intialization method.

void GeneticAlgorithm::initialize_weighted_population()
{
    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Tensor<type, 2> correlations = data_set_pointer->calculate_input_target_columns_correlations_values();

    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

    Tensor<type, 1> final_correlations = correlations.sum(rows_sum).abs();

    // Neural network

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

//    const Index inputs_number = neural_network_pointer->get_inputs_number();

    const Index inputs_number = data_set_pointer->get_input_columns_number();

    // Optimization algortihm stuff

    Tensor<bool, 1> inputs(inputs_number);
    inputs.setConstant(false);

    Tensor<type, 0> sum;

    Tensor<type, 1> correlations_sum;

    Index zero_ocurrences;

    type random;

    //type random_uniform;

    Index random_loops = 0;

    for(Index i = 0; i < final_correlations.size(); i++)
    {
        if(final_correlations(i) < static_cast<type>(1.0)/population_size)
        {
            final_correlations(i) = static_cast<type>(1.0)/population_size;
        }
    }

    sum = final_correlations.sum();

    correlations_sum = final_correlations.cumsum(0);//cumulative(final_correlations);

    for(Index i = 0; i < population_size; i++)
    {
        zero_ocurrences = 0;

        inputs.resize(inputs_number);
        inputs.setConstant(false);

        for(Index j = 0; j < inputs_number; j++)
        {
            random = sum(0)*static_cast<type>(rand() /(RAND_MAX + 1.0));

            for(Index k = 0; k < correlations_sum.size(); k++)
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
            inputs[static_cast<Index>(rand())%inputs_number] = true;
        }

        bool contains = true;

        for(Index k = 0; k < population.dimension(1); k++)
        {
            for(Index l = 0; l < inputs.size(); l++)
            {
                if(population(l,k) != inputs(l)) contains = false;

                contains = true;
            }
        }

        if(contains && random_loops <= 5)
        {
            random_loops++;

            i--;
        }
        else
        {
            for(Index k = 0; k < population.dimension(0); k++)
            {
                population(k,i) = inputs(k);
            }

            random_loops = 0;
        }

    }
}


/// Evaluate a population.
/// Training all the neural networks in the population and calculate their fitness.

void GeneticAlgorithm::evaluate_population()
{
#ifdef __OPENNN_DEBUG__

    check();

    if(population.size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void evaluate_population() method.\n"
               << "Population size must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Tensor<DataSet::VariableUse, 1> current_uses(original_uses);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm

    Tensor<bool, 1> current_inputs;

    Index index = 0;

    Tensor<type, 1> errors(2);

    loss.resize(population_size,2);
    loss.setZero();

    for(Index i = 0; i < population_size; i++)
    {
        current_inputs = population.chip(i,0);

        for(Index j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);

            Index inputs_number = 0;

            if(current_inputs[j] == false)
            {
                current_uses[index] = DataSet::UnusedVariable;
            }
            else
            {
                current_uses[index] = DataSet::Input;
                inputs_number++;
            }
        }

        data_set_pointer->set_columns_uses(current_uses);

        neural_network_pointer->set_inputs_number(data_set_pointer->get_input_variables_number());

        // Training Neural networks

        errors = calculate_losses(population.chip(i,0));

        for(Index k = 0; k < loss.dimension(1); k++)
        {
            loss(i,k) = errors(k);
        }

    }

    calculate_fitness();
}


/// Calculate the fitness with the errors depending on the fitness assignment method.

void GeneticAlgorithm::calculate_fitness()
{
#ifdef __OPENNN_DEBUG__

    if(loss.dimension(0) == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void calculate_fitness() method.\n"
               << "Loss size must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(fitness_assignment_method)
    {
    case ObjectiveBased:
    {
        calculate_objective_fitness();
        break;
    }
    case RankBased:
    {
        calculate_rank_fitness();
        break;
    }
    }
}


/// Calculate the fitness with error based fitness assignment method.

void GeneticAlgorithm::calculate_objective_fitness()
{

    fitness.resize(loss.dimension(0));
    fitness.setConstant(0.);

    for(Index i = 0; i < fitness.size(); i++)
    {
        fitness(i) = 1/(1 + loss(i,1));
    }

}


/// Calculate the fitness with rank based fitness assignment method.

void GeneticAlgorithm::calculate_rank_fitness()
{

    const Tensor<type, 1> column = loss.chip(1,1);

    const Index column_size = column.size();

    Tensor<Index, 1> rank(column_size);

    Tensor<type, 1> sorted_vector(column);

    sort(sorted_vector.data(), sorted_vector.data() + sorted_vector.size(), greater<type>());

    Tensor<Index, 1> previous_rank(column_size);

    for(Index i = 0; i < column_size; i++)
    {
        for(Index j = 0; j < column_size; j++)
        {
//            for(Index k = 0; k < previous_rank.size(); k++)
//            {
//                if(previous_rank(k) == j) continue;
//            }

            if(fabsf(static_cast<type>(column(i)) - static_cast<type>(sorted_vector(j))) <= static_cast<type>(1e-6))
            {
                rank(i) = j;

//                previous_rank(i) = j;

                break;
            }
        }
    }

    fitness.resize(loss.dimension(0));

    for(Index i = 0; i < population_size; i++)
    {
        fitness(i) = selective_pressure * rank(i);
    }
}


/// Evolve the population to a new generation.
/// Perform the selection, crossover and mutation of the current generation.

void GeneticAlgorithm::evolve_population()
{
    Index zero_ocurrences;

    perform_selection();

    cout << "perform selection" << endl;

    perform_crossover();

    cout << "perform crossover" << endl;

    perform_mutation();

    cout << "perform mutation" << endl;

    for(Index i = 0; i < population.dimension(0); i++)
    {
        zero_ocurrences = 0;

        const Tensor<bool, 1> population_row = population.chip(i,0);

        for(Index j = 0; j < population_row.size(); j++)
        {
            if(population(i,j) == false)
            {
                zero_ocurrences++;
            }
        }

        if(zero_ocurrences == population_row.size())
        {
            population(i,static_cast<Index>(rand())%population_row.size()) = true;
        }

    }
}


/// Selects for crossover some individuals from the population.

void GeneticAlgorithm::perform_selection()
{
#ifdef __OPENNN_DEBUG__

    if(population.size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_selection() method.\n"
               << "Population size must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(fitness.dimension(0) == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_selection() method.\n"
               << "No fitness found.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index selected_population_size = static_cast<Index>(population_size/2);

    Tensor<bool, 2> population_copy;

    vector<vector<bool>> population_vector_copy;

    Tensor<bool, 1> selected_population(population.dimension(0));
    selected_population.setConstant(false);

    Index selected_index = 0;

    Tensor<type, 1> fitness_sum = fitness.cumsum(0);    

    const Tensor<type, 0> sum = fitness.sum();

    type random;

    Index random_loops = 0;

    Tensor<type, 1> fitness_copy(fitness);

    // Elitist selection

    if(elitism_size >= 1)
    {
        selected_index = get_optimal_individual_index();

        selected_population[selected_index] = true;

        population_vector_copy.push_back(tensor_to_vector(population.chip(selected_index, 0)));

        fitness_copy[selected_index] = -1;
    }

    // Natural selection

    while(static_cast<Index>(population_vector_copy.size()) < elitism_size
       && static_cast<Index>(population_vector_copy.size()) < selected_population_size)
    {
        selected_index = maximal_index(fitness_copy);

        const Tensor<bool, 1> selected_population_row = population.chip(selected_index,0);

        Index count = 0;

        for(size_t i = 0; i < population_vector_copy.size(); i++)
        {
            count = 0;

            for(Index j = 0; j < selected_population_row.size(); j++)
            {
                if(population_vector_copy[i][static_cast<size_t>(j)] == selected_population_row(j)) count++;
            }
        }

        if(count < selected_population_row.size())
        {
            selected_population[selected_index] = true;

            population_vector_copy.push_back(tensor_to_vector(selected_population_row));
        }

        fitness_copy[selected_index] = -1;
    }

    // Roulette wheel

    while(static_cast<Index>(population_vector_copy.size()) != selected_population_size)
    {
        random = sum(0)*static_cast<type>(rand() /(RAND_MAX + 1.0));

        for(Index k = 0; k < fitness_sum.size(); k++)
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

        if(selected_population[selected_index] == false || random_loops == 5)
        {
            selected_population[selected_index] = true;

            population_vector_copy.push_back(tensor_to_vector(population.chip(selected_index,0)));

            random_loops = 0;
        }
        else
        {
            random_loops++;
        }
    }
cout << "wheel" << endl;
    population_copy.resize(static_cast<Index>(population_vector_copy.size()), static_cast<Index>(population_vector_copy[0].size()));

    population.setZero();

    for(size_t i = 0; i < population_vector_copy.size(); i++)
    {
        for(size_t j = 0; j < population_vector_copy[0].size(); j++)
        {
            population(static_cast<Index>(i),static_cast<Index>(j)) = population_vector_copy[i][j];
        }
    }

}


/// Perform the crossover depending on the crossover method.

void GeneticAlgorithm::perform_crossover()
{
#ifdef __OPENNN_DEBUG__

    if(population.size() <= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_crossover() method.\n"
               << "Selected population size must be greater than 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(crossover_method)
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
    }
}


/// Perform the OnePoint crossover method.

void GeneticAlgorithm::perform_1point_crossover()
{

    const Index inputs_number = population.dimension(1);
    const Index selected_population = population.dimension(0);

    Index parent1_index;
    Tensor<bool, 1> parent1(inputs_number);

    Index parent2_index;
    Tensor<bool, 1> parent2(inputs_number);

    Tensor<bool, 1> offspring1(inputs_number);
    Tensor<bool, 1> offspring2(inputs_number);

    Index random_loops = 0;

    Index first_point = crossover_first_point;

    vector<vector<bool>> new_population;

    while(static_cast<Index>(new_population.size()) < population_size)
    {
        parent1_index = static_cast<Index>(rand())%selected_population;

        parent2_index = static_cast<Index>(rand())%selected_population;

        random_loops = 0;

        const Tensor<type, 1 > parent_1 = population.chip(parent1_index,0).cast<type>();
        const Tensor<type, 1 > parent_2 = population.chip(parent2_index,0).cast<type>();

        while(euclidean_distance(parent_1, parent_2)
                <= incest_prevention_distance)
        {
            parent2_index = static_cast<Index>(rand())%selected_population;

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

        parent1 = population.chip(parent1_index,1);
        parent2 = population.chip(parent2_index,1);

        if(crossover_first_point == 0)
        {
            first_point = 1 + static_cast<Index>(rand())%(inputs_number-1);
        }

        for(Index i = 0; i < inputs_number; i++)
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

        new_population.push_back(tensor_to_vector(offspring1));

        if(static_cast<Index>(new_population.size()) != population_size)
        {
            new_population.push_back(tensor_to_vector(offspring2));
        }
    }

    Tensor<bool, 2> population_copy(static_cast<Index>(new_population.size()), static_cast<Index>(new_population[0].size()));

    for(size_t i = 0; i < new_population.size(); i++)
    {
        for(size_t j = 0; j < new_population[0].size(); j++)
        {
            population_copy(static_cast<Index>(i),static_cast<Index>(j)) = new_population[i][j];
        }
    }

    set_population(population_copy);
}


/// Perform the TwoPoint crossover method.

void GeneticAlgorithm::perform_2point_crossover()
{

    const Index inputs_number = population.dimension(1);
    const Index selected_population = population.dimension(0);

    Index parent1_index;
    Tensor<bool, 1> parent1(inputs_number);
    Index parent2_index;
    Tensor<bool, 1> parent2(inputs_number);

    Tensor<bool, 1> offspring1(inputs_number);
    Tensor<bool, 1> offspring2(inputs_number);

    Index random_loops = 0;

    Index first_point = crossover_first_point;
    Index second_point = crossover_second_point;

    vector<vector<bool>> new_population;

    while(static_cast<Index>(new_population.size()) < population_size)
    {
        parent1_index = static_cast<Index>(rand())%selected_population;
        parent2_index = static_cast<Index>(rand())%selected_population;

        random_loops = 0;

        const Tensor<type, 1 > parent_1 = population.chip(parent1_index,0).cast<type>();
        const Tensor<type, 1 > parent_2 = population.chip(parent2_index,0).cast<type>();

        while(euclidean_distance(parent_1, parent_2)
                <= incest_prevention_distance)
        {
            parent2_index = static_cast<Index>(rand())%selected_population;
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

        parent1 = population.chip(parent1_index,0);
        parent2 = population.chip(parent2_index,0);

        if(crossover_first_point == 0)
        {
            first_point = 1 + static_cast<Index>(rand())%(inputs_number-2);
        }

        if(crossover_second_point == 0)
        {
            second_point = first_point + static_cast<Index>(rand())%(inputs_number-1-first_point);
        }

        for(Index i = 0; i < inputs_number; i++)
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

        new_population.push_back(tensor_to_vector(offspring1));

        if(static_cast<Index>(new_population.size()) < population_size)
        {
            new_population.push_back(tensor_to_vector(offspring2));
        }
    }

    Tensor<bool, 2> population_copy(static_cast<Index>(new_population.size()), static_cast<Index>(new_population[0].size()));

    for(size_t i = 0; i < new_population.size(); i++)
    {
        for(size_t j = 0; j < new_population[0].size(); j++)
        {
            population_copy(static_cast<Index>(i),static_cast<Index>(j)) = new_population[i][j];
        }
    }

    set_population(population_copy);

}


/// Perform the uniform crossover method.

void GeneticAlgorithm::perform_uniform_crossover()
{

    const Index inputs_number = population.dimension(1);
    const Index selected_population = static_cast<Index>(population_size/2);

    Index parent1_index;
    Tensor<bool, 1> parent1(inputs_number);

    Index parent2_index;
    Tensor<bool, 1> parent2(inputs_number);

    Tensor<bool, 1> offspring1(inputs_number);
    Tensor<bool, 1> offspring2(inputs_number);

    type random_uniform;
    Index random_loops = 0;

    vector<vector<bool>> new_population;

    while(static_cast<Index>(new_population.size()) < population_size)
    {
        parent1_index = static_cast<Index>(rand())%selected_population;
        parent2_index = static_cast<Index>(rand())%selected_population;

        random_loops = 0;

        const Tensor<type, 1 > parent_1 = population.chip(parent1_index,0).cast<type>();
        const Tensor<type, 1 > parent_2 = population.chip(parent2_index,0).cast<type>();

        while(euclidean_distance(parent_1, parent_2)
                <= incest_prevention_distance)
        {
            parent2_index = static_cast<Index>(rand())%selected_population;
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

        parent1 = population.chip(parent1_index,0);
        parent2 = population.chip(parent2_index,0);

        for(Index i = 0; i < inputs_number; i++)
        {
            random_uniform = static_cast<type>(1.0)*static_cast<type>(rand() /(RAND_MAX + 1.0));

            if(random_uniform > static_cast<type>(0.5))
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

        new_population.push_back(tensor_to_vector(offspring1));

        if(static_cast<Index>(new_population.size()) != population_size)
        {
            new_population.push_back(tensor_to_vector(offspring2));
        }
    }

    Tensor<bool, 2> population_copy(static_cast<Index>(new_population.size()), static_cast<Index>(new_population[0].size()));

    for(size_t i = 0; i < new_population.size(); i++)
    {
        for(size_t j = 0; j < new_population[0].size(); j++)
        {
            population_copy(static_cast<Index>(i),static_cast<Index>(j)) = new_population[i][j];
        }
    }

    set_population(population_copy);
}


/// Perform the mutation of the individuals generated in the crossover.

void GeneticAlgorithm::perform_mutation()
{
#ifdef __OPENNN_DEBUG__

    if(population.dimension(0) != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_mutation() method.\n"
               << "Population size must be equal to "<< population_size <<".\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index selected_population_size = population_size - elitism_size;

    type random;

    for(Index i = selected_population_size; i < population.dimension(0); i++)
    {
        for(Index j = 0; j < population.dimension(1); j++)
        {
            random = static_cast<type>(1.0)*static_cast<type>(rand() /(RAND_MAX + 1.0));

            if(random <= mutation_rate)
            {
                population(i,j) = !population(i,j);
            }
        }
    }
}


/// Return the index of the optimal individual of the population considering the tolerance.

Index GeneticAlgorithm::get_optimal_individual_index() const
{
    Index index = 0;

    Tensor<bool, 1> optimal_inputs = population.chip(0,0);

    type optimum_error = loss(0,1);

    Tensor<bool, 1> current_inputs(optimal_inputs.size());

    type current_error = 0;

    for(Index i = 1; i < population_size; i++)
    {
        current_inputs = population.chip(i,0);
        current_error = loss(i,1);

        Index count_inputs = 0;
        Index count_optimal = 0;

        for(Index j = 0; j < optimal_inputs.size(); j++)
        {
            if(current_inputs(j) == true) count_inputs++;
            if(optimal_inputs(j) == true) count_optimal++;
        }

        if((abs(optimum_error-current_error) < tolerance &&
                count_inputs > count_optimal) ||
                (abs(optimum_error-current_error) >= tolerance &&
                 current_error < optimum_error)  )
        {
            optimal_inputs = current_inputs;
            optimum_error = current_error;

            index = i;
        }
    }

    return index;
}


/// Select the inputs with best generalization properties using the genetic algorithm.

GeneticAlgorithm::GeneticAlgorithmResults* GeneticAlgorithm::perform_inputs_selection()
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    GeneticAlgorithmResults* results = new GeneticAlgorithmResults();

    if(display) cout << "Performing genetic inputs selection..." << endl;

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    type optimum_selection_error = numeric_limits<type>::max();
    type optimum_training_error = numeric_limits<type>::max();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Tensor<Index, 1> original_input_columns_indices = data_set_pointer->get_input_columns_indices();

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Tensor<Descriptives, 1> original_input_variables_descriptives = neural_network_pointer->get_scaling_layer_pointer()->get_descriptives();

    const Tensor<ScalingLayer::ScalingMethod, 1> original_scaling_methods = neural_network_pointer->get_scaling_layer_pointer()->get_scaling_methods();

    // Optimization algorithm

    Index minimal_index;
    type current_training_error = 0;
    type current_selection_error = 0;

    Tensor<bool, 1> current_inputs;
    Tensor<DataSet::VariableUse, 1> current_uses;
    Tensor<type, 0> current_mean;
    type current_standard_deviation;
    Tensor<type, 1> current_parameters;

    Index optimal_generation = 0;

    bool end_algortihm = false;

    Index index = 0;

    time_t beginning_time, current_time;
    type elapsed_time = 0;

    original_uses = data_set_pointer->get_columns_uses();

    current_uses = original_uses;

    Index count = data_set_pointer->get_input_columns_number();

    Tensor<bool, 1> optimal_inputs(count);
    optimal_inputs.setZero();

    Tensor<type, 1> optimal_parameters;

    Tensor<type, 2>  test(100,4);

    time(&beginning_time);

    initialize_population();

    for(Index iteration = 0; iteration < maximum_epochs_number; iteration++)
    {
        cout << "iteration: " << iteration << endl;

        if(iteration != 0)
        {
            evolve_population();
        }

        evaluate_population();

        minimal_index = get_optimal_individual_index();

        current_mean = loss.chip(1,1).mean();

        current_standard_deviation = standard_deviation(loss.chip(1,1));

        current_inputs = population.chip(minimal_index,0);

        current_selection_error = loss(minimal_index,1);

        current_training_error = loss(minimal_index,0);

        current_parameters = parameters_history((population_size * iteration) + minimal_index);

        Index count_optimal = 0;
        Index count_inputs = 0;

        for(Index k = 0; k < optimal_inputs.size(); k++)
        {
            if(optimal_inputs(k) == true) count_optimal++;
            if(current_inputs(k) == true) count_inputs++;
        }

        if((abs(optimum_selection_error - current_selection_error) >= tolerance &&
                optimum_selection_error > current_selection_error) ||
            (abs(optimum_selection_error - current_selection_error) < tolerance &&
            count_inputs < count_optimal))
        {
            optimal_inputs = current_inputs;
            optimum_training_error = current_training_error;
            optimum_selection_error = current_selection_error;
            optimal_generation = iteration;
            optimal_parameters = current_parameters;
        }

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(reserve_generation_mean)
        {
            results->generation_selection_error_mean_history = insert_result(current_mean(), results->generation_selection_error_mean_history);
        }

        if(reserve_generation_standard_deviation)
        {
            results->generation_selection_error_standard_deviation_history = insert_result(current_standard_deviation, results->generation_selection_error_standard_deviation_history);
        }

        if(reserve_generation_minimum_selection)
        {
            results->generation_minimum_selection_error_history = insert_result(current_selection_error, results->generation_minimum_selection_error_history);
        }

        if(reserve_generation_optimum_loss)
        {
            results->generation_optimum_training_error_history = insert_result(current_training_error, results->generation_optimum_training_error_history);
        }

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end_algortihm = true;

            if(display)
            {
                cout << "Maximum time reached." << endl;
            }

            results->stopping_condition = InputsSelection::MaximumTime;
        }
        else if(current_selection_error <= selection_error_goal)
        {
            end_algortihm = true;

            if(display)
            {
                cout << "selection error reached." << endl;
            }

            results->stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(iteration >= maximum_epochs_number-1)
        {
            end_algortihm = true;

            if(display)
            {
                cout << "Maximum number of epochs reached." << endl;
            }

            results->stopping_condition = InputsSelection::MaximumEpochs;
        }

        current_uses.setConstant(DataSet::UnusedVariable);
        for(Index j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);
            if(current_inputs[j] == false)
            {
                current_uses(index) = DataSet::UnusedVariable;
            }
            else
            {
                current_uses(index) = DataSet::Input;
            }
        }

        // Print results

        data_set_pointer->set_columns_uses(current_uses);
        Tensor<type, 0> mean = loss.chip(1,1).mean();

        if(display)
        {
            cout << "Generation: " << iteration+1 << endl;
            cout << "Generation optimal inputs: " << data_set_pointer->get_input_variables_names().cast<string>() << " " << endl;
            cout << "Generation optimal number of inputs: " << data_set_pointer->get_input_variables_names().size() << endl;
            cout << "Generation optimum selection error: " << current_selection_error << endl;
            cout << "Corresponding training loss: " << current_training_error << endl;
            cout << "Generation selection mean = " << mean(0) << endl;
            cout << "Generation selection standard deviation = " << standard_deviation(loss.chip(1,1)) << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            cout << endl;
        }

        if(end_algortihm == true)
        {
            // Save results
            results->optimal_inputs = optimal_inputs;
            results->final_selection_error = optimum_selection_error;
            results->final_training_error = optimum_training_error;
            results->iterations_number = iteration + 1;
            results->elapsed_time = write_elapsed_time(elapsed_time);
            results->minimal_parameters = optimal_parameters;            
            break;
        }
    }

    Index original_index;

    Tensor<DataSet::VariableUse, 1> optimal_uses = original_uses;

    for(Index i = 0; i < optimal_inputs.size(); i++)

    {
        original_index = get_input_index(original_uses, i);
        if(optimal_inputs[i] == 1)
        {
            optimal_uses(original_index) = DataSet::Input;
        }
        else
        {
            optimal_uses(original_index) = DataSet::UnusedVariable;
        }
    }

    // Set Data set results

    data_set_pointer->set_columns_uses(optimal_uses);

    const Index optimal_inputs_number = data_set_pointer->get_input_variables_number();

    results->optimal_inputs_indices = data_set_pointer->get_input_variables_indices();

    const Index optimal_input_variables_number = data_set_pointer->get_input_variables_names().size();

    data_set_pointer->set_input_variables_dimensions(Tensor<Index, 1> (1).setConstant(optimal_input_variables_number));

//    // Set Neural network results

    neural_network_pointer->set_inputs_number(optimal_input_variables_number);

    neural_network_pointer->set_parameters(optimal_parameters);

    neural_network_pointer->set_inputs_names(data_set_pointer->get_input_variables_names());

    Tensor<Descriptives, 1> new_input_descriptives(optimal_input_variables_number);
    Tensor<ScalingLayer::ScalingMethod, 1> new_scaling_methods(optimal_input_variables_number);

    Index descriptive_index = 0;
    Index unused = 0;

    for(Index i = 0; i < original_input_columns_indices.size(); i++)
    {
        const Index current_column_index = original_input_columns_indices(i);

        if(data_set_pointer->get_column_use(current_column_index) == DataSet::Input)
        {
            if(data_set_pointer->get_column_type(current_column_index) != DataSet::ColumnType::Categorical)
            {
                new_input_descriptives(descriptive_index) = original_input_variables_descriptives(descriptive_index + unused);
                new_scaling_methods(descriptive_index) = original_scaling_methods(descriptive_index + unused);
                descriptive_index++;
            }
            else
            {
                for(Index j = 0; j < data_set_pointer->get_columns()[current_column_index].get_categories_number(); j++)
                {
                    new_input_descriptives(descriptive_index) = original_input_variables_descriptives(descriptive_index + unused);
                    new_scaling_methods(descriptive_index) = original_scaling_methods(descriptive_index + unused);
                    descriptive_index++;
                }
            }
        }
        else if(data_set_pointer->get_column_use(current_column_index) == DataSet::UnusedVariable)
        {
            if(data_set_pointer->get_column_type(current_column_index) != DataSet::ColumnType::Categorical) unused ++;
            else
            {
                for(Index j = 0; j < data_set_pointer->get_columns()[current_column_index].get_categories_number(); j++) unused ++;
            }
        }
    }

    neural_network_pointer->get_scaling_layer_pointer()->set_descriptives(new_input_descriptives);
    neural_network_pointer->get_scaling_layer_pointer()->set_scaling_methods(new_scaling_methods);

    time(&current_time);
    elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

    if(display)
    {
        cout << "Optimal inputs: " << data_set_pointer->get_input_variables_names().cast<string>() << endl;
        cout << "Optimal generation: " << optimal_generation << endl;
        cout << "Optimal number of inputs: " << optimal_inputs_number << endl;
        cout << "Optimum training error: " << optimum_training_error << endl;
        cout << "Optimum selection error: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    return results;
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> GeneticAlgorithm::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(18, 2);

    ostringstream buffer;

    Tensor<string, 1> labels(18);
    Tensor<string, 1> values(18);

    // Trials number

//    string_matrix(0, 0) = "Trials number";
    labels(0) = "Trials number";

    buffer.str("");
    buffer << trials_number;
    values(0) = buffer.str();

    // Tolerance

    string_matrix(1, 0) = "Tolerance";
    labels(1) = "Tolerance";

    buffer.str("");
    buffer << tolerance;
    values(1) = buffer.str();


    // Population size

//    string_matrix(2, 0) = "Population size";
    labels(2) = "Population size";

    buffer.str("");
    buffer << population_size;
    values(2) = buffer.str();

    // Initialization method

//    string_matrix(3, 0) = "Initialization method";
    labels(3) = "Initialization method";

    const string initialization_method = write_initialization_method();
    values(3) = initialization_method;

    // Fitness assignment method

//    labels.push_back("Fitness assignment method");
    labels(4) = "Fitness assignment method";

    const string fitnes_assignment_method = write_fitness_assignment_method();

//    values.push_back(fitnes_assignment_method);
    values(4) = fitnes_assignment_method;

    // Crossover method

//    labels.push_back("Crossover method");
    labels(5) ="Crossover method";

    const string crossover_method = write_crossover_method();

//    values.push_back(crossover_method);
    values(5) = crossover_method;

    // Elitism size

//    labels.push_back("Elitism size");
    labels(6) = "Elitism size";

    buffer.str("");
    buffer << elitism_size;
    values(6) = buffer.str();


    // Crossover first point

//    labels.push_back("Crossover first point");
    labels(7) = "Crossover first point";

    buffer.str("");
    buffer << crossover_first_point;
    values(7) = buffer.str();

    // Crossover second point

//    labels.push_back("Crossover second point");
    labels(8) = "Crossover second point";

    buffer.str("");
    buffer << crossover_second_point;
    values(8) = buffer.str();

    // Selective pressure

//    labels.push_back("Selective pressure");
    labels(9) = "Selective pressure";

    buffer.str("");
    buffer << selective_pressure;
    values(9) = buffer.str();

    // Mutation rate

//    labels.push_back("Mutation rate");
    labels(10) = "Mutation rate";

    buffer.str("");
    buffer << mutation_rate;
    values(10) = buffer.str();


    // Selection loss goal

//    labels.push_back("Selection loss goal");
    labels(11) = "Selection loss goal";

    buffer.str("");
    buffer << selection_error_goal;
    values(11) = buffer.str();

    // Maximum Generations number

//    labels.push_back("Maximum Generations number");
    labels(12) = "Maximum Generations number";

    buffer.str("");
    buffer << maximum_epochs_number;
    values(12) = buffer.str();

    // Maximum time

//    labels.push_back("Maximum time");
    labels(13) = "Maximum time";

    buffer.str("");
    buffer << maximum_time;
    values(13) = buffer.str();

    // Plot training error history

//    labels.push_back("Plot training error history");
    labels(14) = "Plot training error history";

    buffer.str("");

    if(reserve_training_error_data)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

//    values.push_back(buffer.str());
    values(14) = buffer.str();

    // Plot selection error history

//    labels.push_back("Plot selection error history");
    labels(15) = "Plot selection error histroy";

    buffer.str("");

    if(reserve_selection_error_data)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(15) = buffer.str();

    // Plot generation mean history

//    labels.push_back("Plot generation mean history");
    labels(16) = "Plot generation mean history";

    buffer.str("");

    if(reserve_generation_mean)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(16) = buffer.str();

    // Plot generation standard deviation history

//    labels.push_back("Plot generation standard deviation history");
    labels(17) = "Plot generation standard deviation history";

    buffer.str("");

    if(reserve_generation_standard_deviation)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(17) = buffer.str();

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

    return string_matrix;
}


/// Serializes the genetic algorithm object into a XML document of the TinyXML library without keep the DOM
/// tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GeneticAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("GeneticAlgorithm");

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

    // selection error goal

    file_stream.OpenElement("SelectionErrorGoal");

    buffer.str("");
    buffer << selection_error_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations

    file_stream.OpenElement("MaximumGenerationsNumber");

    buffer.str("");
    buffer << maximum_epochs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve generation optimum loss

    file_stream.OpenElement("ReserveGenerationOptimumLoss");

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

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this genetic algorithm object.
/// @param document TinyXML document containing the member data.

void GeneticAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GeneticAlgorithm");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GeneticAlgorithm element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Regression
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Approximation");

        if(element)
        {
            const string new_approximation = element->GetText();

            try
            {
                set_approximation(new_approximation != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Trials number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
            const Index new_trials_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_trials_number(new_trials_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Initialization method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InitializationMethod");

        if(element)
        {
            const string new_initialization_method = element->GetText();

            try
            {
                set_inicialization_method(new_initialization_method);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Crossover method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CrossoverMethod");

        if(element)
        {
            const string new_crossover_method = element->GetText();

            try
            {
                set_crossover_method(new_crossover_method);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Fitness assignment method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("FitnessAssignmentMethod");

        if(element)
        {
            const string new_fitness_assignment_method = element->GetText();

            try
            {
                set_fitness_assignment_method(new_fitness_assignment_method);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Population size
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PopulationSize");

        if(element)
        {
            const Index new_population_size = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_population_size(new_population_size);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Incest prevention distance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("IncestPreventionDistance");

        if(element)
        {
            const type new_incest_prevention_rate = static_cast<type>(atof(element->GetText()));

            try
            {
                set_incest_prevention_distance(new_incest_prevention_rate);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Mutation rate
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MutationRate");

        if(element)
        {
            const type new_mutation_rate = static_cast<type>(atof(element->GetText()));

            try
            {
                set_mutation_rate(new_mutation_rate);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Elitism size
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ElitismSize");

        if(element)
        {
            const Index new_elitism_size = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_elitism_size(new_elitism_size);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Crossover first point
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CrossoverFirstPoint");

        if(element)
        {
            const Index new_crossover_first_point = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_crossover_first_point(new_crossover_first_point);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Crossover second point
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CrossoverSecondPoint");

        if(element)
        {
            const Index new_crossover_second_point = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_crossover_second_point(new_crossover_second_point);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Selective pressure
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectivePressure");

        if(element)
        {
            const type new_selective_pressure = static_cast<type>(atof(element->GetText()));

            try
            {
                set_selective_pressure(new_selective_pressure);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve generation mean
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationMean");

        if(element)
        {
            const string new_reserve_generation_mean = element->GetText();

            try
            {
                set_reserve_generation_mean(new_reserve_generation_mean != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve generation standard deviation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationStandardDeviation");

        if(element)
        {
            const string new_reserve_generation_standard_deviation = element->GetText();

            try
            {
                set_reserve_generation_standard_deviation(new_reserve_generation_standard_deviation != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve generation minimum selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationMinimumSelection");

        if(element)
        {
            const string new_reserve_generation_minimum_selection = element->GetText();

            try
            {
                set_reserve_generation_minimum_selection(new_reserve_generation_minimum_selection != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve generation optimum loss
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveGenerationOptimumLoss");

        if(element)
        {
            const string new_reserve_generation_optimum_loss = element->GetText();

            try
            {
                set_reserve_generation_optimum_loss(new_reserve_generation_optimum_loss != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve error data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

        if(element)
        {
            const string new_reserve_training_error_data = element->GetText();

            try
            {
                set_reserve_training_error_data(new_reserve_training_error_data != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve selection error data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
            const string new_reserve_selection_error_data = element->GetText();

            try
            {
                set_reserve_selection_error_data(new_reserve_selection_error_data != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve minimal parameters
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMinimalParameters");

        if(element)
        {
            const string new_reserve_minimal_parameters = element->GetText();

            try
            {
                set_reserve_minimal_parameters(new_reserve_minimal_parameters != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

        if(element)
        {
            const string new_display = element->GetText();

            try
            {
                set_display(new_display != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // selection error goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionErrorGoal");

        if(element)
        {
            const type new_selection_error_goal = static_cast<type>(atof(element->GetText()));

            try
            {
                set_selection_error_goal(new_selection_error_goal);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum iterations number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumGenerationsNumber");

        if(element)
        {
            const Index new_maximum_iterations_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_iterations_number(new_maximum_iterations_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumCorrelation");

        if(element)
        {
            const type new_maximum_correlation = static_cast<type>(atof(element->GetText()));

            try
            {
                set_maximum_correlation(new_maximum_correlation);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Minimum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumCorrelation");

        if(element)
        {
            const type new_minimum_correlation = static_cast<type>(atof(element->GetText()));

            try
            {
                set_minimum_correlation(new_minimum_correlation);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum time
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

        if(element)
        {
            const type new_maximum_time = atoi(element->GetText());

            try
            {
                set_maximum_time(new_maximum_time);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Tolerance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Tolerance");

        if(element)
        {
            const type new_tolerance = static_cast<type>(atof(element->GetText()));

            try
            {
                set_tolerance(new_tolerance);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

}


/// Saves to a XML-type file the members of the genetic algorithm object.
/// @param file_name Name of genetic algorithm XML-type file.

void GeneticAlgorithm::save(const string& file_name) const
{
//    tinyxml2::XMLDocument* document = to_XML();

//    document->SaveFile(file_name.c_str());

//    delete document;
}


/// Loads a genetic algorithm object from a XML-type file.
/// @param file_name Name of genetic algorithm XML-type file.

void GeneticAlgorithm::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
}


type GeneticAlgorithm::euclidean_distance(const Tensor<type, 1>& tensor, const Tensor<type, 1>& other_tensor)
{
    const Index x_size = tensor.size();

#ifdef __OPENNN_DEBUG__

    const Index y_size = other_tensor.size();

    if(y_size != x_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double euclidean_distance(const Vector<double>&) const "
               "method.\n"
               << "Size must be equal to this size.\n";

        throw logic_error(buffer.str());
    }

#endif

    type distance = 0.0;
    type error;

    for(Index i = 0; i < x_size; i++)
    {
        error = tensor(i) - other_tensor(i);

        distance += error * error;
    }

    return sqrt(distance);
}


vector<bool> GeneticAlgorithm::tensor_to_vector(const Tensor<bool, 1>& tensor)
{
    const size_t size = static_cast<size_t>(tensor.dimension(0));

    vector<bool> new_vector(static_cast<size_t>(size));

    for(size_t i = 0; i < size; i++)
    {
        new_vector[i] = tensor(static_cast<Index>(i));
    }

    return new_vector;
}


bool GeneticAlgorithm::contains(const vector<vector<bool>>&values, const vector<bool>&values_2) const
{

  if(values.empty()) {
    return false;
  }

  vector<bool> copy(values_2);

  const size_t values_size = values.size();

  for(size_t i = 0; i < values_size; i++)
  {
  for(size_t j = 0; j < values[i].size(); j++)
  {
      typename vector<bool>::iterator it = find(copy.begin(), copy.end(), values[i][j]);

      if(it != copy.end())
      {
          return true;
      }
  }
  }

  return false;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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



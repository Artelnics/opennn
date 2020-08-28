//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "genetic_algorithm.h"

namespace OpenNN {

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


/// File constructor.
/// @param file_name Name of XML order selection file.

GeneticAlgorithm::GeneticAlgorithm(const string& file_name)
    : InputsSelection(file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/// @param genetic_algorithm_document Pointer to a TinyXML document containing the genetic algorithm data.

GeneticAlgorithm::GeneticAlgorithm(const tinyxml2::XMLDocument& genetic_algorithm_document)
    : InputsSelection(genetic_algorithm_document)
{
    from_XML(genetic_algorithm_document);
}


/// Destructor.

GeneticAlgorithm::~GeneticAlgorithm()
{
}


/// Returns the population matrix.

const Vector<Vector<bool>>& GeneticAlgorithm::get_population() const
{
    return(population);
}


/// Returns the training and selection losses of the population.

const Matrix<double>& GeneticAlgorithm::get_loss() const
{
    return(loss);
}


/// Returns the fitness of the population.

const Vector<double>& GeneticAlgorithm::get_fitness() const
{
    return(fitness);
}


/// Returns the method for the initialization of the population.

const GeneticAlgorithm::InitializationMethod& GeneticAlgorithm::get_initialization_method() const
{
    return(initialization_method);
}


/// Returns the method for the crossover of the population.

const GeneticAlgorithm::CrossoverMethod& GeneticAlgorithm::get_crossover_method() const
{
    return(crossover_method);
}


/// Returns the method for the fitness assignment of the population.

const GeneticAlgorithm::FitnessAssignment& GeneticAlgorithm::get_fitness_assignment_method() const
{
    return(fitness_assignment_method);
}


/// Returns the size of the population.

const size_t& GeneticAlgorithm::get_population_size() const
{
    return(population_size);
}


/// Returns the rate used in the mutation.

const double& GeneticAlgorithm::get_mutation_rate() const
{
    return(mutation_rate);
}


/// Returns the size of the elite in the selection.

const size_t& GeneticAlgorithm::get_elitism_size() const
{
    return(elitism_size);
}


/// Returns the first point used for the crossover.

const size_t& GeneticAlgorithm::get_crossover_first_point() const
{
    return(crossover_first_point);
}


/// Returns the second point used for the crossover.

const size_t& GeneticAlgorithm::get_crossover_second_point() const
{
    return(crossover_second_point);
}


/// Returns the selective pressure used for the fitness assignment.

const double& GeneticAlgorithm::get_selective_pressure() const
{
    return(selective_pressure);
}


/// Returns the incest prevention distance used for the crossover.

const double& GeneticAlgorithm::get_incest_prevention_distance() const
{
    return(incest_prevention_distance);
}


/// Returns true if the generation mean of the selection losses are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_mean() const
{
    return(reserve_generation_mean);
}


/// Returns true if the generation standard deviation of the selection losses are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_standard_deviation() const
{
    return(reserve_generation_standard_deviation);
}


/// Returns true if the generation minimum selection error are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_minimum_selection() const
{
    return(reserve_generation_minimum_selection);
}


/// Returns true if the generation optimum loss error are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_optimum_loss() const
{
    return(reserve_generation_optimum_loss);
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
    size_t inputs_number;

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
        maximum_epochs_number = static_cast<size_t>(max(100.,inputs_number*5.));

        mutation_rate = 1.0/inputs_number;

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


/// Sets a new popualtion.
/// @param new_population New population matrix.

void GeneticAlgorithm::set_population(const Vector<Vector<bool>>& new_population)
{
#ifdef __OPENNN_DEBUG__

    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }
/*
    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    if(new_population[0].size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population(const Matrix<double>&) method.\n"
               << "Population columns("<<new_population[0].size()<< ") must be equal to inputs number("<<inputs_number<<").\n";

        throw logic_error(buffer.str());
    }
*/
    if(new_population.size() != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population(const Matrix<double>&) method.\n"
               << "Population rows("<<new_population.size()<< ") must be equal to population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    population.set(new_population);
}


/// Sets a new training losses of the population.
/// @param new_loss New training losses.

void GeneticAlgorithm::set_loss(const Matrix<double>& new_loss)
{
#ifdef __OPENNN_DEBUG__

    if(new_loss.get_columns_number() != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_loss(const Matrix<double>&&) method.\n"
               << "Performance columns must be equal to 2.\n";

        throw logic_error(buffer.str());
    }

    if(new_loss.get_rows_number() != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_loss(const Matrix<double>&&) method.\n"
               << "Performance rows("<<new_loss.get_rows_number()<< ") must be equal to population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    loss.set(new_loss);
}


/// Sets a new fitness for the population.
/// @param new_fitness New fitness values.

void GeneticAlgorithm::set_fitness(const Vector<double>& new_fitness)
{
#ifdef __OPENNN_DEBUG__

    if(new_fitness.size() != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_fitness(const Matrix<double>&&) method.\n"
               << "Fitness size("<<new_fitness.size()<< ") must be equal to population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < new_fitness.size(); i++)
    {
        if(new_fitness[i] < 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
                   << "void set_fitness(const Matrix<double>&&) method.\n"
                   << "Fitness must be greater than 0.\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    fitness.set(new_fitness);
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
/// @param new_crossover_method Method to perform the crossover of the selected population(Uniform, OnePoint or TwoPoint).

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

void GeneticAlgorithm::set_population_size(const size_t& new_population_size)
{
#ifdef __OPENNN_DEBUG__

    if(new_population_size < 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population_size(const size_t&) method.\n"
               << "Population size must be greater than 4.\n";

        throw logic_error(buffer.str());
    }

#endif

    population_size = new_population_size;
}


/// Sets a new rate used in the mutation.
/// It is a number between 0 and 1.
/// @param new_mutation_rate Rate used for the mutation.

void GeneticAlgorithm::set_mutation_rate(const double& new_mutation_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_mutation_rate < 0 || new_mutation_rate > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_mutation_rate(const double&) method.\n"
               << "Mutation rate must be between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    mutation_rate = new_mutation_rate;
}


/// Sets the number of individuals with the greatest fitness selected.
/// @param new_elitism_size Size of the elitism.

void GeneticAlgorithm::set_elitism_size(const size_t& new_elitism_size)
{
#ifdef __OPENNN_DEBUG__

    if(new_elitism_size > population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_elitism_size(const size_t&) method.\n"
               << "Elitism size("<< new_elitism_size<<") must be lower than the population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    elitism_size = new_elitism_size;
}


/// Sets the point of for the OnePoint and TwoPoint crossover.
/// If it is set to 0, the algorithm will select it randomly for each crossover.
/// @param new_crossover_first_point Point for the OnePoint and first point for TwoPoint crossover.

void GeneticAlgorithm::set_crossover_first_point(const size_t& new_crossover_first_point)
{
#ifdef __OPENNN_DEBUG__

    if(new_crossover_first_point > population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_first_point(const size_t&) method.\n"
               << "First crossover point("<< new_crossover_first_point<<") must be lower than the population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    crossover_first_point = new_crossover_first_point;
}


/// Sets the point of for the TwoPoint crossover.
/// If it is set to 0, the algorithm will select it randomly for each crossover.
/// @param new_crossover_second_point Second point for the TwoPoint crossover.

void GeneticAlgorithm::set_crossover_second_point(const size_t& new_crossover_second_point)
{
#ifdef __OPENNN_DEBUG__

    if(new_crossover_second_point > population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_second_point(const size_t&) method.\n"
               << "Second crossover point("<< new_crossover_second_point<<") must be lower than the population size("<<population_size<<").\n";

        throw logic_error(buffer.str());
    }

    if(new_crossover_second_point <= crossover_first_point && new_crossover_second_point != 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_crossover_second_point(const size_t&) method.\n"
               << "Second crossover point("<< new_crossover_second_point<<") must be greater than the first point("<<crossover_first_point<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    crossover_second_point = new_crossover_second_point;
}


/// Sets a new value for the selective pressure parameter.
/// Linear ranking allows values for the selective pressure greater than 0.
/// @param new_selective_pressure Selective pressure value.

void GeneticAlgorithm::set_selective_pressure(const double& new_selective_pressure)
{
#ifdef __OPENNN_DEBUG__

    if(new_selective_pressure <= 0.0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_selective_pressure(const double&) method. "
               << "Selective pressure must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selective_pressure = new_selective_pressure;
}


/// Sets a new value for the incest prevention distance used in the crossover.
/// @param new_incest_prevention_distance Incest prevention distance value.

void GeneticAlgorithm::set_incest_prevention_distance(const double& new_incest_prevention_distance)
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

    population.set(population_size);

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
    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm stuff

    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    Vector<bool> inputs(inputs_number,true);

    size_t zero_ocurrences;

    size_t random;

    size_t random_loops = 0;

    for(size_t i = 0; i < population_size; i++)
    {
        zero_ocurrences = 0;

        for(size_t j = 0; j < inputs_number; j++)
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
            inputs[static_cast<size_t>(rand())%inputs_number] = true;
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


/// Initialize the population with the weighted intialization method.

void GeneticAlgorithm::initialize_weighted_population()
{    

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    Matrix<double> correlations = data_set_pointer->calculate_input_target_columns_correlations_double();

    Vector<double> final_correlations = absolute_value(correlations.calculate_rows_sum());

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    // Optimization algortihm stuff

    Vector<bool> inputs(inputs_number, false);

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

    correlations_sum = cumulative(final_correlations);

    for(size_t i = 0; i < population_size; i++)
    {
        zero_ocurrences = 0;

        inputs.set(inputs_number,false);

        for(size_t j = 0; j < inputs_number; j++)
        {
            random = calculate_random_uniform(0.,sum);

            for(size_t k = 0; k < correlations_sum.size(); k++)
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
            inputs[static_cast<size_t>(rand())%inputs_number] = true;
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

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set stuff

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();
    
    Vector<DataSet::VariableUse> current_uses(original_uses);

    // Neural network stuff

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm stuff

    Vector<bool> current_inputs;

    size_t index;

    Vector<double> errors(2);

    loss.set(population_size,2);

    for(size_t i = 0; i < population_size; i++)
    {
        current_inputs = population[i];

        for(size_t j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);

            if(current_inputs[j] == false)
            {
                current_uses[index] = DataSet::UnusedVariable;
            }
            else
            {
                current_uses[index] = DataSet::Input;
            }
        }

        data_set_pointer->set_columns_uses(current_uses);

//        data_set_pointer->set_variables_uses(current_uses);

        neural_network_pointer->set_inputs_number(current_inputs.count_equal_to(true));

        // Training Neural networks

        errors = calculate_losses(population[i]);

        loss.set_row(i, errors);
    }

    calculate_fitness();
}


/// Calculate the fitness with the errors depending on the fitness assignment method.

void GeneticAlgorithm::calculate_fitness()
{
#ifdef __OPENNN_DEBUG__

    if(loss.get_rows_number() == 0)
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
            calculate_objetive_fitness();
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

void GeneticAlgorithm::calculate_objetive_fitness()
{
    fitness.set(loss.get_rows_number(),0.);

    for(size_t i = 0; i < fitness.size(); i++)
    {
        fitness[i] = 1/(1 + loss(i,1));
    }
}


/// Calculate the fitness with rank based fitness assignment method.

void GeneticAlgorithm::calculate_rank_fitness()
{
    const Vector<size_t> rank = loss.get_column(1).calculate_greater_rank();

    fitness.set(loss.get_rows_number(),0.);

    for(size_t i = 0; i < population_size; i++)
    {
        fitness[i] = selective_pressure*rank[i];
    }
}


/// Evolve the population to a new generation.
/// Perform the selection, crossover and mutation of the current generation.

void GeneticAlgorithm::evolve_population()
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
            population[i][static_cast<size_t>(rand())%population[i].size()] = true;
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

    if(fitness.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_selection() method.\n"
               << "No fitness found.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t selected_population_size = static_cast<size_t>(population_size/2);

    Vector<Vector<bool>> population_copy;

    Vector<bool> selected_population(population.size(), false);

    size_t selected_index = 0;

    Vector<double> fitness_sum = cumulative(fitness);

    double sum = fitness.calculate_sum();

    double random;

    size_t random_loops = 0;

    Vector<double> fitness_copy(fitness);

    population_copy.set();

    // Elitist selection

    if(elitism_size >= 1)
    {
        selected_index = get_optimal_individual_index();

        selected_population[selected_index] = true;

        population_copy.push_back(population[selected_index]);

        fitness_copy[selected_index] = -1;
    }

    // Natural selection

    while(population_copy.size() < elitism_size && population_copy.size() < selected_population_size)
    {
        selected_index = maximal_index(fitness_copy);

        if(!population_copy.contains(population[selected_index]))
        {
            selected_population[selected_index] = true;

            population_copy.push_back(population[selected_index]);
        }

        fitness_copy[selected_index] = -1;
    }

    // Roulette wheel

    while(population_copy.size() != selected_population_size)
    {
        random = calculate_random_uniform(0.,sum);

        for(size_t k = 0; k < fitness_sum.size(); k++)
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

    Vector<Vector<bool>> new_population;

    while(new_population.size() < population_size)
    {
        parent1_index = static_cast<size_t>(rand())%selected_population;

        parent2_index = static_cast<size_t>(rand())%selected_population;

        random_loops = 0;

        while(euclidean_distance(population[parent1_index].to_double_vector(), population[parent2_index].to_double_vector())
              <= incest_prevention_distance)
        {
            parent2_index = static_cast<size_t>(rand())%selected_population;

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
            first_point = 1 + static_cast<size_t>(rand())%(inputs_number-1);
        }

        for(size_t i = 0; i < inputs_number; i++)
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


/// Perform the TwoPoint crossover method.

void GeneticAlgorithm::perform_2point_crossover()
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

    Vector<Vector<bool>> new_population;

    cout << population_size << endl;

    while(new_population.size() < population_size)
    {
        parent1_index = static_cast<size_t>(rand())%selected_population;
        parent2_index = static_cast<size_t>(rand())%selected_population;

        random_loops = 0;

        while(euclidean_distance(population[parent1_index].to_double_vector(), population[parent2_index].to_double_vector()) <= incest_prevention_distance)
        {
            parent2_index = static_cast<size_t>(rand())%selected_population;
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
            first_point = 1 + static_cast<size_t>(rand())%(inputs_number-2);
        }

        if(crossover_second_point == 0)
        {
            second_point = first_point + static_cast<size_t>(rand())%(inputs_number-1-first_point);
        }

        for(size_t i = 0; i < inputs_number; i++)
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
        if(new_population.size() < population_size)
        {
            new_population.push_back(offspring2);
        }
    }

    cout << new_population.size() << endl;
    set_population(new_population);
}


/// Perform the uniform crossover method.

void GeneticAlgorithm::perform_uniform_crossover()
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

    Vector<Vector<bool>> new_population;

    while(new_population.size() < population_size)
    {
        parent1_index = static_cast<size_t>(rand())%selected_population;
        parent2_index = static_cast<size_t>(rand())%selected_population;

        random_loops = 0;

        while(euclidean_distance(population[parent1_index].to_double_vector(), population[parent2_index].to_double_vector())
              <= incest_prevention_distance)
        {
            parent2_index = static_cast<size_t>(rand())%selected_population;
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

        for(size_t i = 0; i < inputs_number; i++)
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


/// Perform the mutation of the individuals generated in the crossover.

void GeneticAlgorithm::perform_mutation()
{
#ifdef __OPENNN_DEBUG__

    if(population.size() != population_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_mutation() method.\n"
               << "Population size must be equal to "<< population_size <<".\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t selected_population_size = population_size - elitism_size;

    double random;

    for(size_t i = selected_population_size; i < population.size(); i++)
    {
        for(size_t j = 0; j < population[i].size(); j++)
        {
            random = calculate_random_uniform(0.,1.);

            if(random <= mutation_rate)
            {
                population[i][j] = !population[i][j];
            }
       }
    }
}


/// Return the index of the optimal individual of the population considering the tolerance.

size_t GeneticAlgorithm::get_optimal_individual_index() const
{
    size_t index = 0;

    Vector<bool> optimal_inputs = population[0];

    double optimum_error = loss(0,1);

    Vector<bool> current_inputs;

    double current_error;

    for(size_t i = 1; i < population_size; i++)
    {
        current_inputs = population[i];
        current_error = loss(i,1);

        if((abs(optimum_error-current_error) < tolerance &&
             current_inputs.count_equal_to(true) > optimal_inputs.count_equal_to(true)) ||
           (abs(optimum_error-current_error) >= tolerance &&
             current_error < optimum_error)   )
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

    if(display)
    {
        cout << "Performing genetic inputs selection..." << endl;
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set stuff

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

//    const size_t targets_number = data_set_pointer->get_target_variables_number();

    // Neural network stuff

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm stuff

    size_t minimal_index;
    double current_training_error;
    double current_selection_error;

    Vector<bool> current_inputs;
    Vector<DataSet::VariableUse> current_uses;
    double current_mean;
    double current_standard_deviation;

//    double previous_minimum_selection_error = 1e10;

    double optimum_selection_error = 1e10;
    double optimum_training_error = 0.0;

    Vector<bool> optimal_inputs;
    Vector<double> optimal_parameters;

    size_t optimal_generation = 0;

    bool end_algortihm = false;

    size_t iterations = 0;

    size_t index = 0;

    time_t beginning_time, current_time;
    double elapsed_time = 0.0;

    original_uses = data_set_pointer->get_columns_uses();

    current_uses = original_uses;

    optimal_inputs.set(original_uses.count_equal_to(DataSet::Input),0);

    Matrix<double>  test(100,4);

    time(&beginning_time);

    initialize_population();

    for(size_t epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        if(epoch != 0)
        {
            evolve_population();
        }

        evaluate_population();

        minimal_index = get_optimal_individual_index();

        current_mean = mean(loss.get_column(1));

        current_standard_deviation = standard_deviation(loss.get_column(1));

        current_inputs = population[minimal_index];

        current_selection_error = loss(minimal_index,1);

        current_training_error = loss(minimal_index,0);

        if((abs(optimum_selection_error - current_selection_error) >= tolerance &&
             optimum_selection_error > current_selection_error) ||
               (abs(optimum_selection_error - current_selection_error) < tolerance &&
                 optimal_inputs.count_equal_to(true) < current_inputs.count_equal_to(true)))
        {
            optimal_inputs = current_inputs;
            optimum_training_error = current_training_error;
            optimum_selection_error = current_selection_error;
            optimal_generation = epoch;
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

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
            results->generation_minimum_selection_history.push_back(current_selection_error);
        }

        if(reserve_generation_optimum_loss)
        {
            results->generation_optimum_loss_history.push_back(current_training_error);
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
        else if(epoch >= maximum_epochs_number)
        {
            end_algortihm = true;

            if(display)
            {
                cout << "Maximum number of iterations reached." << endl;
            }

            results->stopping_condition = InputsSelection::MaximumIterations;
        }

        for(size_t j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);

            if(current_inputs[j] == false)
            {
                current_uses[index] = DataSet::UnusedVariable;
            }
            else
            {
                current_uses[index] = DataSet::Input;
            }
        }

        // Print results

        data_set_pointer->set_columns_uses(current_uses);

        if(display)
        {
            cout << "Generation: " << epoch << endl;
            cout << "Generation optimal inputs: " << data_set_pointer->get_input_variables_names().vector_to_string() << " " << endl;
            cout << "Generation optimal number of inputs: " << current_inputs.count_equal_to(true) << endl;
            cout << "Generation optimum selection error: " << current_selection_error << endl;
            cout << "Corresponding training loss: " << current_training_error << endl;
            cout << "Generation selection mean = " << mean(loss.get_column(1)) << endl;
            cout << "Generation selection standard deviation = " << standard_deviation(loss.get_column(1)) << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            cout << endl;
        }

        if(end_algortihm == true) break;
    }

    // Save results

    results->inputs_data.set(inputs_history);

    if(reserve_error_data)
    {
        results->loss_data.set(training_error_history);
    }

    if(reserve_selection_error_data)
    {
        results->selection_error_data.set(selection_error_history);
    }

    optimal_parameters = get_parameters_inputs(optimal_inputs);

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }

    results->optimal_inputs = optimal_inputs;
    results->final_selection_error = optimum_selection_error;
    results->final_training_error = optimum_training_error;
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    size_t original_index;

    Vector<DataSet::VariableUse> optimal_uses = original_uses;

    for(size_t i = 0; i < optimal_inputs.size(); i++)
    {
        original_index = get_input_index(original_uses, i);
        if(optimal_inputs[i] == 1)
        {
            optimal_uses[original_index] = DataSet::Input;
        }
        else
        {
            optimal_uses[original_index] = DataSet::UnusedVariable;
        }
    }

    // Set Data set results

    data_set_pointer->set_columns_uses(optimal_uses);

    // Set Neural network results

    neural_network_pointer->set_inputs_number(optimal_inputs);

    neural_network_pointer->set_parameters(optimal_parameters);

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);


    if(display)
    {
        cout << "Optimal inputs: " << data_set_pointer->get_input_variables_names().vector_to_string() << endl;
        cout << "Optimal generation: " << optimal_generation << endl;
        cout << "Optimal number of inputs: " << optimal_inputs.count_equal_to(true) << endl;
        cout << "Optimum training error: " << optimum_training_error << endl;
        cout << "Optimum selection error: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    return results;


/*
    while(!end)
    {
        if(iterations != 0)
        {
            evolve_population();
        }

        evaluate_population();

        minimal_index = get_optimal_individual_index();

        current_mean = mean(loss.get_column(1));

        current_standard_deviation = standard_deviation(loss.get_column(1));

        current_inputs = population[minimal_index];

        current_selection_error = loss(minimal_index,1);

        current_training_error = loss(minimal_index,0);

        if((abs(optimum_selection_error - current_selection_error) >= tolerance &&
             optimum_selection_error > current_selection_error) ||
               (abs(optimum_selection_error - current_selection_error) < tolerance &&
                 optimal_inputs.count_equal_to(true) < current_inputs.count_equal_to(true)))
        {
            optimal_inputs = current_inputs;
            optimum_training_error = current_training_error;
            optimum_selection_error = current_selection_error;
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

//        previous_minimum_selection_error = current_minimum_selection_error;

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
            results->generation_minimum_selection_history.push_back(current_selection_error);
        }

        if(reserve_generation_optimum_loss)
        {
            results->generation_optimum_loss_history.push_back(current_training_error);
        }

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                cout << "Maximum time reached." << endl;
            }

            results->stopping_condition = InputsSelection::MaximumTime;
        }
        else if(current_selection_error <= selection_error_goal)
        {
            end = true;

            if(display)
            {
                cout << "selection error reached." << endl;
            }

            results->stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(iterations >= maximum_epochs_number)
        {
            end = true;

            if(display)
            {
                cout << "Maximum number of iterations reached." << endl;
            }

            results->stopping_condition = InputsSelection::MaximumIterations;
        }

        for(size_t j = 0; j < current_inputs.size(); j++)
        {
            index = get_input_index(original_uses,j);

            if(current_inputs[j] == false)
            {
                current_uses[index] = DataSet::UnusedVariable;
            }
            else
            {
                current_uses[index] = DataSet::Input;
            }
        }

//        data_set_pointer->set_variables_uses(current_uses);

//        neural_network_pointer->set_inputs(current_inputs, targets_number);

        if(display)
        {
            cout << "Generation: " << iterations << endl;
            cout << "Generation optimal inputs: " << data_set_pointer->get_input_variables_names().vector_to_string() << " " << endl;
            cout << "Generation optimal number of inputs: " << current_inputs.count_equal_to(true) << endl;
            cout << "Generation optimum selection error: " << current_selection_error << endl;
            cout << "Corresponding training loss: " << current_training_error << endl;
            cout << "Generation selection mean = " << mean(loss.get_column(1)) << endl;
            cout << "Generation selection standard deviation = " << standard_deviation(loss.get_column(1)) << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            cout << endl;
        }
    }

    results->inputs_data.set(inputs_history);

    if(reserve_error_data)
    {
        results->loss_data.set(training_error_history);
    }

    if(reserve_selection_error_data)
    {
        results->selection_error_data.set(selection_error_history);
    }

    optimal_parameters = get_parameters_inputs(optimal_inputs);

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }

//    results->optimal_inputs_indices = optimal_inputs;
    results->final_selection_error = optimum_selection_error;
    results->final_training_error = calculate_losses(optimal_inputs)[0];
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    size_t original_index;

    Vector<DataSet::VariableUse> optimal_uses = original_uses;

    for(size_t i = 0; i < optimal_inputs.size(); i++)
    {
        original_index = get_input_index(original_uses, i);
        if(optimal_inputs[i] == 1)
        {
            optimal_uses[original_index] = DataSet::Input;            
        }
        else
        {
            optimal_uses[original_index] = DataSet::UnusedVariable;
        }
    }

    data_set_pointer->set_columns_uses(optimal_uses);

//    neural_network_pointer->set_inputs(optimal_inputs, targets_number);
    neural_network_pointer->set_parameters(optimal_parameters);

    neural_network_pointer->set_inputs_names(data_set_pointer->get_input_variables_names());

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        cout << "Optimal inputs: " << neural_network_pointer->get_inputs_names().vector_to_string() << endl;

        cout << "Optimal number of inputs: " << optimal_inputs.count_equal_to(true) << endl;
        cout << "Optimum training error: " << optimum_training_error << endl;
        cout << "Optimum selection error: " << optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    return results;*/
}


/// Writes as matrix of strings the most representative atributes.

Matrix<string> GeneticAlgorithm::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

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

   const string initialization_method = write_initialization_method();

   values.push_back(initialization_method);

   // Fitness assignment method

   labels.push_back("Fitness assignment method");

   const string fitnes_assignment_method = write_fitness_assignment_method();

   values.push_back(fitnes_assignment_method);

   // Crossover method

   labels.push_back("Crossover method");

   const string crossover_method = write_crossover_method();

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
   buffer << selection_error_goal;

   values.push_back(buffer.str());

   // Maximum Generations number

   labels.push_back("Maximum Generations number");

   buffer.str("");
   buffer << maximum_epochs_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Plot training error history

   labels.push_back("Plot training error history");

   buffer.str("");

   if(reserve_error_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot selection error history

   labels.push_back("Plot selection error history");

   buffer.str("");

   if(reserve_selection_error_data)
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

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return string_matrix;
}


/// Prints to the screen the genetic algorithm parameters, the stopping criteria
/// and other user stuff concerning the genetic algorithm object.

tinyxml2::XMLDocument* GeneticAlgorithm::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Order Selection algorithm

    tinyxml2::XMLElement* root_element = document->NewElement("GeneticAlgorithm");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = nullptr;
    tinyxml2::XMLText* text = nullptr;

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

    // selection error goal
    {
        element = document->NewElement("SelectionErrorGoal");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << selection_error_goal;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Maximum iterations
    {
        element = document->NewElement("MaximumGenerationsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_epochs_number;

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
        element = document->NewElement("ReserveGenerationOptimumLoss");
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

    // Incest prevention distance
//    {
//        element = document->NewElement("IncestPreventionDistance");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << incest_prevention_distance;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }


    // Reserve loss data
//    {
//        element = document->NewElement("ReserveTrainingErrorHistory");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << reserve_error_data;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    // Reserve selection error data
//    {
//        element = document->NewElement("ReserveSelectionErrorHistory");
//        root_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << reserve_selection_error_data;

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

    return document;
}


/// Serializes the genetic algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GeneticAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

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


    //file_stream.CloseElement();
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
            const size_t new_trials_number = static_cast<size_t>(atoi(element->GetText()));

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
            const size_t new_population_size = static_cast<size_t>(atoi(element->GetText()));

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
            const double new_incest_prevention_rate = atof(element->GetText());

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
            const double new_mutation_rate = atof(element->GetText());

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
            const size_t new_elitism_size = static_cast<size_t>(atoi(element->GetText()));

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
            const size_t new_crossover_first_point = static_cast<size_t>(atoi(element->GetText()));

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
            const size_t new_crossover_second_point = static_cast<size_t>(atoi(element->GetText()));

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
            const double new_selective_pressure = atof(element->GetText());

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
            const string new_reserve_error_data = element->GetText();

            try
            {
                set_reserve_error_data(new_reserve_error_data != "0");
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
            const double new_selection_error_goal = atof(element->GetText());

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
            const size_t new_maximum_iterations_number = static_cast<size_t>(atoi(element->GetText()));

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
            const double new_maximum_correlation = atof(element->GetText());

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
            const double new_minimum_correlation = atof(element->GetText());

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
            const double new_maximum_time = atoi(element->GetText());

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
            const double new_tolerance = atof(element->GetText());

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
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
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


/// Returns a string representation of the current genetic algorithm results structure.

string GeneticAlgorithm::GeneticAlgorithmResults::object_to_string() const
{
    ostringstream buffer;

    // Loss history

    if(!loss_data.empty())
    {
        buffer << "% Loss history:\n"
               << loss_data.to_row_matrix() << "\n";
    }

    // selection error history

    if(!selection_error_data.empty())
    {
        buffer << "% Selection loss history:\n"
               << selection_error_data.to_row_matrix() << "\n";
    }

    // Generation optimum loss history

    if(!generation_optimum_loss_history.empty())
    {
        buffer << "% Generation optimum loss history:\n"
               << generation_optimum_loss_history.vector_to_string() << "\n";
    }

    // Generation minimum selection history

    if(!generation_minimum_selection_history.empty())
    {
        buffer << "% Generation minimum selection history:\n"
               << generation_minimum_selection_history.vector_to_string() << "\n";
    }

    // Generation mean history

    if(!generation_mean_history.empty())
    {
        buffer << "% Generation mean history:\n"
               << generation_mean_history.vector_to_string() << "\n";
    }

    // Generation standard deviation history

    if(!generation_standard_deviation_history.empty())
    {
        buffer << "% Generation standard deviation history:\n"
               << generation_standard_deviation_history.vector_to_string() << "\n";
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

    // Optimum selection error

    if(abs(final_selection_error - 0) > numeric_limits<double>::epsilon())
    {
        buffer << "% Optimum selection error:\n"
               << final_selection_error << "\n";
    }

    // Final training loss

    if(abs(final_training_error - 0) > numeric_limits<double>::epsilon())
    {
        buffer << "% Final training loss:\n"
               << final_training_error << "\n";
    }

    // Optimal input

    if(!optimal_inputs_indices.empty())
    {
        buffer << "% Optimal input:\n"
               << optimal_inputs_indices << "\n";
    }

    // Iterations number


    buffer << "% Number of iterations:\n"
           << iterations_number << "\n";

    // Elapsed time

    buffer << "% Elapsed time:\n"
           << write_elapsed_time(elapsed_time) << "\n";

    return buffer.str();
}
}



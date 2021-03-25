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


/// Returns the fitness of the population.

const Tensor<type, 1>& GeneticAlgorithm::get_fitness() const
{
    return fitness;
}


const Tensor<bool, 1>& GeneticAlgorithm::get_selection() const
{
    return selection;
}


/// Returns the size of the population.

Index GeneticAlgorithm::get_individuals_number() const
{
    return population.dimension(0);
}


Index GeneticAlgorithm::get_genes_number() const
{
    return population.dimension(1);
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


/// Returns the selective pressure used for the fitness assignment.

const type& GeneticAlgorithm::get_selective_pressure() const
{
    return selective_pressure;
}


/// Returns true if the generation mean of the selection losses are to be reserved, and false otherwise.

const bool& GeneticAlgorithm::get_reserve_generation_mean() const
{
    return reserve_generation_mean;
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


/// Sets the members of the genetic algorithm object to their default values.

void GeneticAlgorithm::set_default()
{
    const Index genes_number = get_genes_number();
    Index individuals_number;

    if(training_strategy_pointer == nullptr
    || !training_strategy_pointer->has_neural_network())
    {
        maximum_epochs_number = 100;

        mutation_rate = 0.5;

        individuals_number = 10;
    }
    else
    {
        maximum_epochs_number = static_cast<Index>(max(100.0,genes_number*5.0));

        mutation_rate = static_cast<type>(1.0/genes_number);

        individuals_number = 10 * genes_number;
    }

    // Population stuff

    population.resize(individuals_number, genes_number);

    training_errors.resize(individuals_number);
    selection_errorss.resize(individuals_number);

    fitness.resize(individuals_number);

    selection.resize(individuals_number);

    // Training operators

    elitism_size = 2;

    selective_pressure = 1.5;

    // Inputs selection results

    reserve_generation_mean = true;

    reserve_generation_minimum_selection = true;

    reserve_generation_optimum_loss = true;
}


/// Sets a new popualtion.
/// @param new_population New population matrix.

void GeneticAlgorithm::set_population(const Tensor<bool, 2>& new_population)
{
#ifdef OPENNN_DEBUG

    const Index individuals_number = get_individuals_number();
    const Index new_individuals_number = new_population.dimension(2);

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

    if(new_individuals_number  != individuals_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_population(const Tensor<type, 2>&) method.\n"
               << "Population rows("<<new_individuals_number
               << ") must be equal to population size("<<individuals_number<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    population = new_population;
}


void GeneticAlgorithm::set_training_errors(const Tensor<type, 1>& new_training_errors)
{
    training_errors = new_training_errors;
}


void GeneticAlgorithm::set_selection_errors(const Tensor<type, 1>& new_selection_errorss)
{
    selection_errorss = new_selection_errorss;
}


/// Sets a new fitness for the population.
/// @param new_fitness New fitness values.

void GeneticAlgorithm::set_fitness(const Tensor<type, 1>& new_fitness)
{
#ifdef OPENNN_DEBUG

    const Index individuals_number = get_individuals_number();

    if(new_fitness.size() != individuals_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_fitness(const Tensor<type, 1>&) method.\n"
               << "Fitness size ("<<new_fitness.size()
               << ") must be equal to population size ("<< individuals_number <<").\n";

        throw logic_error(buffer.str());
    }

    for(Index i = 0; i < individuals_number; i++)
    {
        if(new_fitness[i] < 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
                   << "void set_fitness(const Tensor<type, 2>&) method.\n"
                   << "Fitness must be greater than 0.\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    fitness = new_fitness;
}


/// Sets a new population size. It must be greater than 4.
/// @param new_population_size Size of the population.

void GeneticAlgorithm::set_individuals_number(const Index& new_population_size)
{
#ifdef OPENNN_DEBUG

    if(new_population_size < 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_individuals_number(const Index&) method.\n"
               << "Population size must be greater than 4.\n";

        throw logic_error(buffer.str());
    }

#endif

// @todo Set population and other matrices

}


/// Sets a new rate used in the mutation.
/// It is a number between 0 and 1.
/// @param new_mutation_rate Rate used for the mutation.

void GeneticAlgorithm::set_mutation_rate(const type& new_mutation_rate)
{
#ifdef OPENNN_DEBUG

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
#ifdef OPENNN_DEBUG

    const Index individuals_number = get_individuals_number();

    if(new_elitism_size > individuals_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void set_elitism_size(const Index&) method.\n"
               << "Elitism size("<< new_elitism_size
               <<") must be lower than the population size("<<individuals_number<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    elitism_size = new_elitism_size;
}


/// Sets a new value for the selective pressure parameter.
/// Linear ranking allows values for the selective pressure greater than 0.
/// @param new_selective_pressure Selective pressure value.

void GeneticAlgorithm::set_selective_pressure(const type& new_selective_pressure)
{
#ifdef OPENNN_DEBUG

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


/// Sets the reserve flag for the generation mean history.
/// @param new_reserve_generation_mean Flag value.

void GeneticAlgorithm::set_reserve_generation_mean(const bool& new_reserve_generation_mean)
{
    reserve_generation_mean = new_reserve_generation_mean;
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


/// Initialize the population depending on the intialization method.

void GeneticAlgorithm::initialize_population()
{
    const Index individuals_number = get_individuals_number();

#ifdef OPENNN_DEBUG

    if(individuals_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void initialize_population() method.\n"
               << "Population size must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index genes_number = get_genes_number();

    Tensor<bool, 1> individual(genes_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        for(Index j = 0; j < genes_number; j++)
        {
            rand()%2 == 0 ? individual[j] = false : individual[j] = true;
        }

        // Prevent no inputs

        if(is_false(individual))
        {
            individual(static_cast<Index>(rand())%genes_number) = true;
        }

        for(Index j = 0; j < genes_number; j++)
        {
            population(i,j) = individual(j);
        }
    }
}


/// Evaluate a population.
/// Training all the neural networks in the population and calculate their fitness.

void GeneticAlgorithm::evaluate_population()
{
#ifdef OPENNN_DEBUG

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

    // Training strategy

    TrainingResults training_results;

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm

    Tensor<bool, 1> individual;

    // Model selection

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    for(Index i = 0; i < individuals_number; i++)
    {
        individual = population.chip(i,0);

        const Tensor<Index, 0> input_columns_number = individual.cast<Index>().sum();

        Tensor<Index, 1> input_columns(input_columns_number(0));

        Index index = 0;

        for(i = 0; i < genes_number; i++)
        {
            if(individual(i))
            {
                input_columns(index) = original_input_columns_indices(i);
                index++;
            }
        }

        data_set_pointer->set_input_target_columns(input_columns, original_target_columns_indices);

        neural_network_pointer->set_inputs_number(data_set_pointer->get_input_variables_number());

        training_results = training_strategy_pointer->perform_training();

        // Set stuff

        parameters(i) = training_results.parameters;

        training_errors(i) = training_results.training_error;
        selection_errorss(i) = training_results.selection_error;
    }
}


/// Calculate the fitness with the errors depending on the fitness assignment method.

void GeneticAlgorithm::perform_fitness_assignment()
{
    const Index individuals_number = get_individuals_number();

    Tensor<Index, 1> selection_errors_rank(individuals_number);

    sort(selection_errors_rank.data(),
         selection_errors_rank.data() + selection_errors_rank.size(),
         [&](Index i, Index j){return selection_errorss[i]<selection_errorss[j];});

    for(Index i = 0; i < individuals_number; i++)
    {
        fitness(i) = selective_pressure*selection_errors_rank(i);
    }
}


/// Selects for crossover some individuals from the population.

void GeneticAlgorithm::perform_selection()
{
#ifdef OPENNN_DEBUG

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

    const Index individuals_number = get_individuals_number();

    const Index selected_individuals_number = static_cast<Index>(individuals_number/2);

    const Tensor<type, 1> cumulative_fitness = fitness.cumsum(0);

    Tensor<Index, 1> fitness_rank_descending(individuals_number);

    sort(fitness_rank_descending.data(),
         fitness_rank_descending.data() + fitness_rank_descending.size(),
         [&](Index i, Index j){return cumulative_fitness[i]<cumulative_fitness[j];});

    const Tensor<type, 0> total_fitness = fitness.sum();

    Index selection_count = 0;

    // Elitism selection

    selection.setConstant(false);


    for(Index i = 0; i < elitism_size ; i++)
    {
        selection[fitness_rank_descending[i]] = true;

        selection_count++;
    }

    // Roulette wheel

    while(selection_count != selected_individuals_number)
    {
        const type pointer = static_cast<type>(rand()/(RAND_MAX+1.0))*total_fitness(0);

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
            for(Index i = 1; i < individuals_number; i++)
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
}


/// Perform the crossover depending on the crossover method.

void GeneticAlgorithm::perform_crossover()
{
#ifdef OPENNN_DEBUG

    if(population.size() <= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GeneticAlgorithm class.\n"
               << "void perform_crossover() method.\n"
               << "Selected population size must be greater than 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index individuals_number = get_individuals_number();
    const Index genes_number = get_genes_number();

    const Index selected_individuals_number = static_cast<Index>(individuals_number/2);

    Index parent_1_index = 0;
    Index parent_2_index = 0;

    Tensor<bool, 1> parent_1(genes_number);
    Tensor<bool, 1> parent_2(genes_number);

    Tensor<bool, 1> offspring_1(genes_number);
    Tensor<bool, 1> offspring_2(genes_number);

    Index offspring_count = 0;

    Tensor<bool, 2> new_population(individuals_number, genes_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        if(!selection(i)) continue;

        parent_1_index = i;

        do{
        parent_2_index = static_cast<Index>(rand())%individuals_number;
        }while(selection(parent_2_index) && parent_1_index != parent_2_index);

        parent_1 = population.chip(parent_1_index, 0);
        parent_2 = population.chip(parent_2_index, 0);

        for(Index j = 0; j < genes_number; j++)
        {
            if(rand()%2 == 0)
            {
                offspring_1(j) = parent_1[j];
                offspring_2(j) = parent_2[j];
            }
            else
            {
                offspring_1(j) = parent_2[j];
                offspring_2(j) = parent_1[j];
            }
        }

        if(is_false(offspring_1))
            offspring_1(static_cast<Index>(rand())%genes_number) = true;

        if(is_false(offspring_2))
            offspring_2(static_cast<Index>(rand())%genes_number) = true;

        for(Index j = 0; j < genes_number; j++)
        {
            new_population(offspring_count, j) = offspring_1(j);
            new_population(offspring_count+1, j) = offspring_2(j);
        }

        offspring_count += 2;
    }

    population = new_population;
}


/// Perform the mutation of the individuals generated in the crossover.

void GeneticAlgorithm::perform_mutation()
{
    const Index individuals_number = get_individuals_number();

    const Index genes_number = get_genes_number();

    Tensor<bool, 1> individual(genes_number);

    for(Index i = 0; i < individuals_number; i++)
    {
        for(Index j = 0; j < genes_number; j++)
        {
            if(static_cast<type>(rand()/(RAND_MAX+1.0)) <= mutation_rate)
                population(i,j) = !population(i,j);
        }

        // Prevent no inputs

        individual = population.chip(i, 0);

        if(is_false(individual))
        {
            individual(static_cast<Index>(rand())%genes_number) = true;
        }
    }
}


/// Select the inputs with best generalization properties using the genetic algorithm.

InputsSelectionResults GeneticAlgorithm::perform_inputs_selection()
{
#ifdef OPENNN_DEBUG

    check();

#endif

    if(display) cout << "Performing genetic inputs selection..." << endl;

    InputsSelectionResults results(maximum_epochs_number);

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    // Optimization algorithm

    Index optimal_individual_index;

    bool stop = false;

    time_t beginning_time, current_time;
    type elapsed_time = 0;

    time(&beginning_time);

    initialize_population();

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        cout << "Generation: " << epoch << endl;

        evaluate_population();

        optimal_individual_index = minimal_index(selection_errorss);

        results.training_errors(epoch) = training_errors(optimal_individual_index);
        results.selection_errors(epoch) = training_errors(optimal_individual_index);

        if(results.optimum_selection_error < selection_errorss(optimal_individual_index))
        {
            // Neural network

            results.optimal_inputs = population.chip(optimal_individual_index,0);

            results.optimal_parameters = parameters(optimal_individual_index);

            // Loss index

            results.optimum_training_error = training_errors(optimal_individual_index);

            results.optimum_selection_error = selection_errorss(optimal_individual_index);
        }

        time(&current_time);

        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            stop = true;

            if(display) cout << "Maximum time reached." << endl;

            results.stopping_condition = InputsSelection::MaximumTime;
        }
        else if(selection_errorss(optimal_individual_index) <= selection_error_goal)
        {
            stop = true;

            if(display) cout << "selection error reached." << endl;

            results.stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(epoch >= maximum_epochs_number-1)
        {
            stop = true;

            if(display) cout << "Maximum number of epochs reached." << endl;

            results.stopping_condition = InputsSelection::MaximumEpochs;
        }

        if(display)
        {
            cout << "Generation: " << epoch + 1 << endl;
            cout << "Generation optimal inputs: " << data_set_pointer->get_input_variables_names().cast<string>() << " " << endl;
            cout << "Generation optimal number of inputs: " << data_set_pointer->get_input_variables_names().size() << endl;
            cout << "Corresponding training error: " << training_errors(optimal_individual_index) << endl;
            cout << "Generation optimum selection error: " << selection_errorss(optimal_individual_index) << endl;
            cout << "Generation selection mean: " << training_errors.mean() << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
            cout << endl;
        }

        if(stop == true)
        {
            results.elapsed_time = write_elapsed_time(elapsed_time);
            break;
        }

        perform_fitness_assignment();

        perform_selection();

        perform_crossover();

        perform_mutation();
    }

    time(&current_time);

    elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

    if(display)
    {
        cout << "Optimal inputs: " << data_set_pointer->get_input_variables_names().cast<string>() << endl;
        //cout << "Optimal generation: " << optimal_generation << endl;
        //cout << "Optimal number of inputs: " << optimal_genes_number << endl;
        cout << "Optimum training error: " << results.optimum_training_error << endl;
        cout << "Optimum selection error: " << results.optimum_selection_error << endl;
        cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
    }

    // Set data set stuff

    data_set_pointer->set_input_columns_binary(results.optimal_inputs);

    // Set neural network stuff

    neural_network_pointer->set_inputs_number(data_set_pointer->get_input_variables_number());

    neural_network_pointer->set_inputs_names(data_set_pointer->get_input_variables_names());

    neural_network_pointer->set_parameters(results.optimal_parameters);

    return results;
}


/// Writes as matrix of strings the most representative atributes.
/// @todo to many rows in string matrix.

Tensor<string, 2> GeneticAlgorithm::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(18, 2);

    const Index individuals_number = get_individuals_number();

    ostringstream buffer;

    Tensor<string, 1> labels(18);
    Tensor<string, 1> values(18);

    // Population size

    labels(2) = "Population size";

    buffer.str("");
    buffer << individuals_number;
    values(2) = buffer.str();

    // Elitism size

    labels(6) = "Elitism size";

    buffer.str("");
    buffer << elitism_size;
    values(6) = buffer.str();

    // Selective pressure

    labels(9) = "Selective pressure";

    buffer.str("");
    buffer << selective_pressure;
    values(9) = buffer.str();

    // Mutation rate

    labels(10) = "Mutation rate";

    buffer.str("");
    buffer << mutation_rate;
    values(10) = buffer.str();

    // Selection loss goal

    labels(11) = "Selection loss goal";

    buffer.str("");
    buffer << selection_error_goal;
    values(11) = buffer.str();

    // Maximum Generations number

    labels(12) = "Maximum Generations number";

    buffer.str("");
    buffer << maximum_epochs_number;
    values(12) = buffer.str();

    // Maximum time

    labels(13) = "Maximum time";

    buffer.str("");
    buffer << maximum_time;
    values(13) = buffer.str();

    // Plot training error history

    labels(14) = "Plot training error history";

    buffer.str("");

    if(reserve_training_errors)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(14) = buffer.str();

    // Plot selection error history

    labels(15) = "Plot selection error histroy";

    buffer.str("");

    if(reserve_selection_errors)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(15) = buffer.str();

    // Plot generation mean history

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

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

    return string_matrix;
}


/// Serializes the genetic algorithm object into a XML document of the TinyXML library without keep the DOM
/// tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GeneticAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    const Index individuals_number = get_individuals_number();

    ostringstream buffer;

    file_stream.OpenElement("GeneticAlgorithm");

    // Population size

    file_stream.OpenElement("PopulationSize");

    buffer.str("");
    buffer << individuals_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Elitism size

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

    // Population size
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PopulationSize");

        if(element)
        {
            const Index new_population_size = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_individuals_number(new_population_size);
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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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

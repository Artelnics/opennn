/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   C L A S S                                                                  */
/*                                                                                                              */ 
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "model_selection.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.

ModelSelection::ModelSelection()
    : training_strategy_pointer(nullptr)
   , incremental_order_pointer(nullptr)
   , golden_section_order_pointer(nullptr)
   , simulated_annelaing_order_pointer(nullptr)
   , growing_inputs_pointer(nullptr)
   , pruning_inputs_pointer(nullptr)
   , genetic_algorithm_pointer(nullptr)
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
   , incremental_order_pointer(nullptr)
   , golden_section_order_pointer(nullptr)
   , simulated_annelaing_order_pointer(nullptr)
   , growing_inputs_pointer(nullptr)
   , pruning_inputs_pointer(nullptr)
   , genetic_algorithm_pointer(nullptr)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML model selection file.

ModelSelection::ModelSelection(const string& file_name)
    : training_strategy_pointer(nullptr)
   , incremental_order_pointer(nullptr)
   , golden_section_order_pointer(nullptr)
   , simulated_annelaing_order_pointer(nullptr)
   , growing_inputs_pointer(nullptr)
   , pruning_inputs_pointer(nullptr)
   , genetic_algorithm_pointer(nullptr)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor. 
/// @param model_selection_document Pointer to a TinyXML document containing the model selection data.

ModelSelection::ModelSelection(const tinyxml2::XMLDocument& model_selection_document)
    : training_strategy_pointer(nullptr)
   , incremental_order_pointer(nullptr)
   , golden_section_order_pointer(nullptr)
   , simulated_annelaing_order_pointer(nullptr)
   , growing_inputs_pointer(nullptr)
   , pruning_inputs_pointer(nullptr)
   , genetic_algorithm_pointer(nullptr)
{
    from_XML(model_selection_document);
}


// DESTRUCTOR

/// Destructor. 

ModelSelection::~ModelSelection()
{
    // Delete inputs selection algorithms

    delete growing_inputs_pointer;
    delete pruning_inputs_pointer;
    delete genetic_algorithm_pointer;

    // Delete order selection algorithms

    delete incremental_order_pointer;
    delete golden_section_order_pointer;
    delete simulated_annelaing_order_pointer;
}


// METHODS


/// Returns a pointer to the training strategy object.

TrainingStrategy* ModelSelection::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "TrainingStrategy* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}


/// Returns true if this model selection has a training strategy associated,
/// and false otherwise.

bool ModelSelection::has_training_strategy() const
{
    if(training_strategy_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns the type of algorithm for the order selection.

const ModelSelection::OrderSelectionMethod& ModelSelection::get_order_selection_method() const
{
    return(order_selection_method);
}


/// Returns the type of algorithm for the inputs selection.

const ModelSelection::InputsSelectionMethod& ModelSelection::get_inputs_selection_method() const
{
    return(inputs_selection_method);
}


/// Returns a pointer to the incremental order selection algorithm.

IncrementalOrder* ModelSelection::get_incremental_order_pointer() const
{
    return(incremental_order_pointer);
}


/// Returns a pointer to the golden section order selection algorithm.

GoldenSectionOrder* ModelSelection::get_golden_section_order_pointer() const
{
    return(golden_section_order_pointer);
}


/// Returns a pointer to the simulated annealing order selection algorithm.

SimulatedAnnealingOrder* ModelSelection::get_simulated_annealing_order_pointer() const
{
    return(simulated_annelaing_order_pointer);
}


/// Returns a pointer to the growing inputs selection algorithm.

GrowingInputs* ModelSelection::get_growing_inputs_pointer() const
{
    return(growing_inputs_pointer);
}


/// Returns a pointer to the pruning inputs selection algorithm.

PruningInputs* ModelSelection::get_pruning_inputs_pointer() const
{
    return(pruning_inputs_pointer);
}


/// Returns a pointer to the genetic inputs selection algorithm.

GeneticAlgorithm* ModelSelection::get_genetic_algorithm_pointer() const
{
    return(genetic_algorithm_pointer);
}


/// Sets the members of the model selection object to their default values.

void ModelSelection::set_default()
{
    set_order_selection_method(ModelSelection::INCREMENTAL_ORDER);
    set_inputs_selection_method(ModelSelection::GROWING_INPUTS);

    display = true;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ModelSelection::set_display(const bool& new_display)
{
    display = new_display;

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        // do nothing

        break;
    }
    case GROWING_INPUTS:
    {
        growing_inputs_pointer->set_display(new_display);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs_pointer->set_display(new_display);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm_pointer->set_display(new_display);

        break;
    }
    }

    switch(order_selection_method)
    {
    case NO_ORDER_SELECTION:
    {
        // do nothing

        break;
    }
    case INCREMENTAL_ORDER:
    {
        incremental_order_pointer->set_display(new_display);

        break;
    }
    case GOLDEN_SECTION:
    {
        golden_section_order_pointer->set_display(new_display);

        break;
    }
    case SIMULATED_ANNEALING:
    {
        simulated_annelaing_order_pointer->set_display(new_display);

        break;
    }
    }
}


/// Sets a new method for selecting the order which have more impact on the targets.
/// @param new_order_selection_method Method for selecting the order(NO_ORDER_SELECTION, INCREMENTAL_ORDER, GOLDEN_SECTION, SIMULATED_ANNEALING).

void ModelSelection::set_order_selection_method(const ModelSelection::OrderSelectionMethod& new_order_selection_method)
{
    destruct_order_selection();

    order_selection_method = new_order_selection_method;

    switch(new_order_selection_method)
    {
    case NO_ORDER_SELECTION:
    {
        // do nothing

        break;
    }
    case INCREMENTAL_ORDER:
    {
        incremental_order_pointer = new IncrementalOrder(training_strategy_pointer);

        break;
    }
    case GOLDEN_SECTION:
    {
        golden_section_order_pointer = new GoldenSectionOrder(training_strategy_pointer);

        break;
    }
    case SIMULATED_ANNEALING:
    {
        simulated_annelaing_order_pointer = new SimulatedAnnealingOrder(training_strategy_pointer);

        break;
    }
    }
}


/// Sets a new order selection algorithm from a string.
/// @param new_order_selection_method String with the order selection type.

void ModelSelection::set_order_selection_method(const string& new_order_selection_method)
{
    if(new_order_selection_method == "NO_ORDER_SELECTION")
    {
        set_order_selection_method(NO_ORDER_SELECTION);
    }
    else if(new_order_selection_method == "INCREMENTAL_ORDER")
    {
        set_order_selection_method(INCREMENTAL_ORDER);
    }
    else if(new_order_selection_method == "GOLDEN_SECTION")
    {
        set_order_selection_method(GOLDEN_SECTION);
    }
    else if(new_order_selection_method == "SIMULATED_ANNEALING")
    {
        set_order_selection_method(SIMULATED_ANNEALING);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_order_selection_method(const string&) method.\n"
               << "Unknown order selection type: " << new_order_selection_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new method for selecting the inputs which have more impact on the targets.
/// @param new_inputs_selection_method Method for selecting the inputs(NO_INPUTS_SELECTION, GROWING_INPUTS, PRUNING_INPUTS, GENETIC_ALGORITHM).

void ModelSelection::set_inputs_selection_method(const ModelSelection::InputsSelectionMethod& new_inputs_selection_method)
{
    destruct_inputs_selection();

    inputs_selection_method = new_inputs_selection_method;

    switch(new_inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        // do nothing

        break;
    }
    case GROWING_INPUTS:
    {        
        growing_inputs_pointer = new GrowingInputs(training_strategy_pointer);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs_pointer = new PruningInputs(training_strategy_pointer);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm_pointer = new GeneticAlgorithm(training_strategy_pointer);

        break;
    }
    }
}


/// Sets a new inputs selection algorithm from a string.
/// @param new_inputs_selection_method String with the inputs selection type.

void ModelSelection::set_inputs_selection_method(const string& new_inputs_selection_method)
{
    if(new_inputs_selection_method == "NO_INPUTS_SELECTION")
    {
        set_inputs_selection_method(NO_INPUTS_SELECTION);
    }
    else if(new_inputs_selection_method == "GROWING_INPUTS")
    {
        set_inputs_selection_method(GROWING_INPUTS);
    }
    else if(new_inputs_selection_method == "PRUNING_INPUTS")
    {
        set_inputs_selection_method(PRUNING_INPUTS);
    }
    else if(new_inputs_selection_method == "GENETIC_ALGORITHM")
    {
        set_inputs_selection_method(GENETIC_ALGORITHM);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_inputs_selection_method(const string&) method.\n"
               << "Unknown inputs selection type: " << new_inputs_selection_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new approximation method.
/// If it is set to true the problem will be taken as a approximation;
/// if it is set to false the problem will be taken as a classification.
/// @param new_approximation Approximation value.

void ModelSelection::set_approximation(const bool& new_approximation)
{
    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        // do nothing

        break;
    }
    case GROWING_INPUTS:
    {
        growing_inputs_pointer->set_approximation(new_approximation);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs_pointer->set_approximation(new_approximation);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm_pointer->set_approximation(new_approximation);

        break;
    }
    }
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void ModelSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    switch(order_selection_method)
    {
        case NO_ORDER_SELECTION:
        {
            // do nothing

            break;
        }
        case INCREMENTAL_ORDER:
        {
            incremental_order_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
        case GOLDEN_SECTION:
        {
            golden_section_order_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
        case SIMULATED_ANNEALING:
        {
            simulated_annelaing_order_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
    }

    switch(inputs_selection_method)
    {
        case NO_INPUTS_SELECTION:
        {
            // do nothing

            break;
        }
        case GROWING_INPUTS:
        {
            growing_inputs_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
        case PRUNING_INPUTS:
        {
            pruning_inputs_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
        case GENETIC_ALGORITHM:
        {
            genetic_algorithm_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
    }
}

#ifdef __OPENNN_MPI__

void ModelSelection::set_MPI(TrainingStrategy* new_training_strategy, const ModelSelection* model_selection)
{

    set_training_strategy_pointer(new_training_strategy);

    // Inputs selection

    set_inputs_selection_MPI(model_selection);

    // Order Selection

    set_order_selection_MPI(model_selection);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank != 0)
    {
        set_display(false);
    }
}

void ModelSelection::set_inputs_selection_MPI(const ModelSelection* model_selection)
{

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int original_inputs_selection_method;

    // Growing/Pruning inputs parameters

    int maximum_selection_failures;
    int max_min_inputs_number;
    double minimum_correlation;
    double maximum_correlation;

    // Genetic algorithm parameters

    int population_size;
    int initialization_method;
    int fitness_assignment_method;
    int crossover_method;
    int elitism_size;
    double selective_pressure;
    double mutation_rate;
    int reserve_generation_mean_history;
    int reserve_generation_standard_deviation_history;

    // General parameters

    int trials_number;
    double tolerance;
    double selection_error_goal;
    int maximum_iterations_number;
    int maximum_time;
    int reserve_loss_loss_history;
    int reserve_selection_error_loss_history;

    if(rank == 0)
    {
        // Variables to send initialization

        original_inputs_selection_method = (int)model_selection->get_inputs_selection_method();

        switch(original_inputs_selection_method)
        {
            case(int)ModelSelection::GROWING_INPUTS:
            {
                GrowingInputs* original_growing_inputs = model_selection->get_growing_inputs_pointer();

                trials_number = (int)original_growing_inputs->get_trials_number();
                tolerance = original_growing_inputs->get_tolerance();
                selection_error_goal = original_growing_inputs->get_selection_error_goal();
                maximum_selection_failures = (int)original_growing_inputs->get_maximum_selection_failures();
                max_min_inputs_number = (int)original_growing_inputs->get_maximum_inputs_number();
                minimum_correlation = original_growing_inputs->get_minimum_correlation();
                maximum_correlation = original_growing_inputs->get_maximum_correlation();
                maximum_iterations_number = (int)original_growing_inputs->get_maximum_iterations_number();
                maximum_time = (int)original_growing_inputs->get_maximum_time();
                reserve_loss_loss_history = original_growing_inputs->get_reserve_error_data();
                reserve_selection_error_loss_history = original_growing_inputs->get_reserve_selection_error_data();
            }
            break;

            case(int)ModelSelection::PRUNING_INPUTS:
            {
                PruningInputs* original_pruning_inputs = model_selection->get_pruning_inputs_pointer();

                trials_number = (int)original_pruning_inputs->get_trials_number();
                tolerance = original_pruning_inputs->get_tolerance();
                selection_error_goal = original_pruning_inputs->get_selection_error_goal();
                maximum_selection_failures = (int)original_pruning_inputs->get_maximum_selection_failures();
                max_min_inputs_number = (int)original_pruning_inputs->get_minimum_inputs_number();
                minimum_correlation = original_pruning_inputs->get_minimum_correlation();
                maximum_correlation = original_pruning_inputs->get_maximum_correlation();
                maximum_iterations_number = (int)original_pruning_inputs->get_maximum_iterations_number();
                maximum_time = (int)original_pruning_inputs->get_maximum_time();
                reserve_loss_loss_history = original_pruning_inputs->get_reserve_error_data();
                reserve_selection_error_loss_history = original_pruning_inputs->get_reserve_selection_error_data();
            }
            break;

            case(int)ModelSelection::GENETIC_ALGORITHM:
            {
                GeneticAlgorithm* original_genetic_algorithm = model_selection->get_genetic_algorithm_pointer();

                trials_number = (int)original_genetic_algorithm->get_trials_number();
                tolerance = original_genetic_algorithm->get_tolerance();
                population_size = (int)original_genetic_algorithm->get_population_size();
                initialization_method = original_genetic_algorithm->get_initialization_method();
                fitness_assignment_method = original_genetic_algorithm->get_fitness_assignment_method();
                crossover_method = original_genetic_algorithm->get_crossover_method();
                elitism_size = (int)original_genetic_algorithm->get_elitism_size();
                selective_pressure = original_genetic_algorithm->get_selective_pressure();
                mutation_rate = original_genetic_algorithm->get_mutation_rate();
                selection_error_goal = original_genetic_algorithm->get_selection_error_goal();
                maximum_iterations_number = (int)original_genetic_algorithm->get_maximum_iterations_number();
                maximum_time = (int)original_genetic_algorithm->get_maximum_time();
                reserve_loss_loss_history  = original_genetic_algorithm->get_reserve_error_data();
                reserve_selection_error_loss_history = original_genetic_algorithm->get_reserve_selection_error_data();
                reserve_generation_mean_history = original_genetic_algorithm->get_reserve_generation_mean();
                reserve_generation_standard_deviation_history = original_genetic_algorithm->get_reserve_generation_standard_deviation();
            }
            break;

            default:
                break;
        }
    }

    // Send variables

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank > 0)
    {
        MPI_Recv(&original_inputs_selection_method, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request req[9];

        switch(original_inputs_selection_method)
        {
            case(int)ModelSelection::GROWING_INPUTS:

                MPI_Irecv(&maximum_selection_failures, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&max_min_inputs_number, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Irecv(&minimum_correlation, 1, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&maximum_correlation, 1, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case(int)ModelSelection::PRUNING_INPUTS:

                MPI_Irecv(&maximum_selection_failures, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&max_min_inputs_number, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Irecv(&minimum_correlation, 1, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&maximum_correlation, 1, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case(int)ModelSelection::GENETIC_ALGORITHM:

                MPI_Irecv(&population_size, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&initialization_method, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Irecv(&fitness_assignment_method, 1, MPI_INT, rank-1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&crossover_method, 1, MPI_INT, rank-1, 4, MPI_COMM_WORLD, &req[3]);
                MPI_Irecv(&elitism_size, 1, MPI_INT, rank-1, 5, MPI_COMM_WORLD, &req[4]);
                MPI_Irecv(&selective_pressure, 1, MPI_DOUBLE, rank-1, 6, MPI_COMM_WORLD, &req[5]);
                MPI_Irecv(&mutation_rate, 1, MPI_DOUBLE, rank-1, 7, MPI_COMM_WORLD, &req[6]);
                MPI_Irecv(&reserve_generation_mean_history, 1, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &req[7]);
                MPI_Irecv(&reserve_generation_standard_deviation_history, 1, MPI_INT, rank-1, 9, MPI_COMM_WORLD, &req[8]);

                MPI_Waitall(9, req, MPI_STATUS_IGNORE);

                break;

            default:
                break;
        }

        MPI_Irecv(&trials_number, 1, MPI_INT, rank-1, 10, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&tolerance, 1, MPI_DOUBLE, rank-1, 11, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&selection_error_goal, 1, MPI_DOUBLE, rank-1, 12, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&maximum_iterations_number, 1, MPI_INT, rank-1, 13, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(&maximum_time, 1, MPI_INT, rank-1, 14, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(&reserve_loss_loss_history, 1, MPI_INT, rank-1, 15, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(&reserve_selection_error_loss_history, 1, MPI_INT, rank-1, 16, MPI_COMM_WORLD, &req[6]);

        MPI_Waitall(7, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Send(&original_inputs_selection_method, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

        MPI_Request req[9];

        switch(original_inputs_selection_method)
        {
            case(int)ModelSelection::GROWING_INPUTS:

                MPI_Isend(&maximum_selection_failures, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&max_min_inputs_number, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&minimum_correlation, 1, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&maximum_correlation, 1, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case(int)ModelSelection::PRUNING_INPUTS:

                MPI_Isend(&maximum_selection_failures, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&max_min_inputs_number, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&minimum_correlation, 1, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&maximum_correlation, 1, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case(int)ModelSelection::GENETIC_ALGORITHM:

                MPI_Isend(&population_size, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&initialization_method, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&fitness_assignment_method, 1, MPI_INT, rank+1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&crossover_method, 1, MPI_INT, rank+1, 4, MPI_COMM_WORLD, &req[3]);
                MPI_Isend(&elitism_size, 1, MPI_INT, rank+1, 5, MPI_COMM_WORLD, &req[4]);
                MPI_Isend(&selective_pressure, 1, MPI_DOUBLE, rank+1, 6, MPI_COMM_WORLD, &req[5]);
                MPI_Isend(&mutation_rate, 1, MPI_DOUBLE, rank+1, 7, MPI_COMM_WORLD, &req[6]);
                MPI_Isend(&reserve_generation_mean_history, 1, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &req[7]);
                MPI_Isend(&reserve_generation_standard_deviation_history, 1, MPI_INT, rank+1, 9, MPI_COMM_WORLD, &req[8]);

                MPI_Waitall(9, req, MPI_STATUS_IGNORE);

                break;

            default:
                break;
        }

        MPI_Isend(&trials_number, 1, MPI_INT, rank+1, 10, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&tolerance, 1, MPI_DOUBLE, rank+1, 11, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(&selection_error_goal, 1, MPI_DOUBLE, rank+1, 12, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&maximum_iterations_number, 1, MPI_INT, rank+1, 13, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(&maximum_time, 1, MPI_INT, rank+1, 14, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(&reserve_loss_loss_history, 1, MPI_INT, rank+1, 15, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(&reserve_selection_error_loss_history, 1, MPI_INT, rank+1, 16, MPI_COMM_WORLD, &req[6]);

        MPI_Waitall(7, req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Set variables

    set_inputs_selection_method((ModelSelection::InputsSelectionType)original_inputs_selection_method);

    switch(original_inputs_selection_method)
    {
        case(int)ModelSelection::GROWING_INPUTS:
        {
            growing_inputs_pointer->set_trials_number(trials_number);
            growing_inputs_pointer->set_tolerance(tolerance);
            growing_inputs_pointer->set_selection_error_goal(selection_error_goal);
            growing_inputs_pointer->set_maximum_selection_failures(maximum_selection_failures);
            growing_inputs_pointer->set_maximum_inputs_number(max_min_inputs_number);
            growing_inputs_pointer->set_minimum_correlation(minimum_correlation);
            growing_inputs_pointer->set_maximum_correlation(maximum_correlation);
            growing_inputs_pointer->set_maximum_iterations_number(maximum_iterations_number);
            growing_inputs_pointer->set_maximum_time(maximum_time);
            growing_inputs_pointer->set_reserve_error_data(reserve_loss_loss_history == 1);
            growing_inputs_pointer->set_reserve_selection_error_data(reserve_selection_error_loss_history == 1);
        }
        break;

        case(int)ModelSelection::PRUNING_INPUTS:
        {
            pruning_inputs_pointer->set_trials_number(trials_number);
            pruning_inputs_pointer->set_tolerance(tolerance);
            pruning_inputs_pointer->set_selection_error_goal(selection_error_goal);
            pruning_inputs_pointer->set_maximum_selection_failures(maximum_selection_failures);
            pruning_inputs_pointer->set_minimum_inputs_number(max_min_inputs_number);
            pruning_inputs_pointer->set_minimum_correlation(minimum_correlation);
            pruning_inputs_pointer->set_maximum_correlation(maximum_correlation);
            pruning_inputs_pointer->set_maximum_iterations_number(maximum_iterations_number);
            pruning_inputs_pointer->set_maximum_time(maximum_time);
            pruning_inputs_pointer->set_reserve_error_data(reserve_loss_loss_history == 1);
            pruning_inputs_pointer->set_reserve_selection_error_data(reserve_selection_error_loss_history == 1);
        }
        break;

        case(int)ModelSelection::GENETIC_ALGORITHM:
        {
            genetic_algorithm_pointer->set_trials_number(trials_number);
            genetic_algorithm_pointer->set_tolerance(tolerance);
            genetic_algorithm_pointer->set_population_size(population_size);
            genetic_algorithm_pointer->set_inicialization_method((GeneticAlgorithm::InitializationMethod)initialization_method);
            genetic_algorithm_pointer->set_fitness_assignment_method((GeneticAlgorithm::FitnessAssignment)fitness_assignment_method);
            genetic_algorithm_pointer->set_crossover_method((GeneticAlgorithm::CrossoverMethod)crossover_method);
            genetic_algorithm_pointer->set_elitism_size(elitism_size);
            genetic_algorithm_pointer->set_selective_pressure(selective_pressure);
            genetic_algorithm_pointer->set_mutation_rate(mutation_rate);
            genetic_algorithm_pointer->set_selection_error_goal(selection_error_goal);
            genetic_algorithm_pointer->set_maximum_iterations_number(maximum_iterations_number);
            genetic_algorithm_pointer->set_maximum_time(maximum_time);
            genetic_algorithm_pointer->set_reserve_error_data(reserve_loss_loss_history == 1);
            genetic_algorithm_pointer->set_reserve_selection_error_data(reserve_selection_error_loss_history == 1);
            genetic_algorithm_pointer->set_reserve_generation_mean(reserve_generation_mean_history == 1);
            genetic_algorithm_pointer->set_reserve_generation_standard_deviation(reserve_generation_standard_deviation_history == 1);
        }
        break;

        default:
            break;
    }
}

void ModelSelection::set_order_selection_MPI(const ModelSelection* model_selection)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int original_order_selection_method;

    // Incremental order parameters

    int step;
    int maximum_selection_failures;

    // Simulated annealing parameters

    double cooling_rate;
    double minimum_temperature;
    int maximum_iterations_number;

    // General parameters

    int minimum_order;
    int maximum_order;
    int trials_number;
    double tolerance;
    double selection_error_goal;
    int maximum_time;
    int reserve_loss_loss_history;
    int reserve_selection_error_loss_history;

    if(rank == 0)
    {
        // Variables to send initialization

        original_order_selection_method = (int)model_selection->get_order_selection_method();

        switch(original_order_selection_method)
        {
            case(int)ModelSelection::INCREMENTAL_ORDER:
            {
                IncrementalOrder* original_incremental_order = model_selection->get_incremental_order_pointer();

                minimum_order = (int)original_incremental_order->get_minimum_order();
                maximum_order = (int)original_incremental_order->get_maximum_order();
                step = (int)original_incremental_order->get_step();
                trials_number = (int)original_incremental_order->get_trials_number();
                tolerance = original_incremental_order->get_tolerance();
                selection_error_goal = original_incremental_order->get_selection_error_goal();
                maximum_selection_failures = (int)original_incremental_order->get_maximum_selection_failures();
                maximum_time = (int)original_incremental_order->get_maximum_time();
                reserve_loss_loss_history = original_incremental_order->get_reserve_error_data();
                reserve_selection_error_loss_history = original_incremental_order->get_reserve_selection_error_data();
            }
            break;

            case(int)ModelSelection::GOLDEN_SECTION:
            {
                GoldenSectionOrder* original_golden_section_order = model_selection->get_golden_section_order_pointer();

                minimum_order = (int)original_golden_section_order->get_minimum_order();
                maximum_order = (int)original_golden_section_order->get_maximum_order();
                trials_number = (int)original_golden_section_order->get_trials_number();
                tolerance = original_golden_section_order->get_tolerance();
                selection_error_goal = original_golden_section_order->get_selection_error_goal();
                maximum_time = (int)original_golden_section_order->get_maximum_time();
                reserve_loss_loss_history = original_golden_section_order->get_reserve_error_data();
                reserve_selection_error_loss_history = original_golden_section_order->get_reserve_selection_error_data();
            }
            break;

            case(int)ModelSelection::SIMULATED_ANNEALING:
            {
                SimulatedAnnealingOrder* original_simulated_annealing = model_selection->get_simulated_annealing_order_pointer();

                minimum_order = (int)original_simulated_annealing->get_minimum_order();
                maximum_order = (int)original_simulated_annealing->get_maximum_order();
                cooling_rate = original_simulated_annealing->get_cooling_rate();
                trials_number = (int)original_simulated_annealing->get_trials_number();
                tolerance = original_simulated_annealing->get_tolerance();
                selection_error_goal = original_simulated_annealing->get_selection_error_goal();
                minimum_temperature = original_simulated_annealing->get_minimum_temperature();
                maximum_iterations_number = (int)original_simulated_annealing->get_maximum_iterations_number();
                maximum_time = (int)original_simulated_annealing->get_maximum_time();
                reserve_loss_loss_history = original_simulated_annealing->get_reserve_error_data();
                reserve_selection_error_loss_history = original_simulated_annealing->get_reserve_selection_error_data();
            }
            break;

            default:
                break;
        }
    }

    // Send variables

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank > 0)
    {
        MPI_Recv(&original_order_selection_method, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request req[8];

        switch(original_order_selection_method)
        {
            case(int)ModelSelection::INCREMENTAL_ORDER:

                MPI_Irecv(&step, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&maximum_selection_failures, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Waitall(2, req, MPI_STATUS_IGNORE);

                break;

            case(int)ModelSelection::SIMULATED_ANNEALING:

                MPI_Irecv(&cooling_rate, 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&minimum_temperature, 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Irecv(&maximum_iterations_number, 1, MPI_INT, rank-1, 3, MPI_COMM_WORLD, &req[2]);

                MPI_Waitall(3, req, MPI_STATUS_IGNORE);

                break;

            default:
                break;
        }

        MPI_Irecv(&minimum_order, 1, MPI_INT, rank-1, 4, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&maximum_order, 1, MPI_INT, rank-1, 5, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&trials_number, 1, MPI_INT, rank-1, 6, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&tolerance, 1, MPI_DOUBLE, rank-1, 7, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(&selection_error_goal, 1, MPI_DOUBLE, rank-1, 8, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(&maximum_time, 1, MPI_INT, rank-1, 9, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(&reserve_loss_loss_history, 1, MPI_INT, rank-1, 10, MPI_COMM_WORLD, &req[6]);
        MPI_Irecv(&reserve_selection_error_loss_history, 1, MPI_INT, rank-1, 11, MPI_COMM_WORLD, &req[7]);

        MPI_Waitall(8, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Send(&original_order_selection_method, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

        MPI_Request req[8];

        switch(original_order_selection_method)
        {
            case(int)ModelSelection::INCREMENTAL_ORDER:

                MPI_Isend(&step, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&maximum_selection_failures, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Waitall(2, req, MPI_STATUS_IGNORE);

                break;

            case(int)ModelSelection::SIMULATED_ANNEALING:

                MPI_Isend(&cooling_rate, 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&minimum_temperature, 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&maximum_iterations_number, 1, MPI_INT, rank+1, 3, MPI_COMM_WORLD, &req[2]);

                MPI_Waitall(3, req, MPI_STATUS_IGNORE);

                break;

            default:
                break;
        }

        MPI_Isend(&minimum_order, 1, MPI_INT, rank+1, 4, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&maximum_order, 1, MPI_INT, rank+1, 5, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(&trials_number, 1, MPI_INT, rank+1, 6, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&tolerance, 1, MPI_DOUBLE, rank+1, 7, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(&selection_error_goal, 1, MPI_DOUBLE, rank+1, 8, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(&maximum_time, 1, MPI_INT, rank+1, 9, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(&reserve_loss_loss_history, 1, MPI_INT, rank+1, 10, MPI_COMM_WORLD, &req[6]);
        MPI_Isend(&reserve_selection_error_loss_history, 1, MPI_INT, rank+1, 11, MPI_COMM_WORLD, &req[7]);

        MPI_Waitall(8, req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Set variables

    set_order_selection_method((ModelSelection::OrderSelectionType)original_order_selection_method);

    switch(original_order_selection_method)
    {
        case(int)ModelSelection::INCREMENTAL_ORDER:
        {
            incremental_order_pointer->set_minimum_order(minimum_order);
            incremental_order_pointer->set_maximum_order(maximum_order);
            incremental_order_pointer->set_step(step);
            incremental_order_pointer->set_trials_number(trials_number);
            incremental_order_pointer->set_tolerance(tolerance);
            incremental_order_pointer->set_selection_error_goal(selection_error_goal);
            incremental_order_pointer->set_maximum_selection_failures(maximum_selection_failures);
            incremental_order_pointer->set_maximum_time(maximum_time);
            incremental_order_pointer->set_reserve_error_data(reserve_loss_loss_history == 1);
            incremental_order_pointer->set_reserve_selection_error_data(reserve_selection_error_loss_history == 1);
        }
        break;

        case(int)ModelSelection::GOLDEN_SECTION:
        {
            golden_section_order_pointer->set_minimum_order(minimum_order);
            golden_section_order_pointer->set_maximum_order(maximum_order);
            golden_section_order_pointer->set_trials_number(trials_number);
            golden_section_order_pointer->set_tolerance(tolerance);
            golden_section_order_pointer->set_selection_error_goal(selection_error_goal);
            golden_section_order_pointer->set_maximum_time(maximum_time);
            golden_section_order_pointer->set_reserve_error_data(reserve_loss_loss_history == 1);
            golden_section_order_pointer->set_reserve_selection_error_data(reserve_selection_error_loss_history == 1);
        }
        break;

        case(int)ModelSelection::SIMULATED_ANNEALING:
        {
            simulated_annelaing_order_pointer->set_minimum_order(minimum_order);
            simulated_annelaing_order_pointer->set_maximum_order(maximum_order);
            simulated_annelaing_order_pointer->set_cooling_rate(cooling_rate);
            simulated_annelaing_order_pointer->set_trials_number(trials_number);
            simulated_annelaing_order_pointer->set_tolerance(tolerance);
            simulated_annelaing_order_pointer->set_selection_error_goal(selection_error_goal);
            simulated_annelaing_order_pointer->set_minimum_temperature(minimum_temperature);
            simulated_annelaing_order_pointer->set_maximum_iterations_number(maximum_iterations_number);
            simulated_annelaing_order_pointer->set_maximum_time(maximum_time);
            simulated_annelaing_order_pointer->set_reserve_error_data(reserve_loss_loss_history == 1);
            simulated_annelaing_order_pointer->set_reserve_selection_error_data(reserve_selection_error_loss_history == 1);
        }
        break;

        default:
            break;
    }
}

#endif


/// This method deletes the order selection algorithm object which composes this model selection object.

void ModelSelection::destruct_order_selection()
{
    delete incremental_order_pointer;
    delete golden_section_order_pointer;
    delete simulated_annelaing_order_pointer;

    incremental_order_pointer = nullptr;
    golden_section_order_pointer = nullptr;
    simulated_annelaing_order_pointer = nullptr;

    order_selection_method = NO_ORDER_SELECTION;
}


/// This method deletes the inputs selection algorithm object which composes this model selection object.

void ModelSelection::destruct_inputs_selection()
{
    delete growing_inputs_pointer;
    delete pruning_inputs_pointer;
    delete genetic_algorithm_pointer;

    growing_inputs_pointer = nullptr;
    pruning_inputs_pointer = nullptr;
    genetic_algorithm_pointer = nullptr;

    inputs_selection_method = NO_INPUTS_SELECTION;
}


/// Checks that the different pointers needed for performing the model selection are not nullptr.

void ModelSelection::check() const
{

    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(multilayer_perceptron_pointer->is_empty())
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw logic_error(buffer.str());
    }

    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.get_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
    }

}


/// Calculate the importance of the inputs, returns a vector with the selection error of the neural network removing one input.

Vector<double> ModelSelection::calculate_inputs_importance() const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

/*
    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();



    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    Vector<double> input_importance(inputs_number, 0.0);

    Vector< Statistics<double> > statistics;

    const bool has_scaling_layer = neural_network_pointer->has_scaling_layer();

    NeuralNetwork neural_network_copy(*neural_network_pointer);

    LossIndex loss_index_copy(*loss_index_pointer);

    loss_index_copy.set_neural_network_pointer(&neural_network_copy);

    TrainingStrategy training_strategy_copy(&loss_index_copy);

    const size_t parameters_number = neural_network_copy.get_parameters_number();

    neural_network_copy.set_parameters(Vector<double>(parameters_number, 0.0));

    if(has_scaling_layer)
    {
        statistics = neural_network_pointer->get_scaling_layer_pointer()->get_statistics();
    }

    training_strategy_copy.set_display(false);

    neural_network_copy.randomize_parameters_normal(0.0, 1.0);

    training_strategy_copy.perform_training();

    const Vector<double> parameters = neural_network_copy.get_parameters();

    const double trained_selection_error = loss_index_copy.calculate_error(selection_instances_indices);

    for(size_t i = 0; i < inputs_number; i++)
    {
        neural_network_copy.prune_input(i);

        const double current_selection_error = loss_index_copy.calculate_error(selection_instances_indices);

//        if(current_selection_error > trained_selection_error)
//        {
//            input_importance[i] = 1;
//        }
//        else if(current_selection_error < trained_selection_error)
//        {
//            input_importance[i] = 0;
//        }
//        else
//        {
          input_importance[i] = current_selection_error/trained_selection_error;
//        }

        neural_network_copy.grow_input();

        neural_network_copy.set_parameters(parameters);
    }

    if(has_scaling_layer)
    {
        neural_network_pointer->get_scaling_layer_pointer()->set_statistics(statistics);
    }

    return input_importance;
*/

    return Vector<double>();
}


/// Perform the order selection, returns a structure with the results of the order selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::Results ModelSelection::perform_order_selection() const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(order_selection_method == NO_ORDER_SELECTION)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "Results perform_order_selection() const method.\n"
               << "None order selection term is used.\n";

        throw logic_error(buffer.str());
    }

    check();

#endif

    Results results;

    switch(order_selection_method)
    {
    case INCREMENTAL_ORDER:
    {
        incremental_order_pointer->set_display(display);

        results.incremental_order_results_pointer = incremental_order_pointer->perform_order_selection();

        break;
    }
    case GOLDEN_SECTION:
    {
        golden_section_order_pointer->set_display(display);

        results.golden_section_order_results_pointer = golden_section_order_pointer->perform_order_selection();

        break;
    }
    case SIMULATED_ANNEALING:
    {
        simulated_annelaing_order_pointer->set_display(display);

        results.simulated_annealing_order_results_pointer = simulated_annelaing_order_pointer->perform_order_selection();

        break;
    }
    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "ModelSelectionResults perform_order_selection() method.\n"
               << "Unknown order selection method.\n";

        throw logic_error(buffer.str());
    }
    }

    return(results);
}


/// Perform the inputs selection, returns a structure with the results of the inputs selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::Results ModelSelection::perform_inputs_selection() const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(inputs_selection_method == NO_INPUTS_SELECTION)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "ModelSelectionResults perform_inputs_selection() const method.\n"
               << "None inputs selection term is used.\n";

        throw logic_error(buffer.str());
    }

    check();

#endif

    Results results;

    switch(inputs_selection_method)
    {
        case GROWING_INPUTS:
        {
            growing_inputs_pointer->set_display(display);

            results.growing_inputs_results_pointer = growing_inputs_pointer->perform_inputs_selection();

            break;
        }
        case PRUNING_INPUTS:
        {
            pruning_inputs_pointer->set_display(display);

            results.pruning_inputs_results_pointer = pruning_inputs_pointer->perform_inputs_selection();

            break;
        }
        case GENETIC_ALGORITHM:
        {
            genetic_algorithm_pointer->set_display(display);

            results.genetic_algorithm_results_pointer = genetic_algorithm_pointer->perform_inputs_selection();

            break;
        }
        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ModelSelection class.\n"
                   << "ModelSelectionResults perform_inputs_selection() method.\n"
                   << "Unknown inputs selection method.\n";

            throw logic_error(buffer.str());
        }
    }

    return(results);
}

/// @todo
/// Perform inputs selection and order selection.

ModelSelection::Results ModelSelection::perform_model_selection() const
{
    Results model_selection_results;

    model_selection_results = perform_order_selection();

    return(model_selection_results);
}


/// Serializes the model selection object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* ModelSelection::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Training strategy

    tinyxml2::XMLElement* model_selection_element = document->NewElement("ModelSelection");

    document->InsertFirstChild(model_selection_element);

    // Inputs Selection

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "NO_INPUTS_SELECTION");
    }
        break;

    case GROWING_INPUTS:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "GROWING_INPUTS");

        const tinyxml2::XMLDocument* growing_inputs_document = growing_inputs_pointer->to_XML();

        const tinyxml2::XMLElement* growing_inputs_element = growing_inputs_document->FirstChildElement("GrowingInputs");

        for( const tinyxml2::XMLNode* nodeFor=growing_inputs_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
            inputs_selection_element->InsertEndChild( copy );
        }

        delete growing_inputs_document;
    }
        break;

    case PRUNING_INPUTS:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "PRUNING_INPUTS");

        const tinyxml2::XMLDocument* pruning_inputs_document = pruning_inputs_pointer->to_XML();

        const tinyxml2::XMLElement* pruning_inputs_element = pruning_inputs_document->FirstChildElement("PruningInputs");

        for( const tinyxml2::XMLNode* nodeFor=pruning_inputs_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
            inputs_selection_element->InsertEndChild( copy );
        }

        delete pruning_inputs_document;
    }
        break;

    case GENETIC_ALGORITHM:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "GENETIC_ALGORITHM");

        const tinyxml2::XMLDocument* genetic_algorithm_document = genetic_algorithm_pointer->to_XML();

        const tinyxml2::XMLElement* genetic_algorithm_element = genetic_algorithm_document->FirstChildElement("GeneticAlgorithm");

        for( const tinyxml2::XMLNode* nodeFor=genetic_algorithm_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
            inputs_selection_element->InsertEndChild( copy );
        }

        delete genetic_algorithm_document;
    }
        break;
    }

    // Order Selection

    switch(order_selection_method)
    {
    case NO_ORDER_SELECTION:
    {
        tinyxml2::XMLElement* order_selection_element = document->NewElement("OrderSelection");
        model_selection_element->LinkEndChild(order_selection_element);

        order_selection_element->SetAttribute("Type", "NO_ORDER_SELECTION");
    }
        break;

    case INCREMENTAL_ORDER:
    {
        tinyxml2::XMLElement* order_selection_element = document->NewElement("OrderSelection");
        model_selection_element->LinkEndChild(order_selection_element);

        order_selection_element->SetAttribute("Type", "INCREMENTAL_ORDER");

        const tinyxml2::XMLDocument* incremental_order_document = incremental_order_pointer->to_XML();

        const tinyxml2::XMLElement* incremental_order_element = incremental_order_document->FirstChildElement("IncrementalOrder");

        for( const tinyxml2::XMLNode* nodeFor=incremental_order_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
            order_selection_element->InsertEndChild( copy );
        }

        delete incremental_order_document;
    }
        break;

    case GOLDEN_SECTION:
    {
        tinyxml2::XMLElement* order_selection_element = document->NewElement("OrderSelection");
        model_selection_element->LinkEndChild(order_selection_element);

        order_selection_element->SetAttribute("Type", "GOLDEN_SECTION");

        const tinyxml2::XMLDocument* golden_section_order_document = golden_section_order_pointer->to_XML();

        const tinyxml2::XMLElement* golden_section_order_element = golden_section_order_document->FirstChildElement("GoldenSectionOrder");

        for( const tinyxml2::XMLNode* nodeFor=golden_section_order_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
            order_selection_element->InsertEndChild( copy );
        }

        delete golden_section_order_document;
    }
        break;

    case SIMULATED_ANNEALING:
    {
        tinyxml2::XMLElement* order_selection_element = document->NewElement("OrderSelection");
        model_selection_element->LinkEndChild(order_selection_element);

        order_selection_element->SetAttribute("Type", "SIMULATED_ANNEALING");

        const tinyxml2::XMLDocument* simulated_annealing_order_document = simulated_annelaing_order_pointer->to_XML();

        const tinyxml2::XMLElement* simulated_annealing_order_element = simulated_annealing_order_document->FirstChildElement("SimulatedAnnealingOrder");

        for( const tinyxml2::XMLNode* nodeFor=simulated_annealing_order_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone( document );
            order_selection_element->InsertEndChild( copy );
        }

        delete simulated_annealing_order_document;
    }
        break;
    }

    return(document);
}


/// Serializes the model selection object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ModelSelection::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("ModelSelection");

    // Inputs Selection

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "NO_INPUTS_SELECTION");

        file_stream.CloseElement();
    }
        break;

    case GROWING_INPUTS:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "GROWING_INPUTS");

        growing_inputs_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case PRUNING_INPUTS:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "PRUNING_INPUTS");

        pruning_inputs_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case GENETIC_ALGORITHM:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "GENETIC_ALGORITHM");

        genetic_algorithm_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;
    }

    // Order Selection

    switch(order_selection_method)
    {
    case NO_ORDER_SELECTION:
    {
        file_stream.OpenElement("OrderSelection");

        file_stream.PushAttribute("Type", "NO_ORDER_SELECTION");

        file_stream.CloseElement();
    }
        break;

    case INCREMENTAL_ORDER:
    {
        file_stream.OpenElement("OrderSelection");

        file_stream.PushAttribute("Type", "INCREMENTAL_ORDER");

        incremental_order_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case GOLDEN_SECTION:
    {
        file_stream.OpenElement("OrderSelection");

        file_stream.PushAttribute("Type", "GOLDEN_SECTION");

        golden_section_order_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case SIMULATED_ANNEALING:
    {
        file_stream.OpenElement("OrderSelection");

        file_stream.PushAttribute("Type", "SIMULATED_ANNEALING");

        simulated_annelaing_order_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;
    }

    file_stream.CloseElement();
}


/// Loads the members of this model selection object from a XML document.
/// @param document XML document of the TinyXML library.

void ModelSelection::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("ModelSelection");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Model Selection element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs Selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InputsSelection");

        if(element)
        {
            const string new_inputs_selection_method = element->Attribute("Type");

            set_inputs_selection_method(new_inputs_selection_method);

            switch(inputs_selection_method)
            {
            case NO_INPUTS_SELECTION:
            {
                // do nothing
            }
                break;
            case GROWING_INPUTS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* growing_element = new_document.NewElement("GrowingInputs");

                for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                    growing_element->InsertEndChild( copy );
                }

                new_document.InsertEndChild(growing_element);

                growing_inputs_pointer->from_XML(new_document);
            }
                break;
            case PRUNING_INPUTS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* pruning_element = new_document.NewElement("PruningInputs");

                for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                    pruning_element->InsertEndChild( copy );
                }

                new_document.InsertEndChild(pruning_element);

                pruning_inputs_pointer->from_XML(new_document);
            }
                break;
            case GENETIC_ALGORITHM:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* genetic_element = new_document.NewElement("GeneticAlgorithm");

                for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                    genetic_element->InsertEndChild( copy );
                }

                new_document.InsertEndChild(genetic_element);

                genetic_algorithm_pointer->from_XML(new_document);
            }
                break;
            }
        }
    }

    // Order Selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("OrderSelection");

        if(element)
        {
            const string new_order_selection_method = element->Attribute("Type");

            set_order_selection_method(new_order_selection_method);

            switch(order_selection_method)
            {
            case NO_ORDER_SELECTION:
            {
                // do nothing
            }
                break;
            case INCREMENTAL_ORDER:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* incremental_element = new_document.NewElement("IncrementalOrder");

                for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                    incremental_element->InsertEndChild( copy );
                }

                new_document.InsertEndChild(incremental_element);

                incremental_order_pointer->from_XML(new_document);
            }
                break;
            case GOLDEN_SECTION:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* golden_section_element = new_document.NewElement("GoldenSectionOrder");

                for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                    golden_section_element->InsertEndChild( copy );
                }

                new_document.InsertEndChild(golden_section_element);

                golden_section_order_pointer->from_XML(new_document);
            }
                break;
            case SIMULATED_ANNEALING:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* simulated_annealing_element = new_document.NewElement("SimulatedAnnealingOrder");

                for( const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone( &new_document );
                    simulated_annealing_element->InsertEndChild( copy );
                }

                new_document.InsertEndChild(simulated_annealing_element);

                simulated_annelaing_order_pointer->from_XML(new_document);
            }
                break;
            }
        }
    }
}


/// Prints to the screen the XML representation of this model selection object. 

void ModelSelection::print() const
{
    cout << to_XML();
}


/// Saves the model selection members to a XML file. 
/// @param file_name Name of model selection XML file. 

void ModelSelection::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


/// Loads the model selection members from a XML file. 
/// @param file_name Name of model selection XML file. 

void ModelSelection::load(const string& file_name)
{
    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
}


/// Saves the results structure to a data file.
/// @param file_name Name of model selection results data file.

void ModelSelection::Results::save(const string& file_name) const
{
    ofstream file(file_name.c_str());

    file << "% Model Selection Results\n";

    file << "\n% Order Selection Results\n";

    if(incremental_order_results_pointer)
    {
        file << incremental_order_results_pointer->object_to_string();
    }

    if(golden_section_order_results_pointer)
    {
        file << golden_section_order_results_pointer->object_to_string();
    }

    if(simulated_annealing_order_results_pointer)
    {
        file << simulated_annealing_order_results_pointer->object_to_string();
    }

    file << "\n% Inputs Selection Results\n";

    if(growing_inputs_results_pointer)
    {
        file << growing_inputs_results_pointer->object_to_string();
    }

    if(pruning_inputs_results_pointer)
    {
        file << pruning_inputs_results_pointer->object_to_string();
    }

    if(genetic_algorithm_results_pointer)
    {
        file << genetic_algorithm_results_pointer->object_to_string();
    }

    file.close();
}


Vector<NeuralNetwork> ModelSelection::perform_k_fold_cross_validation(const size_t& k)
{
    DataSet* data_set_pointer = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();

    const Vector<size_t> selection_instances_indices = data_set_pointer->get_instances().get_selection_indices();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(k < 2)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t&).\n"
              << "Number of iterations must be grater or equal than 2.\n";

       throw logic_error(buffer.str());
    }

    if(!data_set_pointer || data_set_pointer->has_data())
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t&).\n"
              << "There is no data set assigned.\n";

       throw logic_error(buffer.str());
    }

    if(!neural_network_pointer)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t&).\n"
              << "There is no neural network assigned.\n";

       throw logic_error(buffer.str());
    }
#endif

    Instances* instances_pointer = data_set_pointer->get_instances_pointer();

    const Vector<Instances::Use> original_uses = instances_pointer->get_uses();

    instances_pointer->split_random_indices(1,0,0);

    Vector<double> minimum_error_parameters;
    double minimum_error = 1.0;

    Vector<NeuralNetwork> neural_network_ensemble(k);
    double cross_validation_error = 0.0;

    for(size_t i = 0; i < k; i++)
    {
        instances_pointer->set_k_fold_cross_validation_uses(k,i);
        neural_network_pointer->randomize_parameters_normal();

        training_strategy_pointer->perform_training();

        instances_pointer->testing_to_selection();

        const double current_error = loss_index_pointer->calculate_selection_error();

        if(i == 0 || current_error < minimum_error)
        {
            minimum_error = current_error;
            minimum_error_parameters = neural_network_pointer->get_parameters();
        }

        neural_network_ensemble[i].set(*neural_network_pointer);
        cross_validation_error += current_error;

        if(display)
        {
            cout << "Iteration: " << i << "/" << k << endl;
            cout << "Current error: " << current_error << endl;
        }
    }

    if(display)
    {
        cout << "Cross validation error: " << cross_validation_error/k << endl;
    }

    instances_pointer->set_uses(original_uses);
    neural_network_pointer->set_parameters(minimum_error_parameters);

    return neural_network_ensemble;
}


Vector<NeuralNetwork> ModelSelection::perform_random_cross_validation(const size_t& k, const double& selection_ratio)
{
    DataSet* data_set_pointer = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();

    const Vector<size_t> selection_instances_indices = data_set_pointer->get_instances().get_selection_indices();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(k < 2)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "Number of iterations must be grater or equal than 2.\n";

       throw logic_error(buffer.str());
    }

    if(selection_ratio <= 0.0 || selection_ratio >= 1.0)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "The ratio of testing instances must be between 0.0 and 1.0.\n";

       throw logic_error(buffer.str());
    }

    if(!data_set_pointer || data_set_pointer->has_data())
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "There is no data set assigned.\n";

       throw logic_error(buffer.str());
    }

    if(!neural_network_pointer)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "There is no neural network assigned.\n";

       throw logic_error(buffer.str());
    }
#endif

    Instances* instances_pointer = data_set_pointer->get_instances_pointer();

    const Vector<Instances::Use> original_uses = instances_pointer->get_uses();

    instances_pointer->split_random_indices(1,0,0);

    Vector<double> minimum_error_parameters;
    double minimum_error = 1.0;

    Vector<NeuralNetwork> neural_network_ensemble(k);
    double cross_validation_error = 0.0;

    for(size_t i = 0; i < k; i++)
    {
        instances_pointer->split_random_indices(1-selection_ratio,0.0,selection_ratio);
        neural_network_pointer->randomize_parameters_normal();

        training_strategy_pointer->perform_training();

        instances_pointer->testing_to_selection();

        const double current_error = loss_index_pointer->calculate_selection_error();

        if(i == 0 || current_error < minimum_error)
        {
            minimum_error = current_error;
            minimum_error_parameters = neural_network_pointer->get_parameters();
        }

        neural_network_ensemble[i].set(*neural_network_pointer);
        cross_validation_error += current_error;

        if(display)
        {
            cout << "Iteration: " << i << "/" << k << endl;
            cout << "Current error: " << current_error << endl;
        }
    }

    if(display)
    {
        cout << "Cross validation error: " << cross_validation_error/k << endl;
    }

    instances_pointer->set_uses(original_uses);
    neural_network_pointer->set_parameters(minimum_error_parameters);

    return neural_network_ensemble;
}


/// @todo Check this method.

Vector<NeuralNetwork> ModelSelection::perform_positives_cross_validation()
{
    DataSet* data_set_pointer = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();
    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!data_set_pointer || data_set_pointer->has_data())
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_positives_cross_validation().\n"
              << "There is no data set assigned.\n";

       throw logic_error(buffer.str());
    }

    if(!neural_network_pointer)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_positives_cross_validation().\n"
              << "There is no neural network assigned.\n";

       throw logic_error(buffer.str());
    }
#endif

    Instances* instances_pointer = data_set_pointer->get_instances_pointer();

    const Vector<size_t> training_indices = instances_pointer->get_training_indices();

    const Vector<Instances::Use> original_uses = instances_pointer->get_uses();

    Variables* variables = data_set_pointer->get_variables_pointer();

    const Vector<size_t> inputs_indices = variables->get_inputs_indices();

    const size_t target_index = variables->get_targets_indices()[0];

    const Vector<size_t> positives_instances_indices = data_set_pointer->get_variable(target_index).calculate_greater_than_indices(0.5);

    const size_t positives_instances_number = positives_instances_indices.size();

    instances_pointer->split_random_indices(1,0,0);

    Vector<double> minimum_error_parameters;
    double minimum_error = 1.0;

    Vector<NeuralNetwork> neural_network_ensemble(positives_instances_number);
    double cross_validation_error = 0.0;

    for(size_t i = 0; i < positives_instances_number; i++)
    {
        const size_t current_selection_instance_index = positives_instances_indices[i];
        const Vector<double> current_selection_instance = data_set_pointer->get_instance(current_selection_instance_index);
        const double targets = current_selection_instance[target_index];
        const Vector<double> current_inputs_selection_instance = current_selection_instance.get_subvector(inputs_indices);

        instances_pointer->set_use(current_selection_instance_index, Instances::Testing);
        neural_network_pointer->randomize_parameters_normal();

        training_strategy_pointer->perform_training();

        const double outputs = neural_network_pointer->calculate_outputs(current_inputs_selection_instance.to_column_matrix())(0,0);

        const double current_error = abs(targets-outputs);
        const double current_loss = loss_index_pointer->calculate_training_loss();

        if(i == 0 || current_error < minimum_error)
        {
            minimum_error = current_error;
            minimum_error_parameters = neural_network_pointer->get_parameters();
        }

        instances_pointer->set_use(current_selection_instance_index, Instances::Training);

        neural_network_ensemble[i].set(*neural_network_pointer);
        cross_validation_error += current_error;

        if(display)
        {
            cout << "Iteration: " << i+1 << "/" << positives_instances_number << endl;
            cout << "instance index: " << current_selection_instance_index << endl;
            cout << "Output data: " << outputs << endl;
            cout << "Current loss: " << current_loss << endl;
            cout << "Current error: " << current_error << endl;
        }
    }

    if(display)
    {
        cout << "Cross validation error: " << cross_validation_error/positives_instances_number << endl;
    }

    instances_pointer->set_uses(original_uses);
    neural_network_pointer->set_parameters(minimum_error_parameters);

    return neural_network_ensemble;

//    return Vector<NeuralNetwork>();

}
}

/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O D E L   S E L E C T I O N   C L A S S                                                                  */
/*                                                                                                              */ 
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "model_selection.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 

ModelSelection::ModelSelection(void)
    : training_strategy_pointer(NULL)
    , incremental_order_pointer(NULL)
    , golden_section_order_pointer(NULL)
    , simulated_annelaing_order_pointer(NULL)
    , growing_inputs_pointer(NULL)
    , pruning_inputs_pointer(NULL)
    , genetic_algorithm_pointer(NULL)
    , f1_score_optimization_threshold_pointer(NULL)
    , matthew_correlation_optimization_threshold_pointer(NULL)
    , youden_index_optimization_threshold_pointer(NULL)
    , kappa_coefficient_optimization_threshold_pointer(NULL)
    , roc_curve_optimization_threshold_pointer(NULL)
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor. 
/// @param new_training_strategy_pointer Pointer to a training strategy object.

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
    , incremental_order_pointer(NULL)
    , golden_section_order_pointer(NULL)
    , simulated_annelaing_order_pointer(NULL)
    , growing_inputs_pointer(NULL)
    , pruning_inputs_pointer(NULL)
    , genetic_algorithm_pointer(NULL)
    , f1_score_optimization_threshold_pointer(NULL)
    , matthew_correlation_optimization_threshold_pointer(NULL)
    , youden_index_optimization_threshold_pointer(NULL)
    , kappa_coefficient_optimization_threshold_pointer(NULL)
    , roc_curve_optimization_threshold_pointer(NULL)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor. 
/// @param file_name Name of XML model selection file. 

ModelSelection::ModelSelection(const std::string& file_name)
    : training_strategy_pointer(NULL)
    , incremental_order_pointer(NULL)
    , golden_section_order_pointer(NULL)
    , simulated_annelaing_order_pointer(NULL)
    , growing_inputs_pointer(NULL)
    , pruning_inputs_pointer(NULL)
    , genetic_algorithm_pointer(NULL)
    , f1_score_optimization_threshold_pointer(NULL)
    , matthew_correlation_optimization_threshold_pointer(NULL)
    , youden_index_optimization_threshold_pointer(NULL)
    , kappa_coefficient_optimization_threshold_pointer(NULL)
    , roc_curve_optimization_threshold_pointer(NULL)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor. 
/// @param model_selection_document Pointer to a TinyXML document containing the model selection data.

ModelSelection::ModelSelection(const tinyxml2::XMLDocument& model_selection_document)
    : training_strategy_pointer(NULL)
    , incremental_order_pointer(NULL)
    , golden_section_order_pointer(NULL)
    , simulated_annelaing_order_pointer(NULL)
    , growing_inputs_pointer(NULL)
    , pruning_inputs_pointer(NULL)
    , genetic_algorithm_pointer(NULL)
    , f1_score_optimization_threshold_pointer(NULL)
    , matthew_correlation_optimization_threshold_pointer(NULL)
    , youden_index_optimization_threshold_pointer(NULL)
    , kappa_coefficient_optimization_threshold_pointer(NULL)
    , roc_curve_optimization_threshold_pointer(NULL)
{
    from_XML(model_selection_document);
}


// DESTRUCTOR

/// Destructor. 

ModelSelection::~ModelSelection(void)
{
    // Delete inputs selection algorithms

    delete growing_inputs_pointer;
    delete pruning_inputs_pointer;
    delete genetic_algorithm_pointer;

    // Delete order selection algorithms

    delete incremental_order_pointer;
    delete golden_section_order_pointer;
    delete simulated_annelaing_order_pointer;

    // Delete threshold selection algorithms

    delete f1_score_optimization_threshold_pointer;
    delete matthew_correlation_optimization_threshold_pointer;
    delete youden_index_optimization_threshold_pointer;
    delete kappa_coefficient_optimization_threshold_pointer;
    delete roc_curve_optimization_threshold_pointer;
}


// METHODS

// TrainingStrategy* get_training_strategy_pointer(void) const method

/// Returns a pointer to the training strategy object.

TrainingStrategy* ModelSelection::get_training_strategy_pointer(void) const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "TrainingStrategy* get_training_strategy_pointer(void) const method.\n"
               << "Training strategy pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}

// bool has_training_strategy(void) const method

/// Returns true if this model selection has a training strategy associated,
/// and false otherwise.

bool ModelSelection::has_training_strategy(void) const
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

// const OrderSelectionType& get_order_selection_type(void) const method

/// Returns the type of algorithm for the order selection.

const ModelSelection::OrderSelectionType& ModelSelection::get_order_selection_type(void) const
{
    return(order_selection_type);
}

// const InputSelectionType& get_inputs_selection_type(void) const method

/// Returns the type of algorithm for the inputs selection.

const ModelSelection::InputsSelectionType& ModelSelection::get_inputs_selection_type(void) const
{
    return(inputs_selection_type);
}

// const ThresholdSelectionType& get_threshold_selection_type(void) const method

/// Returns the type of algorithm for the threshold selection.

const ModelSelection::ThresholdSelectionType& ModelSelection::get_threshold_selection_type(void) const
{
    return(threshold_selection_type);
}

// IncrementalOrder* get_incremental_order_pointer(void) const method

/// Returns a pointer to the incremental order selection algorithm.

IncrementalOrder* ModelSelection::get_incremental_order_pointer(void) const
{
    return(incremental_order_pointer);
}

// GoldenSectionOrder* get_golden_section_order_pointer(void) const method

/// Returns a pointer to the golden section order selection algorithm.

GoldenSectionOrder* ModelSelection::get_golden_section_order_pointer(void) const
{
    return(golden_section_order_pointer);
}

// SimulatedAnnealingOrder* get_simulated_annealing_order_pointer(void) const method

/// Returns a pointer to the simulated annealing order selection algorithm.

SimulatedAnnealingOrder* ModelSelection::get_simulated_annealing_order_pointer(void) const
{
    return(simulated_annelaing_order_pointer);
}

// GrowingInputs* get_growing_inputs_pointer(void) const method

/// Returns a pointer to the growing inputs selection algorithm.

GrowingInputs* ModelSelection::get_growing_inputs_pointer(void) const
{
    return(growing_inputs_pointer);
}

// PruningInputs* get_pruning_inputs_pointer(void) const method

/// Returns a pointer to the pruning inputs selection algorithm.

PruningInputs* ModelSelection::get_pruning_inputs_pointer(void) const
{
    return(pruning_inputs_pointer);
}

// GeneticAlgorithm* get_genetic_algorithm_pointer(void) const method

/// Returns a pointer to the genetic inputs selection algorithm.

GeneticAlgorithm* ModelSelection::get_genetic_algorithm_pointer(void) const
{
    return(genetic_algorithm_pointer);
}

// F1ScoreOptimizationThreshold* get_f1_score_optimization_threshold_pointer(void) const method

/// Returns a pointer to the f1 score optimiztion threshold selection algorithm.

F1ScoreOptimizationThreshold* ModelSelection::get_f1_score_optimization_threshold_pointer(void) const
{
    return(f1_score_optimization_threshold_pointer);
}

// MatthewCorrelationOptimizationThreshold* get_matthew_correlation_optimization_threshold(void) const method

/// Returns a pointer to the matthew correlation optimiztion threshold selection algorithm.

MatthewCorrelationOptimizationThreshold* ModelSelection::get_matthew_correlation_optimization_threshold(void) const
{
    return(matthew_correlation_optimization_threshold_pointer);
}

// YoudenIndexOptimizationThreshold* get_youden_index_optimization_threshold(void) const method

/// Returns a pointer to the youden index optimiztion threshold selection algorithm.

YoudenIndexOptimizationThreshold* ModelSelection::get_youden_index_optimization_threshold(void) const
{
    return(youden_index_optimization_threshold_pointer);
}

// KappaCoefficientOptimizationThreshold* get_kappa_coefficient_optimization_threshold(void) const method

/// Returns a pointer to the kappa coefficient optimiztion threshold selection algorithm.

KappaCoefficientOptimizationThreshold* ModelSelection::get_kappa_coefficient_optimization_threshold(void) const
{
    return(kappa_coefficient_optimization_threshold_pointer);
}

// ROCCurveOptimizationThreshold* get_roc_curve_optimization_threshold(void) const method

/// Returns a pointer to the roc curve optimiztion threshold selection algorithm.

ROCCurveOptimizationThreshold* ModelSelection::get_roc_curve_optimization_threshold(void) const
{
    return(roc_curve_optimization_threshold_pointer);
}

// void set_default(void) method

/// Sets the members of the model selection object to their default values.

void ModelSelection::set_default(void)
{
    set_order_selection_type(ModelSelection::INCREMENTAL_ORDER);
    set_inputs_selection_type(ModelSelection::GROWING_INPUTS);
    set_threshold_selection_type(ModelSelection::YOUDEN_INDEX);

    display = true;
}

// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ModelSelection::set_display(const bool& new_display)
{
    display = new_display;

    switch (inputs_selection_type)
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
    default:
    {
        break;
    }
    }

    switch (order_selection_type)
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
    default:
    {
        break;
    }
    }

    switch (threshold_selection_type)
    {
    case NO_THRESHOLD_SELECTION:
    {
        // do nothing

        break;
    }
    case F1_SCORE_OPTIMIZATION:
    {
        f1_score_optimization_threshold_pointer->set_display(new_display);

        break;
    }
    case MATTHEW_CORRELATION:
    {
        matthew_correlation_optimization_threshold_pointer->set_display(new_display);

        break;
    }
    case YOUDEN_INDEX:
    {
        youden_index_optimization_threshold_pointer->set_display(new_display);

        break;
    }
    case KAPPA_COEFFICIENT:
    {
        kappa_coefficient_optimization_threshold_pointer->set_display(new_display);

        break;
    }
    case ROC_CURVE_DISTANCE:
    {
        roc_curve_optimization_threshold_pointer->set_display(new_display);

        break;
    }
    default:
    {
        break;
    }
    }
}

// void set_order_selection_type(const OrderSelectionType&) method

/// Sets a new method for selecting the order which have more impact on the targets.
/// @param new_order_selection_type Method for selecting the order (NO_ORDER_SELECTION, INCREMENTAL_ORDER, GOLDEN_SECTION, SIMULATED_ANNEALING).

void ModelSelection::set_order_selection_type(const ModelSelection::OrderSelectionType& new_order_selection_type)
{
    destruct_order_selection();

    order_selection_type = new_order_selection_type;

    switch (new_order_selection_type)
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
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_order_selection_method(const OrderSelectionType&) const method.\n"
               << "Unknow order selection method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }


}

// void set_order_selection_type(const std::string&) method

/// Sets a new order selection algorithm from a string.
/// @param new_order_selection_type String with the order selection type.

void ModelSelection::set_order_selection_type(const std::string& new_order_selection_type)
{
    if(new_order_selection_type == "NO_ORDER_SELECTION")
    {
        set_order_selection_type(NO_ORDER_SELECTION);
    }
    else if(new_order_selection_type == "INCREMENTAL_ORDER")
    {
        set_order_selection_type(INCREMENTAL_ORDER);
    }
    else if(new_order_selection_type == "GOLDEN_SECTION")
    {
        set_order_selection_type(GOLDEN_SECTION);
    }
    else if(new_order_selection_type == "SIMULATED_ANNEALING")
    {
        set_order_selection_type(SIMULATED_ANNEALING);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_order_selection_type(const std::string&) method.\n"
               << "Unknown order selection type: " << new_order_selection_type << ".\n";

        throw std::logic_error(buffer.str());
    }
}

// void set_inputs_selection_type(const InputSelectionType&) method

/// Sets a new method for selecting the inputs which have more impact on the targets.
/// @param new_inputs_selection_type Method for selecting the inputs (NO_INPUTS_SELECTION, GROWING_INPUTS, PRUNING_INPUTS, GENETIC_ALGORITHM).

void ModelSelection::set_inputs_selection_type(const ModelSelection::InputsSelectionType& new_inputs_selection_type)
{
    destruct_inputs_selection();

    inputs_selection_type = new_inputs_selection_type;

    switch (new_inputs_selection_type)
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
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_inputs_selection_method(const InputsSelectionType&) const method.\n"
               << "Unknow inputs selection method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}

// void set_inputs_selection_type(const std::string&) method

/// Sets a new inputs selection algorithm from a string.
/// @param new_inputs_selection_type String with the inputs selection type.

void ModelSelection::set_inputs_selection_type(const std::string& new_inputs_selection_type)
{
    if(new_inputs_selection_type == "NO_INPUTS_SELECTION")
    {
        set_inputs_selection_type(NO_INPUTS_SELECTION);
    }
    else if(new_inputs_selection_type == "GROWING_INPUTS")
    {
        set_inputs_selection_type(GROWING_INPUTS);
    }
    else if(new_inputs_selection_type == "PRUNING_INPUTS")
    {
        set_inputs_selection_type(PRUNING_INPUTS);
    }
    else if(new_inputs_selection_type == "GENETIC_ALGORITHM")
    {
        set_inputs_selection_type(GENETIC_ALGORITHM);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_inputs_selection_type(const std::string&) method.\n"
               << "Unknown inputs selection type: " << new_inputs_selection_type << ".\n";

        throw std::logic_error(buffer.str());
    }
}

// void set_threshold_selection_type(const ThresholdSelectionType&) method

/// Sets a new method for selecting the threshold to improve the final model.
/// @param new_threshold_selection_type Method for selecting the threshold.

void ModelSelection::set_threshold_selection_type(const ModelSelection::ThresholdSelectionType& new_threshold_selection_type)
{
    destruct_threshold_selection();

    threshold_selection_type = new_threshold_selection_type;

    switch (new_threshold_selection_type)
    {
    case NO_THRESHOLD_SELECTION:
    {
        // do nothing

        break;
    }
    case F1_SCORE_OPTIMIZATION:
    {
        f1_score_optimization_threshold_pointer = new F1ScoreOptimizationThreshold(training_strategy_pointer);

        break;
    }
    case MATTHEW_CORRELATION:
    {
        matthew_correlation_optimization_threshold_pointer = new MatthewCorrelationOptimizationThreshold(training_strategy_pointer);

        break;
    }
    case YOUDEN_INDEX:
    {
        youden_index_optimization_threshold_pointer = new YoudenIndexOptimizationThreshold(training_strategy_pointer);

        break;
    }
    case KAPPA_COEFFICIENT:
    {
        kappa_coefficient_optimization_threshold_pointer = new KappaCoefficientOptimizationThreshold(training_strategy_pointer);

        break;
    }
    case ROC_CURVE_DISTANCE:
    {
        roc_curve_optimization_threshold_pointer = new ROCCurveOptimizationThreshold(training_strategy_pointer);

        break;
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_threshold_selection_method(const ThresholdSelectionType&) const method.\n"
               << "Unknow threshold selection method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}

// void set_threshold_selection_type(const std::string&) method

/// Sets a new threshold selection algorithm from a string.
/// @param new_threshold_selection_type String with the threshold selection type.

void ModelSelection::set_threshold_selection_type(const std::string& new_threshold_selection_type)
{
    if(new_threshold_selection_type == "NO_THRESHOLD_SELECTION")
    {
        set_threshold_selection_type(NO_THRESHOLD_SELECTION);
    }
    else if(new_threshold_selection_type == "F1_SCORE_OPTIMIZATION")
    {
        set_threshold_selection_type(F1_SCORE_OPTIMIZATION);
    }
    else if(new_threshold_selection_type == "MATTHEW_CORRELATION")
    {
        set_threshold_selection_type(MATTHEW_CORRELATION);
    }
    else if(new_threshold_selection_type == "YOUDEN_INDEX")
    {
        set_threshold_selection_type(YOUDEN_INDEX);
    }
    else if(new_threshold_selection_type == "KAPPA_COEFFICIENT")
    {
        set_threshold_selection_type(KAPPA_COEFFICIENT);
    }
    else if(new_threshold_selection_type == "ROC_CURVE_DISTANCE")
    {
        set_threshold_selection_type(ROC_CURVE_DISTANCE);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_threshold_selection_type(const std::string&) method.\n"
               << "Unknown threshold selection type: " << new_threshold_selection_type << ".\n";

        throw std::logic_error(buffer.str());
    }
}

// void set_approximation(const bool&) method

/// Sets a new regression value.
/// If it is set to true the problem will be taken as a approximation;
/// if it is set to false the problem will be taken as a classification.
/// @param new_approximation Approximation value.

void ModelSelection::set_approximation(const bool& new_approximation)
{
    switch (inputs_selection_type)
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
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_approximation(const bool&) const method.\n"
               << "Unknow inputs selection method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}

// void set_training_strategy_pointer(TrainingStrategy*) method

/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void ModelSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    switch(order_selection_type)
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
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_training_strategy_pointer(TrainingStrategy*) method.\n"
               << "Unknown order selection type.\n";

        throw std::logic_error(buffer.str());
    }
    }

    switch(inputs_selection_type)
    {
    case NO_ORDER_SELECTION:
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
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_training_strategy_pointer(TrainingStrategy*) method.\n"
               << "Unknown inputs selection type.\n";

        throw std::logic_error(buffer.str());
    }
    }

    switch(threshold_selection_type)
    {
    case NO_THRESHOLD_SELECTION:
    {
        // do nothing

        break;
    }
    case F1_SCORE_OPTIMIZATION:
    {
        f1_score_optimization_threshold_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
        break;
    }
    case MATTHEW_CORRELATION:
    {
        matthew_correlation_optimization_threshold_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
        break;
    }
    case YOUDEN_INDEX:
    {
        youden_index_optimization_threshold_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
        break;
    }
    case KAPPA_COEFFICIENT:
    {
        kappa_coefficient_optimization_threshold_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
        break;
    }
    case ROC_CURVE_DISTANCE:
    {
        roc_curve_optimization_threshold_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
        break;
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_training_strategy_pointer(TrainingStrategy*) method.\n"
               << "Unknown threshold selection type.\n";

        throw std::logic_error(buffer.str());
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

    // Threshold selection

    set_threshold_selection_MPI(model_selection);

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

    int original_inputs_selection_type;

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
    double selection_loss_goal;
    int maximum_iterations_number;
    int maximum_time;
    int reserve_loss_loss_history;
    int reserve_selection_loss_loss_history;

    if(rank == 0)
    {
        // Variables to send initialization

        original_inputs_selection_type = (int)model_selection->get_inputs_selection_type();

        switch (original_inputs_selection_type)
        {
            case (int)ModelSelection::GROWING_INPUTS:
            {
                GrowingInputs* original_growing_inputs = model_selection->get_growing_inputs_pointer();

                trials_number = (int)original_growing_inputs->get_trials_number();
                tolerance = original_growing_inputs->get_tolerance();
                selection_loss_goal = original_growing_inputs->get_selection_loss_goal();
                maximum_selection_failures = (int)original_growing_inputs->get_maximum_selection_failures();
                max_min_inputs_number = (int)original_growing_inputs->get_maximum_inputs_number();
                minimum_correlation = original_growing_inputs->get_minimum_correlation();
                maximum_correlation = original_growing_inputs->get_maximum_correlation();
                maximum_iterations_number = (int)original_growing_inputs->get_maximum_iterations_number();
                maximum_time = (int)original_growing_inputs->get_maximum_time();
                reserve_loss_loss_history = original_growing_inputs->get_reserve_loss_data();
                reserve_selection_loss_loss_history = original_growing_inputs->get_reserve_selection_loss_data();
            }
            break;

            case (int)ModelSelection::PRUNING_INPUTS:
            {
                PruningInputs* original_pruning_inputs = model_selection->get_pruning_inputs_pointer();

                trials_number = (int)original_pruning_inputs->get_trials_number();
                tolerance = original_pruning_inputs->get_tolerance();
                selection_loss_goal = original_pruning_inputs->get_selection_loss_goal();
                maximum_selection_failures = (int)original_pruning_inputs->get_maximum_selection_failures();
                max_min_inputs_number = (int)original_pruning_inputs->get_minimum_inputs_number();
                minimum_correlation = original_pruning_inputs->get_minimum_correlation();
                maximum_correlation = original_pruning_inputs->get_maximum_correlation();
                maximum_iterations_number = (int)original_pruning_inputs->get_maximum_iterations_number();
                maximum_time = (int)original_pruning_inputs->get_maximum_time();
                reserve_loss_loss_history = original_pruning_inputs->get_reserve_loss_data();
                reserve_selection_loss_loss_history = original_pruning_inputs->get_reserve_selection_loss_data();
            }
            break;

            case (int)ModelSelection::GENETIC_ALGORITHM:
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
                selection_loss_goal = original_genetic_algorithm->get_selection_loss_goal();
                maximum_iterations_number = (int)original_genetic_algorithm->get_maximum_iterations_number();
                maximum_time = (int)original_genetic_algorithm->get_maximum_time();
                reserve_loss_loss_history  = original_genetic_algorithm->get_reserve_loss_data();
                reserve_selection_loss_loss_history = original_genetic_algorithm->get_reserve_selection_loss_data();
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
        MPI_Recv(&original_inputs_selection_type, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request req[9];

        switch (original_inputs_selection_type)
        {
            case (int)ModelSelection::GROWING_INPUTS:

                MPI_Irecv(&maximum_selection_failures, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&max_min_inputs_number, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Irecv(&minimum_correlation, 1, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&maximum_correlation, 1, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case (int)ModelSelection::PRUNING_INPUTS:

                MPI_Irecv(&maximum_selection_failures, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&max_min_inputs_number, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Irecv(&minimum_correlation, 1, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&maximum_correlation, 1, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case (int)ModelSelection::GENETIC_ALGORITHM:

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
        MPI_Irecv(&selection_loss_goal, 1, MPI_DOUBLE, rank-1, 12, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&maximum_iterations_number, 1, MPI_INT, rank-1, 13, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(&maximum_time, 1, MPI_INT, rank-1, 14, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(&reserve_loss_loss_history, 1, MPI_INT, rank-1, 15, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(&reserve_selection_loss_loss_history, 1, MPI_INT, rank-1, 16, MPI_COMM_WORLD, &req[6]);

        MPI_Waitall(7, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Send(&original_inputs_selection_type, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

        MPI_Request req[9];

        switch (original_inputs_selection_type)
        {
            case (int)ModelSelection::GROWING_INPUTS:

                MPI_Isend(&maximum_selection_failures, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&max_min_inputs_number, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&minimum_correlation, 1, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&maximum_correlation, 1, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case (int)ModelSelection::PRUNING_INPUTS:

                MPI_Isend(&maximum_selection_failures, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&max_min_inputs_number, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&minimum_correlation, 1, MPI_DOUBLE, rank+1, 3, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&maximum_correlation, 1, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &req[3]);

                MPI_Waitall(4, req, MPI_STATUS_IGNORE);

                break;

            case (int)ModelSelection::GENETIC_ALGORITHM:

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
        MPI_Isend(&selection_loss_goal, 1, MPI_DOUBLE, rank+1, 12, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&maximum_iterations_number, 1, MPI_INT, rank+1, 13, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(&maximum_time, 1, MPI_INT, rank+1, 14, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(&reserve_loss_loss_history, 1, MPI_INT, rank+1, 15, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(&reserve_selection_loss_loss_history, 1, MPI_INT, rank+1, 16, MPI_COMM_WORLD, &req[6]);

        MPI_Waitall(7, req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Set variables

    set_inputs_selection_type((ModelSelection::InputsSelectionType)original_inputs_selection_type);

    switch (original_inputs_selection_type)
    {
        case (int)ModelSelection::GROWING_INPUTS:
        {
            growing_inputs_pointer->set_trials_number(trials_number);
            growing_inputs_pointer->set_tolerance(tolerance);
            growing_inputs_pointer->set_selection_loss_goal(selection_loss_goal);
            growing_inputs_pointer->set_maximum_selection_failures(maximum_selection_failures);
            growing_inputs_pointer->set_maximum_inputs_number(max_min_inputs_number);
            growing_inputs_pointer->set_minimum_correlation(minimum_correlation);
            growing_inputs_pointer->set_maximum_correlation(maximum_correlation);
            growing_inputs_pointer->set_maximum_iterations_number(maximum_iterations_number);
            growing_inputs_pointer->set_maximum_time(maximum_time);
            growing_inputs_pointer->set_reserve_loss_data(reserve_loss_loss_history == 1);
            growing_inputs_pointer->set_reserve_selection_loss_data(reserve_selection_loss_loss_history == 1);
        }
        break;

        case (int)ModelSelection::PRUNING_INPUTS:
        {
            pruning_inputs_pointer->set_trials_number(trials_number);
            pruning_inputs_pointer->set_tolerance(tolerance);
            pruning_inputs_pointer->set_selection_loss_goal(selection_loss_goal);
            pruning_inputs_pointer->set_maximum_selection_failures(maximum_selection_failures);
            pruning_inputs_pointer->set_minimum_inputs_number(max_min_inputs_number);
            pruning_inputs_pointer->set_minimum_correlation(minimum_correlation);
            pruning_inputs_pointer->set_maximum_correlation(maximum_correlation);
            pruning_inputs_pointer->set_maximum_iterations_number(maximum_iterations_number);
            pruning_inputs_pointer->set_maximum_time(maximum_time);
            pruning_inputs_pointer->set_reserve_loss_data(reserve_loss_loss_history == 1);
            pruning_inputs_pointer->set_reserve_selection_loss_data(reserve_selection_loss_loss_history == 1);
        }
        break;

        case (int)ModelSelection::GENETIC_ALGORITHM:
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
            genetic_algorithm_pointer->set_selection_loss_goal(selection_loss_goal);
            genetic_algorithm_pointer->set_maximum_iterations_number(maximum_iterations_number);
            genetic_algorithm_pointer->set_maximum_time(maximum_time);
            genetic_algorithm_pointer->set_reserve_loss_data(reserve_loss_loss_history == 1);
            genetic_algorithm_pointer->set_reserve_selection_loss_data(reserve_selection_loss_loss_history == 1);
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

    int original_order_selection_type;

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
    double selection_loss_goal;
    int maximum_time;
    int reserve_loss_loss_history;
    int reserve_selection_loss_loss_history;

    if(rank == 0)
    {
        // Variables to send initialization

        original_order_selection_type = (int)model_selection->get_order_selection_type();

        switch (original_order_selection_type)
        {
            case (int)ModelSelection::INCREMENTAL_ORDER:
            {
                IncrementalOrder* original_incremental_order = model_selection->get_incremental_order_pointer();

                minimum_order = (int)original_incremental_order->get_minimum_order();
                maximum_order = (int)original_incremental_order->get_maximum_order();
                step = (int)original_incremental_order->get_step();
                trials_number = (int)original_incremental_order->get_trials_number();
                tolerance = original_incremental_order->get_tolerance();
                selection_loss_goal = original_incremental_order->get_selection_loss_goal();
                maximum_selection_failures = (int)original_incremental_order->get_maximum_selection_failures();
                maximum_time = (int)original_incremental_order->get_maximum_time();
                reserve_loss_loss_history = original_incremental_order->get_reserve_loss_data();
                reserve_selection_loss_loss_history = original_incremental_order->get_reserve_selection_loss_data();
            }
            break;

            case (int)ModelSelection::GOLDEN_SECTION:
            {
                GoldenSectionOrder* original_golden_section_order = model_selection->get_golden_section_order_pointer();

                minimum_order = (int)original_golden_section_order->get_minimum_order();
                maximum_order = (int)original_golden_section_order->get_maximum_order();
                trials_number = (int)original_golden_section_order->get_trials_number();
                tolerance = original_golden_section_order->get_tolerance();
                selection_loss_goal = original_golden_section_order->get_selection_loss_goal();
                maximum_time = (int)original_golden_section_order->get_maximum_time();
                reserve_loss_loss_history = original_golden_section_order->get_reserve_loss_data();
                reserve_selection_loss_loss_history = original_golden_section_order->get_reserve_selection_loss_data();
            }
            break;

            case (int)ModelSelection::SIMULATED_ANNEALING:
            {
                SimulatedAnnealingOrder* original_simulated_annealing = model_selection->get_simulated_annealing_order_pointer();

                minimum_order = (int)original_simulated_annealing->get_minimum_order();
                maximum_order = (int)original_simulated_annealing->get_maximum_order();
                cooling_rate = original_simulated_annealing->get_cooling_rate();
                trials_number = (int)original_simulated_annealing->get_trials_number();
                tolerance = original_simulated_annealing->get_tolerance();
                selection_loss_goal = original_simulated_annealing->get_selection_loss_goal();
                minimum_temperature = original_simulated_annealing->get_minimum_temperature();
                maximum_iterations_number = (int)original_simulated_annealing->get_maximum_iterations_number();
                maximum_time = (int)original_simulated_annealing->get_maximum_time();
                reserve_loss_loss_history = original_simulated_annealing->get_reserve_loss_data();
                reserve_selection_loss_loss_history = original_simulated_annealing->get_reserve_selection_loss_data();
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
        MPI_Recv(&original_order_selection_type, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request req[8];

        switch (original_order_selection_type)
        {
            case (int)ModelSelection::INCREMENTAL_ORDER:

                MPI_Irecv(&step, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&maximum_selection_failures, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Waitall(2, req, MPI_STATUS_IGNORE);

                break;

            case (int)ModelSelection::SIMULATED_ANNEALING:

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
        MPI_Irecv(&selection_loss_goal, 1, MPI_DOUBLE, rank-1, 8, MPI_COMM_WORLD, &req[4]);
        MPI_Irecv(&maximum_time, 1, MPI_INT, rank-1, 9, MPI_COMM_WORLD, &req[5]);
        MPI_Irecv(&reserve_loss_loss_history, 1, MPI_INT, rank-1, 10, MPI_COMM_WORLD, &req[6]);
        MPI_Irecv(&reserve_selection_loss_loss_history, 1, MPI_INT, rank-1, 11, MPI_COMM_WORLD, &req[7]);

        MPI_Waitall(8, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Send(&original_order_selection_type, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

        MPI_Request req[8];

        switch (original_order_selection_type)
        {
            case (int)ModelSelection::INCREMENTAL_ORDER:

                MPI_Isend(&step, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&maximum_selection_failures, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);

                MPI_Waitall(2, req, MPI_STATUS_IGNORE);

                break;

            case (int)ModelSelection::SIMULATED_ANNEALING:

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
        MPI_Isend(&selection_loss_goal, 1, MPI_DOUBLE, rank+1, 8, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(&maximum_time, 1, MPI_INT, rank+1, 9, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(&reserve_loss_loss_history, 1, MPI_INT, rank+1, 10, MPI_COMM_WORLD, &req[6]);
        MPI_Isend(&reserve_selection_loss_loss_history, 1, MPI_INT, rank+1, 11, MPI_COMM_WORLD, &req[7]);

        MPI_Waitall(8, req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Set variables

    set_order_selection_type((ModelSelection::OrderSelectionType)original_order_selection_type);

    switch (original_order_selection_type)
    {
        case (int)ModelSelection::INCREMENTAL_ORDER:
        {
            incremental_order_pointer->set_minimum_order(minimum_order);
            incremental_order_pointer->set_maximum_order(maximum_order);
            incremental_order_pointer->set_step(step);
            incremental_order_pointer->set_trials_number(trials_number);
            incremental_order_pointer->set_tolerance(tolerance);
            incremental_order_pointer->set_selection_loss_goal(selection_loss_goal);
            incremental_order_pointer->set_maximum_selection_failures(maximum_selection_failures);
            incremental_order_pointer->set_maximum_time(maximum_time);
            incremental_order_pointer->set_reserve_loss_data(reserve_loss_loss_history == 1);
            incremental_order_pointer->set_reserve_selection_loss_data(reserve_selection_loss_loss_history == 1);
        }
        break;

        case (int)ModelSelection::GOLDEN_SECTION:
        {
            golden_section_order_pointer->set_minimum_order(minimum_order);
            golden_section_order_pointer->set_maximum_order(maximum_order);
            golden_section_order_pointer->set_trials_number(trials_number);
            golden_section_order_pointer->set_tolerance(tolerance);
            golden_section_order_pointer->set_selection_loss_goal(selection_loss_goal);
            golden_section_order_pointer->set_maximum_time(maximum_time);
            golden_section_order_pointer->set_reserve_loss_data(reserve_loss_loss_history == 1);
            golden_section_order_pointer->set_reserve_selection_loss_data(reserve_selection_loss_loss_history == 1);
        }
        break;

        case (int)ModelSelection::SIMULATED_ANNEALING:
        {
            simulated_annelaing_order_pointer->set_minimum_order(minimum_order);
            simulated_annelaing_order_pointer->set_maximum_order(maximum_order);
            simulated_annelaing_order_pointer->set_cooling_rate(cooling_rate);
            simulated_annelaing_order_pointer->set_trials_number(trials_number);
            simulated_annelaing_order_pointer->set_tolerance(tolerance);
            simulated_annelaing_order_pointer->set_selection_loss_goal(selection_loss_goal);
            simulated_annelaing_order_pointer->set_minimum_temperature(minimum_temperature);
            simulated_annelaing_order_pointer->set_maximum_iterations_number(maximum_iterations_number);
            simulated_annelaing_order_pointer->set_maximum_time(maximum_time);
            simulated_annelaing_order_pointer->set_reserve_loss_data(reserve_loss_loss_history == 1);
            simulated_annelaing_order_pointer->set_reserve_selection_loss_data(reserve_selection_loss_loss_history == 1);
        }
        break;

        default:
            break;
    }
}

void ModelSelection::set_threshold_selection_MPI(const ModelSelection* model_selection)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int original_threshold_selection_type;

    // General parameters

    double minimum_threshold;
    double maximum_threshold;
    double step;
    int reserve_function_history;

    if(rank == 0)
    {
        // Variables to send initialization

        original_threshold_selection_type = (int)model_selection->get_threshold_selection_type();

        switch (original_threshold_selection_type)
        {
            case (int)ModelSelection::F1_SCORE_OPTIMIZATION:
            {
                F1ScoreOptimizationThreshold* original_f1_score_optimization = model_selection->get_f1_score_optimization_threshold_pointer();

                minimum_threshold = original_f1_score_optimization->get_minimum_threshold();
                maximum_threshold = original_f1_score_optimization->get_maximum_threshold();
                step = original_f1_score_optimization->get_step();
                reserve_function_history = original_f1_score_optimization->get_reserve_function_data();
            }
            break;

            case (int)ModelSelection::MATTHEW_CORRELATION:
            {
                MatthewCorrelationOptimizationThreshold* original_matthew_correlation_optimization = model_selection->get_matthew_correlation_optimization_threshold();

                minimum_threshold = original_matthew_correlation_optimization->get_minimum_threshold();
                maximum_threshold = original_matthew_correlation_optimization->get_maximum_threshold();
                step = original_matthew_correlation_optimization->get_step();
                reserve_function_history = original_matthew_correlation_optimization->get_reserve_function_data();
            }
            break;

            case (int)ModelSelection::YOUDEN_INDEX:
            {
                YoudenIndexOptimizationThreshold* original_youden_index_optimization = model_selection->get_youden_index_optimization_threshold();

                minimum_threshold = original_youden_index_optimization->get_minimum_threshold();
                maximum_threshold = original_youden_index_optimization->get_maximum_threshold();
                step = original_youden_index_optimization->get_step();
                reserve_function_history = original_youden_index_optimization->get_reserve_function_data();
            }
            break;

            case (int)ModelSelection::KAPPA_COEFFICIENT:
            {
                KappaCoefficientOptimizationThreshold* original_kappa_coefficient_optimization = model_selection->get_kappa_coefficient_optimization_threshold();

                minimum_threshold = original_kappa_coefficient_optimization->get_minimum_threshold();
                maximum_threshold = original_kappa_coefficient_optimization->get_maximum_threshold();
                step = original_kappa_coefficient_optimization->get_step();
                reserve_function_history = original_kappa_coefficient_optimization->get_reserve_function_data();
            }
            break;

            case (int)ModelSelection::ROC_CURVE_DISTANCE:
            {
                ROCCurveOptimizationThreshold* original_roc_curve_optimization = model_selection->get_roc_curve_optimization_threshold();

                minimum_threshold = original_roc_curve_optimization->get_minimum_threshold();
                maximum_threshold = original_roc_curve_optimization->get_maximum_threshold();
                step = original_roc_curve_optimization->get_step();
                reserve_function_history = original_roc_curve_optimization->get_reserve_function_data();
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
        MPI_Recv(&original_threshold_selection_type, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request req[4];

        MPI_Irecv(&minimum_threshold, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&maximum_threshold, 1, MPI_INT, rank-1, 3, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&step, 1, MPI_INT, rank-1, 4, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&reserve_function_history, 1, MPI_INT, rank-1, 5, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Send(&original_threshold_selection_type, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

        MPI_Request req[4];

        MPI_Isend(&minimum_threshold, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&maximum_threshold, 1, MPI_INT, rank+1, 3, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(&step, 1, MPI_INT, rank+1, 4, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&reserve_function_history, 1, MPI_INT, rank+1, 5, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Set variables

    set_threshold_selection_type((ModelSelection::ThresholdSelectionType)original_threshold_selection_type);

    switch (original_threshold_selection_type)
    {
        case (int)ModelSelection::F1_SCORE_OPTIMIZATION:
        {
            f1_score_optimization_threshold_pointer->set_minimum_threshold(minimum_threshold);
            f1_score_optimization_threshold_pointer->set_maximum_threshold(maximum_threshold);
            f1_score_optimization_threshold_pointer->set_step(step);
            f1_score_optimization_threshold_pointer->set_reserve_function_data(reserve_function_history == 1);
        }
        break;

        case (int)ModelSelection::MATTHEW_CORRELATION:
        {
            matthew_correlation_optimization_threshold_pointer->set_minimum_threshold(minimum_threshold);
            matthew_correlation_optimization_threshold_pointer->set_maximum_threshold(maximum_threshold);
            matthew_correlation_optimization_threshold_pointer->set_step(step);
            matthew_correlation_optimization_threshold_pointer->set_reserve_function_data(reserve_function_history == 1);
        }
            break;

        case (int)ModelSelection::YOUDEN_INDEX:
        {
            youden_index_optimization_threshold_pointer->set_minimum_threshold(minimum_threshold);
            youden_index_optimization_threshold_pointer->set_maximum_threshold(maximum_threshold);
            youden_index_optimization_threshold_pointer->set_step(step);
            youden_index_optimization_threshold_pointer->set_reserve_function_data(reserve_function_history == 1);
        }
            break;

        case (int)ModelSelection::KAPPA_COEFFICIENT:
        {
            kappa_coefficient_optimization_threshold_pointer->set_minimum_threshold(minimum_threshold);
            kappa_coefficient_optimization_threshold_pointer->set_maximum_threshold(maximum_threshold);
            kappa_coefficient_optimization_threshold_pointer->set_step(step);
            kappa_coefficient_optimization_threshold_pointer->set_reserve_function_data(reserve_function_history == 1);
        }
            break;

        case (int)ModelSelection::ROC_CURVE_DISTANCE:
        {
            roc_curve_optimization_threshold_pointer->set_minimum_threshold(minimum_threshold);
            roc_curve_optimization_threshold_pointer->set_maximum_threshold(maximum_threshold);
            roc_curve_optimization_threshold_pointer->set_step(step);
            roc_curve_optimization_threshold_pointer->set_reserve_function_data(reserve_function_history == 1);
        }
        break;

        default:
            break;
    }
}
#endif

// void destruct_order_selection(void) method

/// This method deletes the order selection algorithm object which composes this model selection object.

void ModelSelection::destruct_order_selection(void)
{
    delete incremental_order_pointer;
    delete golden_section_order_pointer;
    delete simulated_annelaing_order_pointer;

    incremental_order_pointer = NULL;
    golden_section_order_pointer = NULL;
    simulated_annelaing_order_pointer = NULL;

    order_selection_type = NO_ORDER_SELECTION;
}

// void destruct_inputs_selection(void) method

/// This method deletes the inputs selection algorithm object which composes this model selection object.

void ModelSelection::destruct_inputs_selection(void)
{
    delete growing_inputs_pointer;
    delete pruning_inputs_pointer;
    delete genetic_algorithm_pointer;

    growing_inputs_pointer = NULL;
    pruning_inputs_pointer = NULL;
    genetic_algorithm_pointer = NULL;

    inputs_selection_type = NO_INPUTS_SELECTION;
}

// void destruct_threshold_selection(void) method

/// This method deletes the threshold selection algorithm object which composes this model selection object.

void ModelSelection::destruct_threshold_selection(void)
{
    delete f1_score_optimization_threshold_pointer;
    delete matthew_correlation_optimization_threshold_pointer;
    delete youden_index_optimization_threshold_pointer;
    delete kappa_coefficient_optimization_threshold_pointer;
    delete roc_curve_optimization_threshold_pointer;

    f1_score_optimization_threshold_pointer = NULL;
    matthew_correlation_optimization_threshold_pointer = NULL;
    youden_index_optimization_threshold_pointer = NULL;
    kappa_coefficient_optimization_threshold_pointer = NULL;
    roc_curve_optimization_threshold_pointer = NULL;

    threshold_selection_type = NO_THRESHOLD_SELECTION;
}

// void check(void) const method

/// Checks that the different pointers needed for performing the model selection are not NULL.

void ModelSelection::check(void) const
{

    // Training algorithm stuff

    std::ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Pointer to training strategy is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Pointer to loss functional is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Pointer to neural network is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Pointer to multilayer perceptron is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    if(multilayer_perceptron_pointer->is_empty())
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw std::logic_error(buffer.str());
    }

    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Pointer to data set is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.count_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check(void) const method.\n"
               << "Number of selection instances is zero.\n";

        throw std::logic_error(buffer.str());
    }

}

// Vector<double> calculate_inputs_importance(void) const method

/// Calculate the importance of the inputs, returns a vector with the selection loss of the neural network removing one input.

Vector<double> ModelSelection::calculate_inputs_importance(void) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

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

    const size_t parameters_number = neural_network_copy.count_parameters_number();

    neural_network_copy.set_parameters(Vector<double>(parameters_number, 0.0));

    if(has_scaling_layer)
    {
        statistics = neural_network_pointer->get_scaling_layer_pointer()->get_statistics();
    }

    training_strategy_copy.set_display(false);

    neural_network_copy.randomize_parameters_normal(0.0, 1.0);

    training_strategy_copy.perform_training();

    const Vector<double> parameters = neural_network_copy.arrange_parameters();

    const double trained_selection_error = loss_index_copy.calculate_selection_error();

    for (size_t i = 0; i < inputs_number; i++)
    {
        neural_network_copy.prune_input(i);

        const double current_selection_error = loss_index_copy.calculate_selection_error();

        if(current_selection_error > trained_selection_error)
        {
            input_importance[i] = 1;
        }
        else if(current_selection_error < trained_selection_error)
        {
            input_importance[i] = 0;
        }
        else
        {
            input_importance[i] = current_selection_error/trained_selection_error;
        }

        neural_network_copy.grow_input();

        neural_network_copy.set_parameters(parameters);
    }

    if(has_scaling_layer)
    {
        neural_network_pointer->get_scaling_layer_pointer()->set_statistics(statistics);
    }

    return input_importance;
}

// ModelSelectionResults perform_order_selection(void) method

/// Perform the order selection, returns a structure with the results of the order selection
/// It also set the neural network of the training strategy pointer with the optimum parameters

ModelSelection::ModelSelectionResults ModelSelection::perform_order_selection(void) const
{
#ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(order_selection_type == NO_ORDER_SELECTION)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "ModelSelectionResults perform_order_selection(void) const method.\n"
               << "None order selection term is used.\n";

        throw std::logic_error(buffer.str());
    }

    check();

#endif

    ModelSelectionResults results;

    switch(order_selection_type)
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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "ModelSelectionResults perform_order_selection(void) method.\n"
               << "Unknown order selection method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }

    return(results);
}

// ModelSelectionResults perform_inputs_selection(void) method

/// Perform the inputs selection, returns a structure with the results of the inputs selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::ModelSelectionResults ModelSelection::perform_inputs_selection(void) const
{
#ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(inputs_selection_type == NO_INPUTS_SELECTION)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "ModelSelectionResults perform_inputs_selection(void) const method.\n"
               << "None inputs selection term is used.\n";

        throw std::logic_error(buffer.str());
    }

    check();

#endif

    ModelSelectionResults results;

    switch(inputs_selection_type)
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
            std::ostringstream buffer;

            buffer << "OpenNN Exception: ModelSelection class.\n"
                   << "ModelSelectionResults perform_inputs_selection(void) method.\n"
                   << "Unknown inputs selection method.\n";

            throw std::logic_error(buffer.str());

            break;
        }
    }

    return(results);
}

// ModelSelectionResults perform_threshold_selection(void) method

/// Perform the threshold selection, returns a structure with the results of the threshold selection.
/// It also set the neural network with the optimum threshold.

ModelSelection::ModelSelectionResults ModelSelection::perform_threshold_selection(void) const
{
#ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(threshold_selection_type == NO_THRESHOLD_SELECTION)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "ModelSelectionResults perform_threshold_selection(void) const method.\n"
               << "None threshold selection term is used.\n";

        throw std::logic_error(buffer.str());
    }

    check();

#endif

    ModelSelectionResults results;

    switch(threshold_selection_type)
    {
        case F1_SCORE_OPTIMIZATION:
        {
            f1_score_optimization_threshold_pointer->set_display(display);

            results.f1_score_opimization_results_pointer = f1_score_optimization_threshold_pointer->perform_threshold_selection();

            break;
        }
        case MATTHEW_CORRELATION:
        {
            matthew_correlation_optimization_threshold_pointer->set_display(display);

            results.matthew_correlation_optimization_results_pointer = matthew_correlation_optimization_threshold_pointer->perform_threshold_selection();

            break;
        }
        case YOUDEN_INDEX:
        {
            youden_index_optimization_threshold_pointer->set_display(display);

            results.youden_index_optimization_results_pointer = youden_index_optimization_threshold_pointer->perform_threshold_selection();

            break;
        }
        case KAPPA_COEFFICIENT:
        {
            kappa_coefficient_optimization_threshold_pointer->set_display(display);

            results.kappa_coefficient_optimization_results_pointer = kappa_coefficient_optimization_threshold_pointer->perform_threshold_selection();

            break;
        }
        case ROC_CURVE_DISTANCE:
        {
            roc_curve_optimization_threshold_pointer->set_display(display);

            results.roc_curve_optimization_results_pointer = roc_curve_optimization_threshold_pointer->perform_threshold_selection();

            break;
        }
        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: ModelSelection class.\n"
                   << "ModelSelectionResults perform_threshold_selection(void) method.\n"
                   << "Unknown threshold selection method.\n";

            throw std::logic_error(buffer.str());

            break;
        }
    }

    return(results);
}

// ModelSelectionResults perform_model_selection(void) const method

/// @todo
/// Perform inputs selection and order selection.

ModelSelection::ModelSelectionResults ModelSelection::perform_model_selection(void) const
{
    ModelSelectionResults model_selection_results;

    model_selection_results = perform_order_selection();

    return(model_selection_results);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the model selection object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* ModelSelection::to_XML(void) const
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Training strategy

    tinyxml2::XMLElement* model_selection_element = document->NewElement("ModelSelection");

    document->InsertFirstChild(model_selection_element);

    // Inputs Selection

    switch(inputs_selection_type)
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

        DeepClone(inputs_selection_element, growing_inputs_element, document, NULL);

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

        DeepClone(inputs_selection_element, pruning_inputs_element, document, NULL);

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

        DeepClone(inputs_selection_element, genetic_algorithm_element, document, NULL);

        delete genetic_algorithm_document;
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
               << "Unknown inputs selection type.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }

    // Order Selection

    switch(order_selection_type)
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

        DeepClone(order_selection_element, incremental_order_element, document, NULL);

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

        DeepClone(order_selection_element, golden_section_order_element, document, NULL);

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

        DeepClone(order_selection_element, simulated_annealing_order_element, document, NULL);

        delete simulated_annealing_order_document;
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
               << "Unknown order selection type.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }

    // Threshold Selection

    switch(threshold_selection_type)
    {
    case NO_ORDER_SELECTION:
    {
        tinyxml2::XMLElement* order_selection_element = document->NewElement("ThresholdSelection");
        model_selection_element->LinkEndChild(order_selection_element);

        order_selection_element->SetAttribute("Type", "NO_ORDER_SELECTION");
    }
        break;

    case F1_SCORE_OPTIMIZATION:
    {
        tinyxml2::XMLElement* threshold_selection_element = document->NewElement("ThresholdSelection");
        model_selection_element->LinkEndChild(threshold_selection_element);

        threshold_selection_element->SetAttribute("Type", "F1_SCORE_OPTIMIZATION");

        const tinyxml2::XMLDocument* f1_score_optimization_document = f1_score_optimization_threshold_pointer->to_XML();

        const tinyxml2::XMLElement* f1_score_optimization_element = f1_score_optimization_document->FirstChildElement("F1ScoreOptimizationThreshold");

        DeepClone(threshold_selection_element, f1_score_optimization_element, document, NULL);

        delete f1_score_optimization_document;
    }
        break;

    case MATTHEW_CORRELATION:
    {
        tinyxml2::XMLElement* threshold_selection_element = document->NewElement("ThresholdSelection");
        model_selection_element->LinkEndChild(threshold_selection_element);

        threshold_selection_element->SetAttribute("Type", "MATTHEW_CORRELATION");

        const tinyxml2::XMLDocument* matthew_correlation_optimization_document = matthew_correlation_optimization_threshold_pointer->to_XML();

        const tinyxml2::XMLElement* matthew_correlation_optimization_element = matthew_correlation_optimization_document->FirstChildElement("MatthewCorrelationOptimizationThreshold");

        DeepClone(threshold_selection_element, matthew_correlation_optimization_element, document, NULL);

        delete matthew_correlation_optimization_document;
    }
        break;

    case YOUDEN_INDEX:
    {
        tinyxml2::XMLElement* threshold_selection_element = document->NewElement("ThresholdSelection");
        model_selection_element->LinkEndChild(threshold_selection_element);

        threshold_selection_element->SetAttribute("Type", "YOUDEN_INDEX");

        const tinyxml2::XMLDocument* youden_index_optimization_document = youden_index_optimization_threshold_pointer->to_XML();

        const tinyxml2::XMLElement* youden_index_optimization_element = youden_index_optimization_document->FirstChildElement("YoudenIndexOptimizationThreshold");

        DeepClone(threshold_selection_element, youden_index_optimization_element, document, NULL);

        delete youden_index_optimization_document;
    }
        break;

    case KAPPA_COEFFICIENT:
    {
        tinyxml2::XMLElement* threshold_selection_element = document->NewElement("ThresholdSelection");
        model_selection_element->LinkEndChild(threshold_selection_element);

        threshold_selection_element->SetAttribute("Type", "KAPPA_COEFFICIENT");

        const tinyxml2::XMLDocument* kappa_coefficient_optimization_document = kappa_coefficient_optimization_threshold_pointer->to_XML();

        const tinyxml2::XMLElement* kappa_coefficient_optimization_element = kappa_coefficient_optimization_document->FirstChildElement("KappaCoefficientOptimizationThreshold");

        DeepClone(threshold_selection_element, kappa_coefficient_optimization_element, document, NULL);

        delete kappa_coefficient_optimization_document;
    }
        break;

    case ROC_CURVE_DISTANCE:
    {
        tinyxml2::XMLElement* threshold_selection_element = document->NewElement("ThresholdSelection");
        model_selection_element->LinkEndChild(threshold_selection_element);

        threshold_selection_element->SetAttribute("Type", "ROC_CURVE_DISTANCE");

        const tinyxml2::XMLDocument* roc_curve_optimization_document = roc_curve_optimization_threshold_pointer->to_XML();

        const tinyxml2::XMLElement* roc_curve_optimization_element = roc_curve_optimization_document->FirstChildElement("ROCCurveOptimizationThreshold");

        DeepClone(threshold_selection_element, roc_curve_optimization_element, document, NULL);

        delete roc_curve_optimization_document;
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "tinyxml2::XMLDocument* to_XML(void) const method.\n"
               << "Unknown threshold selection type.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the model selection object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ModelSelection::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("ModelSelection");

    // Inputs Selection

    switch(inputs_selection_type)
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

    default:
    {
        std::ostringstream buffer;

        file_stream.CloseElement();

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
               << "Unknown inputs selection type.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }

    // Order Selection

    switch(order_selection_type)
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

    default:
    {
        std::ostringstream buffer;

        file_stream.CloseElement();

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
               << "Unknown order selection type.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }

    // Threshold Selection

    switch(threshold_selection_type)
    {
    case NO_THRESHOLD_SELECTION:
    {
        file_stream.OpenElement("ThresholdSelection");

        file_stream.PushAttribute("Type", "NO_THRESHOLD_SELECTION");

        file_stream.CloseElement();
    }
        break;

    case F1_SCORE_OPTIMIZATION:
    {
        file_stream.OpenElement("ThresholdSelection");

        file_stream.PushAttribute("Type", "F1_SCORE_OPTIMIZATION");

        f1_score_optimization_threshold_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case MATTHEW_CORRELATION:
    {
        file_stream.OpenElement("ThresholdSelection");

        file_stream.PushAttribute("Type", "MATTHEW_CORRELATION");

        matthew_correlation_optimization_threshold_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case YOUDEN_INDEX:
    {
        file_stream.OpenElement("ThresholdSelection");

        file_stream.PushAttribute("Type", "YOUDEN_INDEX");

        youden_index_optimization_threshold_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case KAPPA_COEFFICIENT:
    {
        file_stream.OpenElement("ThresholdSelection");

        file_stream.PushAttribute("Type", "KAPPA_COEFFICIENT");

        kappa_coefficient_optimization_threshold_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case ROC_CURVE_DISTANCE:
    {
        file_stream.OpenElement("ThresholdSelection");

        file_stream.PushAttribute("Type", "ROC_CURVE_DISTANCE");

        roc_curve_optimization_threshold_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    default:
    {
        std::ostringstream buffer;

        file_stream.CloseElement();

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
               << "Unknown order selection type.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }

    file_stream.CloseElement();
}

// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads the members of this model selection object from a XML document.
/// @param document XML document of the TinyXML library.

void ModelSelection::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("ModelSelection");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Model Selection element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Inputs Selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InputsSelection");

        if(element)
        {
            const std::string new_inputs_selection_type = element->Attribute("Type");

            set_inputs_selection_type(new_inputs_selection_type);

            switch(inputs_selection_type)
            {
            case NO_INPUTS_SELECTION:
            {
                // do nothing
            }
                break;
            case GROWING_INPUTS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("GrowingInputs");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                growing_inputs_pointer->from_XML(new_document);
            }
                break;
            case PRUNING_INPUTS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("PruningInputs");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                pruning_inputs_pointer->from_XML(new_document);
            }
                break;
            case GENETIC_ALGORITHM:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("GeneticAlgorithm");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                genetic_algorithm_pointer->from_XML(new_document);
            }
                break;
            default:
            {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: ModelSelection class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown inputs selection type.\n";

                throw std::logic_error(buffer.str());
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
            const std::string new_order_selection_type = element->Attribute("Type");

            set_order_selection_type(new_order_selection_type);

            switch(order_selection_type)
            {
            case NO_ORDER_SELECTION:
            {
                // do nothing
            }
                break;
            case INCREMENTAL_ORDER:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("IncrementalOrder");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                incremental_order_pointer->from_XML(new_document);
            }
                break;
            case GOLDEN_SECTION:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("GoldenSectionOrder");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                golden_section_order_pointer->from_XML(new_document);
            }
                break;
            case SIMULATED_ANNEALING:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("SimulatedAnnealingOrder");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                simulated_annelaing_order_pointer->from_XML(new_document);
            }
                break;
            default:
            {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: ModelSelection class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown order selection type.\n";

                throw std::logic_error(buffer.str());
            }
                break;
            }
        }
    }

    // Threshold Selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ThresholdSelection");

        if(element)
        {
            const std::string new_threshold_selection_type = element->Attribute("Type");

            set_threshold_selection_type(new_threshold_selection_type);

            switch(threshold_selection_type)
            {
            case NO_THRESHOLD_SELECTION:
            {
                // do nothing
            }
                break;
            case F1_SCORE_OPTIMIZATION:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("F1ScoreOptimizationThreshold");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                f1_score_optimization_threshold_pointer->from_XML(new_document);
            }
                break;
            case MATTHEW_CORRELATION:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("MatthewCorrelationOptimizationThreshold");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                matthew_correlation_optimization_threshold_pointer->from_XML(new_document);
            }
                break;
            case YOUDEN_INDEX:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("YoudenIndexOptimizationThreshold");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                youden_index_optimization_threshold_pointer->from_XML(new_document);
            }
                break;
            case KAPPA_COEFFICIENT:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("KappaCoefficientOptimizationThreshold");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                kappa_coefficient_optimization_threshold_pointer->from_XML(new_document);
            }
                break;
            case ROC_CURVE_DISTANCE:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* element_clone = new_document.NewElement("ROCCurveOptimizationThreshold");
                new_document.InsertFirstChild(element_clone);

                DeepClone(element_clone, element, &new_document, NULL);

                roc_curve_optimization_threshold_pointer->from_XML(new_document);
            }
                break;
            default:
            {
                std::ostringstream buffer;

                buffer << "OpenNN Exception: ModelSelection class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Unknown order selection type.\n";

                throw std::logic_error(buffer.str());
            }
                break;
            }
        }
    }
}


// void print(void) method

/// Prints to the screen the XML representation of this model selection object. 

void ModelSelection::print(void) const
{
    std::cout << to_XML();
}


// void save(const std::string&) const method

/// Saves the model selection members to a XML file. 
/// @param file_name Name of model selection XML file. 

void ModelSelection::save(const std::string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


// void load(const std::string&) method

/// Loads the model selection members from a XML file. 
/// @param file_name Name of model selection XML file. 

void ModelSelection::load(const std::string& file_name)
{
    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void load(const std::string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw std::logic_error(buffer.str());
    }

    from_XML(document);
}

// void ModelSelectionResults::save(const std::string&) const method

/// Saves the results structure to a data file.
/// @param file_name Name of model selection results data file.

void ModelSelection::ModelSelectionResults::save(const std::string& file_name) const
{
    std::ofstream file(file_name.c_str());

    file << "% Model Selection Results\n";

    file << "\n% Order Selection Results\n";

    if(incremental_order_results_pointer)
    {
        file << incremental_order_results_pointer->to_string();
    }

    if(golden_section_order_results_pointer)
    {
        file << golden_section_order_results_pointer->to_string();
    }

    if(simulated_annealing_order_results_pointer)
    {
        file << simulated_annealing_order_results_pointer->to_string();
    }

    file << "\n% Inputs Selection Results\n";

    if(growing_inputs_results_pointer)
    {
        file << growing_inputs_results_pointer->to_string();
    }

    if(pruning_inputs_results_pointer)
    {
        file << pruning_inputs_results_pointer->to_string();
    }

    if(genetic_algorithm_results_pointer)
    {
        file << genetic_algorithm_results_pointer->to_string();
    }

    file << "\n% Threhold Selection Results\n";

    if(f1_score_opimization_results_pointer)
    {
        file << f1_score_opimization_results_pointer->to_string();
    }

    if(matthew_correlation_optimization_results_pointer)
    {
        file << matthew_correlation_optimization_results_pointer->to_string();
    }

    if(youden_index_optimization_results_pointer)
    {
        file << youden_index_optimization_results_pointer->to_string();
    }

    if(kappa_coefficient_optimization_results_pointer)
    {
        file << kappa_coefficient_optimization_results_pointer->to_string();
    }

    if(roc_curve_optimization_results_pointer)
    {
        file << roc_curve_optimization_results_pointer->to_string();
    }

    file.close();
}

}

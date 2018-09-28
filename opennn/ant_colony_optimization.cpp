/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   A N T   C O L O N Y   O P T I M I Z A T I O N   C L A S S                                                  */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "ant_colony_optimization.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

AntColonyOptimization::AntColonyOptimization()
    : OrderSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a gradient descent object.

AntColonyOptimization::AntColonyOptimization(TrainingStrategy* new_training_strategy_pointer)
    : OrderSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}

// XML CONSTRUCTOR

/// XML constructor.
/// @param ant_colony_optimization_document Pointer to a TinyXML document containing the ant colony optimization algorithm data.

AntColonyOptimization::AntColonyOptimization(const tinyxml2::XMLDocument& ant_colony_optimization_document)
    : OrderSelectionAlgorithm(ant_colony_optimization_document)
{
    from_XML(ant_colony_optimization_document);
}

// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML ant colony optimization file.

AntColonyOptimization::AntColonyOptimization(const string& file_name)
    : OrderSelectionAlgorithm(file_name)
{
    load(file_name);
}

// DESTRUCTOR

/// Destructor.

AntColonyOptimization::~AntColonyOptimization()
{
}

// METHODS

// const size_t& get_maximum_selection_failures() const method

/// Returns the maximum number of selection failures in the model order selection algorithm.

const size_t& AntColonyOptimization::get_maximum_selection_failures() const
{
    return(maximum_selection_failures);
}

// void set_default() method

/// Sets the members of the model selection object to their default values:

void AntColonyOptimization::set_default()
{
    maximum_selection_failures = 10;

    default_activation_function = Perceptron::HyperbolicTangent;

    ants_number = 4;

    evaporation_rate = 3;

    scaling_parameter = 2;

    maximum_layers = 3;

    pheromone_trail.set(maximum_layers);

    const size_t columns_number = maximum_order - minimum_order + 1;

    const size_t squared_columns_number = columns_number*columns_number;

    pheromone_trail[0].set(1, columns_number, 1.0);

    for(size_t i = 1; i < maximum_layers; i++)
    {
        pheromone_trail[i].set(squared_columns_number, columns_number, 1.0);
    }

    architectures.set(ants_number, maximum_layers, minimum_order);

    model_loss.set(ants_number);
}

// void set_maximum_selection_failures(const size_t&) method

/// Sets the maximum selection failures for the ant colony optimization.
/// @param new_maximum_loss_failures Maximum number of selection failures in the ant colony optimization.

void AntColonyOptimization::set_maximum_selection_failures(const size_t& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AntColonyOptimization class.\n"
               << "void set_maximum_selection_failures(const size_t&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}

// Vector<double> perform_minimum_model_evaluation(const Vector<size_t>&) method

/// Returns the minimum of the loss and selection loss in trials_number trainings
/// @param architecture Architecture of the multilayer perceptron to be trained.

Vector<double> AntColonyOptimization::perform_minimum_model_evaluation(const Vector<size_t>& architecture)
{
#ifdef __OPENNN_DEBUG__

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AntColonyOptimization class.\n"
               << "Vector<double> perform_minimum_model_evaluation(Vector<size_t>&) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> final(2, 10);

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for(size_t i = 0; i < architecture_history.size(); i++)
    {
        if(architecture_history[i] == architecture)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < architecture_history.size(); i++)
    {
        if(architecture_history[i] == architecture)
        {
            final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();

    multilayer_perceptron->set(architecture);

    const size_t layers_number = multilayer_perceptron->get_layers_number();

    // Set activation functions

    multilayer_perceptron->set_layer_activation_function(0, default_activation_function);

    for(size_t i = 1; i < layers_number-1; i++)
    {
        multilayer_perceptron->set_layer_activation_function(i, default_activation_function);
    }

    multilayer_perceptron->set_layer_activation_function(layers_number-1, Perceptron::Linear);

    for(size_t i = 0; i < trials_number; i++)
    {

        neural_network->randomize_parameters_normal();
#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losses(training_strategy_results);

        if(!flag_loss && final[0] > current_loss[0])
        {
            final[0] = current_loss[0];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(!flag_selection && final[1] > current_loss[1])
        {
            final[1] = current_loss[1];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(display)
        {
            cout << "Trial number: " << i+1 << endl;
            cout << "Training loss: " << final[0] << endl;
            cout << "Selection loss: " << final[1] << endl;
        }
    }

    architecture_history.push_back(architecture);

    loss_history.push_back(final[0]);

    selection_loss_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


// Vector<double> perform_maximum_model_evaluation(const Vector<size_t>&) const method

/// Returns the maximum of the loss and selection loss in trials_number trainings
/// @param architecture Architecture of the multilayer perceptron to be trained.

Vector<double> AntColonyOptimization::perform_maximum_model_evaluation(const Vector<size_t>& architecture)
{
#ifdef __OPENNN_DEBUG__

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AntColonyOptimization class.\n"
               << "Vector<double> perform_maximum_model_evaluation(Vector<size_t>&) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> final(2);
    final[0] = 0;
    final[1] = 0;

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for(size_t i = 0; i < architecture_history.size(); i++)
    {
        if(architecture_history[i] == architecture)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }



    for(size_t i = 0; i < architecture_history.size(); i++)
    {
        if(architecture_history[i] == architecture)
        {
            final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }


    if(flag_loss && flag_selection)
    {
        return(final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();

    multilayer_perceptron->set(architecture);

    const size_t layers_number = multilayer_perceptron->get_layers_number();

    // Set activation functions

    multilayer_perceptron->set_layer_activation_function(0, default_activation_function);

    for(size_t i = 1; i < layers_number-1; i++)
    {
        multilayer_perceptron->set_layer_activation_function(i, default_activation_function);
    }

    multilayer_perceptron->set_layer_activation_function(layers_number-1, Perceptron::Linear);

    for(size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            cout << "Trial number: " << i << endl;
            cout << "Training loss: " << final[0] << endl;
            cout << "Selection loss: " << final[1] << endl;
        }

        neural_network->randomize_parameters_normal();

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losses(training_strategy_results);

        if(!flag_loss && final[0] < current_loss[0])
        {
            final[0] = current_loss[0];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(!flag_selection && final[1] < current_loss[1])
        {
            final[1] = current_loss[1];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(i == trials_number - 1 && display)
        {
            cout << "Trial number: " << trials_number << endl;
            cout << "Training loss: " << final[0] << endl;
            cout << "Selection loss: " << final[1] << endl;
        }

    }

    architecture_history.push_back(architecture);

    loss_history.push_back(final[0]);

    selection_loss_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


// Vector<double> perform_mean_model_evaluation(const Vector<size_t>&) method

/// Returns the mean of the loss and selection loss in trials_number trainings
/// @param architecture Architecture of the multilayer perceptron to be trained.

Vector<double> AntColonyOptimization::perform_mean_model_evaluation(const Vector<size_t>& architecture)
{
#ifdef __OPENNN_DEBUG__

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AntColonyOptimization class.\n"
               << "Vector<double> perform_mean_model_evaluation(Vector<size_t>&) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> mean_final(2);
    mean_final[0] = 0;
    mean_final[1] = 0;

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;


    for(size_t i = 0; i < architecture_history.size(); i++)
    {
        if(architecture_history[i] == architecture)
        {
            mean_final[0] = loss_history[i];
            flag_loss = true;
        }
    }



    for(size_t i = 0; i < architecture_history.size(); i++)
    {
        if(architecture_history[i] == architecture)
        {
            mean_final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }


    if(flag_loss && flag_selection)
    {
        return(mean_final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();

    multilayer_perceptron->set(architecture);

    const size_t layers_number = multilayer_perceptron->get_layers_number();

    // Set activation functions

    multilayer_perceptron->set_layer_activation_function(0, default_activation_function);

    for(size_t i = 1; i < layers_number-1; i++)
    {
        multilayer_perceptron->set_layer_activation_function(i, default_activation_function);
    }

    multilayer_perceptron->set_layer_activation_function(layers_number-1, Perceptron::Linear);

    for(size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            cout << "Trial number: " << i << endl;
            cout << "Training loss: " << mean_final[0] << endl;
            cout << "Selection loss: " << mean_final[1] << endl;
        }

        neural_network->randomize_parameters_normal();

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losses(training_strategy_results);

        if(!flag_loss)
        {
            mean_final[0] += current_loss[0]/trials_number;
        }

        if(!flag_selection)
        {
            mean_final[1] += current_loss[1]/trials_number;
        }

        if(i == trials_number - 1 && display)
        {
            cout << "Trial number: " << trials_number << endl;
            cout << "Training loss: " << mean_final[0] << endl;
            cout << "Selection loss: " << mean_final[1] << endl;
        }

    }

    architecture_history.push_back(architecture);

    loss_history.push_back(mean_final[0]);

    selection_loss_history.push_back(mean_final[1]);

    parameters_history.push_back(final_parameters);

    return mean_final;
}

// Vector<double> perform_model_evaluation(const Vector<size_t>&) method

/// Return loss and selection depending on the loss calculation method.
/// @param architecture Architecture of the multilayer perceptron to be trained.

Vector<double> AntColonyOptimization::perform_model_evaluation(const Vector<size_t>& architecture)
{
    switch(loss_calculation_method)
    {
    case Maximum:
    {
        return(perform_maximum_model_evaluation(architecture));
    }
    case Minimum:
    {
        return(perform_minimum_model_evaluation(architecture));
    }
    case Mean:
    {
        return(perform_mean_model_evaluation(architecture));
    }
    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_model_evaluation(const Vector<size_t>&) method.\n"
               << "Unknown loss calculation method.\n";

        throw logic_error(buffer.str());
    }
    }
}


// void chose_paths() method

/// Chose the path that each ant will follow.

void AntColonyOptimization::chose_paths()
{
    double random;

    architectures.initialize(0);

    for(size_t i = 0; i < ants_number; i++)
    {
        Vector<double> initial_trial = pheromone_trail[0].get_row(0);

        Vector<double> initial_trial_sum = initial_trial.calculate_cumulative();

        double sum = initial_trial.calculate_sum();

        random = calculate_random_uniform(0.,sum);

        size_t selected_order;

        for(size_t k = 0; k < initial_trial_sum.size(); k++)
        {
            if(k == 0 && random < initial_trial_sum[0])
            {
                selected_order = k;
                k = initial_trial_sum.size();
            }
            else if(random < initial_trial_sum[k] && random >= initial_trial_sum[k-1])
            {
                selected_order = k;
                k = initial_trial_sum.size();
            }
        }

        architectures(i,0) = selected_order + minimum_order;

        for(size_t j = 1; j < maximum_layers; j++)
        {
            size_t previous_order = architectures(i, j-1);

            Vector<double> trial = pheromone_trail[0].get_row(previous_order - minimum_order);

            Vector<double> trial_sum = trial.calculate_cumulative();

            double sum = trial.calculate_sum();

            random = calculate_random_uniform(0.,sum);

            size_t selected_order;

            for(size_t k = 0; k < trial_sum.size(); k++)
            {
                if(k == 0 && random < trial_sum[0])
                {
                    selected_order = k;
                    k = trial_sum.size();
                }
                else if(random < trial_sum[k] && random >= trial_sum[k-1])
                {
                    selected_order = k;
                    k = trial_sum.size();
                }
            }

            architectures(i,j) = selected_order + minimum_order;
        }
    }
}

// void evaluate_ants() method

/// Evaluate the model chosen by each ant.

void AntColonyOptimization::evaluate_ants()
{
    for(size_t i = 0; i < ants_number; i++)
    {

    }
}

// AntColonyOptimizationResults* perform_order_selection() method

/// Perform the order selection with the Incremental method.

AntColonyOptimization::AntColonyOptimizationResults* AntColonyOptimization::perform_order_selection()
{
    AntColonyOptimizationResults* results = new AntColonyOptimizationResults();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    Vector<double> loss(2);
    double prev_selection_loss = 1.0e99;

    size_t optimal_order;
    Vector<double> optimum_parameters;
    double optimum_training_loss;
    double optimum_selection_loss;

    Vector<double> parameters_history_row;
    double current_training_loss, current_selection_loss;

    size_t order = minimum_order;
    size_t iterations = 0;
    size_t selection_failures = 0;

    bool end = false;

    time_t beginning_time, current_time;
    double elapsed_time;

    if(display)
    {
        cout << "Performing ant colony optimization..." << endl;
        cout.flush();
    }

    time(&beginning_time);

    while(!end)
    {
        current_training_loss = loss[0];
        current_selection_loss = loss[1];

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        results->order_data.push_back(order);

        if(reserve_loss_data)
        {
            results->loss_data.push_back(current_training_loss);
        }

        if(reserve_selection_loss_data)
        {
            results->selection_loss_data.push_back(current_selection_loss);
        }

        if(reserve_parameters_data)
        {
            parameters_history_row = get_parameters_order(order);
            results->parameters_data.push_back(parameters_history_row);
        }

        if(iterations == 0
        ||(optimum_selection_loss > current_selection_loss
        && fabs(optimum_selection_loss - current_selection_loss) > tolerance))
        {
            optimal_order = order;
            optimum_training_loss = current_training_loss;
            optimum_selection_loss = current_selection_loss;
            optimum_parameters = get_parameters_order(optimal_order);

        }
        else if(prev_selection_loss < current_selection_loss)
        {
            selection_failures++;
        }

        prev_selection_loss = current_selection_loss;
        iterations++;

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display)
            {
                cout << "Maximum time reached." << endl;
            }

            results->stopping_condition = AntColonyOptimization::MaximumTime;
        }
        else if(loss[1] <= selection_loss_goal)
        {
            end = true;

            if(display)
            {
                cout << "Selection loss reached." << endl;
            }

            results->stopping_condition = AntColonyOptimization::SelectionLossGoal;

        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                cout << "Maximum number of iterations reached." << endl;
            }

            results->stopping_condition = AntColonyOptimization::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display)
            {
                cout << "Maximum selection failures("<<selection_failures<<") reached." << endl;
            }

            results->stopping_condition = AntColonyOptimization::MaximumSelectionFailures;
        }
        else if(order == maximum_order)
        {
            end = true;

            if(display)
            {
                cout << "Algorithm finished" << endl;
            }

            results->stopping_condition = AntColonyOptimization::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iterations << endl
                      << "Hidden neurons number: " << order << endl
                      << "Training loss: " << loss[0] << endl
                      << "Selection loss: " << loss[1] << endl
                      << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
        }
    }

    if(display)
    {
        cout << endl
                  << "Optimal order: " << optimal_order <<  endl
                  << "Optimum selection loss: " << optimum_selection_loss << endl
                  << "Corresponding training loss: " << optimum_training_loss << endl;
    }

    const size_t last_hidden_layer = multilayer_perceptron_pointer->get_layers_number()-2;
    const size_t perceptrons_number = multilayer_perceptron_pointer->get_layer_pointer(last_hidden_layer)->get_perceptrons_number();

    if(optimal_order > perceptrons_number)
    {
        multilayer_perceptron_pointer->grow_layer_perceptron(last_hidden_layer,optimal_order-perceptrons_number);
    }
    else
    {
        for(size_t i = 0; i <(perceptrons_number-optimal_order); i++)
        {
            multilayer_perceptron_pointer->prune_layer_perceptron(last_hidden_layer,0);
        }
    }

    multilayer_perceptron_pointer->set_parameters(optimum_parameters);

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimum_parameters;
    }

    results->optimal_order = optimal_order;
    results->final_selection_loss = optimum_selection_loss;
//    results->final_loss = perform_model_evaluation(optimal_order)[0];
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    return(results);
}

// Matrix<string> to_string_matrix() const method

/// Writes as matrix of strings the most representative atributes.

Matrix<string> AntColonyOptimization::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

   // Minimum order

   labels.push_back("Minimum order");

   buffer.str("");
   buffer << minimum_order;

   values.push_back(buffer.str());

   // Maximum order

   labels.push_back("Maximum order");

   buffer.str("");
   buffer << maximum_order;

   values.push_back(buffer.str());

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

   // Selection loss goal

   labels.push_back("Selection loss goal");

   buffer.str("");
   buffer << selection_loss_goal;

   values.push_back(buffer.str());

   // Maximum selection failures

   labels.push_back("Maximum selection failures");

   buffer.str("");
   buffer << maximum_selection_failures;

   values.push_back(buffer.str());

   // Maximum iterations number

   labels.push_back("Maximum iterations number");

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

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML() const method

/// Prints to the screen the ant colony optimization parameters, the stopping criteria
/// and other user stuff concerning the ant colony optimization object.

tinyxml2::XMLDocument* AntColonyOptimization::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("AntColonyOptimization");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Minimum order
   {
   element = document->NewElement("MinimumOrder");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_order;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum order
   {
   element = document->NewElement("MaximumOrder");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_order;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Parameters assays number
   {
   element = document->NewElement("TrialsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << trials_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Performance calculation method
//   {
//   element = document->NewElement("PerformanceCalculationMethod");
//   root_element->LinkEndChild(element);

//   text = document->NewText(write_loss_calculation_method().c_str());
//   element->LinkEndChild(text);
//   }

   // Reserve parameters data
//   {
//   element = document->NewElement("ReserveParametersData");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_parameters_data;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Reserve minimal parameters
//   {
//   element = document->NewElement("ReserveMinimalParameters");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_minimal_parameters;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Display
//   {
//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Tolerance
   {
   element = document->NewElement("Tolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << tolerance;

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
//   {
//   element = document->NewElement("MaximumIterationsNumber");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << maximum_iterations_number;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Maximum selection failures
   {
   element = document->NewElement("MaximumSelectionFailures");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_failures;

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

   // Reserve loss data
   {
   element = document->NewElement("ReservePerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_loss_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection loss data
   {
   element = document->NewElement("ReserveSelectionLossHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_loss_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Prints to the screen the ant colony optimization parameters, the stopping criteria
/// and other user stuff concerning the ant colony optimization object.

void AntColonyOptimization::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("AntColonyOptimization");

    // Minimum order

    file_stream.OpenElement("MinimumOrder");

    buffer.str("");
    buffer << minimum_order;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum order

    file_stream.OpenElement("MaximumOrder");

    buffer.str("");
    buffer << maximum_order;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Parameters assays number

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

    // selection loss goal

    file_stream.OpenElement("SelectionLossGoal");

    buffer.str("");
    buffer << selection_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection failures

    file_stream.OpenElement("MaximumSelectionFailures");

    buffer.str("");
    buffer << maximum_selection_failures;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve loss data

    file_stream.OpenElement("ReservePerformanceHistory");

    buffer.str("");
    buffer << reserve_loss_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection loss data

    file_stream.OpenElement("ReserveSelectionLossHistory");

    buffer.str("");
    buffer << reserve_selection_loss_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this ant colony optimization object.
/// @param document TinyXML document containing the member data.

void AntColonyOptimization::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("AntColonyOptimization");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AntColonyOptimization class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "AntColonyOptimization element is NULL.\n";

        throw logic_error(buffer.str());
    }

    // Minimum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumOrder");

        if(element)
        {
           const size_t new_minimum_order = atoi(element->GetText());

           try
           {
              minimum_order = new_minimum_order;
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Maximum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumOrder");

        if(element)
        {
           const size_t new_maximum_order = atoi(element->GetText());

           try
           {
              maximum_order = new_maximum_order;
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Parameters assays number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
           const size_t new_trials_number = atoi(element->GetText());

           try
           {
              set_trials_number(new_trials_number);
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Performance calculation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PerformanceCalculationMethod");

        if(element)
        {
           const string new_loss_calculation_method = element->GetText();

           try
           {
              set_loss_calculation_method(new_loss_calculation_method);
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Reserve parameters data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersData");

        if(element)
        {
           const string new_reserve_parameters_data = element->GetText();

           try
           {
              set_reserve_parameters_data(new_reserve_parameters_data != "0");
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

        if(element)
        {
           const string new_reserve_loss_data = element->GetText();

           try
           {
              set_reserve_loss_data(new_reserve_loss_data != "0");
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Reserve selection loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

        if(element)
        {
           const string new_reserve_selection_loss_data = element->GetText();

           try
           {
              set_reserve_selection_loss_data(new_reserve_selection_loss_data != "0");
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
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
              cout << e.what() << endl;
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
              cout << e.what() << endl;
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
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }

    // Maximum iterations number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumIterationsNumber");

        if(element)
        {
           const size_t new_maximum_iterations_number = atoi(element->GetText());

           try
           {
              set_maximum_iterations_number(new_maximum_iterations_number);
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
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
              cout << e.what() << endl;
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
              cout << e.what() << endl;
           }
        }
    }

    // Maximum selection failures
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionFailures");

        if(element)
        {
           const size_t new_maximum_selection_failures = atoi(element->GetText());

           try
           {
              set_maximum_selection_failures(new_maximum_selection_failures);
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
    }
}

// void save(const string&) const method

/// Saves to a XML-type file the members of the ant colony optimization object.
/// @param file_name Name of ant colony optimization XML-type file.

void AntColonyOptimization::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads a ant colony optimization object from a XML-type file.
/// @param file_name Name of ant colony optimization XML-type file.

void AntColonyOptimization::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: AntColonyOptimization class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
}



}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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

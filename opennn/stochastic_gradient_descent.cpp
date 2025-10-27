//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "neural_network.h"
#include "loss_index.h"
#include "stochastic_gradient_descent.h"

namespace opennn
{

StochasticGradientDescent::StochasticGradientDescent(const LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


const type& StochasticGradientDescent::get_initial_learning_rate() const
{
    return initial_learning_rate;
}


const type& StochasticGradientDescent::get_initial_decay() const
{
    return initial_decay;
}


const type& StochasticGradientDescent::get_momentum() const
{
    return momentum;
}


const bool& StochasticGradientDescent::get_nesterov() const
{
    return nesterov;
}


const type& StochasticGradientDescent::get_loss_goal() const
{
    return training_loss_goal;
}


void StochasticGradientDescent::set_default()
{
    name = "StochasticGradientDescent";

    // TRAINING OPERATORS

    initial_learning_rate = type(0.001);
    initial_decay = type(0.001);
    momentum = type(0);
    nesterov = false;

    // Stopping criteria

    training_loss_goal = type(0);
    maximum_time = type(3600);
    maximum_epochs_number = 1000;

    // UTILITIES

    display_period = 100;
}


void StochasticGradientDescent::set_batch_size(const Index& new_batch_size)
{
    batch_size = new_batch_size;
}


Index StochasticGradientDescent::get_samples_number() const
{
    return batch_size;
}


void StochasticGradientDescent::set_initial_learning_rate(const type& new_learning_rate)
{
    initial_learning_rate = new_learning_rate;
}


void StochasticGradientDescent::set_initial_decay(const type& new_decay)
{
    initial_decay = new_decay;
}


void StochasticGradientDescent::set_momentum(const type& new_momentum)
{
    momentum = new_momentum;
}


void StochasticGradientDescent::set_nesterov(const bool& new_nesterov_momentum)
{
    nesterov = new_nesterov_momentum;
}


void StochasticGradientDescent::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void StochasticGradientDescent::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void StochasticGradientDescent::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  StochasticGradientDescentData& optimization_data,
                                                  const type& learning_rate) const
{
    NeuralNetwork* neural_network = loss_index->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        Layer* layer = neural_network->get_layer(i).get();

        if (!layer->get_is_trainable())
            continue;

        LayerBackPropagation* layer_back_propagation = back_propagation.neural_network.layers[i].get();

        const vector<ParameterView> layer_parameter_pairs = layer->get_parameter_views();
        const vector<ParameterView> layer_parameter_delta_pairs = layer_back_propagation->get_parameter_delta_views();

        // #pragma omp parallel for #@todo check pragma vs thread_pool_device
        for(Index j = 0; j < Index(layer_parameter_pairs.size()); j++)
        {
            type* parameter_data = layer_parameter_pairs[j].data;
            const Index parameter_size = layer_parameter_pairs[j].size;
            type* delta_data = layer_parameter_delta_pairs[j].data;

            TensorMap<Tensor<type, 1>> parameters(parameter_data, parameter_size);
            TensorMap<Tensor<type, 1>> gradient(delta_data, parameter_size);

            Tensor<type, 1>& parameters_increment = optimization_data.parameters_increment[i][j];
            Tensor<type, 1>& last_parameters_increment = optimization_data.last_parameters_increment[i][j];

            if (momentum <= type(0))
            {
                parameters_increment.device(*thread_pool_device) = gradient * (-learning_rate);
                parameters.device(*thread_pool_device) += parameters_increment;
            }
            else if (momentum > type(0) && !nesterov)
            {
                parameters_increment.device(*thread_pool_device) =
                    gradient * (-learning_rate) + momentum * last_parameters_increment;
                last_parameters_increment.device(*thread_pool_device) = parameters_increment;
                parameters.device(*thread_pool_device) += parameters_increment;
            }
            else if (momentum > type(0) && nesterov)
            {
                parameters_increment.device(*thread_pool_device)
                    = gradient * (-learning_rate) + momentum * last_parameters_increment;
                last_parameters_increment.device(*thread_pool_device) = parameters_increment;
                parameters.device(*thread_pool_device) += parameters_increment * momentum - gradient * learning_rate;
            }
        }
    }
}


TrainingResults StochasticGradientDescent::train()
{
    if (!loss_index || !loss_index->has_neural_network() || !loss_index->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs_number+1);

    check();

    // Start training

    if(display) cout << "Training with stochastic gradient descent (SGD)...\n";

    // Dataset

    Dataset* dataset = loss_index->get_dataset();

    const bool has_selection = dataset->has_selection();

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    // const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    const vector<Index> training_samples_indices = dataset->get_sample_indices("Training");
    const vector<Index> selection_samples_indices = dataset->get_sample_indices("Selection");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index selection_samples_number = dataset->get_samples_number("Selection");

    const Index training_batch_samples_number = min(training_samples_number, batch_size);

    const Index selection_batch_samples_number = (selection_samples_number != 0)
                                                     ? min(selection_samples_number, batch_size)
                                                     : 0;

    Batch training_batch(training_batch_samples_number, dataset);
    Batch selection_batch(selection_batch_samples_number, dataset);

    const Index training_batches_number = (training_batch_samples_number != 0)
                                              ? training_samples_number / training_batch_samples_number
                                              : 0;

    const Index selection_batches_number = (selection_batch_samples_number != 0)
                                               ? selection_samples_number / selection_batch_samples_number
                                               : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> selection_batches(selection_batches_number);

    vector<Index> training_batch_indices(training_batch_samples_number);
    vector<Index> selection_batch_indices(training_batch_samples_number);

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();

    set_scaling();

    set_vocabularies();

    ForwardPropagation training_forward_propagation(training_batch_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_batch_samples_number, neural_network);

    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_batch_samples_number, loss_index);

    //type training_loss = type(0);
    type training_error = type(0);
    type selection_error = type(0);

    Index selection_failures = 0;

    // Optimization algorithm

    StochasticGradientDescentData optimization_data(this);

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    bool shuffle = !neural_network->has("Recurrent");

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_samples_indices, training_batch_samples_number, shuffle);

        const Index batches_number = training_batches.size();

        const type current_learning_rate = initial_learning_rate / (type(1) + type(epoch) * initial_decay);

        //training_loss = type(0);
        training_error = type(0);

        optimization_data.iteration = 0;

        for(Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;

            // Dataset

            training_batch.fill(training_batches[iteration],
                                input_variable_indices,
                                // decoder_variable_indices,
                                target_variable_indices);

            // Neural network

            neural_network->forward_propagate(training_batch.get_input_pairs(),
                                              training_forward_propagation,
                                              is_training);

            // Loss index

            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            loss_index->add_regularization_to_deltas(training_back_propagation);

            results.training_error_history(epoch) = training_back_propagation.error();

            training_error += training_back_propagation.error();
            //training_loss += training_back_propagation.loss;

            // Gradient

            update_parameters(training_back_propagation, optimization_data, current_learning_rate);

            //if(display && epoch % display_period == 0)      display_progress_bar(iteration, batches_number - 1);
        }


        // Loss

        training_error /= type(batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_selection)
        {
            selection_batches = dataset->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);

            selection_error = type(0);

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Dataset

                selection_batch.fill(selection_batches[iteration],
                                     input_variable_indices,
                                     // decoder_variable_indices,
                                     target_variable_indices);

                // Neural network

                neural_network->forward_propagate(selection_batch.get_input_pairs(),
                                                  selection_forward_propagation,
                                                  false);

                // Loss

                loss_index->calculate_error(selection_batch,
                                            selection_forward_propagation,
                                            selection_back_propagation);

                selection_error += selection_back_propagation.error();
            }

            selection_error /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if(has_selection) cout << "Selection error: " << selection_error << endl<<endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping criteria

        stop_training = true;

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }
        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;
            results.stopping_condition = StoppingCondition::MaximumTime;
        }
        else if(results.training_error_history(epoch) < training_loss_goal)
        {
            if (display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
            results.stopping_condition  = StoppingCondition::LossGoal;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else
        {
            stop_training = false;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;
            results.selection_failures = selection_failures;
            results.elapsed_time = write_time(elapsed_time);

            results.resize_training_error_history(epoch+1);
            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            break;
        }

        // Update stuff

        if(epoch != 0 && epoch%save_period == 0)
            neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    if(display) results.print();

    return results;
}


Tensor<string, 2> StochasticGradientDescent::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(7, 2);

    const string apply_momentum = momentum > type(0)
                                      ? "true"
                                      : "false";

    string_matrix.setValues({{"Inital learning rate", to_string(double(initial_learning_rate))},
                             {"Inital decay", to_string(double(initial_decay))},
                             {"Apply momentum", apply_momentum},
                             {"Training loss goal", to_string(double(training_loss_goal))},
                             {"Maximum epochs number", to_string(maximum_epochs_number)},
                             {"Maximum time", write_time(maximum_time)},
                             {"Batch samples number", to_string(batch_size)}});

    return string_matrix;
}


void StochasticGradientDescent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("StochasticGradientDescent");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    add_xml_element(printer, "ApplyMomentum", to_string(momentum > type(0)));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));
    add_xml_element(printer, "HardwareUse", hardware_use);

    printer.CloseElement();
}


void StochasticGradientDescent::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("StochasticGradientDescent");

    if(!root_element)
        throw runtime_error("Stochastic gradient descent element is nullptr.\n");

    set_batch_size(read_xml_index(root_element, "BatchSize"));

    const bool apply_momentum = read_xml_bool(root_element, "ApplyMomentum");
    set_momentum(apply_momentum ? type(0.9) : type(0));

    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
    set_hardware_use(read_xml_string(root_element, "HardwareUse"));
}


StochasticGradientDescentData::StochasticGradientDescentData(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    set(new_stochastic_gradient_descent);
}


void StochasticGradientDescentData::set(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    stochastic_gradient_descent = new_stochastic_gradient_descent;

    const LossIndex* loss_index = stochastic_gradient_descent->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index layers_number = neural_network->get_layers_number();

    parameters_increment.resize(layers_number);

    parameters_increment.resize(layers_number);
    last_parameters_increment.resize(layers_number);

# pragma omp parallel for
    for(Index i = 0; i < layers_number; i++)
    {
        Layer* layer = neural_network->get_layer(i).get();
        const vector<ParameterView> parameter_views = layer->get_parameter_views();

        parameters_increment[i].resize(parameter_views.size());
        last_parameters_increment[i].resize(parameter_views.size());

        for(Index j = 0; j < (Index)parameter_views.size(); j++)
        {
            const Index size = parameter_views[j].size;

            parameters_increment[i][j].resize(array<Index, 1>{size});
            last_parameters_increment[i][j].resize(array<Index, 1>{size});

            parameters_increment[i][j].setZero();
            last_parameters_increment[i][j].setZero();
        }
    }
}


#ifdef OPENNN_CUDA

TrainingResults StochasticGradientDescent::train_cuda()
{
    if (!loss_index || !loss_index->has_neural_network() || !loss_index->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs_number + 1);

    check();

    // Start training

    if (display) cout << "Training with stochastic gradient descent (SGD) CUDA...\n";

    // Dataset

    Dataset* dataset = loss_index->get_dataset();

    const bool has_selection = dataset->has_selection();

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    //const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    const vector<Index> training_samples_indices = dataset->get_sample_indices("Training");
    const vector<Index> selection_samples_indices = dataset->get_sample_indices("Selection");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index selection_samples_number = dataset->get_samples_number("Selection");

    const Index training_batch_samples_number = min(training_samples_number, batch_size);

    const Index selection_batch_samples_number = (selection_samples_number != 0)
                                                     ? min(selection_samples_number, batch_size)
                                                     : 0;

    BatchCuda training_batch_cuda(training_batch_samples_number, dataset);
    unique_ptr<BatchCuda> selection_batch_cuda;

    const Index training_batches_number = (training_batch_samples_number != 0)
                                              ? training_samples_number / training_batch_samples_number
                                              : 0;

    const Index selection_batches_number = (selection_batch_samples_number != 0)
                                               ? selection_samples_number / selection_batch_samples_number
                                               : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> selection_batches(selection_batches_number);

    vector<Index> training_batch_indices(training_batch_samples_number);
    vector<Index> selection_batch_indices(training_batch_samples_number);

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();

    set_scaling();

    set_vocabularies();

    ForwardPropagationCuda training_forward_propagation_cuda(training_batch_samples_number, neural_network);
    unique_ptr<ForwardPropagationCuda> selection_forward_propagation_cuda;

    neural_network->allocate_parameters_device();
    neural_network->copy_parameters_device();

    // Loss Index

    loss_index->set_normalization_coefficient();

    BackPropagationCuda training_back_propagation_cuda(training_batch_samples_number, loss_index);
    unique_ptr<BackPropagationCuda> selection_back_propagation_cuda;

    if (has_selection)
    {
        selection_batch_cuda = make_unique<BatchCuda>(selection_batch_samples_number, dataset);
        selection_forward_propagation_cuda = make_unique<ForwardPropagationCuda>(selection_batch_samples_number, neural_network);
        selection_back_propagation_cuda = make_unique<BackPropagationCuda>(selection_batch_samples_number, loss_index);
    }

    //type training_loss = type(0);
    type training_error = type(0);
    type selection_error = type(0);

    Index selection_failures = 0;

    // Optimization algorithm

    SGDOptimizationDataCuda optimization_data(this);

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    bool shuffle = false;

    if (neural_network->has("Recurrent"))
        shuffle = false;

    // Main loop

    for (Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if (display && epoch % display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_samples_indices, training_batch_samples_number, shuffle);

        const Index batches_number = training_batches.size();

        //training_loss = type(0);
        training_error = type(0);

        optimization_data.iteration = 0;

        for (Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;

            // Dataset

            training_batch_cuda.fill(training_batches[iteration],
                                     input_variable_indices,
                                     //decoder_variable_indices,
                                     target_variable_indices);

            // Neural network

            neural_network->forward_propagate_cuda(training_batch_cuda.get_input_device(),
                                                   training_forward_propagation_cuda,
                                                   is_training);

            // Loss index

            loss_index->back_propagate_cuda(training_batch_cuda,
                                            training_forward_propagation_cuda,
                                            training_back_propagation_cuda);

            results.training_error_history(epoch) = training_back_propagation_cuda.error();

            training_error += training_back_propagation_cuda.error();
            //training_loss += training_back_propagation.loss;

            // Gradient

            update_parameters_cuda(training_back_propagation_cuda, optimization_data);

            //if(display && epoch % display_period == 0)      display_progress_bar(iteration, batches_number - 1);
        }


        // Loss

        //training_loss /= type(batches_number);
        training_error /= type(batches_number);

        results.training_error_history(epoch) = training_error;

        if (has_selection)
        {
            selection_batches = dataset->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);

            selection_error = type(0);

            for (Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Dataset

                selection_batch_cuda->fill(selection_batches[iteration],
                                           input_variable_indices,
                                           //decoder_variable_indices,
                                           target_variable_indices);

                // Neural network

                neural_network->forward_propagate_cuda(selection_batch_cuda->get_input_device(),
                                                       *selection_forward_propagation_cuda,
                                                       is_training);

                results.selection_error_history(epoch) = selection_error;

                // Loss

                loss_index->calculate_error_cuda(*selection_batch_cuda,
                                                 *selection_forward_propagation_cuda,
                                                 *selection_back_propagation_cuda);

                selection_error += selection_back_propagation_cuda->error();
            }

            selection_error /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if (epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch - 1)) selection_failures++;
        }

        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if (display && epoch % display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if (has_selection) cout << "Selection error: " << selection_error << endl << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping criteria

        stop_training = true;

        if (epoch == maximum_epochs_number)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }
        else if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;
            results.stopping_condition = StoppingCondition::MaximumTime;
        }
        else if (results.training_error_history(epoch) < training_loss_goal)
        {
            if (display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
            results.stopping_condition = StoppingCondition::LossGoal;
        }
        else if (selection_failures >= maximum_selection_failures)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else
        {
            stop_training = false;
        }

        if (stop_training)
        {
            results.loss = training_back_propagation_cuda.loss;
            results.selection_failures = selection_failures;
            results.elapsed_time = write_time(elapsed_time);

            results.resize_training_error_history(epoch + 1);
            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            break;
        }
    }

    set_unscaling();

    neural_network->copy_parameters_host();
    neural_network->free_parameters_device();

    if (display) results.print();

    return results;
}


void StochasticGradientDescent::update_parameters_cuda(BackPropagationCuda& back_propagation_cuda,
                                                       SGDOptimizationDataCuda& optimization_data_cuda) const
{
    NeuralNetwork* neural_network = back_propagation_cuda.loss_index->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    const float current_learning_rate = static_cast<float>(initial_learning_rate / (1.0 + static_cast<double>(optimization_data_cuda.iteration) * initial_decay));
    const float momentum_f = static_cast<float>(momentum);

    for (Index layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        Layer* layer = neural_network->get_layer(layer_index).get();

        if (!layer->get_is_trainable())
            continue;

        const vector<ParameterView> parameter_views = layer->get_parameter_views_device();
        LayerBackPropagationCuda* layer_back_prop = back_propagation_cuda.neural_network.layers[layer_index].get();
        const vector<ParameterView> delta_views = layer_back_prop->get_parameter_delta_views_device();

        for (Index parameter_index = 0; parameter_index < Index(parameter_views.size()); ++parameter_index)
        {
            float* params_d = parameter_views[parameter_index].data;
            const Index param_size = parameter_views[parameter_index].size;
            const float* grads_d = delta_views[parameter_index].data;
            float* velocity_d = optimization_data_cuda.velocity[layer_index][parameter_index];

            sgd_update_device(
                param_size,
                params_d,
                velocity_d,
                grads_d,
                current_learning_rate,
                momentum_f,
                nesterov
                );
        }
    }
}


SGDOptimizationDataCuda::SGDOptimizationDataCuda(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    set(new_stochastic_gradient_descent);
}


void SGDOptimizationDataCuda::set(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    stochastic_gradient_descent = new_stochastic_gradient_descent;

    NeuralNetwork* neural_network = stochastic_gradient_descent->get_loss_index()->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    velocity.resize(layers_number);

    for (Index i = 0; i < layers_number; ++i)
    {
        Layer* layer = neural_network->get_layer(i).get();
        if (!layer->get_is_trainable()) continue;

        const auto parameter_views = layer->get_parameter_views_device();
        const size_t param_blocks_count = parameter_views.size();

        velocity[i].resize(param_blocks_count, nullptr);

        for (Index j = 0; j < Index(param_blocks_count); ++j)
        {
            const Index param_size = parameter_views[j].size;
            if (param_size > 0)
            {
                const size_t memory_size_bytes = param_size * sizeof(float);
                CHECK_CUDA(cudaMalloc(&velocity[i][j], memory_size_bytes));
                CHECK_CUDA(cudaMemset(velocity[i][j], 0, memory_size_bytes));
            }
        }
    }
}


void SGDOptimizationDataCuda::free()
{
    for (auto& layer_velocity : velocity)
    {
        for (float*& ptr : layer_velocity)
        {
            if (ptr != nullptr)
            {
                CHECK_CUDA(cudaFree(ptr));
                ptr = nullptr;
            }
        }
    }
    velocity.clear();
}


void SGDOptimizationDataCuda::print() const
{
    cout << "--- SGD Optimization Data (CUDA) ---" << endl;
    NeuralNetwork* neural_network = stochastic_gradient_descent->get_loss_index()->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    for (Index i = 0; i < layers_number; ++i)
    {
        Layer* layer = neural_network->get_layer(i).get();
        if (!layer->get_is_trainable()) continue;

        cout << "Layer " << i << " (" << layer->get_name() << "):" << endl;
        const auto parameter_views = layer->get_parameter_views_device();

        for (Index j = 0; j < Index(parameter_views.size()); ++j)
        {
            const Index param_size = parameter_views[j].size;
            if (param_size == 0) continue;

            cout << "  - Parameter Block " << j << " (Size: " << param_size << "):" << endl;
            cout << "    velocity_host:" << endl << "      ";
            const float* v_device_ptr = velocity[i][j];
            cout << vector_from_device(v_device_ptr, param_size) << endl;
        }
    }
    cout << "------------------------------------" << endl;
}

#endif

REGISTER(OptimizationAlgorithm, StochasticGradientDescent, "StochasticGradientDescent");

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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

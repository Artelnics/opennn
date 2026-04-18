//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "dataset.h"
#include "loss.h"
#include "batch.h"
#include "dense_layer.h"
#include "levenberg_marquardt_algorithm.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void LevenbergMarquardtAlgorithm::set_default()
{
    name = "LevenbergMarquardt";

    // Stopping criteria

    minimum_loss_decrease = type(0);
    training_loss_goal = type(0);
    maximum_validation_failures = 1000;

    maximum_epochs = 1000;
    maximum_time = type(3600);

    // UTILITIES

    display_period = 10;

    // Training parameters

    damping_parameter = type(1.0e-3);

    damping_parameter_factor = type(10.0);

    minimum_damping_parameter = type(1.0e-6);
    maximum_damping_parameter = type(1.0e6);
}

void LevenbergMarquardtAlgorithm::set_damping_parameter(const type new_damping_parameter)
{
    damping_parameter = clamp(new_damping_parameter, minimum_damping_parameter, maximum_damping_parameter);
}

void LevenbergMarquardtAlgorithm::set_damping_parameter_factor(const type new_damping_parameter_factor)
{
    damping_parameter_factor = new_damping_parameter_factor;
}

void LevenbergMarquardtAlgorithm::set_minimum_damping_parameter(const type new_minimum_damping_parameter)
{
    minimum_damping_parameter = new_minimum_damping_parameter;
}

void LevenbergMarquardtAlgorithm::set_maximum_damping_parameter(const type new_maximum_damping_parameter)
{
    maximum_damping_parameter = new_maximum_damping_parameter;
}

void LevenbergMarquardtAlgorithm::set_minimum_loss_decrease(const type new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}

void LevenbergMarquardtAlgorithm::check() const
{
    if(!loss)
        throw runtime_error("LevenbergMarquardtAlgorithm error: loss is not set.");

    const Dataset* dataset = loss->get_dataset();

    if(!dataset)
        throw runtime_error("LevenbergMarquardtAlgorithm error: dataset is not set.");

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if(!neural_network)
        throw runtime_error("LevenbergMarquardtAlgorithm error: neural network is not set.");
}

void LevenbergMarquardtAlgorithm::back_propagate(const Batch& batch,
                                                  const ForwardPropagation& forward_propagation,
                                                  BackPropagationLM& back_propagation_lm)
{
    if(batch.is_empty()) return;

    calculate_errors(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

    calculate_error(batch, forward_propagation, back_propagation_lm);

    compute_jacobian(batch, forward_propagation, back_propagation_lm);

    const MatrixR& J = back_propagation_lm.squared_errors_jacobian;
    const VectorR& e = back_propagation_lm.errors;
    const type factor = type(2) / type(e.size());

    back_propagation_lm.gradient.noalias() = factor * J.transpose() * e;
    back_propagation_lm.hessian.noalias() = factor * J.transpose() * J;

    back_propagation_lm.loss_value = back_propagation_lm.error;
}

void LevenbergMarquardtAlgorithm::calculate_errors(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagationLM& back_propagation_lm) const
{
    const VectorMap output = forward_propagation.get_last_trainable_layer_outputs().as_vector();
    const VectorMap target = batch.get_targets().as_vector();

    back_propagation_lm.errors.noalias() = output - target;
}

void LevenbergMarquardtAlgorithm::calculate_squared_errors(const Batch&,
                                                           const ForwardPropagation&,
                                                           BackPropagationLM& back_propagation_lm) const
{
    back_propagation_lm.squared_errors = back_propagation_lm.errors.array().square();
}

void LevenbergMarquardtAlgorithm::calculate_error(const Batch&,
                                                   const ForwardPropagation&,
                                                   BackPropagationLM& back_propagation_lm) const
{
    const Index size = back_propagation_lm.squared_errors.size();

    back_propagation_lm.error = back_propagation_lm.squared_errors.sum() / type(size);
}

void LevenbergMarquardtAlgorithm::compute_jacobian(const Batch& batch,
                                                   const ForwardPropagation& fp,
                                                   BackPropagationLM& bp_lm)
{

    NeuralNetwork* nn = loss->get_neural_network();
    const auto& layers = nn->get_layers();

    bp_lm.squared_errors_jacobian.setZero();

    Index parameter_offset = 0;

    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (!layers[i]->get_is_trainable()) continue;

        if (auto* dense = dynamic_cast<Dense<2>*>(layers[i].get()))
            insert_dense_jacobian(dense, fp, i, parameter_offset, bp_lm.squared_errors_jacobian);        

        parameter_offset += layers[i]->get_parameters_number();
    }
}

VectorR LevenbergMarquardtAlgorithm::calculate_numerical_gradient()
{
    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices = dataset->get_sample_indices("Training");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, loss);

    VectorR& parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    type h = 0;

    type error_forward = 0;
    type error_backward = 0;

    VectorR numerical_gradient_lm(parameters_number);
    numerical_gradient_lm.setZero();

    for(Index i = 0; i < parameters_number; ++i)
    {
        h = Loss::calculate_h(parameters(i));

        parameters(i) += h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

        calculate_error(batch, forward_propagation, back_propagation_lm);

        error_forward = back_propagation_lm.error;

        parameters(i) -= type(2) * h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

        calculate_error(batch, forward_propagation, back_propagation_lm);

        error_backward = back_propagation_lm.error;

        parameters(i) += h;

        numerical_gradient_lm(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient_lm;
}


MatrixR LevenbergMarquardtAlgorithm::calculate_numerical_jacobian()
{
    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index samples_number = dataset->get_samples_number("Training");
    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, {}, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagationLM back_propagation_lm(samples_number, loss);

    VectorR& parameters = neural_network->get_parameters();
    const Index parameters_number = parameters.size();

    const Index total_error_terms = back_propagation_lm.squared_errors.size();

    type perturbation;

    VectorR error_terms_forward(total_error_terms);
    VectorR error_terms_backward(total_error_terms);

    MatrixR jacobian(total_error_terms, parameters_number);

    for(Index j = 0; j < parameters_number; ++j)
    {
        perturbation = Loss::calculate_h(parameters(j));

        parameters(j) -= perturbation;
        neural_network->forward_propagate(batch.get_inputs(), parameters, forward_propagation);
        calculate_errors(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        error_terms_backward = back_propagation_lm.squared_errors;

        parameters(j) += type(2) * perturbation;
        neural_network->forward_propagate(batch.get_inputs(), parameters, forward_propagation);
        calculate_errors(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        error_terms_forward = back_propagation_lm.squared_errors;

        parameters(j) -= perturbation;

        for(Index i = 0; i < total_error_terms; ++i)
            jacobian(i, j) = (error_terms_forward(i) - error_terms_backward(i)) / (type(2.0) * perturbation);
    }

    return jacobian;
}

MatrixR LevenbergMarquardtAlgorithm::calculate_numerical_hessian()
{
    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, {}, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, loss);

    VectorR& parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    neural_network->forward_propagate(batch.get_inputs(),
                                      parameters,
                                      forward_propagation);

    calculate_errors(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

    calculate_error(batch, forward_propagation, back_propagation_lm);

    const type y = back_propagation_lm.error;

    MatrixR H(parameters_number, parameters_number);
    H.setZero();

    type h_i;
    type h_j;

    type y_backward_2i;
    type y_backward_i;

    type y_forward_i;
    type y_forward_2i;

    type y_backward_ij;
    type y_forward_ij;

    type y_backward_i_forward_j;
    type y_forward_i_backward_j;

    for(Index i = 0; i < parameters_number; ++i)
    {
        h_i = Loss::calculate_h(parameters(i));

        // parameters(i) - 2*h_i
        parameters(i) -= type(2) * h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        calculate_error(batch, forward_propagation, back_propagation_lm);

        y_backward_2i = back_propagation_lm.error;

        // parameters(i) - h_i
        parameters(i) += h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        calculate_error(batch, forward_propagation, back_propagation_lm);

        y_backward_i = back_propagation_lm.error;

        // parameters(i) + h_i
        parameters(i) += type(2) * h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        calculate_error(batch, forward_propagation, back_propagation_lm);

        y_forward_i = back_propagation_lm.error;

        // parameters(i) + 2*h_i
        parameters(i) += h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        calculate_error(batch, forward_propagation, back_propagation_lm);

        y_forward_2i = back_propagation_lm.error;

        // restore parameters(i)
        parameters(i) -= type(2) * h_i;

        H(i, i) = (-y_forward_2i + type(16.0) * y_forward_i - type(30.0) * y + type(16.0) * y_backward_i - y_backward_2i) / (type(12.0) * h_i * h_i);

        for(Index j = i; j < parameters_number; ++j)
        {
            h_j = Loss::calculate_h(parameters(j));

            // parameters(i) - h_i, parameters(j) - h_j
            parameters(i) -= h_i;
            parameters(j) -= h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              parameters,
                                              forward_propagation);

            calculate_errors(batch, forward_propagation, back_propagation_lm);
            calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
            calculate_error(batch, forward_propagation, back_propagation_lm);

            y_backward_ij = back_propagation_lm.error;

            // parameters(i) + h_i, parameters(j) + h_j
            parameters(i) += type(2) * h_i;
            parameters(j) += type(2) * h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              parameters,
                                              forward_propagation);

            calculate_errors(batch, forward_propagation, back_propagation_lm);
            calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
            calculate_error(batch, forward_propagation, back_propagation_lm);

            y_forward_ij = back_propagation_lm.error;

            // parameters(i) - h_i, parameters(j) + h_j
            parameters(i) -= type(2) * h_i;

            neural_network->forward_propagate(batch.get_inputs(),
                                              parameters,
                                              forward_propagation);

            calculate_errors(batch, forward_propagation, back_propagation_lm);
            calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
            calculate_error(batch, forward_propagation, back_propagation_lm);

            y_backward_i_forward_j = back_propagation_lm.error;

            // parameters(i) + h_i, parameters(j) - h_j
            parameters(i) += type(2) * h_i;
            parameters(j) -= type(2) * h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              parameters,
                                              forward_propagation);

            calculate_errors(batch, forward_propagation, back_propagation_lm);
            calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
            calculate_error(batch, forward_propagation, back_propagation_lm);

            y_forward_i_backward_j = back_propagation_lm.error;

            // restore parameters(i) and parameters(j)
            parameters(i) -= h_i;
            parameters(j) += h_j;

            H(i, j) = (y_forward_ij - y_forward_i_backward_j - y_backward_i_forward_j + y_backward_ij) / (type(4.0) * h_i * h_j);
        }
    }

    for(Index i = 0; i < parameters_number; ++i)
        for(Index j = 0; j < i; ++j)
            H(i, j) = H(j, i);

    return H;
}

void LevenbergMarquardtAlgorithm::insert_dense_jacobian(const Dense<2>* layer,
                                                        const ForwardPropagation& fp,
                                                        Index layer_index,
                                                        Index parameter_offset,
                                                        MatrixR& jacobian)
{
    const Index batch_size = fp.batch_size;
    const Index num_neurons = layer->get_outputs_number();
    const Index num_inputs = layer->get_input_shape().size();

    const MatrixMap inputs = fp.views[layer_index][0][0].as_matrix();

    // Biases: identity pattern repeated for each sample
    for(Index j = 0; j < num_neurons; ++j)
        for(Index s = 0; s < batch_size; ++s)
            jacobian(s * num_neurons + j, parameter_offset + j) = type(1);

    // Weights: column-oriented fill for better cache locality
    for(Index k = 0; k < num_inputs; ++k)
        for(Index j = 0; j < num_neurons; ++j)
        {
            const Index col = parameter_offset + num_neurons + k * num_neurons + j;
            for(Index s = 0; s < batch_size; ++s)
                jacobian(s * num_neurons + j, col) = inputs(s, k);
        }
}

TrainingResults LevenbergMarquardtAlgorithm::train()
{
    if(!loss || !loss->get_neural_network() || !loss->get_dataset())
        return TrainingResults();

    if(loss->get_name() == "MinkowskiError")
        throw runtime_error("Levenberg-Marquard algorithm cannot work with Minkowski error.");
    else if(loss->get_name() == "CrossEntropy")
        throw runtime_error("Levenberg-Marquard algorithm cannot work with cross-entropy error.");
    else if(loss->get_name() == "WeightedSquaredError")
        throw runtime_error("Levenberg-Marquard algorithm is not implemented with weighted squared error.");

    // Start training

    if(display) cout << "Training with Levenberg-Marquardt algorithm..." << "\n";;

    TrainingResults results(maximum_epochs+1);

    // Dataset

    Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();

    set_scaling();

    Batch training_batch(training_samples_number, dataset);
    training_batch.fill(training_sample_indices, input_feature_indices, {}, target_feature_indices, true);

    Batch validation_batch(validation_samples_number, dataset);
    validation_batch.fill(validation_sample_indices, input_feature_indices, {}, target_feature_indices);

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);
    ForwardPropagation validation_forward_propagation(validation_samples_number, neural_network);

    // Loss index

    loss->set_normalization_coefficient();

    type old_loss = type(0);
    type loss_decrease = MAX;

    Index validation_failures = 0;

    BackPropagationLM training_back_propagation_lm(training_samples_number, loss);
    BackPropagationLM validation_back_propagation_lm(validation_samples_number, loss);

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    LevenbergMarquardtAlgorithmData optimization_data(this);

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << "\n";

        neural_network->forward_propagate(training_batch.get_inputs(),
                                          training_forward_propagation,
                                          true);

        back_propagate(training_batch,
                       training_forward_propagation,
                       training_back_propagation_lm);

        results.training_error_history(epoch) = training_back_propagation_lm.error;

        if(has_validation)
        {
            neural_network->forward_propagate(validation_batch.get_inputs(),
                                              validation_forward_propagation,
                                              false);

            calculate_errors(validation_batch, validation_forward_propagation, validation_back_propagation_lm);
            calculate_squared_errors(validation_batch, validation_forward_propagation, validation_back_propagation_lm);
            calculate_error(validation_batch, validation_forward_propagation, validation_back_propagation_lm);

            results.validation_error_history(epoch) = validation_back_propagation_lm.error;

            if(epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch-1))
                ++validation_failures;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation_lm.loss_value;

        old_loss = training_back_propagation_lm.loss_value;

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << results.training_error_history(epoch) << "\n";
            if(has_validation) cout << "Validation error: " << results.validation_error_history(epoch) << "\n";
            cout << "Damping parameter: " << damping_parameter << "\n";
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        bool stop = false;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached: " << loss_decrease << "\n";
            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
            stop = true;
        }
        else
        {
            stop = check_stopping_condition(results, epoch, elapsed_time,
                                            results.training_error_history(epoch),
                                            validation_failures);
        }

        if(stop)
        {
            results.loss = training_back_propagation_lm.loss_value;
            results.loss_decrease = loss_decrease;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch+1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = get_time(elapsed_time);
            break;
        }

        update_parameters(training_batch,
                          training_forward_propagation,
                          training_back_propagation_lm,
                          optimization_data);
    }

    set_unscaling();

    if(display) results.print();

    return results;
}

void LevenbergMarquardtAlgorithm::update_parameters(const Batch& batch,
                                                    ForwardPropagation& forward_propagation,
                                                    BackPropagationLM& back_propagation_lm,
                                                    LevenbergMarquardtAlgorithmData& optimization_data)
{

    NeuralNetwork* neural_network = loss->get_neural_network();

    VectorR& parameters = neural_network->get_parameters();

    type& error = back_propagation_lm.error;
    type& loss_value = back_propagation_lm.loss_value;

    const VectorR& gradient = back_propagation_lm.gradient;
    MatrixR& hessian = back_propagation_lm.hessian;

    VectorR& potential_parameters = optimization_data.potential_parameters;
    VectorR& parameter_updates = optimization_data.parameter_updates;

    const Index parameters_number = parameters.size();

    bool success = false;

    const VectorR neg_gradient = type(-1) * gradient;

    do
    {
        hessian.diagonal().array() += damping_parameter;

        parameter_updates = perform_Householder_QR_decomposition(hessian, neg_gradient);

        potential_parameters = parameters + parameter_updates;

        neural_network->forward_propagate(batch.get_inputs(),
                                          potential_parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

        calculate_error(batch, forward_propagation, back_propagation_lm);

        type new_loss_value = error + loss->calculate_regularization(potential_parameters);

        if(!isfinite(new_loss_value))
            new_loss_value = loss_value;

        if(new_loss_value < loss_value) // succesfull step
        {
            set_damping_parameter(damping_parameter/damping_parameter_factor);

            parameters = potential_parameters;

            loss_value = new_loss_value;

            success = true;

            break;
        }
        else
        {
            hessian.diagonal().array() -= damping_parameter;

            set_damping_parameter(damping_parameter*damping_parameter_factor);
        }

    }while(damping_parameter < maximum_damping_parameter);

    if(!success)
    {
        parameter_updates = (gradient.array().abs() >= type(EPSILON))
                                .select(-gradient.array().sign() * type(EPSILON), type(0));
        parameters += parameter_updates;
    }

    neural_network->set_parameters(parameters);

}

void LevenbergMarquardtAlgorithm::to_XML(XmlPrinter& printer) const
{
    printer.open_element("LevenbergMarquardt");

    write_xml_properties(printer, {
        {"DampingParameterFactor", to_string(damping_parameter_factor)},
        {"MinimumLossDecrease", to_string(minimum_loss_decrease)}
    });
    write_common_xml(printer);

    printer.close_element();
}

void LevenbergMarquardtAlgorithm::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "LevenbergMarquardt");

    set_damping_parameter_factor(read_xml_type(root_element, "DampingParameterFactor"));
    set_minimum_loss_decrease(read_xml_type(root_element, "MinimumLossDecrease"));
    read_common_xml(root_element);
}

LevenbergMarquardtAlgorithmData::LevenbergMarquardtAlgorithmData(LevenbergMarquardtAlgorithm *new_Levenberg_Marquardt_method)
{
    set(new_Levenberg_Marquardt_method);
}

void LevenbergMarquardtAlgorithmData::set(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method)
{
    Levenberg_Marquardt_algorithm = new_Levenberg_Marquardt_method;

    const Loss* loss = Levenberg_Marquardt_algorithm->get_loss();

    const NeuralNetwork* neural_network = loss->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_size();

    potential_parameters.resize(parameters_number);
    parameter_updates.resize(parameters_number);
}

REGISTER(Optimizer, LevenbergMarquardtAlgorithm, "LevenbergMarquardt");

// @todo Stub implementations for LM backpropagation (not yet refactored)

BackPropagationLM::BackPropagationLM(const Index, Loss*) {}
void BackPropagationLM::set(const Index, Loss*) {}
void BackPropagationLM::print() const {}
TensorView BackPropagationLM::get_output_gradients() const { return {}; }
vector<vector<TensorView>> BackPropagationLM::get_layer_gradients() const { return {}; }

vector<TensorView*> LayerBackPropagationLM::get_gradient_views() { return {}; }
vector<TensorView*> LayerBackPropagationLM::get_workspace_views() { return {}; }
vector<TensorView> LayerBackPropagationLM::get_input_gradients() const { return input_gradients; }

NeuralNetworkBackPropagationLM::NeuralNetworkBackPropagationLM(NeuralNetwork*) {}
void NeuralNetworkBackPropagationLM::set(const Index, NeuralNetwork*) {}
const vector<unique_ptr<LayerBackPropagationLM>>& NeuralNetworkBackPropagationLM::get_layers() const { return layers; }
const NeuralNetwork* NeuralNetworkBackPropagationLM::get_neural_network() const { return neural_network; }
void NeuralNetworkBackPropagationLM::print() {}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

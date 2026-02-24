//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "cross_entropy_error_3d.h"
#include "adaptive_moment_estimation.h"

namespace opennn
{

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;

public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(item);
        lock.unlock();
        cond_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]() { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        return item;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

AdaptiveMomentEstimation::AdaptiveMomentEstimation(const Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}


Index AdaptiveMomentEstimation::get_samples_number() const
{
    return batch_size;
}


type AdaptiveMomentEstimation::get_beta_1() const
{
    return beta_1;
}


type AdaptiveMomentEstimation::get_beta_2() const
{
    return beta_2;
}


type AdaptiveMomentEstimation::get_learning_rate() const
{
    return learning_rate;
}


type AdaptiveMomentEstimation::get_loss_goal() const
{
    return training_loss_goal;
}


void AdaptiveMomentEstimation::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}


void AdaptiveMomentEstimation::set_beta_1(const type new_beta_1)
{
    beta_1 = new_beta_1;
}


void AdaptiveMomentEstimation::set_beta_2(const type new_beta_2)
{
    beta_2= new_beta_2;
}


void AdaptiveMomentEstimation::set_default()
{
    display_period = 100;
    name = "AdaptiveMomentEstimation";
}


void AdaptiveMomentEstimation::set_display(bool new_display)
{
    display = new_display;
}


void AdaptiveMomentEstimation::set_learning_rate(const type new_learning_rate)
{
    learning_rate = new_learning_rate;
}


void AdaptiveMomentEstimation::set_loss_goal(const type new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void AdaptiveMomentEstimation::set_accuracy_goal(const type new_accuracy_goal)
{
    training_accuracy_goal = new_accuracy_goal;
}


void AdaptiveMomentEstimation::set_maximum_epochs(const Index new_maximum_epochs)
{
    maximum_epochs = new_maximum_epochs;
}


void AdaptiveMomentEstimation::set_maximum_time(const type new_maximum_time)
{
    maximum_time = new_maximum_time;
}


TrainingResults AdaptiveMomentEstimation::train()
{
    if(!loss_index || !loss_index->has_neural_network() || !loss_index->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs + 1);

    check();

    if(display) cout << "Training with adaptive moment estimation \"Adam\" ..." << endl;

    // Dataset

    Dataset* dataset = loss_index->get_dataset();

    if(!dataset)
        throw runtime_error("Dataset is null.");

    const bool has_validation = dataset->has_validation();

    const bool is_classification_model = is_instance_of<CrossEntropyError3d>(loss_index);

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    // const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const Index training_batch_size = min(training_samples_number, batch_size);

    const Index validation_batch_size = (validation_samples_number != 0)
                                         ? min(validation_samples_number, batch_size)
                                         : 0;

    const Index training_batches_number = (training_batch_size != 0)
        ? training_samples_number / training_batch_size
        : 0;

    const Index validation_batches_number = (validation_batch_size != 0)
       ? validation_samples_number / validation_batch_size
       : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches(validation_batches_number);

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();

    set_scaling();

    set_vocabularies();

    Batch training_batch(training_batch_size, dataset);
    unique_ptr<Batch> validation_batch;

    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);
    unique_ptr<ForwardPropagation> validation_forward_propagation;

    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss_index);
    unique_ptr<BackPropagation> validation_back_propagation;

    if (has_validation)
    {
        validation_batch = make_unique<Batch>(validation_batch_size, dataset);
        validation_forward_propagation = make_unique<ForwardPropagation>(validation_batch_size, neural_network);
        validation_back_propagation = make_unique<BackPropagation>(validation_batch_size, loss_index);
    }

    type training_error = type(0);
    type training_accuracy = type(0);

    type validation_error = type(0);
    type validation_accuracy = type(0);

    Index validation_failures = 0;

    // Optimization algorithm

    AdaptiveMomentEstimationData optimization_data(this);

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);

    type elapsed_time = type(0);

    bool shuffle = true;

    if(neural_network->has("Recurrent"))
        shuffle = false;

    // Main loop
    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;
        
        training_batches = dataset->get_batches(training_sample_indices, training_batch_size, shuffle);
        
        training_error = type(0);

        if(is_classification_model) training_accuracy = type(0);

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            training_back_propagation.neural_network.gradient.setZero();

            // Dataset

            training_batch.fill(training_batches[iteration],
                                input_feature_indices,
                                // decoder_feature_indices,
                                target_feature_indices);

            // Neural network

            neural_network->forward_propagate(training_batch.get_inputs(),
                                              training_forward_propagation,
                                              is_training);

            // Loss index

            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            training_error += training_back_propagation.error;

            if(is_classification_model) training_accuracy += training_back_propagation.accuracy(0);

            update_parameters(training_back_propagation, optimization_data);
        }

        // Loss

        training_error /= type(training_batches_number);
        if(is_classification_model)
            training_accuracy /= type(training_batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_validation)
        {
            validation_batches = dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle);

            validation_error = type(0);

            if(is_classification_model)
                validation_accuracy = type(0);

            for(Index iteration = 0; iteration < validation_batches_number; iteration++)
            {
                // Dataset

                validation_batch->fill(validation_batches[iteration],
                                      input_feature_indices,
                                      // decoder_feature_indices,
                                      target_feature_indices);

                // Neural network

                neural_network->forward_propagate(validation_batch->get_inputs(),
                                                  *validation_forward_propagation,
                                                  is_training);

                // Loss

                loss_index->calculate_error(*validation_batch,
                                            *validation_forward_propagation,
                                            *validation_back_propagation);

                validation_error += validation_back_propagation->error;

                if(is_classification_model)
                    validation_accuracy += validation_back_propagation->accuracy(0);
            }

            validation_error /= type(validation_batches_number);
            if(is_classification_model) validation_accuracy /= type(validation_batches_number);

            results.validation_error_history(epoch) = validation_error;

            if(epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch-1)) validation_failures++;
        }

        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if(is_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if(has_validation) cout << "Validation error: " << validation_error << endl;
            if(has_validation && is_classification_model) cout << "Validation accuracy: " << validation_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        stop_training = true;

        if(epoch == maximum_epochs)
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
            results.stopping_condition  = StoppingCondition::LossGoal;
            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }
        else if(training_accuracy >= training_accuracy_goal)
        {
            results.stopping_condition  = StoppingCondition::LossGoal;
            if(display) cout << "Epoch " << epoch << "\nAccuracy goal reached: " << training_accuracy << endl;
        }
        else if(validation_failures >= maximum_validation_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << validation_failures << endl;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else
        {
            stop_training = false;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;

            results.validation_failures = validation_failures;

            results.resize_training_error_history(epoch+1);

            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    if(display) results.print();

    return results;
}


Tensor<string, 2> AdaptiveMomentEstimation::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(8, 2);

    string_matrix.setValues({
    {"Learning rate", to_string(double(learning_rate))},
    {"Beta 1", to_string(double(beta_1))},
    {"Beta 2", to_string(double(beta_2))},
    {"Epsilon", to_string(numeric_limits<type>::epsilon())},
    {"Training loss goal", to_string(double(training_loss_goal))},
    {"Maximum epochs number", to_string(maximum_epochs)},
    {"Maximum time", write_time(maximum_time)},
    {"Batch samples number", to_string(batch_size)}});

    return string_matrix;
}


void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 AdaptiveMomentEstimationData& optimization_data) const
{
    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    optimization_data.iteration++;
    const type iteration = static_cast<type>(optimization_data.iteration);

    const type bias_correction_1 = type(1) - pow(beta_1, iteration);
    const type bias_correction_2 = type(1) - pow(beta_2, iteration);

    VectorR& parameters = neural_network->get_parameters();

    VectorMap gradient_exponential_decay(optimization_data.gradient_exponential_decay.data(), optimization_data.gradient_exponential_decay.size());
    VectorMap square_gradient_exponential_decay(optimization_data.square_gradient_exponential_decay.data(), optimization_data.square_gradient_exponential_decay.size());
    const VectorMap gradient(back_propagation.neural_network.gradient.data(), back_propagation.neural_network.gradient.size());

    gradient_exponential_decay.array() = beta_1 * gradient_exponential_decay.array() + (type(1) - beta_1) * gradient.array();
    square_gradient_exponential_decay.array() = beta_2 * square_gradient_exponential_decay.array() + (type(1) - beta_2) * gradient.array().square();

    parameters.array() -= learning_rate * (gradient_exponential_decay.array() / bias_correction_1) /
                          ((square_gradient_exponential_decay.array() / bias_correction_2).sqrt() + numeric_limits<type>::epsilon());
}


void AdaptiveMomentEstimation::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("AdaptiveMomentEstimation");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));
    add_xml_element(printer, "HardwareUse", get_hardware_use());

    printer.CloseElement();
}


void AdaptiveMomentEstimation::from_XML(const XMLDocument& document)
{

    const XMLElement* root_element = document.FirstChildElement("AdaptiveMomentEstimation");

    if(!root_element)
        throw runtime_error("Adaptive moment estimation element is nullptr.\n");

    set_batch_size(read_xml_index(root_element, "BatchSize"));
    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_epochs(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
    set_hardware_use(read_xml_string(root_element, "HardwareUse"));
}


AdaptiveMomentEstimationData::AdaptiveMomentEstimationData(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void AdaptiveMomentEstimationData::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    Loss* loss_index = new_adaptive_moment_estimation->get_loss_index();
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters().size();

    gradient_exponential_decay.resize(parameters_number);
    gradient_exponential_decay.setZero();

    square_gradient_exponential_decay.resize(parameters_number);
    square_gradient_exponential_decay.setZero();
}


void AdaptiveMomentEstimationData::print() const
{
    // cout << "Gradient exponential decay:" << endl
    //      << gradient_exponential_decay << endl
    //      << "Square gradient exponential decay:" << endl
    //      << square_gradient_exponential_decay << endl;
}


#ifdef OPENNN_CUDA

TrainingResults AdaptiveMomentEstimation::train_cuda()
{
    if(!loss_index || !loss_index->has_neural_network() || !loss_index->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs + 1);
    
    check();

    if (display) cout << "Training with adaptive moment estimation \"Adam\" CUDA ...\n";

    Dataset* dataset = loss_index->get_dataset();
    if(!dataset) throw runtime_error("Dataset is null.");

    const bool has_validation = dataset->has_validation();
    const bool is_classification_model = is_instance_of<CrossEntropyError3d>(loss_index);

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const Index training_batch_size = min(training_samples_number, batch_size);
    const Index validation_batch_size = (validation_samples_number != 0) ? min(validation_samples_number, batch_size) : 0;

    const Index training_batches_number = (training_batch_size != 0) ? training_samples_number / training_batch_size : 0;
    const Index validation_batches_number = (validation_batch_size != 0) ? validation_samples_number / validation_batch_size : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches(validation_batches_number);

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();
    set_scaling();
    set_vocabularies();

    // 1 Worker en background + 3 Buffers es la configuración óptima para OpenMP + CUDA
    const int PREFETCH_BATCHES = 3;
    
    ThreadSafeQueue<BatchCuda*> empty_training_queue;
    ThreadSafeQueue<BatchCuda*> ready_training_queue;
    vector<BatchCuda*> training_batch_pool;

    for (int i = 0; i < PREFETCH_BATCHES; i++) 
    {
        BatchCuda* b = new BatchCuda(training_batch_size, dataset);
        training_batch_pool.push_back(b);
        empty_training_queue.push(b);
    }

    ThreadSafeQueue<BatchCuda*> empty_validation_queue;
    ThreadSafeQueue<BatchCuda*> ready_validation_queue;
    vector<BatchCuda*> validation_batch_pool;

    if (has_validation) 
    {
        for (int i = 0; i < PREFETCH_BATCHES; i++) 
        {
            BatchCuda* b = new BatchCuda(validation_batch_size, dataset);
            validation_batch_pool.push_back(b);
            empty_validation_queue.push(b);
        }
    }

    cudaStream_t memory_stream;
    cudaStreamCreate(&memory_stream);
    cudaEvent_t batch_ready_event;
    cudaEventCreate(&batch_ready_event);

    ForwardPropagationCuda training_forward_propagation(training_batch_size, neural_network);
    unique_ptr<ForwardPropagationCuda> validation_forward_propagation;

    neural_network->copy_parameters_device();
    loss_index->set_normalization_coefficient();

    BackPropagationCuda training_back_propagation(training_batch_size, loss_index);
    unique_ptr<BackPropagationCuda> validation_back_propagation;

    if (has_validation) 
    {
        validation_forward_propagation = make_unique<ForwardPropagationCuda>(validation_batch_size, neural_network);
        validation_back_propagation = make_unique<BackPropagationCuda>(validation_batch_size, loss_index);
    }

    type training_error = type(0);
    type training_accuracy = type(0);
    type validation_error = type(0);
    type validation_accuracy = type(0);
    Index validation_failures = 0;

    ADAMOptimizationDataCuda optimization_data(this);

    bool stop_training = false;
    bool is_training = true;
    bool shuffle = true;

    if(neural_network->has("Recurrent")) shuffle = false;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);
    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_sample_indices, training_batch_size, shuffle);
        training_error = type(0);
        if (is_classification_model) training_accuracy = type(0);
        
        std::thread training_worker([&]() 
        {
            for(Index iteration = 0; iteration < training_batches_number; iteration++) 
            {
                BatchCuda* batch = empty_training_queue.pop();
                batch->fill_host(training_batches[iteration], input_feature_indices, target_feature_indices);
                ready_training_queue.push(batch);
            }
        });

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            BatchCuda* current_batch = ready_training_queue.pop();

            current_batch->copy_device_async(training_batches[iteration].size(), memory_stream);
            cudaEventRecord(batch_ready_event, memory_stream);
            cudaStreamWaitEvent(0, batch_ready_event, 0);

            neural_network->forward_propagate(current_batch->get_inputs_device(), training_forward_propagation, is_training);
            loss_index->back_propagate(*current_batch, training_forward_propagation, training_back_propagation);
            
            training_error += training_back_propagation.error;

            if (is_classification_model)
                training_accuracy += training_back_propagation.accuracy();
            
            update_parameters(training_back_propagation, optimization_data);

            cudaStreamSynchronize(0);

            empty_training_queue.push(current_batch);
        }

        training_worker.join();

        training_error /= type(training_batches_number);
        if (is_classification_model) training_accuracy /= type(training_batches_number);
        results.training_error_history(epoch) = training_error;

        if (has_validation)
        {
            validation_batches = dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle);
            validation_error = type(0);
            if (is_classification_model) validation_accuracy = type(0);

            std::thread validation_worker([&]() 
            {
                for(Index iteration = 0; iteration < validation_batches_number; iteration++) 
                {
                    BatchCuda* batch = empty_validation_queue.pop();
                    batch->fill_host(validation_batches[iteration], input_feature_indices, target_feature_indices);
                    ready_validation_queue.push(batch);
                }
            });

            for(Index iteration = 0; iteration < validation_batches_number; iteration++)
            {
                BatchCuda* current_batch = ready_validation_queue.pop();

                current_batch->copy_device_async(validation_batches[iteration].size(), memory_stream);
                cudaEventRecord(batch_ready_event, memory_stream);
                cudaStreamWaitEvent(0, batch_ready_event, 0);

                neural_network->forward_propagate(current_batch->get_inputs_device(), *validation_forward_propagation, is_training);
                loss_index->calculate_error(*current_batch, *validation_forward_propagation, *validation_back_propagation);

                validation_error += validation_back_propagation->error;
                if (is_classification_model) validation_accuracy += validation_back_propagation->accuracy();

                cudaStreamSynchronize(0);

                empty_validation_queue.push(current_batch);
            }

            validation_worker.join();

            validation_error /= type(validation_batches_number);
            if (is_classification_model) validation_accuracy /= type(validation_batches_number);
            results.validation_error_history(epoch) = validation_error;

            if (epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch - 1)) 
                validation_failures++;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (display && epoch % display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if (is_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if (has_validation) cout << "Validation error: " << validation_error << endl;
            if (has_validation && is_classification_model) cout << "Validation accuracy: " << validation_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        stop_training = true;

        if (epoch == maximum_epochs) 
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
            results.stopping_condition = StoppingCondition::LossGoal;
            if (display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        } 
        else if (training_accuracy >= training_accuracy_goal) 
        {
            results.stopping_condition = StoppingCondition::LossGoal;
            if (display) cout << "Epoch " << epoch << "\nAccuracy goal reached: " << training_accuracy << endl;
        } 
        else if (validation_failures >= maximum_validation_failures) 
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << validation_failures << endl;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        } 
        else 
            stop_training = false;

        if (stop_training) 
        {
            results.loss = training_back_propagation.loss;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch + 1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = write_time(elapsed_time);
            break;
        }

        if (epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }

    cudaStreamDestroy(memory_stream);
    cudaEventDestroy(batch_ready_event);

    for (BatchCuda* b : training_batch_pool) delete b;
    for (BatchCuda* b : validation_batch_pool) delete b;

    neural_network->copy_parameters_host();
    set_unscaling();

    if (display) results.print();

    return results;
}



void AdaptiveMomentEstimation::update_parameters(BackPropagationCuda& back_propagation,
                                                      ADAMOptimizationDataCuda& optimization_data) const
{
    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    const int parameters_number = static_cast<int>(neural_network->get_parameters_number());

    float* parameters_device_data = neural_network->get_parameters_device().data;
    const float* gradients_device = back_propagation.neural_network.workspace.data;

    optimization_data.iteration++;
    const int iteration = static_cast<int>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - powf(beta_1, static_cast<float>(iteration));
    const float bias_correction_2 = 1.0f - powf(beta_2, static_cast<float>(iteration));

    adam_update_device(
        parameters_number,
        parameters_device_data,
        optimization_data.gradient_exponential_decay.data,
        optimization_data.square_gradient_exponential_decay.data,
        gradients_device,
        beta_1,
        beta_2,
        learning_rate,
        numeric_limits<float>::epsilon(),
        bias_correction_1,
        bias_correction_2);
}


ADAMOptimizationDataCuda::ADAMOptimizationDataCuda(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void ADAMOptimizationDataCuda::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    NeuralNetwork* neural_network = adaptive_moment_estimation->get_loss_index()->get_neural_network();
    const Index parameters_number = neural_network->get_parameters_number();

    gradient_exponential_decay.resize({parameters_number});
    square_gradient_exponential_decay.resize({parameters_number});

    CHECK_CUDA(cudaMemset(gradient_exponential_decay.data, 0, parameters_number * sizeof(float)));
    CHECK_CUDA(cudaMemset(square_gradient_exponential_decay.data, 0, parameters_number * sizeof(float)));
}


void ADAMOptimizationDataCuda::print() const
{
    cout << "--- ADAM Optimization Data (CUDA) ---" << endl;

    NeuralNetwork* neural_network = adaptive_moment_estimation->get_loss_index()->get_neural_network();

    cout << "-----------------------------------" << endl;
}

#endif

REGISTER(Optimizer, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

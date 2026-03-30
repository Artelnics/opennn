#include "pch.h"
#include "variable.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/cross_entropy_error_3d.h"
#include "../opennn/dense_layer.h"
#include "../opennn/dataset.h"
#include "gtest/gtest.h"

using namespace opennn;

TEST(CrossEntropyError3DTest, DefaultConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    CrossEntropyError3d cross_entropy_error_3d(&neural_network, &dataset);

    EXPECT_TRUE(cross_entropy_error_3d.has_neural_network());
    EXPECT_TRUE(cross_entropy_error_3d.has_dataset());
}


TEST(CrossEntropyError3DTest, MultipleClassification_CPU_vs_GPU)
{
    const Index batch_size = 4;
    const Index sequence_length = 5;
    const Index vocab_size = 8;
    const Index input_dim = 6;

    // 1. Create a minimal neural network just to generate valid forward predictions
    NeuralNetwork nn;
    nn.add_layer(std::unique_ptr<Layer>(new opennn::Dense<3>(Shape{sequence_length, input_dim}, Shape{vocab_size}, "Softmax")));
    nn.set_parameters_random();

    // 2. Create generic dataset and fill it with random values
    Dataset dataset(batch_size, {sequence_length, input_dim}, {sequence_length});

    MatrixR data = MatrixR::Random(batch_size, dataset.get_features_number());
    vector<Index> target_indices = dataset.get_feature_indices("Target");

    // Fix targets to be integer class indices in range [0, vocab_size - 1]
    // Note: CrossEntropyError3d considers '0' as padding/mask for multiple classification
    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j : target_indices)
        {
            data(i, j) = static_cast<type>(rand() % vocab_size);
        }
    }
    dataset.set_data(data);

    vector<Index> sample_indices(batch_size);
    iota(sample_indices.begin(), sample_indices.end(), 0);

    // 3. Prepare CPU Batch and Forward Propagation
    Batch batch(batch_size, &dataset);
    batch.fill(sample_indices, dataset.get_feature_indices("Input"), {}, target_indices);

    ForwardPropagation fp_cpu(batch_size, &nn);
    nn.forward_propagate(batch.get_inputs(), fp_cpu, true);

    // 4. Calculate CPU Loss and Gradients
    CrossEntropyError3d loss(&nn, &dataset);
    BackPropagation bp_cpu(batch_size, &loss);

    loss.calculate_error(batch, fp_cpu, bp_cpu);
    loss.calculate_output_gradients(batch, fp_cpu, bp_cpu);

#ifdef CUDA

    // 5. Prepare GPU Batch and Forward Propagation
    BatchCuda batch_cuda(batch_size, &dataset);
    batch_cuda.fill(sample_indices, dataset.get_feature_indices("Input"), {}, target_indices);

    ForwardPropagationCuda fp_cuda(batch_size, &nn);

    nn.allocate_parameters_device();
    nn.copy_parameters_device();
    nn.forward_propagate(batch_cuda.get_inputs_device(), fp_cuda, true);

    // 6. Calculate GPU Loss and Gradients
    BackPropagationCuda bp_cuda(batch_size, &loss);

    // Call via base pointer to bypass 'private' access specifier of derived class
    Loss* base_loss = &loss;
    base_loss->calculate_error(batch_cuda, fp_cuda, bp_cuda);
    base_loss->calculate_output_gradients(batch_cuda, fp_cuda, bp_cuda);

    // 7. Compare CPU and GPU Error and Accuracy
    EXPECT_NEAR(bp_cpu.error, bp_cuda.error, 1e-4);
    EXPECT_NEAR(bp_cpu.accuracy(0), bp_cuda.accuracy(0), 1e-4);

    // 8. Compare CPU and GPU Output Gradients
    TensorView cpu_grads_view = bp_cpu.get_output_gradients();
    VectorMap cpu_grads = vector_map(cpu_grads_view);

    TensorViewCuda gpu_grads_view = bp_cuda.get_output_gradients_device();
    vector<type> gpu_grads_host(gpu_grads_view.size());

    CHECK_CUDA(cudaMemcpy(gpu_grads_host.data(), gpu_grads_view.data, gpu_grads_view.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for(Index i = 0; i < cpu_grads.size(); ++i)
    {
        EXPECT_NEAR(cpu_grads[i], gpu_grads_host[i], 1e-4);
    }

#endif
}


TEST(CrossEntropyError3DTest, BinaryClassification_CPU_vs_GPU)
{
    const Index batch_size = 4;
    const Index sequence_length = 5;
    const Index vocab_size = 1;
    const Index input_dim = 6;

    // 1. Create a minimal neural network
    NeuralNetwork nn;
    nn.add_layer(std::unique_ptr<Layer>(new opennn::Dense<3>(Shape{sequence_length, input_dim}, Shape{vocab_size}, "Sigmoid")));
    nn.set_parameters_random();

    // 2. Create generic dataset
    Dataset dataset(batch_size, {sequence_length, input_dim}, {sequence_length});

    MatrixR data = MatrixR::Random(batch_size, dataset.get_features_number());
    vector<Index> target_indices = dataset.get_feature_indices("Target");

    // Fix targets to be 0.0 or 1.0.
    // Note: CrossEntropyError3d considers '-1.0' as padding/mask for binary classification
    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j : target_indices)
        {
            int rnd = rand() % 3;
            data(i, j) = (rnd == 0) ? -1.0f : ((rnd == 1) ? 0.0f : 1.0f);
        }
    }
    dataset.set_data(data);

    vector<Index> sample_indices(batch_size);
    iota(sample_indices.begin(), sample_indices.end(), 0);

    // 3. Prepare CPU Batch and Forward Propagation
    Batch batch(batch_size, &dataset);
    batch.fill(sample_indices, dataset.get_feature_indices("Input"), {}, target_indices);

    ForwardPropagation fp_cpu(batch_size, &nn);
    nn.forward_propagate(batch.get_inputs(), fp_cpu, true);

    // 4. Calculate CPU Loss and Gradients
    CrossEntropyError3d loss(&nn, &dataset);
    BackPropagation bp_cpu(batch_size, &loss);

    loss.calculate_error(batch, fp_cpu, bp_cpu);
    loss.calculate_output_gradients(batch, fp_cpu, bp_cpu);

#ifdef CUDA

    // 5. Prepare GPU Batch and Forward Propagation
    BatchCuda batch_cuda(batch_size, &dataset);
    batch_cuda.fill(sample_indices, dataset.get_feature_indices("Input"), {}, target_indices);

    ForwardPropagationCuda fp_cuda(batch_size, &nn);

    nn.allocate_parameters_device();
    nn.copy_parameters_device();
    nn.forward_propagate(batch_cuda.get_inputs_device(), fp_cuda, true);

    // 6. Calculate GPU Loss and Gradients
    BackPropagationCuda bp_cuda(batch_size, &loss);

    // Call via base pointer
    Loss* base_loss = &loss;
    base_loss->calculate_error(batch_cuda, fp_cuda, bp_cuda);
    base_loss->calculate_output_gradients(batch_cuda, fp_cuda, bp_cuda);

    // 7. Compare CPU and GPU Error and Accuracy
    EXPECT_NEAR(bp_cpu.error, bp_cuda.error, 1e-4);
    EXPECT_NEAR(bp_cpu.accuracy(0), bp_cuda.accuracy(0), 1e-4);

    // 8. Compare CPU and GPU Output Gradients
    TensorView cpu_grads_view = bp_cpu.get_output_gradients();
    VectorMap cpu_grads = vector_map(cpu_grads_view);

    TensorViewCuda gpu_grads_view = bp_cuda.get_output_gradients_device();
    vector<type> gpu_grads_host(gpu_grads_view.size());

    CHECK_CUDA(cudaMemcpy(gpu_grads_host.data(), gpu_grads_view.data, gpu_grads_view.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for(Index i = 0; i < cpu_grads.size(); ++i)
    {
        EXPECT_NEAR(cpu_grads[i], gpu_grads_host[i], 1e-4);
    }

#endif
}

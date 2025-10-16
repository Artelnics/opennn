#include "pch.h"

#include "../opennn/cross_entropy_error.h"
#include "../opennn/tensors.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/language_dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/convolutional_layer.h"


using namespace opennn;

TEST(CrossEntropyError2d, DefaultConstructor)
{

    CrossEntropyError2d cross_entropy_error;

    EXPECT_TRUE(!cross_entropy_error.has_dataset());
    EXPECT_TRUE(!cross_entropy_error.has_neural_network());
}


TEST(CrossEntropyError2d, BackPropagate)
{

    
    const Index samples_number = get_random_index(2, 10);

    const Index inputs_number = get_random_index(1, 10);
    const Index targets_number = 1;
    const Index neurons_number = get_random_index(1, 10);

    Dataset dataset(samples_number, { inputs_number }, { targets_number });

    dataset.set_data_random();

    dataset.set_sample_uses("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(dimensions{ inputs_number }, dimensions{ targets_number }, "Logistic"));
    neural_network.set_parameters_random();


    CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);

    const Tensor<type, 1> gradient = cross_entropy_error.calculate_gradient();

    const Tensor<type, 1> numerical_gradient = cross_entropy_error.calculate_numerical_gradient();

    EXPECT_EQ(are_equal(gradient, numerical_gradient, type(1.0e-3)), true);
    }
   

/*
CrossEntropyError2d::calculate_error(const Batch& batch,
    const ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation) const

*/

TEST(CrossEntropyError2d, calculate_binary_error)
{
    Tensor<type, 2> data;
    Dataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data.setValues({
      {type(2), type(5), type(6), type(0)},
      {type(2), type(9), type(1), type(0)},
      {type(2), type(9), type(1), type(1)},
      {type(6), type(5), type(6), type(1)},
      {type(0), type(1), type(0), type(1)}
        });

    dataset.set_data(data);

    vector<Index> input_raw_variable_indices(3);
    input_raw_variable_indices = { 0,1,2 };

    vector<Index> target_raw_variable_indices(1);
    target_raw_variable_indices[0] = Index(3);

    dataset.set_raw_variable_indices(input_raw_variable_indices, target_raw_variable_indices);
    dataset.set_sample_uses("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(dimensions{ 3 }, dimensions{ 1 }, "Logistic"));
    neural_network.set_parameters_random();

    const Batch batch(5, &dataset);

    CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);
    ForwardPropagation forward_propagation(5, &neural_network);
    BackPropagation back_propagation(5, &cross_entropy_error);

    const vector<TensorView> batch_input_pairs = batch.get_input_pairs();
    neural_network.forward_propagate(batch_input_pairs, forward_propagation, true);

    cross_entropy_error.calculate_binary_error(batch, forward_propagation, back_propagation);
  
    EXPECT_GE(abs(back_propagation.error()), 0.0);
  }


TEST(CrossEntropyError2d, calculate_multiple_error) 
{
    Tensor<type, 2> data;
    Dataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data.setValues({
      {type(2), type(5), type(6), type(2)},
      {type(2), type(9), type(1), type(3)},
      {type(2), type(9), type(1), type(1)},
      {type(6), type(5), type(6), type(4)},
      {type(0), type(1), type(0), type(1)}
        });

    dataset.set_data(data);

    vector<Index> input_raw_variable_indices(3);
    input_raw_variable_indices = { 0,1,2 };

    vector<Index> target_raw_variable_indices(1);
    target_raw_variable_indices[0] = Index(3);

    dataset.set_raw_variable_indices(input_raw_variable_indices, target_raw_variable_indices);
    dataset.set_sample_uses("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(dimensions{ 3 }, dimensions{ 1 }, "Logistic"));
    neural_network.set_parameters_random();

    const Batch batch(5, &dataset);

    CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);
    ForwardPropagation forward_propagation(5, &neural_network);
    BackPropagation back_propagation(5, &cross_entropy_error);

    const vector<TensorView> batch_input_pairs = batch.get_input_pairs();
    neural_network.forward_propagate(batch_input_pairs, forward_propagation, true);

    cross_entropy_error.calculate_multiple_error(batch, forward_propagation, back_propagation);

    EXPECT_GE(abs(back_propagation.error()), 0.0);


  }


/*
    CrossEntropyError2d::calculate_output_delta(const Batch& batch,
        ForwardPropagation& forward_propagation,
        BackPropagation& back_propagation) const

    CrossEntropyError2d::calculate_binary_output_delta(const Batch& batch,
        ForwardPropagation& forward_propagation,
        BackPropagation& back_propagation) const

        /*
    CrossEntropyError2d::calculate_multiple_output_delta(const Batch& batch,
        ForwardPropagation& forward_propagation,
        BackPropagation& back_propagation) const

    CrossEntropyError2d::get_name()

    CrossEntropyError2d::to_XML(XMLPrinter& file_stream)

    CrossEntropyError2d::from_XML(const XMLDocument& document)

*/



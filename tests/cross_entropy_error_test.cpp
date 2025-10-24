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

    type difference = 0;
    for (int i = 0; i < gradient.size(); i++)
    {
        difference += ((gradient[i] - numerical_gradient[i]) * (gradient[i] - numerical_gradient[i]));
    }
    type error = sqrt(difference);

    EXPECT_NEAR(error,0, type(1.0e-1));
}
   

TEST(CrossEntropyError2d, calculate_binary_error)
{
    Tensor<type, 2> data;
    Dataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data.setValues({
      {type(0), type(1), type(0), type(1)},
      {type(1), type(1), type(0), type(0)},
      {type(0), type(1), type(1), type(1)},
      {type(0), type(1), type(1), type(1)},
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
  

    Batch batch(5, &dataset);

    const vector<Index> training_indices = dataset.get_sample_indices("Training");
    batch.fill(training_indices, input_raw_variable_indices, target_raw_variable_indices);

    CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);
    ForwardPropagation forward_propagation(5, &neural_network);
    BackPropagation back_propagation(5, &cross_entropy_error);

    const vector<TensorView> batch_input_pairs = batch.get_input_pairs();
    neural_network.forward_propagate(batch_input_pairs, forward_propagation, false);

    cross_entropy_error.calculate_binary_error(batch, forward_propagation, back_propagation);

    const type binary_error = back_propagation.error();

    EXPECT_FALSE(std::isnan(binary_error));
    EXPECT_GE(abs(binary_error), 0.0);

    cross_entropy_error.calculate_error(batch, forward_propagation, back_propagation);

    const type calculate_error = back_propagation.error();

    EXPECT_EQ(binary_error, calculate_error);
}


TEST(CrossEntropyError2d, calculate_multiple_error) 
{
    Tensor<type, 2> data;
    Dataset multipledataset( 5, {3}, {2});

    data.resize(5, 4);
    data.setValues({
      {type(2), type(5), type(0), type(1)},
      {type(2), type(9), type(1), type(1)},
      {type(2), type(9), type(1), type(1)},
      {type(6), type(5), type(0), type(0)},
      {type(0), type(1), type(0), type(1)}
        });
  
    multipledataset.set_data(data);

    vector<Index> input_raw_variable_indices = { 0,1 };

    vector<Index> target_raw_variable_indices = {2, 3};
  

    multipledataset.set_raw_variable_indices(input_raw_variable_indices, target_raw_variable_indices);
    multipledataset.set_sample_uses("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Dense2d>(dimensions{ 2 }, dimensions{ 2 }, "Logistic"));

    Batch batch(5, &multipledataset);

    const vector<Index> training_indices = multipledataset.get_sample_indices("Training");
    batch.fill(training_indices, input_raw_variable_indices, target_raw_variable_indices);

    CrossEntropyError2d cross_entropy_error(&neural_network, &multipledataset);
    ForwardPropagation forward_propagation(5, &neural_network);
    BackPropagation back_propagation(5, &cross_entropy_error);

    const vector<TensorView> batch_input_pairs = batch.get_input_pairs();
    neural_network.forward_propagate(batch_input_pairs, forward_propagation, true);

    cross_entropy_error.calculate_multiple_error(batch, forward_propagation, back_propagation);

    const type multiple_error = back_propagation.error();
    EXPECT_FALSE(std::isnan(multiple_error));
    EXPECT_GE(abs(multiple_error), 0.0);

    cross_entropy_error.calculate_error(batch, forward_propagation, back_propagation);

    const type calculate_error = back_propagation.error();
    EXPECT_EQ(multiple_error, calculate_error);
}


 TEST(CrossEntropyError2d, calculate_binary_output_delta)
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

     Batch batch(5, &dataset);

     const vector<Index> training_indices = dataset.get_sample_indices("Training");
     batch.fill(training_indices, input_raw_variable_indices, target_raw_variable_indices);


     CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);
     ForwardPropagation forward_propagation(5, &neural_network);
     BackPropagation back_propagation(5, &cross_entropy_error);

     const vector<TensorView> batch_input_pairs = batch.get_input_pairs();
     neural_network.forward_propagate(batch_input_pairs, forward_propagation, true);

     cross_entropy_error.calculate_binary_output_delta(batch, forward_propagation, back_propagation);

     auto deltas = tensor_map<2>(back_propagation.get_output_deltas_pair());
     auto outputs = tensor_map<2>(forward_propagation.get_last_trainable_layer_outputs_pair());

     EXPECT_EQ(deltas.dimension(0), outputs.dimension(0)); 
     EXPECT_EQ(deltas.dimension(1), outputs.dimension(1));
 }

 TEST(CrossEntropyError2d, calculate_multiple_output_delta) 
 {
     Tensor<type, 2> data;
     Dataset dataset(5, { 3 }, { 1 });

     data.resize(5, 4);
     data.setValues({
       {type(2), type(5), type(0), type(0)},
       {type(2), type(9), type(1), type(0)},
       {type(2), type(9), type(1), type(1)},
       {type(6), type(5), type(0), type(1)},
       {type(0), type(1), type(0), type(1)}
         });

     dataset.set_data(data);

     vector<Index> input_raw_variable_indices(2);
     input_raw_variable_indices = { 0,1 };

     vector<Index> target_raw_variable_indices(2);
     target_raw_variable_indices = {2,3};

     dataset.set_raw_variable_indices(input_raw_variable_indices, target_raw_variable_indices);
     dataset.set_sample_uses("Training");

     NeuralNetwork neural_network;
     neural_network.add_layer(make_unique<Dense2d>(dimensions{ 2 }, dimensions{ 2 }, "Logistic"));

     Batch batch(5, &dataset);

     const vector<Index> training_indices = dataset.get_sample_indices("Training");
     batch.fill(training_indices, input_raw_variable_indices, target_raw_variable_indices);


     CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);
     ForwardPropagation forward_propagation(5, &neural_network);
     BackPropagation back_propagation(5, &cross_entropy_error);

     const vector<TensorView> batch_input_pairs = batch.get_input_pairs();
     neural_network.forward_propagate(batch_input_pairs, forward_propagation, true);

     cross_entropy_error.calculate_multiple_output_delta(batch, forward_propagation, back_propagation);

     auto deltas = tensor_map<2>(back_propagation.get_output_deltas_pair());
     auto outputs = tensor_map<2>(forward_propagation.get_last_trainable_layer_outputs_pair());

     EXPECT_EQ(deltas.dimension(0), outputs.dimension(0));
     EXPECT_EQ(deltas.dimension(1), outputs.dimension(1));


 }
   
 
 TEST(CrossEntropyError2d, get_name)
 {
     Tensor<type, 2> data;
     Dataset dataset(5, { 3 }, { 1 });

     data.resize(5, 4);
     data.setValues({
       {type(2), type(5), type(0), type(0)},
       {type(2), type(9), type(1), type(0)},
       {type(2), type(9), type(1), type(1)},
       {type(6), type(5), type(0), type(1)},
       {type(0), type(1), type(0), type(1)}
         });

     dataset.set_data(data);

     vector<Index> input_raw_variable_indices(2);
     input_raw_variable_indices = { 0,1 };

     vector<Index> target_raw_variable_indices(2);
     target_raw_variable_indices = { 2,3 };

     dataset.set_raw_variable_indices(input_raw_variable_indices, target_raw_variable_indices);
     dataset.set_sample_uses("Training");

     NeuralNetwork neural_network;
     neural_network.add_layer(make_unique<Dense2d>(dimensions{ 2 }, dimensions{ 2 }, "Logistic"));
 
     CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);

     string name = cross_entropy_error.get_name();

     EXPECT_EQ(name, "CrossEntropyError2d");
 }

 TEST(CrossEntropyError2d, to_XML)
 {
     Dataset dataset(5, { 2 }, { 1 });
     Tensor<type, 2> data;
     data.resize(5, 3);
     data.setValues({
         {type(0.1), type(0.2), type(1)},
         {type(0.5), type(0.4), type(0)},
         {type(0.9), type(0.1), type(1)},
         {type(0.3), type(0.8), type(0)},
         {type(0.6), type(0.7), type(1)}
         });
     dataset.set_data(data);

     std::vector<Index> input_indices = { 0, 1 };
     std::vector<Index> target_indices = { 2 };
     dataset.set_raw_variable_indices(input_indices, target_indices);
     dataset.set_sample_uses("Training");

     NeuralNetwork neural_network;
     neural_network.add_layer(make_unique<Dense2d>(dimensions{ 2 }, dimensions{ 1 }, "Logistic"));

     CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);

     XMLPrinter printer;
     EXPECT_NO_THROW(cross_entropy_error.to_XML(printer));

     std::string xml_output = printer.CStr();

     EXPECT_TRUE(
         xml_output.find("<CrossEntropyError2d>") != std::string::npos ||
         xml_output.find("<CrossEntropyError2d/>") != std::string::npos
     );

     XMLDocument document;
     XMLError parse_result = document.Parse(xml_output.c_str());
     EXPECT_EQ(parse_result, XML_SUCCESS);

     const XMLElement* root = document.FirstChildElement("CrossEntropyError2d");
     ASSERT_NE(root, nullptr);
 }

 TEST(CrossEntropyError2d, from_XML_valid_document)
 {
     const char* xml_text = R"(
        <CrossEntropyError2d>
        </CrossEntropyError2d>
    )";

     XMLDocument document;
     ASSERT_EQ(document.Parse(xml_text), XML_SUCCESS);

     NeuralNetwork neural_network;
     Dataset dataset;
     CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);

     EXPECT_NO_THROW(cross_entropy_error.from_XML(document));
 }

 TEST(CrossEntropyError2d, from_XML_invalid_document)
 {
     const char* xml_text = R"(
        <InvalidTag></InvalidTag>
    )";

     XMLDocument document;
     ASSERT_EQ(document.Parse(xml_text), XML_SUCCESS);

     NeuralNetwork neural_network;
     Dataset dataset;
     CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);

     EXPECT_THROW(cross_entropy_error.from_XML(document), std::runtime_error);
 }



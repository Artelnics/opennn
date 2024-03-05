//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com


#ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif


// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>
#include <random>
#include <regex>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <list>
#include <vector>
#include <string_view>

// OpenNN includes

#include "../../opennn/opennn.h"
#include "../../opennn/opennn_strings.h"

constexpr Index numb_of_training_data = 60'000;
constexpr Index numb_of_columns_per_image = 28;
constexpr Index numb_of_rows_per_image = 28;
constexpr Index numb_of_input_variables = numb_of_columns_per_image * numb_of_rows_per_image;
constexpr Index numb_of_testing_data = 10'000;
constexpr Index numb_of_labels = 10;

constexpr auto path_to_data = "data/"sv;
constexpr auto name_of_training_file = "train-images.idx3-ubyte"sv;
constexpr auto name_of_training_label_file = "train-labels.idx1-ubyte"sv;
constexpr auto name_of_testing_input_file = "t10k-images.idx3-ubyte"sv;
constexpr auto name_of_testing_label_file = "t10k-labels.idx1-ubyte"sv;


using namespace opennn;

static string operator+(const string_view s0, const string_view s1)
{
    return string(s0) + string(s1);
}

Tensor<type, 2> get_data_from_file()
{
    Tensor<type, 2U> data(numb_of_training_data + numb_of_testing_data, numb_of_input_variables + numb_of_labels);
    ifstream training_data_input_file(path_to_data + name_of_training_file, iostream::in | iostream::binary);
    ifstream training_labeling_file(path_to_data + name_of_training_label_file, iostream::in | iostream::binary);
    //ignore headers
    training_data_input_file.ignore(16U);
    training_labeling_file.ignore(8U);
    auto fill = [&data](ifstream& input, ifstream& target, Index begin, Index end)
    {
        Tensor<type, 1U> labels(numb_of_labels);
        for(Index k = begin; k < end; ++k)
        {
            Index i = 0U;
            char c{};
            for(; i < numb_of_input_variables; ++i)
            {
                input.get(c);
                //min max scaling
                //data(k, i) = static_cast<type>(2) * (static_cast<type>(abs(c)) / type(255)) - static_cast<type>(1);
                data(k, i) = static_cast<type>(abs(c)) / type(255);
            }
            target.get(c);
            Index idx = static_cast<Index>(c);
            labels.setZero();
            labels(idx) = static_cast<type>(1);
            for(Index j = 0U; j < numb_of_labels; ++j, ++i)
            {
                data(k, i) = labels(j);
            }
        }

    };
    fill(training_data_input_file, training_labeling_file, 0U, numb_of_training_data);
    std::ifstream testing_data_input_file(path_to_data + name_of_testing_input_file, std::iostream::in | std::iostream::binary);
    std::ifstream testing_data_label_file(path_to_data + name_of_testing_label_file, std::iostream::in | std::iostream::binary);
    //ignore headers
    testing_data_input_file.ignore(16U);
    testing_data_label_file.ignore(8U);
    fill(testing_data_input_file, testing_data_label_file, numb_of_training_data, numb_of_training_data + numb_of_testing_data);
    return data;
}

DataSet get_data_set()
{
    Tensor<type, 2> data = get_data_from_file();
    DataSet data_set(data);
    for(Index sample_indx = 0; sample_indx < static_cast<Index>(numb_of_training_data * 0.8); sample_indx++)
    {
        data_set.set_sample_use(sample_indx, DataSet::SampleUse::Training);
    }
    for(Index sample_indx = static_cast<Index>(numb_of_training_data * 0.8); sample_indx < numb_of_training_data; sample_indx++)
    {
        data_set.set_sample_use(sample_indx, DataSet::SampleUse::Selection);
    }
    for(Index sample_indx = numb_of_training_data; sample_indx < numb_of_training_data + numb_of_testing_data; sample_indx++)
    {
        data_set.set_sample_use(sample_indx, DataSet::SampleUse::Testing);
    }
    for(Index column_indx = 0; column_indx < numb_of_input_variables; column_indx++)
    {
        data_set.set_column_use(column_indx, DataSet::VariableUse::Input);
        data_set.set_column_type(column_indx, DataSet::ColumnType::Numeric);
    }
    for(Index column_indx = numb_of_input_variables; column_indx < numb_of_input_variables + numb_of_labels; column_indx++)
    {
        data_set.set_column_type(column_indx, DataSet::ColumnType::Binary);
        data_set.set_column_use(column_indx, DataSet::VariableUse::Target);
    }
    data_set.set_columns_scalers(Scaler::NoScaling);
    return data_set;
}

void test(const TestingAnalysis& ta)
{
    auto errors = ta.calculate_multiple_classification_testing_errors();
    cout << "Sum squared error: " << errors(0) << endl;
    cout << "Mean squared error: " << errors(1) << endl;
    cout << "Root mean squared error: " << errors(2) << endl;
    cout << "Normalized squared error: " << errors(3) << endl;
    cout << "Cross-entropy erro: " << errors(4) << endl;

    auto confusion_matrix = ta.calculate_confusion();
    cout << "Confusion matrix:\n";
    Index correct_predicted_values_number = 0;
    for(Index i = 0; i < confusion_matrix.dimension(0); i++)
    {
        for(Index j = 0; j < confusion_matrix.dimension(1); j++)
        {
            if(i == j && i != confusion_matrix.dimension(0) - 1 && j != confusion_matrix.dimension(1) - 1)
            {
                correct_predicted_values_number += confusion_matrix(i, j);
            }
            cout << confusion_matrix(i, j) << "\t";
        }
        cout << endl;
    }
    const Index testing_samples_number = ta.get_data_set_pointer()->get_testing_samples_number();
    const type accuracy = static_cast<type>(correct_predicted_values_number) / static_cast<type>(testing_samples_number);
    
    cout << "Accuracy: " << accuracy << endl;
    cout << endl;
}

TrainingResults do_example1(DataSet& ds)
{
    NeuralNetwork nn;
    TrainingStrategy ts(&nn, &ds);
    TestingAnalysis ta(&nn, &ds);

    ts.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    ts.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    ts.set_default();
    ts.set_maximum_epochs_number(10);
    ts.set_display(false);
    
    Tensor<Index, 1> input_variable_dimension(1);
    input_variable_dimension.setValues({numb_of_input_variables});
    ds.set_input_variables_dimensions(input_variable_dimension);

    PerceptronLayer* pcll = new PerceptronLayer(numb_of_input_variables, 128U, PerceptronLayer::ActivationFunction::RectifiedLinear);
    ProbabilisticLayer* pl = new ProbabilisticLayer(pcll->get_neurons_number(), numb_of_labels);
    pl->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
    nn.add_layer(pcll);
    nn.add_layer(pl);
    
    TrainingResults r = ts.perform_training();
   
    test(ta);

    return r;
}

TrainingResults do_example2(DataSet& ds)
{
    NeuralNetwork nn;
    TrainingStrategy ts(&nn, &ds);
    TestingAnalysis ta(&nn, &ds);

    ts.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    ts.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    ts.set_default();
    ts.set_maximum_epochs_number(10);
    ts.set_display(false);

    Tensor<Index, 1> input_variable_dimension(3);
    input_variable_dimension.setValues({1, numb_of_columns_per_image, numb_of_rows_per_image});
    ds.set_input_variables_dimensions(input_variable_dimension);

    const Index batch_samples = 100U;
    Tensor<Index, 1> conv_input_dimension(4);
    conv_input_dimension[Convolutional4dDimensions::sample_index] = batch_samples;
    conv_input_dimension[Convolutional4dDimensions::row_index] = numb_of_rows_per_image;
    conv_input_dimension[Convolutional4dDimensions::column_index] = numb_of_columns_per_image;
    conv_input_dimension[Convolutional4dDimensions::channel_index] = 1;

    Tensor<Index, 1> conv_kernel_dimension(4);
    conv_kernel_dimension[Kernel4dDimensions::row_index] = 5;
    conv_kernel_dimension[Kernel4dDimensions::column_index] = 5;
    conv_kernel_dimension[Kernel4dDimensions::channel_index] = 1;
    conv_kernel_dimension[Kernel4dDimensions::kernel_index] = 6;

    Tensor<Index, 1> conv_kernel_dimension0(4);
    conv_kernel_dimension0[Kernel4dDimensions::row_index] = 4;
    conv_kernel_dimension0[Kernel4dDimensions::column_index] = 4;
    conv_kernel_dimension0[Kernel4dDimensions::channel_index] = 6;
    conv_kernel_dimension0[Kernel4dDimensions::kernel_index] = 12;
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({
        2,
        2,
    });
    ConvolutionalLayer* conv_layer = new ConvolutionalLayer(conv_input_dimension, conv_kernel_dimension);
    conv_layer->set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv_layer->set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    PoolingLayer* pooling_layer = new PoolingLayer(conv_layer->get_outputs_dimensions(), pooling_dimension);
    pooling_layer->set_column_stride(2);
    pooling_layer->set_row_stride(2);
    pooling_layer->set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayer* conv_layer0 = new ConvolutionalLayer(pooling_layer->get_outputs_dimensions(), conv_kernel_dimension0);
    conv_layer0->set_convolution_type(ConvolutionalLayer::ConvolutionType::Valid);
    conv_layer0->set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);


    FlattenLayer* fl = new FlattenLayer(conv_layer0->get_outputs_dimensions());

    PerceptronLayer* pcl = new PerceptronLayer(fl->get_outputs_number(), 32U, PerceptronLayer::ActivationFunction::RectifiedLinear);

    ProbabilisticLayer* pll = new ProbabilisticLayer(pcl->get_neurons_number(), 10);
    pll->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    nn.add_layer(conv_layer);
    nn.add_layer(pooling_layer);
    nn.add_layer(conv_layer0);
    nn.add_layer(fl);
    nn.add_layer(pcl);
    nn.add_layer(pll);

    ts.set_display(true);
    ts.set_display_period(1);
    TrainingResults r = ts.perform_training();
   
    test(ta);

    return r;
}

TrainingResults do_example3(DataSet& ds)
{
    NeuralNetwork nn;
    TrainingStrategy ts(&nn, &ds);
    TestingAnalysis ta(&nn, &ds);

    ts.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    ts.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    ts.set_default();
    ts.set_maximum_epochs_number(50);
    ts.set_maximum_time(60* 60* 60 * 10);
    ts.set_display(true);
    ts.set_display_period(1);

    Tensor<Index, 1> input_variable_dimension(3);
    input_variable_dimension.setValues({1, numb_of_columns_per_image, numb_of_rows_per_image});
    ds.set_input_variables_dimensions(input_variable_dimension);

    const Index batch_samples = 100U;
    Tensor<Index, 1> conv_input_dimension(4);
    conv_input_dimension[Convolutional4dDimensions::sample_index] = batch_samples;
    conv_input_dimension[Convolutional4dDimensions::row_index] = numb_of_rows_per_image;
    conv_input_dimension[Convolutional4dDimensions::column_index] = numb_of_columns_per_image;
    conv_input_dimension[Convolutional4dDimensions::channel_index] = 1;

    Tensor<Index, 1> conv_kernel_dimension0(4);
    conv_kernel_dimension0[Kernel4dDimensions::row_index] = 5;
    conv_kernel_dimension0[Kernel4dDimensions::column_index] = 5;
    conv_kernel_dimension0[Kernel4dDimensions::channel_index] = 1;
    conv_kernel_dimension0[Kernel4dDimensions::kernel_index] = 6;
    
    Tensor<Index, 1> conv_kernel_dimension1(4);
    conv_kernel_dimension1[Kernel4dDimensions::row_index] = 5;
    conv_kernel_dimension1[Kernel4dDimensions::column_index] = 5;
    conv_kernel_dimension1[Kernel4dDimensions::channel_index] = 6;
    conv_kernel_dimension1[Kernel4dDimensions::kernel_index] = 16;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({
        2,
        2,
    });
    ConvolutionalLayer* conv_layer0 = new ConvolutionalLayer(conv_input_dimension, conv_kernel_dimension0);
    conv_layer0->set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv_layer0->set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    PoolingLayer* pooling_layer0 = new PoolingLayer(conv_layer0->get_outputs_dimensions(), pooling_dimension);
    pooling_layer0->set_column_stride(2);
    pooling_layer0->set_row_stride(2);
    pooling_layer0->set_pooling_method(PoolingLayer::PoolingMethod::AveragePooling);

    ConvolutionalLayer* conv_layer1 = new ConvolutionalLayer(pooling_layer0->get_outputs_dimensions(), conv_kernel_dimension1);
    conv_layer1->set_convolution_type(ConvolutionalLayer::ConvolutionType::Valid);
    conv_layer1->set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    PoolingLayer* pooling_layer1 = new PoolingLayer(conv_layer1->get_outputs_dimensions(), pooling_dimension);
    pooling_layer1->set_column_stride(2);
    pooling_layer1->set_row_stride(2);
    pooling_layer1->set_pooling_method(PoolingLayer::PoolingMethod::AveragePooling);

    FlattenLayer* fl = new FlattenLayer(pooling_layer1->get_outputs_dimensions());

    PerceptronLayer* pcl0 = new PerceptronLayer(fl->get_outputs_number(), 120U, PerceptronLayer::ActivationFunction::RectifiedLinear);
    PerceptronLayer* pcl1 = new PerceptronLayer(pcl0->get_neurons_number(), 84U, PerceptronLayer::ActivationFunction::RectifiedLinear);

    ProbabilisticLayer* pll = new ProbabilisticLayer(pcl1->get_neurons_number(), 10);
    pll->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    nn.add_layer(conv_layer0);
    nn.add_layer(pooling_layer0);
    nn.add_layer(conv_layer1);
    nn.add_layer(pooling_layer1);
    nn.add_layer(fl);
    nn.add_layer(pcl0);
    nn.add_layer(pcl1);
    nn.add_layer(pll);

    TrainingResults r = ts.perform_training();
   
    test(ta);

    return r;
}

void save_errors(const TrainingResults& tr, string_view sv)        
{
    ofstream of(sv.data(), ios::out);
    for(Index i = 0; i < tr.selection_error_history.size(); i++)
    {
        of << i + 1 << ' ' << tr.training_error_history(i) << ' ' << tr.selection_error_history(i) << '\n';
    }
}

int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));
        
        DataSet ds = get_data_set();

        //cout << "Model 1: \n";
        //TrainingResults tr0 = do_example1(ds);
        //tr0.print();
        //cout << "Training time: " << tr0.elapsed_time << " \n";
        //save_errors(tr0, "tr0");

        cout << "Model 2: \n";
        TrainingResults tr1 = do_example2(ds);
        save_errors(tr1, "tr1");
        tr1.print();
        cout << "Training time: " << tr1.elapsed_time << " \n";

        cout << "Model 3: \n";
        TrainingResults tr2 = do_example3(ds);
        save_errors(tr2, "tr2");
        tr2.print();
        cout << "Training time: " << tr2.elapsed_time << " \n";

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL
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

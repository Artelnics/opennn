#include "yolo_error.h"
#include "yolo_dataset.h"
#include "yolo_network.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "tensors.h"

namespace opennn
{

YoloError::YoloError() : LossIndex()
{
}


YoloError::YoloError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


void YoloError::calculate_error(const Batch& batch,
                                const ForwardPropagation& forward_propagation,
                                BackPropagation& back_propagation) const
{

    YOLODataset* yolo_dataset = static_cast<YOLODataset*>(data_set);

    const vector<Tensor<type, 1>> anchors = yolo_dataset->get_anchors();
    // Batch

    // cout<<"=========error========="<<endl;


    const Index batch_samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    // cout<<targets_pair.second[0]<<", "<< targets_pair.second[1]<<", "<<targets_pair.second[2]<<", "<< targets_pair.second[3]<<endl;

    const TensorMap<Tensor<type, 4>> targets = tensor_map_4(targets_pair);

    // cout<<targets<<endl<<endl<<endl<<endl<<endl<<endl<<endl;

    // throw runtime_error("I wanted to see the targets");

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);


    // cout<<outputs<<endl<<endl<<endl<<endl<<endl<<endl<<endl;

    // Back propagation

    // const Index batch_samples_number = outputs.dimension(0);
    const type epsilon = 1e-06;
    const Index grid_size = outputs.dimension(1);
    const Index boxes_per_cell = anchors.size();
    const Index classes_number = (outputs.dimension(3) / boxes_per_cell) - 5;
    const Index box_data_size = 5 + classes_number;
    type coordinate_loss = 0, confidence_loss_object = 0, confidence_loss_noobject = 0, confidence_loss = 0, class_loss = 0;
    const type lambda_coord = 5.0;
    const type lambda_noobject = 0.5;

    // cout << "Outputs:\n" << outputs << endl;
    // throw runtime_error("ya");

    for(Index i = 0; i < batch_samples_number; i++)
    {
        for(Index j = 0; j < grid_size; j++)
        {
            for(Index k = 0; k < grid_size; k++)
            {
                for(Index l = 0; l < boxes_per_cell; l++)
                {
                    if(targets(i,j, k, l * box_data_size + 4) == 1)
                    {
                        coordinate_loss += pow(targets(i, j, k, l * box_data_size + 0) - outputs(i, j, k, l * box_data_size + 0), 2)      // @TODO check if I'm not mistaking the x and y coordinates
                                           + pow(targets(i, j, k, l * box_data_size + 1) - outputs(i, j, k, l * box_data_size + 1), 2)
                                           + pow(sqrt(targets(i, j, k, l * box_data_size + 2)) - sqrt(outputs(i, j, k, l * box_data_size + 2)), 2)
                                           + pow(sqrt(targets(i, j, k, l * box_data_size + 3)) - sqrt(outputs(i, j, k, l * box_data_size + 3)), 2);


                        Tensor<type, 1> ground_truth_box(4);
                        Tensor<type, 1> predicted_box(4);

                        for(Index data = 0; data < 4; data++)
                        {
                            ground_truth_box(data) = targets(i, j, k, l * box_data_size + data);
                            predicted_box(data) = outputs(i, j, k, l * box_data_size + data);
                        }

                        type confidence = calculate_intersection_over_union(ground_truth_box, predicted_box);

                        // cout<<"ground_truth_box: "<<ground_truth_box<<endl;
                        // cout<<"predicted_box: "<<predicted_box<<endl;

                        // cout<<"IOU between prediction and ground truth: "<<confidence<<endl<<endl;

                        confidence_loss_object += pow(confidence - outputs(i, j, k, l * box_data_size + 4), 2);

                        for(Index c = 0; c < classes_number; c++)
                            class_loss += targets(i,j, k, l * box_data_size + (5 + c)) * log(/*epsilon +*/ outputs(i,j, k, l * box_data_size + (5 + c)));
                    }
                    else
                    {
                        confidence_loss_noobject += pow(outputs(i, j, k, l * box_data_size + 4), 2);
                    }
                }
                // cout<<"coordinate loss: "<<coordinate_loss<<endl;
                // cout<<"confidence_loss_object: "<<confidence_loss_object<<endl;
                // cout<<"class_loss: "<<class_loss<<endl;
                // cout<<"confidence_loss_noobject: "<<confidence_loss_noobject<<endl;



            }
        }
    }

    coordinate_loss = lambda_coord * coordinate_loss;
    confidence_loss = confidence_loss_object + lambda_noobject * confidence_loss_noobject;

    Tensor<type, 0> total_loss;

    // cout<<"coordinate loss: "<<coordinate_loss<<endl<<endl;
    // cout<<"class_loss: "<<-class_loss<<endl<<endl;
    // cout<<"confidence_loss: "<<confidence_loss<<endl<<endl;

    total_loss() = (coordinate_loss - class_loss + confidence_loss);

    Tensor<type, 0>& error = back_propagation.error;

    // cout<<"=========error========="<<endl;

    error = total_loss / (type)batch_samples_number;

    // cout<<"Error: "<<back_propagation.error<<endl;

    // cout<<"=========error========="<<endl;

    // cout<<error<<endl;

    if(isnan(error())) throw runtime_error("\nError is NAN.");

}

void YoloError::calculate_output_delta(const Batch& batch,
                                               ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
    // Batch

    YOLODataset* yolo_dataset = static_cast<YOLODataset*>(data_set);

    const vector<Tensor<type, 1>> anchors = yolo_dataset->get_anchors();
    const Index batch_samples_number = batch.get_samples_number();

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    // cout<<"Outputs:\n"<<outputs<<endl;

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    cout<<targets_pair.second.size()<<endl;

    const TensorMap<Tensor<type, 4>> targets = tensor_map_4(targets_pair);


    cout<<"========delta4=========="<<endl;


    // Back propagation

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    // cout<<output_deltas_pair.second.size()<<endl;

    TensorMap<Tensor<type, 4>> output_deltas = tensor_map_4(output_deltas_pair);
    output_deltas.setZero();

    // cout<<output_deltas<<endl;

    const Index grid_size = outputs.dimension(1);
    const Index boxes_per_cell = anchors.size();
    const Index box_data_size = (outputs.dimension(3) / boxes_per_cell);
    const Index classes_number = box_data_size - 5;
    const type lambda_coord = 5.0;
    const type lambda_noobject = 0.5;
    // const type epsilon = 1e-05;

    cout<<"========delta4=========="<<endl;

    // cout<<batch_samples_number<<endl;

    for(Index i = 0; i < batch_samples_number; i++)
    {
        for(Index j = 0; j < grid_size; j++)
        {
            for(Index k = 0; k < grid_size; k++)
            {
                for(Index l = 0; l < boxes_per_cell; l++)
                {
                    const Index base_index = l * box_data_size;

                    if(targets(i, j, k, base_index + 4) == 1)
                    {

                        // Center deltas
                        output_deltas(i, j, k, base_index + 0) = lambda_coord * 2 * (outputs(i, j, k, base_index + 0) - targets(i, j, k, base_index + 0))
                                                                 * outputs(i,j,k, base_index + 0) * (1 - outputs(i,j,k, base_index + 0));                                // chain rule

                        output_deltas(i, j, k, base_index + 1) = lambda_coord * 2 * (outputs(i, j, k, base_index + 1) - targets(i, j, k, base_index + 1))
                                                                  * outputs(i,j,k, base_index + 1) * (1 - outputs(i,j,k, base_index + 1));                               // chain rule

                        // Width and height deltas
                        output_deltas(i, j, k, base_index + 2) = lambda_coord * (sqrt(outputs(i, j, k, base_index + 2)) - sqrt(targets(i, j, k, base_index + 2)))
                                                                 * sqrt(outputs(i, j, k, base_index + 2))
                                                                /* * (1 / sqrt(outputs(i, j, k, base_index + 2)))
                                                                 * outputs(i,j,k, base_index + 2)*/;                                    // chain rule

                        output_deltas(i, j, k, base_index + 3) = lambda_coord * (sqrt(outputs(i, j, k, base_index + 3)) - sqrt(targets(i, j, k, base_index + 3)))
                                                                 * sqrt(outputs(i, j, k, base_index + 3))
                                                                 /* * (1 / sqrt(outputs(i, j, k, base_index + 3)))
                                                                 * outputs(i,j,k, base_index + 3)*/;                                    // chain rule

                        // Confidence deltas (there is an object)
                        Tensor<type, 1> ground_truth_box(4);
                        Tensor<type, 1> predicted_box(4);

                        for(Index data = 0; data < 4; data++)
                        {
                            ground_truth_box(data) = targets(i, j, k, base_index + data);
                            predicted_box(data) = outputs(i, j, k, base_index + data);
                        }

                        // cout<<ground_truth_box<<endl;
                        // cout<<predicted_box<<endl;

                        type confidence = calculate_intersection_over_union(ground_truth_box, predicted_box);

                        output_deltas(i, j, k, base_index + 4) = 2 * (outputs(i, j, k, base_index + 4) - confidence)
                                                                 * outputs(i, j, k, base_index + 4) * (1 - outputs(i, j, k, base_index + 4));                           // chain rule

                        // Classes deltas
                        for(Index c = 0; c < classes_number; c++)
                        {
                            output_deltas(i, j, k, base_index + 5 + c) = outputs(i, j, k, base_index + 5 + c) - targets(i, j, k, base_index + 5 + c);
                        }
                    }
                    else
                    {
                        // Confidence deltas (there is no object)
                        output_deltas(i, j, k, base_index + 4) = lambda_noobject * 2 * (outputs(i, j, k, base_index + 4))
                                                                 * outputs(i, j, k, base_index + 4) * (1 - outputs(i, j, k, base_index + 4));                           // chain rule
                    }
                }
            }
        }
    }
    // cout<<output_deltas<<endl;
    output_deltas = output_deltas / (type) batch_samples_number;

    // cout<<"Yolo error output deltas:\n"<<output_deltas<<endl;
    // throw runtime_error("ya");
}

string YoloError::get_loss_method() const
{
    return "YOLO_ERROR";
}


string YoloError::get_error_type_text() const
{
    return "YOLO error";
}


void YoloError::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("YOLOError");

    file_stream.CloseElement();
}


void YoloError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("YOLOError");

    if(!root_element)
        throw runtime_error("YOLO error element is nullptr.\n");

    // Regularization

    tinyxml2::XMLDocument regularization_document;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");
    regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));

    regularization_from_XML(regularization_document);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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

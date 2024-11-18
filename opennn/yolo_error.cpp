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

    cout<<"=========error========="<<endl;


    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 4>> targets = tensor_map_4(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    cout<<outputs<<endl<<endl<<endl<<endl<<endl<<endl<<endl;

    // Back propagation

    // const Index batch_samples_number = outputs.dimension(0);
    const Index grid_size = outputs.dimension(1);
    const Index boxes_per_cell = anchors.size();
    const Index classes_number = (outputs.dimension(3) / anchors.size()) - 5;
    const Index box_data_size = 5 + classes_number;
    type coordinate_loss = 0, confidence_loss_object = 0, confidence_loss_noobject = 0, confidence_loss = 0, class_loss = 0;
    const type lambda_coord = 5.0;
    const type lambda_noobject = 0.5;

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
                        coordinate_loss += pow(targets(i, j, k, l * box_data_size + 0) - (outputs(i, j, k, l * box_data_size + 0) + j), 2)      // @TODO check if I'm not mistaking the x and y coordinates
                                           + pow(targets(i, j, k, l * box_data_size + 1) - (outputs(i, j, k, l * box_data_size + 1) + k), 2)
                                           + pow(sqrt(targets(i, j, k, l * box_data_size + 2)) - sqrt(outputs(i, j, k, l * box_data_size + 2) * anchors[l](0)), 2)
                                           + pow(sqrt(targets(i, j, k, l * box_data_size + 3)) - sqrt(outputs(i, j, k, l * box_data_size + 3) * anchors[l](1)), 2);



                        confidence_loss_object += pow(1 - outputs(i, j, k, l * box_data_size + 4), 2);

                        for(Index c = 0; c < classes_number; c++){
                            cout<<outputs(i,j, k, l * box_data_size + (5 + c))<<endl;
                            class_loss += targets(i,j, k, l * box_data_size + (5 + c)) * log(outputs(i,j, k, l * box_data_size + (5 + c)));}
                    }
                    else
                    {
                        confidence_loss_noobject += pow(outputs(i, j, k, l * box_data_size + 4), 2);
                    }
                }
                cout<<"coordinate loss: "<<coordinate_loss<<endl;
                cout<<"confidence_loss_object: "<<confidence_loss_object<<endl;
                cout<<"class_loss: "<<class_loss<<endl;
                cout<<"confidence_loss_noobject: "<<confidence_loss_noobject<<endl;



            }
        }
    }

    coordinate_loss = lambda_coord * coordinate_loss;
    confidence_loss = confidence_loss_object + lambda_noobject * confidence_loss_noobject;

    Tensor<type, 0> total_loss;

    cout<<coordinate_loss<<endl<<endl;
    cout<<-class_loss<<endl<<endl;
    cout<<confidence_loss<<endl<<endl;

    total_loss() = (coordinate_loss - class_loss + confidence_loss) / batch_samples_number;

    Tensor<type, 0>& error = back_propagation.error;

    cout<<"=========error========="<<endl;

    error = total_loss;

    cout<<back_propagation.error<<endl;


    cout<<"=========error========="<<endl;

    cout<<error<<endl;

    if(isnan(error())) throw runtime_error("\nError is NAN.");


}
/*
Tensor<type, 0> YoloError::loss_function(const Tensor<type, 4>& targets, const vector<Tensor<type, 1>>& anchors, const Tensor<type, 4>& outputs) const
{
    const Index batch_size = outputs.dimension(0);
    const Index grid_size = outputs.dimension(1);
    const Index boxes_per_cell = anchors.size();
    const Index classes_number = (outputs.dimension(3) / anchors.size()) - 5;
    const Index box_data_size = 5 + classes_number;
    type coordinate_loss = 0, confidence_loss_object = 0, confidence_loss_noobject = 0, confidence_loss = 0, class_loss = 0;
    const type lambda_coord = 5.0;
    const type lambda_noobject = 0.5;


    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j = 0; j < grid_size; j++)
        {
            for(Index k = 0; k < grid_size; k++)
            {
                for(Index l = 0; l < boxes_per_cell; l++)
                {
                    if(targets(i,j, k, l * box_data_size + 4) == 1)
                    {
                        coordinate_loss += pow(targets(i, j, k, l * box_data_size + 0) - outputs(i, j, k, l * box_data_size + 0), 2)
                                           + pow(targets(i, j, k, l * box_data_size + 1) - outputs(i, j, k, l * box_data_size + 1), 2)
                                           + pow(sqrt(targets(i, j, k, l * box_data_size + 2)) - sqrt(outputs(i, j, k, l * box_data_size + 2) * anchors[l](0)), 2)
                                           + pow(sqrt(targets(i, j, k, l * box_data_size + 3)) - sqrt(outputs(i, j, k, l * box_data_size + 3) * anchors[l](1)), 2);

                        confidence_loss_object += pow(1 - outputs(i, j, k, l * box_data_size + 4), 2);

                        for(Index c = 0; c < classes_number; c++)
                            class_loss += targets(i,j, k, l * box_data_size + (5 + c)) * log(outputs(i,j, k, l * box_data_size + (5 + c)));
                    }
                    else
                    {
                        confidence_loss_noobject += pow(outputs(i, j, k, l * box_data_size + 4), 2);
                    }
                }

            }
        }
    }

    coordinate_loss = lambda_coord * coordinate_loss;
    // class_loss = -class_loss;
    confidence_loss = confidence_loss_object + lambda_noobject * confidence_loss_noobject;

    // total_loss = coord_loss + class_loss + conf_loss;
    const Tensor<type, 0> total_loss = coordinate_loss - class_loss + confidence_loss;

    return total_loss;
}
*/

void YoloError::calculate_output_delta(const Batch& batch,
                                               ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
    // Batch

    YOLODataset* yolo_dataset = static_cast<YOLODataset*>(data_set);

    const vector<Tensor<type, 1>> anchors = yolo_dataset->get_anchors();
    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 4>> outputs = tensor_map_4(outputs_pair);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 4>> targets = tensor_map_4(targets_pair);

    // Back propagation

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 4>> output_deltas = tensor_map_4(output_deltas_pair);

    const Index grid_size = outputs.dimension(1);
    const Index boxes_per_cell = anchors.size();
    const Index classes_number = (outputs.dimension(3) / anchors.size()) - 5;
    const Index box_data_size = 5 + classes_number;
    const type lambda_coord = 5.0;
    const type lambda_noobject = 0.5;

    for(Index i = 0; i < batch_samples_number; i++)
    {
        for(Index j = 0; j < grid_size; j++)
        {
            for(Index k = 0; k < grid_size; k++)
            {
                for(Index l = 0; l < boxes_per_cell; l++)
                {
                    Index base_index = l * box_data_size;

                    if(targets(i, j, k, base_index + 4) == 1)
                    {
                        // Coordinate deltas (delta of mean_squared_error)
                        output_deltas(i, j, k, base_index + 0) = lambda_coord * 2 * (outputs(i, j, k, base_index + 0) - targets(i, j, k, base_index + 0));
                        output_deltas(i, j, k, base_index + 1) = lambda_coord * 2 * (outputs(i, j, k, base_index + 1) - targets(i, j, k, base_index + 1));

                        // Width and height deltas (similar to delta of mean_squared_error)
                        output_deltas(i, j, k, base_index + 2) = lambda_coord * 2 * (sqrt(outputs(i, j, k, base_index + 2) * anchors[l](0) ) - sqrt(targets(i, j, k, base_index + 2)))
                                                                 * (0.5 / sqrt(outputs(i, j, k, base_index + 2) * anchors[l](0) )) * anchors[l](0) ;

                        output_deltas(i, j, k, base_index + 3) = lambda_coord * 2 * (sqrt(outputs(i, j, k, base_index + 3) * anchors[l](1) ) - sqrt(targets(i, j, k, base_index + 3)))
                                                                 * (0.5 / sqrt(outputs(i, j, k, base_index + 3) * anchors[l](1) )) * anchors[l](1) ;

                        // Confidence deltas (there is object) (delta of mean_squared_error)
                        output_deltas(i, j, k, base_index + 4) = 2 * (1 - outputs(i, j, k, base_index + 4));

                        // Classes deltas (delta for the cross entropy)
                        for(Index c = 0; c < classes_number; c++)
                        {
                            output_deltas(i, j, k, base_index + c) = - targets(i, j, k, base_index + 5 + c) / outputs(i, j, k, base_index + 5 + c);
                        }
                    }
                    else
                    {
                        // Confidence deltas (there is no object) (delta of mean_squared_error)
                        output_deltas(i, j, k, base_index + 4) = lambda_noobject * 2 * (outputs(i, j, k, base_index + 4));
                    }
                }
            }
        }
    }
    // output_deltas = - output_deltas;
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

    file_stream.OpenElement("YOLOv2Error");

    file_stream.CloseElement();
}


void YoloError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("YOLOv2Error");

    if(!root_element)
        throw runtime_error("YOLOv2 error element is nullptr.\n");

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

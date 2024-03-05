//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer.h"
#include "perceptron_layer.h"

namespace opennn
{


/// Default constructor.
/// It creates a flatten layer.
/// This constructor also initializes the rest of the class members to their default values.

FlattenLayer::FlattenLayer() : Layer()
{
    layer_type = Layer::Type::Flatten;
}


FlattenLayer::FlattenLayer(const Tensor<Index, 1>& new_input_variables_dimensions) : Layer()
{
    set(new_input_variables_dimensions);

    layer_type = Type::Flatten;
}


void FlattenLayer::set_parameters(const Tensor<type, 1>&, const Index&)
{
    return;
}


Tensor<Index, 1> FlattenLayer::get_input_variables_dimensions() const
{
    return input_variables_dimensions;
}


void FlattenLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// @todo
/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's kernels to an image.

//Tensor<Index, 1> FlattenLayer::get_outputs_dimensions() const
Index FlattenLayer::get_outputs_number() const
{
    return  input_variables_dimensions(Convolutional4dDimensions::channel_index) * 
            input_variables_dimensions(Convolutional4dDimensions::column_index) * 
            input_variables_dimensions(Convolutional4dDimensions::row_index);
}

Tensor<Index, 1> FlattenLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(2);

    outputs_dimensions(0) = get_inputs_batch_number(); 
    outputs_dimensions(1) = get_input_height() * get_input_width() * get_inputs_channels_number();

    return outputs_dimensions;
}

/// @todo
Index FlattenLayer::get_inputs_number() const
{
    return get_input_height() * 
        get_input_width() * 
        get_inputs_channels_number() * 
        get_inputs_batch_number();
}


Index FlattenLayer::get_input_height() const
{
    return input_variables_dimensions(Convolutional4dDimensions::row_index);
}


Index FlattenLayer::get_input_width() const
{
    return input_variables_dimensions(Convolutional4dDimensions::column_index);
}


Index FlattenLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions(Convolutional4dDimensions::channel_index);
}


Index FlattenLayer::get_inputs_batch_number() const
{
    return input_variables_dimensions(Convolutional4dDimensions::sample_index);
}


/// Returns the number of neurons

Index FlattenLayer::get_neurons_number() const
{
    return  input_variables_dimensions(Convolutional4dDimensions::row_index) * 
            input_variables_dimensions(Convolutional4dDimensions::channel_index) * 
            input_variables_dimensions(Convolutional4dDimensions::column_index);
}


Tensor<type, 1> FlattenLayer::get_parameters() const
{
    return Tensor<type,1>();
}


/// Returns the number of parameters of the layer.

Index FlattenLayer::get_parameters_number() const
{
    return 0;
}

Tensor< TensorMap< Tensor<type, 1> >*, 1> FlattenLayer::get_layer_parameters()
{
    Tensor< TensorMap< Tensor<type, 1> >*, 1> layer_parameters;

    return layer_parameters;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.

/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (number of images (batch), number of channels, width, height).

void FlattenLayer::set(const Tensor<Index, 1>& new_inputs_dimensions)
{
    layer_name = "flatten_layer";

    input_variables_dimensions = new_inputs_dimensions;

}


/// Obtain the connection between the convolutional and the conventional part
/// of a neural network. That is a matrix which links to the perceptron layer.
/// @param inputs 4d tensor(batch, channels, width, height)
/// @return result 2d tensor(batch, number of pixels)

void FlattenLayer::calculate_outputs(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                     type* outputs_data, const Tensor<Index, 1>& outputs_dimensions)
{
    const Index rows_number = inputs_dimensions[Convolutional4dDimensions::row_index];
    const Index columns_number = inputs_dimensions[Convolutional4dDimensions::column_index];
    const Index channels_number = inputs_dimensions[Convolutional4dDimensions::channel_index];
    const Index batch_size = inputs_dimensions[Convolutional4dDimensions::sample_index];
    const Index variable_size = rows_number * columns_number * channels_number;


    const TensorMap<Tensor<type, 4>> inputs(inputs_data, [&](){
        DSizes<Index, 4> dim{};
        dim[Convolutional4dDimensions::channel_index] = channels_number;
        dim[Convolutional4dDimensions::row_index] = rows_number;
        dim[Convolutional4dDimensions::column_index] = columns_number;
        dim[Convolutional4dDimensions::sample_index] = batch_size;
        return dim;
    }());


    const Index output_batch_numbers = outputs_dimensions(0);
    const Index variable_size_numbers = outputs_dimensions(1);
    TensorMap<Tensor<type, 2>> outputs(outputs_data, output_batch_numbers, variable_size_numbers);

    outputs.device(*thread_pool_device) = inputs.reshape(outputs.dimensions());
}



void FlattenLayer::forward_propagate(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                     LayerForwardPropagation* forward_propagation,
                                     bool& switch_train)
{
    FlattenLayerForwardPropagation* flatten_layer_forward_propagation
            = static_cast<FlattenLayerForwardPropagation*>(forward_propagation);

#ifdef OPENNN_DEBUG

    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    if(outputs_dimensions[0] != flatten_layer_forward_propagation->outputs.dimension(0))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer::forward_propagate.\n"
               << "outputs_dimensions[0]" <<outputs_dimensions[0] <<"must be equal to" << flatten_layer_forward_propagation->outputs.dimension(0)<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs_dimensions[1] != flatten_layer_forward_propagation->outputs.dimension(1))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer::forward_propagate.\n"
               << "outputs_dimensions[1]" <<outputs_dimensions[1] <<"must be equal to" << flatten_layer_forward_propagation->outputs.dimension(1)<<".\n";

        throw invalid_argument(buffer.str());
    }

#endif

    calculate_outputs(
        inputs_data, 
        inputs_dimensions, 
        forward_propagation->outputs_data, 
        forward_propagation->outputs_dimensions);
    
}




void FlattenLayer::calculate_hidden_delta(
    FlattenLayerForwardPropagation* next_flatten_layer_forwardpropagation,
    FlattenLayerBackPropagation* next_flatten_layer_back_propagation,
    LayerBackPropagation* back_propagation) const
{
    (void)next_flatten_layer_forwardpropagation;
    const Index images_number = next_flatten_layer_back_propagation->deltas_dimensions(0);
    const Index next_delta_pixel_numbers = next_flatten_layer_back_propagation->deltas_dimensions(1);
    
    TensorMap<Tensor<type, 2>> next_delta(
        next_flatten_layer_back_propagation->deltas_data, 
        images_number, 
        next_delta_pixel_numbers);

    const Index delta_row_numbers = back_propagation->deltas_dimensions(Convolutional4dDimensions::row_index);
    const Index delta_column_numbers = back_propagation->deltas_dimensions(Convolutional4dDimensions::column_index);
    const Index delta_channel_numbers = back_propagation->deltas_dimensions(Convolutional4dDimensions::channel_index);

    TensorMap<Tensor<type, 4>> delta(
        back_propagation->deltas_data,
        [&](){
            DSizes<Index, 4> dim{};
            dim[Convolutional4dDimensions::row_index] = delta_row_numbers;
            dim[Convolutional4dDimensions::column_index] = delta_column_numbers;
            dim[Convolutional4dDimensions::channel_index] = delta_channel_numbers;
            dim[Convolutional4dDimensions::sample_index] = images_number;
            return dim;
        }());

    delta.device(*thread_pool_device) = next_delta.reshape(delta.dimensions());
}
void FlattenLayer::calculate_hidden_delta(
    LayerForwardPropagation* next_layer_forwardpropagation,
    LayerBackPropagation* next_layer_back_propagation,
    LayerBackPropagation* back_propagation) const
{
    switch(next_layer_forwardpropagation->layer_pointer->get_type())
    {
        case Layer::Type::Flatten:
        {
            FlattenLayerForwardPropagation* next_flatten_layer_forward_propagation = 
                static_cast<FlattenLayerForwardPropagation*>(next_layer_forwardpropagation);
            FlattenLayerBackPropagation* next_flatten_layer_back_propagation = 
                static_cast<FlattenLayerBackPropagation*>(next_layer_back_propagation);
            calculate_hidden_delta(
                next_flatten_layer_forward_propagation, 
                next_flatten_layer_back_propagation,
                back_propagation);
            
        }
        break;
        case Layer::Type::Perceptron:
        {
            static_cast<const PerceptronLayer*>(next_layer_forwardpropagation->layer_pointer)->calculate_hidden_delta(
                next_layer_forwardpropagation,
                next_layer_back_propagation,
                back_propagation);
        }
        break;
        default:
        {
            //Forwarding
            next_layer_forwardpropagation->layer_pointer->calculate_hidden_delta(
                next_layer_forwardpropagation,
                next_layer_back_propagation,
                back_propagation);
        }
        break;
    }
}
/// Serializes the flatten layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void FlattenLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("FlattenLayer");

    file_stream.OpenElement("InputVariablesDimension");

    file_stream.OpenElement("InputHeight");
    buffer.str("");
    buffer << get_input_height();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputWidth");
    buffer.str("");
    buffer << get_input_width();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.OpenElement("InputChannels");
    buffer.str("");
    buffer << get_inputs_channels_number();

    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this flatten layer object.
/// @param document TinyXML document containing the member data.

void FlattenLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* flatten_layer_element = document.FirstChildElement("FlattenLayer");

    if(!flatten_layer_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenLayer element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Flatten layer input variables dimenison

    const tinyxml2::XMLElement* input_variables_dimensions_element = flatten_layer_element->FirstChildElement("InputVariablesDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputVariablesDimensions element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Input height

    const tinyxml2::XMLElement* input_height_element = input_variables_dimensions_element->NextSiblingElement("InputHeight");

    if(!input_height_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputHeight element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index input_height = static_cast<Index>(atoi(input_height_element->GetText()));

    // Input width

    const tinyxml2::XMLElement* input_width_element = input_variables_dimensions_element->NextSiblingElement("InputWidth");

    if(!input_width_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputWidth element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index input_width = static_cast<Index>(atoi(input_width_element->GetText()));

    // Input channels number

    const tinyxml2::XMLElement* input_channels_number_element = input_variables_dimensions_element->NextSiblingElement("InputChannels");

    if(!input_channels_number_element)
    {
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "FlattenInputChannelsNumber element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index input_channels_number = static_cast<Index>(atoi(input_channels_number_element->GetText()));

    Tensor<Index,1> inputsDimensionTensor(4);

    inputsDimensionTensor.setValues({input_height, input_width, input_channels_number, 0});

    set(inputsDimensionTensor);
}

FlattenLayerForwardPropagation::FlattenLayerForwardPropagation() : LayerForwardPropagation()
{
}

FlattenLayerForwardPropagation::FlattenLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer_pointer);
}

void FlattenLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
{
    layer_pointer = new_layer_pointer;
    
    batch_samples_number = new_batch_samples_number;

    const FlattenLayer* flatten_layer_pointer = static_cast<const FlattenLayer*>(layer_pointer);

    const Index rows_number = flatten_layer_pointer->get_input_height();
    const Index columns_number = flatten_layer_pointer->get_input_width();
    const Index channels_number = flatten_layer_pointer->get_inputs_channels_number();


    outputs_data = static_cast<type*>(malloc(batch_samples_number * rows_number * columns_number * channels_number * sizeof(type)));
    
    outputs_dimensions.resize(2);
    outputs_dimensions.setValues({
        batch_samples_number,
        rows_number * columns_number * channels_number
    });
}

void FlattenLayerForwardPropagation::print() const
{
    //TODO: output
    cout << "Outputs:" << endl;
}

FlattenLayerBackPropagation::FlattenLayerBackPropagation() : LayerBackPropagation()
{
}

FlattenLayerBackPropagation::~FlattenLayerBackPropagation()
{
}


FlattenLayerBackPropagation::FlattenLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer_pointer);
}


void FlattenLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
{
    layer_pointer = new_layer_pointer;

    batch_samples_number = new_batch_samples_number;

    const FlattenLayer* flatten_layer_pointer = static_cast<const FlattenLayer*>(layer_pointer);

    const Index rows_number = flatten_layer_pointer->get_input_height();
    const Index columns_number = flatten_layer_pointer->get_input_width();
    const Index channels_number = flatten_layer_pointer->get_inputs_channels_number();

    deltas_data = static_cast<type*>(malloc(batch_samples_number * rows_number * columns_number * channels_number * sizeof(type)));
    deltas_dimensions.resize(2);
    deltas_dimensions.setValues({
        batch_samples_number,
        rows_number *
        columns_number *
        channels_number,
    });
}


void FlattenLayerBackPropagation::print() const
{
    //TODO: output
    cout << "Deltas: " << endl;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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

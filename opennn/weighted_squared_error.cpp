//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "weighted_squared_error.h"
#include "tensor_utilities.h"

namespace opennn
{

/// Default constructor.
/// It creates a weighted squared error term not associated with any
/// neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

WeightedSquaredError::WeightedSquaredError() : LossIndex()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a weighted squared error term object associated with a
/// neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


/// Returns the weight of the positives.

type WeightedSquaredError::get_positives_weight() const
{
    return positives_weight;
}


/// Returns the weight of the negatives.

type WeightedSquaredError::get_negatives_weight() const
{
    return negatives_weight;
}


type WeightedSquaredError::get_normalizaton_coefficient() const
{
    return normalization_coefficient;
}


/// Set the default values for the object.

void WeightedSquaredError::set_default()
{
    if(has_data_set() && !data_set_pointer->is_empty())
    {
        set_weights();

        set_normalization_coefficient();
    }
    else
    {
        negatives_weight = type(-1.0);
        positives_weight = type(-1.0);

        normalization_coefficient = type(-1.0);
    }
}


/// Set a new weight for the positives values.
/// @param new_positives_weight New weight for the positives.

void WeightedSquaredError::set_positives_weight(const type& new_positives_weight)
{
    positives_weight = new_positives_weight;
}


/// Set a new weight for the negatives values.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_negatives_weight(const type& new_negatives_weight)
{
    negatives_weight = new_negatives_weight;
}


/// Set new weights for the positives and negatives values.
/// @param new_positives_weight New weight for the positives.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_weights(const type& new_positives_weight, const type& new_negatives_weight)
{
    positives_weight = new_positives_weight;
    negatives_weight = new_negatives_weight;
}


/// Calculates of the weights for the positives and negatives values with the data of the data set.

void WeightedSquaredError::set_weights()
{
    // Control sentence

#ifdef OPENNN_DEBUG

    //    check();

#endif

    if(data_set_pointer->get_target_variables_number() == 0)
    {
        positives_weight = type(1);
        negatives_weight = type(1);
    }
    else if(data_set_pointer && data_set_pointer->get_target_columns()(0).type == DataSet::ColumnType::Binary)
    {
        const Tensor<Index, 1> target_distribution = data_set_pointer->calculate_target_distribution();

        const Index negatives = target_distribution[0];
        const Index positives = target_distribution[1];

        if(positives == 0 || negatives == 0)
        {
            positives_weight = type(1);
            negatives_weight = type(1);

            return;
        }

        negatives_weight = type(1);
        positives_weight = static_cast<type>(negatives)/static_cast<type>(positives);
    }
}


/// Calculates of the normalization coefficient with the data of the data set.

void WeightedSquaredError::set_normalization_coefficient()
{
    // Control sentence

#ifdef OPENNN_DEBUG

    check();

#endif

    if(data_set_pointer->get_target_columns().size()==0)
    {
        normalization_coefficient = static_cast<type>(1);
    }
    else if(data_set_pointer && data_set_pointer->get_target_columns()(0).type == DataSet::ColumnType::Binary)
    {
        const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_variables_indices();

        const Index negatives = data_set_pointer->calculate_used_negatives(target_variables_indices[0]);

        normalization_coefficient = type(negatives)*negatives_weight*static_cast<type>(0.5);
    }
    else
    {
        normalization_coefficient = static_cast<type>(1);
    }
}



/// \brief set_data_set_pointer
/// \param new_data_set_pointer

void WeightedSquaredError::set_data_set_pointer(DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;

    set_weights();

    set_normalization_coefficient();
}


void WeightedSquaredError::calculate_error(const DataSetBatch& batch,
                                           const NeuralNetworkForwardPropagation& forward_propagation,
                                           LossIndexBackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(trainable_layers_number-1);

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

    const TensorMap<Tensor<type, 2>> targets(batch.targets_data, batch.targets_dimensions(0), batch.targets_dimensions(1));

    const Tensor<Index, 1> outputs_dimensions = probabilistic_layer_forward_propagation->outputs_dimensions;

    const TensorMap<Tensor<type, 2>> outputs(probabilistic_layer_forward_propagation->outputs_data, outputs_dimensions(0), outputs_dimensions(1));

    const Tensor<bool, 2> if_sentence = elements_are_equal(targets, targets.constant(type(1)));
    const Tensor<bool, 2> else_sentence = elements_are_equal(targets, targets.constant(type(0)));

    Tensor<type, 2> f_1(targets.dimension(0), targets.dimension(1));
    f_1 = back_propagation.errors.square()*positives_weight;

    Tensor<type, 2> f_2(targets.dimension(0), targets.dimension(1));
    f_2 = back_propagation.errors.square()*negatives_weight;

    Tensor<type, 2> f_3(targets.dimension(0), targets.dimension(1));
    f_3 = outputs.constant(type(0));

    const Tensor<type, 0> weighted_sum_squared_error = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = (static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient;

    back_propagation.error = weighted_sum_squared_error(0)/coefficient;
}


void WeightedSquaredError::calculate_error_lm(const DataSetBatch& batch,
                                              const NeuralNetworkForwardPropagation&,
                                              LossIndexBackPropagationLM &back_propagation) const
{
    Tensor<type, 0> error;
    error.device(*thread_pool_device) = (back_propagation.squared_errors*back_propagation.squared_errors).sum();

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = (static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient;

    back_propagation.error = error()/coefficient;
}


void WeightedSquaredError::calculate_output_delta(const DataSetBatch& batch,
                                                  NeuralNetworkForwardPropagation& ,
                                                  LossIndexBackPropagation& back_propagation) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);

    const TensorMap<Tensor<type, 2>> targets(batch.targets_data, batch.targets_dimensions(0), batch.targets_dimensions(1));

    const Index batch_samples_number = batch.targets_dimensions(0);
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = static_cast<type>(2.0)/((static_cast<type>(batch_samples_number)/static_cast<type>(total_samples_number))*normalization_coefficient);

    const Tensor<bool, 2> if_sentence = elements_are_equal(targets, targets.constant(type(1)));
    const Tensor<bool, 2> else_sentence = elements_are_equal(targets, targets.constant(type(0)));

    Tensor<type, 2> f_1(targets.dimension(0), targets.dimension(1));
    f_1 = (coefficient*positives_weight)*back_propagation.errors;

    Tensor<type, 2> f_2(targets.dimension(0), targets.dimension(1));
    f_2 = coefficient*negatives_weight*back_propagation.errors;

    Tensor<type, 2> f_3(targets.dimension(0), targets.dimension(1));
    f_3 = targets.constant(type(0));

    TensorMap<Tensor<type, 2>> deltas(output_layer_back_propagation->deltas_data, output_layer_back_propagation->deltas_dimensions(0), output_layer_back_propagation->deltas_dimensions(1));

    deltas.device(*thread_pool_device) = if_sentence.select(f_1, else_sentence.select(f_2, f_3));
}


/// @todo Add gradient and hessian weighted squared error code (insted of normalized squared error)

void WeightedSquaredError::calculate_error_gradient_lm(const DataSetBatch& batch,
                                              LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = type(2)/((static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors, AT_B);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device) = coefficient * loss_index_back_propagation_lm.gradient;
}


void WeightedSquaredError::calculate_error_hessian_lm(const DataSetBatch& batch,
                                                           LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index batch_size = batch.get_batch_size();
    const Index total_samples_number = data_set_pointer->get_samples_number();

    const type coefficient = type(2)/((static_cast<type>(batch_size)/static_cast<type>(total_samples_number))*normalization_coefficient);

    loss_index_back_propagation_lm.hessian.device(*thread_pool_device)
            = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors_jacobian, AT_B);

    loss_index_back_propagation_lm.hessian.device(*thread_pool_device) = coefficient*loss_index_back_propagation_lm.hessian;
}


/// Returns a string with the name of the weighted squared error loss type, "WEIGHTED_SQUARED_ERROR".

string WeightedSquaredError::get_error_type() const
{
    return "WEIGHTED_SQUARED_ERROR";
}


/// Returns a string with the name of the weighted squared error loss type in text format.

string WeightedSquaredError::get_error_type_text() const
{
    return "Weighted squared error";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.
/// @param file_stream

void WeightedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Error type

    file_stream.OpenElement("WeightedSquaredError");

    // Positives weight

    file_stream.OpenElement("PositivesWeight");

    buffer.str("");
    buffer << positives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Negatives weight

    file_stream.OpenElement("NegativesWeight");

    buffer.str("");
    buffer << negatives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();
}


/// Loads a weighted squared error object from an XML document.
/// @param document Pointer to a TinyXML document with the object data.

void WeightedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("WeightedSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Weighted squared element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Positives weight

    const tinyxml2::XMLElement* positives_weight_element = root_element->FirstChildElement("PositivesWeight");

    if(positives_weight_element)
    {
        const string string = positives_weight_element->GetText();

        try
        {
            set_positives_weight(static_cast<type>(atof(string.c_str())));
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Negatives weight

    const tinyxml2::XMLElement* negatives_weight_element = root_element->FirstChildElement("NegativesWeight");

    if(negatives_weight_element)
    {
        const string string = negatives_weight_element->GetText();

        try
        {
            set_negatives_weight(static_cast<type>(atof(string.c_str())));
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }
}


type WeightedSquaredError::weighted_sum_squared_error(const Tensor<type, 2>& x, const Tensor<type, 2>& y) const
{
#ifdef __OPENNN_DEBUG__

    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1);

    const Index other_rows_number = y.dimension(0);

    if(other_rows_number != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double minkowski_error(const Matrix<double>&, const double&) method.\n"
               << "Other number of rows must be equal to this number of rows.\n";

        throw invalid_argument(buffer.str());
    }

    const Index other_columns_number = y.dimension(1);

    if(other_columns_number != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Metrics functions.\n"
               << "double minkowski_error(const Matrix<double>&, const double&) method.\n"
               << "Other number of columns must be equal to this number of columns.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Tensor<bool, 2> if_sentence = elements_are_equal(y, y.constant(type(1)));
    const Tensor<bool, 2> else_sentence = elements_are_equal(y, y.constant(type(0)));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_3(x.dimension(0), x.dimension(1));

    f_1 = (x - y).square()*positives_weight;

    f_2 = (x - y).square()*negatives_weight;

    f_3 = x.constant(type(0));

    const Tensor<type, 0> weighted_sum_squared_error = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    return weighted_sum_squared_error(0);
}


void WeightedSquaredError::calculate_squared_errors_lm(const DataSetBatch& batch,
                                                       const NeuralNetworkForwardPropagation& forward_propagation,
                                                       LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(trainable_layers_number-1);

    const TensorMap<Tensor<type, 2>> targets(batch.targets_data, batch.targets_dimensions(0), batch.targets_dimensions(1));

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

    const Tensor<Index, 1> outputs_dimensions = probabilistic_layer_forward_propagation->outputs_dimensions;

    const TensorMap<Tensor<type, 2>> outputs(probabilistic_layer_forward_propagation->outputs_data, outputs_dimensions(0), outputs_dimensions(1));

    const Tensor<bool, 2> if_sentence = elements_are_equal(outputs, outputs.constant(type(1)));

    Tensor<type, 2> f_1(outputs.dimension(0), outputs.dimension(1));
    f_1 = (outputs - targets)*positives_weight;

    Tensor<type, 2> f_2(outputs.dimension(0), outputs.dimension(1));
    f_2 = (outputs - targets)*negatives_weight;

    loss_index_back_propagation_lm.squared_errors = ((if_sentence.select(f_1, f_2)).sum(rows_sum).square()).sqrt();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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

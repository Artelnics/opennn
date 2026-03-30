//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "statistics.h"
#include "scaling.h"
#include "string_utilities.h"
#include "neural_network.h"

namespace opennn
{

template<int Rank> struct ScalingForwardPropagationCuda;

template<int Rank>
class Scaling final : public Layer
{

public:

    Scaling(const Shape& new_input_shape = Shape(Rank - 1, 0))
    {
        set(new_input_shape);
    }


    Shape get_input_shape() const override
    {
        return input_shape;
    }


    Shape get_output_shape() const override
    {
        return input_shape;
    }


    vector<Descriptives> get_descriptives() const
    {
        return descriptives;
    }


    Descriptives get_descriptives(const Index index) const
    {
        return descriptives[index];
    }


    VectorR get_minimums() const
    {
        const Index outputs_number = get_outputs_number();

        VectorR minimums(outputs_number);

        #pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            minimums[i] = descriptives[i].minimum;

        return minimums;
    }


    VectorR get_maximums() const
    {
        const Index outputs_number = get_outputs_number();

        VectorR maximums(outputs_number);

        #pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            maximums[i] = descriptives[i].maximum;

        return maximums;
    }


    VectorR get_means() const
    {
        const Index outputs_number = get_outputs_number();

        VectorR means(outputs_number);

        #pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            means[i] = descriptives[i].mean;

        return means;
    }


    VectorR get_standard_deviations() const
    {
        const Index outputs_number = get_outputs_number();

        VectorR standard_deviations(outputs_number);

        #pragma omp parallel for
        for(Index i = 0; i < outputs_number; i++)
            standard_deviations[i] = descriptives[i].standard_deviation;

        return standard_deviations;
    }


    vector<string> get_scalers() const
    {
        return scalers;
    }


    void set(const Shape& new_input_shape = {})
    {
        if (new_input_shape.size() != Rank -1) 
        {
           ostringstream buffer;
           buffer << "OpenNN Exception: Scaling Layer.\n"
                  << "void set(const Shape& new_input_shape) method.\n"
                  << "Input shape size must be " << Rank - 1 << ", but is " << new_input_shape.size() << ".\n";
           throw logic_error(buffer.str());
        }

        input_shape = new_input_shape;

        const Index new_inputs_number = new_input_shape.count();

        descriptives.resize(new_inputs_number);

        for(Index i = 0; i < new_inputs_number; i++)
            descriptives[i].set(type(-1.0), type(1), type(0), type(1));

        scalers.resize(new_inputs_number, "MeanStandardDeviation");

        label = "scaling_layer";

        set_scalers("MeanStandardDeviation");

        set_min_max_range(type(-1), type(1));

        name = "Scaling" + to_string(Rank) + "d";

        is_trainable = false;
    }


    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape);
    }


    void set_output_shape(const Shape& new_output_shape) override
    {
        set_input_shape(new_output_shape);
    }


    void set_descriptives(const vector<Descriptives>& new_descriptives)
    {
        descriptives = new_descriptives;
    }

    void set_min_max_range(const type min, type max)
    {
        min_range = min;
        max_range = max;
    }


    void set_scalers(const vector<string>& new_scalers)
    {
        scalers = new_scalers;
    }


    void set_scalers(const string& new_scaler)
    {
        for(string& scaler : scalers)
            scaler = new_scaler;
    }


    void forward_propagate(ForwardPropagation& forward_propagation, size_t index, bool) override
    {
/*
        const Index features = scalers.size();
        const Index total_rows = forward_propagation->inputs[0].size() / features;

        const MatrixMap inputs(forward_propagation->inputs[0].data, total_rows, features);
        MatrixMap outputs(forward_propagation->outputs.data, total_rows, features);

        outputs.noalias() = inputs;

        for(Index i = 0; i < features; i++)
        {
            const string& scaler = scalers[i];
            if(scaler == "None") continue;

            const Descriptives& descriptive = descriptives[i];

            if(scaler == "MeanStandardDeviation")
                outputs.col(i).array() = (outputs.col(i).array() - descriptive.mean) / (descriptive.standard_deviation + EPSILON);
            else if(scaler == "MinimumMaximum")
                outputs.col(i).array() = (outputs.col(i).array() - descriptive.minimum) / ((descriptive.maximum - descriptive.minimum) + EPSILON) * (max_range - min_range) + min_range;
            else if(scaler == "StandardDeviation")
                outputs.col(i).array() /= (descriptive.standard_deviation + EPSILON);
            else if(scaler == "Logarithm")
                outputs.col(i).array() = outputs.col(i).array().log();
            else if(scaler == "ImageMinMax")
                outputs.col(i).array() /= type(255.0);
            else
                throw runtime_error("Unknown scaling method in Scaling Layer: " + scaler);
        }
*/
    }


    string write_no_scaling_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().empty() ? 0 : get_output_shape().count();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = " << input_names[i] << ";\n";

        return buffer.str();
    }


    string write_minimum_maximum_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().empty() ? 0 : get_output_shape().count();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = 2*(" << input_names[i] << "-(" << descriptives[i].minimum << "))/(" << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-1;\n";

        return buffer.str();
    }


    string write_mean_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_inputs_number();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = (" << input_names[i] << "-(" << descriptives[i].mean << "))/" << descriptives[i].standard_deviation << ";\n";

        return buffer.str();
    }


    string write_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().empty() ? 0 : get_output_shape().count();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = " << input_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";

        return buffer.str();
    }


    string get_expression(const vector<string>& new_input_names = vector<string>(), const vector<string>& = vector<string>()) const override
    {
        const vector<string> input_names = new_input_names.empty()
                                               ? get_default_feature_names()
                                               : new_input_names;

        const Index outputs_number = get_outputs_number();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < outputs_number; i++)
        {
            const string& scaler = scalers[i];

            if(scaler == "None")
                buffer << "scaled_" << input_names[i] << " = " << input_names[i] << ";\n";
            else if(scaler == "MinimumMaximum")
                buffer << "scaled_" << input_names[i]
                       << " = " << input_names[i] << "*(" << max_range << "-" << min_range << ")/("
                       << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-" << descriptives[i].minimum << "*("
                       << max_range << "-" << min_range << ")/("
                       << descriptives[i].maximum << "-" << descriptives[i].minimum << ")+" << min_range << ";\n";
            else if(scaler == "MeanStandardDeviation")
                buffer << "scaled_" << input_names[i] << " = (" << input_names[i] << "-" << descriptives[i].mean << ")/" << descriptives[i].standard_deviation << ";\n";
            else if(scaler == "StandardDeviation")
                buffer << "scaled_" << input_names[i] << " = " << input_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";
            else if(scaler == "Logarithm")
                buffer << "scaled_" << input_names[i] << " = log(" << input_names[i] << ");\n";
            else
                throw runtime_error("Unknown inputs scaling method.\n");
        }

        string expression = buffer.str();

        expression = std::regex_replace(expression, std::regex("\\+-"), "-");
        expression = std::regex_replace(expression, std::regex("--"), "+");

        return expression;
    }


    void print() const override
    {
        cout << "Scaling layer" << endl;

        const Index inputs_number = get_inputs_number();

        for(Index i = 0; i < inputs_number; i++)
        {
            cout << "Neuron " << i << endl
                 << "string " << scalers[i] << endl;

            descriptives[i].print();
        }
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* scaling_layer_element = document.FirstChildElement(name.c_str());

        if(!scaling_layer_element)
            throw runtime_error("Scaling element is nullptr.\n");

        const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");
        
        if constexpr (Rank == 2)
            set({ neurons_number });
        else
            set(Shape(Rank - 1, neurons_number));

        const XMLElement* start_element = scaling_layer_element->FirstChildElement("NeuronsNumber");

        for(Index i = 0; i < neurons_number; i++)
        {
            const XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
            if(!scaling_neuron_element)
                throw runtime_error("Scaling neuron " + to_string(i + 1) + " is nullptr.\n");

            const XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");
            if (descriptives_element && descriptives_element->GetText()) {
                const vector<string> tokens = get_tokens(descriptives_element->GetText(), " ");
                descriptives[i].set(
                    type(stof(tokens[0])),
                    type(stof(tokens[1])),
                    type(stof(tokens[2])),
                    type(stof(tokens[3]))
                    );
            }

            scalers[i] = read_xml_string(scaling_neuron_element, "Scaler");
            start_element = scaling_neuron_element;
        }
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement(name.c_str());

        const Index outputs_number = get_outputs_number();

        add_xml_element(printer, "NeuronsNumber", to_string(outputs_number));

        for(Index i = 0; i < outputs_number; i++)
        {
            printer.OpenElement("ScalingNeuron");
            printer.PushAttribute("Index", int(i + 1));
            add_xml_element(printer, "Descriptives", vector_to_string(descriptives[i].to_tensor()));
            add_xml_element(printer, "Scaler", scalers[i]);
            printer.CloseElement();
        }

        printer.CloseElement();
    }

#ifdef OPENNN_CUDA

    void forward_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation, bool) override
    {
        ScalingForwardPropagationCuda<Rank>* scaling_forward_propagation =
            static_cast<ScalingForwardPropagationCuda<Rank>*>(forward_propagation.get());

        const Index outputs_number = get_outputs_number();
        const size_t size = outputs_number * forward_propagation->batch_size;

        scale_2d_cuda(size,
                      forward_propagation->batch_size,
                      outputs_number,
                      forward_propagation->inputs[0].data,
                      forward_propagation->outputs.data,
                      scaling_forward_propagation->scalers_device,
                      scaling_forward_propagation->minimums_device,
                      scaling_forward_propagation->maximums_device,
                      scaling_forward_propagation->means_device,
                      scaling_forward_propagation->standard_deviations_device,
                      min_range,
                      max_range);
    }

#endif

private:



    Shape input_shape;

    vector<Descriptives> descriptives;

    vector<string> scalers;

    type min_range;
    type max_range;
};


template<int Rank>
struct ScalingForwardPropagation final : LayerForwardPropagation
{
    void initialize() override
    {
        outputs.shape = Shape{batch_size}.append(layer->get_output_shape());
    }
};


#ifdef OPENNN_CUDA

template<int Rank>
struct ScalingForwardPropagationCuda : public LayerForwardPropagationCuda
{
    void initialize() override
    {
        const Index outputs_number = scaling_layer->get_outputs_number();

        outputs.set_descriptor({static_cast<int>(batch_size), static_cast<int>(outputs_number)});

        const VectorR minimums_host = scaling_layer->get_minimums();
        const VectorR maximums_host = scaling_layer->get_maximums();
        const VectorR means_host = scaling_layer->get_means();
        const VectorR std_devs_host = scaling_layer->get_standard_deviations();
        const vector<string> scalers_host_vec = scaling_layer->get_scalers();

        Tensor<int, 1> scalers_host_tensor(outputs_number);

        for(Index i = 0; i < outputs_number; ++i)
        {
            const string & scaler_str = scalers_host_vec[i];

            if (scaler_str == "None")
                scalers_host_tensor(i) = 0;
            else if (scaler_str == "MinimumMaximum")
                scalers_host_tensor(i) = 1;
            else if (scaler_str == "MeanStandardDeviation")
                scalers_host_tensor(i) = 2;
            else if (scaler_str == "StandardDeviation")
                scalers_host_tensor(i) = 3;
            else if (scaler_str == "Logarithm")
                scalers_host_tensor(i) = 4;
            else if (scaler_str == "ImageMinMax")
                scalers_host_tensor(i) = 5;
            else
                throw runtime_error("Unknown scaler method for CUDA: " + scaler_str);
        }

        CHECK_CUDA(cudaMalloc(&minimums_device, outputs_number * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&maximums_device, outputs_number * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&means_device, outputs_number * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&standard_deviations_device, outputs_number * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&scalers_device, outputs_number * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(minimums_device, minimums_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(maximums_device, maximums_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(means_device, means_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(standard_deviations_device, std_devs_host.data(), outputs_number * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(scalers_device, scalers_host_tensor.data(), outputs_number * sizeof(int), cudaMemcpyHostToDevice));
    }

    void print() const override
    {
        // @todo
    }

    void free() override
    {
        cudaFree(scalers_device);
        scalers_device = nullptr;

        cudaFree(minimums_device);
        minimums_device = nullptr;

        cudaFree(maximums_device);
        maximums_device = nullptr;

        cudaFree(means_device);
        means_device = nullptr;

        cudaFree(standard_deviations_device);
        standard_deviations_device = nullptr;
    }

    int* scalers_device = nullptr;
    type* minimums_device = nullptr;
    type* maximums_device = nullptr;
    type* means_device = nullptr;
    type* standard_deviations_device = nullptr;
};

#endif


void reference_scaling_layer();

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

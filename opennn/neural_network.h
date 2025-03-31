//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"

namespace opennn
{

struct ForwardPropagation;
struct NeuralNetworkBackPropagation;
struct NeuralNetworkBackPropagationLM;

#ifdef OPENNN_CUDA
struct ForwardPropagationCuda;
struct NeuralNetworkBackPropagationCuda;
#endif

class NeuralNetwork
{

public:

   enum class ModelType{Default,
                        AutoAssociation,
                        Approximation,
                        Classification,
                        Forecasting,
                        ImageClassification,
                        TextClassification};

   NeuralNetwork(const NeuralNetwork::ModelType& = NeuralNetwork::ModelType::Default,
                 const dimensions& = {},
                 const dimensions& = {},
                 const dimensions& = {});

   NeuralNetwork(const filesystem::path&);

   void add_layer(unique_ptr<Layer>, 
                  const vector<Index>& = vector<Index>());

   bool validate_layer_type(const Layer::Type&) const;

   // Get

   bool has(const Layer::Type&) const;

   bool is_empty() const;

   const vector<string>& get_input_names() const;
   Index get_input_index(const string&) const;

   ModelType get_model_type() const;
   string get_model_type_string() const;

   const vector<string>& get_output_names() const;
   Index get_output_index(const string&) const;

   const vector<unique_ptr<Layer>>& get_layers() const;
   const unique_ptr<Layer>& get_layer(const Index&) const;
   const unique_ptr<Layer>& get_layer(const string&) const;

   Index get_layer_index(const string&) const;

   const vector<vector<Index>>& get_layer_input_indices() const;
   vector<vector<Index>> get_layer_output_indices() const;

   Index find_input_index(const vector<Index>&, const Index&) const;

   Layer* get_first(const Layer::Type&) const;

   const bool& get_display() const;

   // Set

   void set(const NeuralNetwork::ModelType& = NeuralNetwork::ModelType::Default,
            const dimensions& = {}, 
            const dimensions& = {},
            const dimensions& = {});

   void set_approximation(const dimensions&, const dimensions&, const dimensions&);
   void set_classification(const dimensions&, const dimensions&, const dimensions&);
   void set_forecasting(const dimensions&, const dimensions&, const dimensions&);
   void set_auto_association(const dimensions&, const dimensions&, const dimensions&);
   void set_image_classification(const dimensions&, const dimensions&, const dimensions&);
   void set_text_classification_transformer(const dimensions&, const dimensions&, const dimensions&);

   void set(const filesystem::path&);

   void set_layers_number(const Index&);

   void set_layer_input_indices(const vector<vector<Index>>&);
   void set_layer_inputs_indices(const Index&, const vector<Index>&);

   void set_layer_inputs_indices(const string&, const vector<string>&);
   void set_layer_inputs_indices(const string&, const initializer_list<string>&);
   void set_layer_inputs_indices(const string&, const string&);

   void set_model_type(const ModelType&);
   void set_model_type_string(const string&);
   void set_input_names(const vector<string>&);
   void set_output_names(const vector<string>&);

   void set_input_dimensions(const dimensions&);

   void set_default();

   void set_threads_number(const int&);

   void set_display(const bool&);

   // Layers

   static bool is_trainable(const Layer::Type&);

   Index get_layers_number() const;
   Index get_layers_number(const Layer::Type&) const;

   Index get_first_trainable_layer_index() const;
   Index get_last_trainable_layer_index() const;

   // Architecture

   Index get_inputs_number() const;
   Index get_outputs_number() const;
   dimensions get_output_dimensions() const;

   // Parameters

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   vector<Index> get_layer_parameter_numbers() const;

   void set_parameters(const Tensor<type, 1>&) const;

   // Parameters initialization

   void set_parameters_constant(const type&) const;

   void set_parameters_random() const;

   // Output

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   Tensor<type, 3> calculate_outputs(const Tensor<type, 3>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 4>&);

   Tensor<type, 2> calculate_scaled_outputs(type*, Tensor<Index, 1>& );

   Tensor<type, 2> calculate_directional_inputs(const Index&, const Tensor<type, 1>&, const type&, const type&, const Index& = 101) const;

   Index calculate_image_output(const filesystem::path&);

   // Serialization

   Tensor<string, 2> get_perceptron_layers_information() const;
   Tensor<string, 2> get_probabilistic_layer_information() const;

   void from_XML(const XMLDocument&);
   void inputs_from_XML(const XMLElement*);
   void layers_from_XML(const XMLElement*);
   void outputs_from_XML(const XMLElement*);

   void to_XML(XMLPrinter&) const;

   virtual void print() const;
   void save(const filesystem::path&) const;
   void save_parameters(const filesystem::path&) const;

   void load(const filesystem::path&);
   void load_parameters_binary(const filesystem::path&);

   vector<string> get_layer_names() const;
   vector<string> get_layer_types_string() const;

   void save_outputs(Tensor<type, 2>&, const filesystem::path&);

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          ForwardPropagation&,
                          const bool& = false) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          const Tensor<type, 1>&, 
                          ForwardPropagation&) const;

   string get_expression() const;


#ifdef OPENNN_CUDA

public:

    NeuralNetworkCuda();
    ~NeuralNetworkCuda();

    void create_cuda();
    void destroy_cuda();

    void allocate_parameters_device();
    void free_parameters_device();
    void copy_parameters_device();
    void copy_parameters_host();

    void forward_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                                NeuralNetwork::ForwardPropagationCuda&,
                                const bool&) const;

    void get_parameters_cuda(Tensor<type, 1>&);
    void set_parameters_cuda(float*);

protected:

    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;

    Index get_first_trainable_layer_index() const;

    Index get_last_trainable_layer_index() const;

    Tensor<Layer*, 1> get_trainable_layers() const;

    Tensor<Index, 1> get_trainable_layers_parameters_numbers() const;

    Tensor<Tensor<Index, 1>, 1> get_layers_inputs_indices() const;

    bool is_input_layer(const Tensor<Index, 1>&) const;

    bool is_context_layer(const Tensor<Index, 1>&) const;

private:

    NeuralNetworkCuda(const NeuralNetworkCuda&) = delete;
    NeuralNetworkCuda& operator=(const NeuralNetworkCuda&) = delete;

#endif

protected:

   string name = "neural_network";

   NeuralNetwork::ModelType model_type = NeuralNetwork::ModelType::Default;

   vector<string> input_names;

   vector<string> output_names;

   vector<unique_ptr<Layer>> layers;

   vector<vector<Index>> layer_input_indices;

   bool display = true;

};

#ifdef OPENNN_CUDA

struct ForwardPropagationCuda
{
    ForwardPropagationCuda();

    ForwardPropagationCuda(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    virtual ~ForwardPropagationCuda();

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    void print();

    void free();

    Index batch_samples_number = 0;
    NeuralNetwork* neural_network = nullptr;
    Tensor<LayerForwardPropagationCuda*, 1> layers;
};


struct NeuralNetworkBackPropagationCuda
{
    NeuralNetworkBackPropagationCuda();

    NeuralNetworkBackPropagationCuda(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    virtual ~NeuralNetworkBackPropagationCuda();

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    void print();

    void free();

    Index batch_samples_number = 0;
    NeuralNetwork* neural_network = nullptr;
    Tensor<LayerBackPropagationCuda*, 1> layers;
};

#endif

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <errno.h>

// OpenNN includes

#include "config.h"
#include "data_set.h"
#include "layer.h"
#include "addition_layer.h"
#include "perceptron_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "probabilistic_layer.h"
#include "convolutional_layer.h"
#include "flatten_layer.h"
//#include "region_proposal_layer.h"
#include "non_max_suppression_layer.h"

#include "pooling_layer.h"
#include "long_short_term_memory_layer.h"
#include "recurrent_layer.h"
#include "text_analytics.h"

namespace opennn
{

struct NeuralNetworkForwardPropagation;
struct NeuralNetworkBackPropagation;

/// This class represents the concept of neural network in the OpenNN library.
///
/// This neural network spans a function space for the problem at hand.

class NeuralNetwork
{

public:

   enum class ProjectType{Approximation,
                          Classification,
                          Forecasting,
                          ImageClassification,
                          TextClassification,
                          TextGeneration,
                          AutoAssociation};

   // Constructors

   explicit NeuralNetwork();

   explicit NeuralNetwork(const NeuralNetwork::ProjectType&, const Tensor<Index, 1>&);

   explicit NeuralNetwork(const NeuralNetwork::ProjectType&, const initializer_list<Index>&);

   explicit NeuralNetwork(const Tensor<Index, 1>&, const Index&, const Tensor<Index, 1>&, const Index&);

   explicit NeuralNetwork(const string&);

   explicit NeuralNetwork(const tinyxml2::XMLDocument&);

   explicit NeuralNetwork(const Tensor<Layer*, 1>&);

   // Destructor

   virtual ~NeuralNetwork();

   // APPENDING LAYERS

   void delete_layers();

   void add_layer(Layer*);

   bool check_layer_type(const Layer::Type);

   // Get methods

   bool has_scaling_layer() const;
   bool has_long_short_term_memory_layer() const;
   bool has_recurrent_layer() const;
   bool has_unscaling_layer() const;
   bool has_bounding_layer() const;
   bool has_probabilistic_layer() const;
   bool has_convolutional_layer() const;
   bool has_flatten_layer() const;
   bool is_empty() const;

   const Tensor<string, 1>& get_inputs_names() const;
   string get_input_name(const Index&) const;
   Index get_input_index(const string&) const;

   ProjectType get_project_type() const;
   string get_project_type_string() const;

   const Tensor<string, 1>& get_outputs_names() const;
   string get_output_name(const Index&) const;
   Index get_output_index(const string&) const;

   Tensor<Layer*, 1> get_layers_pointers() const;
   Layer* get_layer_pointer(const Index&) const;
   Tensor<Layer*, 1> get_trainable_layers_pointers() const;
   Tensor<Index, 1> get_trainable_layers_indices() const;   

   Index get_layer_index(const string&) const;

   Tensor<Tensor<Index, 1>, 1> get_layers_inputs_indices() const;

   ScalingLayer* get_scaling_layer_pointer() const;
   UnscalingLayer* get_unscaling_layer_pointer() const;
   BoundingLayer* get_bounding_layer_pointer() const;
   FlattenLayer* get_flatten_layer_pointer() const;
   //ConvolutionalLayer* get_convolutional_layer_pointer() const;
   PoolingLayer* get_pooling_layer_pointer() const;
   ProbabilisticLayer* get_probabilistic_layer_pointer() const;
   LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const;
   RecurrentLayer* get_recurrent_layer_pointer() const;

   Layer* get_last_trainable_layer_pointer() const;
   PerceptronLayer* get_first_perceptron_layer_pointer() const;

   Index get_batch_samples_number() const;

   const bool& get_display() const;

   // Set methods

   void set();

   void set(const NeuralNetwork::ProjectType&, const Tensor<Index, 1>&);
   void set(const NeuralNetwork::ProjectType&, const initializer_list<Index>&);
   void set(const Tensor<Index, 1>&, const Index&, const Tensor<Index, 1>&, const Index&);

   void set(const string&);

   void set_layers_pointers(Tensor<Layer*, 1>&);

   void set_layers_inputs_indices(const Tensor<Tensor<Index, 1>, 1>&);
   void set_layer_inputs_indices(const Index&, const Tensor<Index, 1>&);

   void set_layer_inputs_indices(const string&, const Tensor<string, 1>&);
   void set_layer_inputs_indices(const string&, const string&);

   void set_project_type(const ProjectType&);
   void set_project_type_string(const string&);
   void set_inputs_names(const Tensor<string, 1>&);
   void set_outputs_names(const Tensor<string, 1>&);

   void set_inputs_number(const Index&);
   void set_inputs_number(const Tensor<bool, 1>&);

   // AANN histogram

   // Histogram parameters

   void set_box_plot_minimum(const type&);
   void set_box_plot_first_quartile(const type&);
   void set_box_plot_median(const type&);
   void set_box_plot_third_quartile(const type&);
   void set_box_plot_maximum(const type&);

   virtual void set_default();

   void set_threads_number(const int&);

   void set_scaling_layer(ScalingLayer&);

   void set_display(const bool&);

   void set_distances_box_plot(BoxPlot&);
   void set_multivariate_distances_box_plot(Tensor<BoxPlot, 1>&);
   void set_variables_distances_names(const Tensor<string, 1>&);
   void set_distances_descriptives(Descriptives&);

   // Layers

   Index get_layers_number() const;
   Tensor<Index, 1> get_layers_neurons_numbers() const;

   Index get_trainable_layers_number() const;
   Index get_first_trainable_layer_index() const;
   Index get_last_trainable_layer_index() const;

   Index get_perceptron_layers_number() const;
   Index get_probabilistic_layers_number() const;
   Index get_flatten_layers_number() const;
   Index get_convolutional_layers_number() const;
   Index get_pooling_layers_number() const;
   Index get_long_short_term_memory_layers_number() const;
   Index get_recurrent_layers_number() const;

   // Architecture

   Index get_inputs_number() const;
   Index get_outputs_number() const;

   Tensor<Index, 1> get_trainable_layers_neurons_numbers() const;
   Tensor<Index, 1> get_trainable_layers_inputs_numbers() const;

   Tensor<Index, 1> get_architecture() const;

   // Parameters

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   Tensor<Index, 1> get_trainable_layers_parameters_numbers() const;

   // AANN histogram

   BoxPlot get_auto_associative_distances_box_plot() const;
   Descriptives get_distances_descriptives() const;

   type get_box_plot_minimum() const;
   type get_box_plot_first_quartile() const;
   type get_box_plot_median() const;
   type get_box_plot_third_quartile() const;
   type get_box_plot_maximum() const;

   Tensor<BoxPlot, 1> get_multivariate_distances_box_plot() const;
   Tensor<type, 1> get_multivariate_distances_box_plot_minimums() const;
   Tensor<type, 1> get_multivariate_distances_box_plot_first_quartile() const;
   Tensor<type, 1> get_multivariate_distances_box_plot_median() const;
   Tensor<type, 1> get_multivariate_distances_box_plot_third_quartile() const;
   Tensor<type, 1> get_multivariate_distances_box_plot_maximums() const;

   void set_parameters(Tensor<type, 1>&) const;

   // Parameters initialization methods

   void set_parameters_constant(const type&) const;

   void set_parameters_random() const;

   // Parameters

   type calculate_parameters_norm() const;

   void perturbate_parameters(const type&);

   // Output

   Tensor<type, 2> calculate_outputs(type*, Tensor<Index, 1>&);
   Tensor<type, 2> calculate_unscaled_outputs(type*, Tensor<Index, 1>&);
   Tensor<type, 2> calculate_outputs(Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(Tensor<type, 4>&);

   Tensor<type, 2> calculate_scaled_outputs(type*, Tensor<Index, 1>&);

   Tensor<type, 2> calculate_multivariate_distances(type* &, Tensor<Index,1>&, type* &, Tensor<Index,1>&);
   Tensor<type, 1> calculate_samples_distances(type* &, Tensor<Index,1>&, type* &, Tensor<Index,1>&);

   Tensor<type, 2> calculate_directional_inputs(const Index&, const Tensor<type, 1>&, const type&, const type&, const Index& = 101) const;

   // Text generation

   string calculate_text_outputs(TextGenerationAlphabet&, const string&, const Index&, const bool&);

   string generate_word(TextGenerationAlphabet&, const string&, const Index&);

   string generate_phrase(TextGenerationAlphabet&, const string&, const Index&);

   // Serialization methods

   Tensor<string, 2> get_information() const;
   Tensor<string, 2> get_perceptron_layers_information() const;
   Tensor<string, 2> get_probabilistic_layer_information() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);
   void inputs_from_XML(const tinyxml2::XMLDocument&);
   void layers_from_XML(const tinyxml2::XMLDocument&);
   void outputs_from_XML(const tinyxml2::XMLDocument&);
   void box_plot_from_XML(const tinyxml2::XMLDocument&);
   void distances_descriptives_from_XML(const tinyxml2::XMLDocument&);
   void multivariate_box_plot_from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML( );

   void print() const;
   void print_layers_inputs_indices() const;
   void summary() const;
   void save(const string&) const;
   void save_parameters(const string&) const;

   virtual void load(const string&);
   void load_parameters_binary(const string&);

   Tensor<string, 1> get_layers_names() const;

   // Expression methods

   string write_expression_autoassociation_distances(string&, string&) const;
   string write_expression_autoassociation_variables_distances(string&, string&) const;
   string write_expression() const;

   string write_expression_python() const;
   string write_expression_c() const;
   string write_expression_api() const;
   string write_expression_javascript(const Tensor<string, 1>&, const Tensor<Index, 1>&, const Tensor<Tensor<string, 1>, 1>&) const;

   void save_expression_c(const string&) const;
   void save_expression_python(const string&) const;
   void save_expression_api(const string&) const;
   void save_expression_javascript(const string&, const Tensor<string, 1>&, const Tensor<Index, 1>&, const Tensor<Tensor<string, 1>, 1>&) const;
   void save_outputs(Tensor<type, 2>&, const string&);

   void save_autoassociation_outputs(const Tensor<type, 1>&,const Tensor<string, 1>&, const string&) const;

   /// Calculate forward propagation in neural network

   void forward_propagate(const DataSetBatch&, NeuralNetworkForwardPropagation&, bool&) const;

   void forward_propagate_deploy(DataSetBatch&, NeuralNetworkForwardPropagation&) const;

   void forward_propagate(const DataSetBatch&, Tensor<type, 1>&, NeuralNetworkForwardPropagation&) const;



protected:

   string name = "neural_network";

   NeuralNetwork::ProjectType project_type;

   /// Names of inputs

   Tensor<string, 1> inputs_names;

   /// Names of ouputs

   Tensor<string, 1> outputs_names;

   /// Layers

   Tensor<Layer*, 1> layers_pointers;

   Tensor<Tensor<Index, 1>, 1> layers_inputs_indices;

   /// AANN distances box plot

   BoxPlot auto_associative_distances_box_plot = BoxPlot();
   Tensor<BoxPlot, 1> multivariate_distances_box_plot;
   Descriptives distances_descriptives = Descriptives();
   Tensor<string, 1> variables_distances_names = Tensor<string, 1>();

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/neural_network_cuda.h"
#endif

};


struct NeuralNetworkForwardPropagation
{
    /// Default constructor.

    NeuralNetworkForwardPropagation() {}

    NeuralNetworkForwardPropagation(const Index& new_batch_samples_number,NeuralNetwork* new_neural_network_pointer)
    {
        set(new_batch_samples_number, new_neural_network_pointer);
    }

    /// Destructor.

    virtual ~NeuralNetworkForwardPropagation()
    {
        const Index layers_number = layers.size();

        for(Index i = 0; i < layers_number; i++)
        {
            delete layers(i);
        }
    }


    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;

        const Tensor<Layer*, 1> layers_pointers = neural_network_pointer->get_layers_pointers();

        const Index layers_number = layers_pointers.size();

        layers.resize(layers_number);

        for(Index i = 0; i < layers_number; i++)
        {
            switch (layers_pointers(i)->get_type())
            {
            case Layer::Type::Perceptron:
            {
                layers(i) = new PerceptronLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;
            case Layer::Type::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::Recurrent:
            {
                layers(i) = new RecurrentLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::LongShortTermMemory:
            {
                layers(i) = new LongShortTermMemoryLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Convolutional:
            {
            //    layers(i) = new ConvolutionalLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Pooling:
            {
                layers(i) = new PoolingLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Flatten:
            {
                layers(i) = new FlattenLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Scaling:
            {
                layers(i) = new ScalingLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Unscaling:
            {
                layers(i) = new UnscalingLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Bounding:
            {
                layers(i) = new BoundingLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::RegionProposal:
            {
//                layers(i) = new RegionProposalLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::NonMaxSuppression:
            {
                layers(i) = new NonMaxSuppressionLayerForwardPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            default: break;
            }
        }
    }


    void print() const
    {
        cout << "Neural network forward propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << ": " << layers(i)->layer_pointer->get_name() << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<LayerForwardPropagation*, 1> layers;
};


struct NeuralNetworkBackPropagation
{
    NeuralNetworkBackPropagation() {}

    virtual ~NeuralNetworkBackPropagation()
    {
        const Index layers_number = layers.size();

        for(Index i = 0; i < layers_number; i++)
        {
            delete layers(i);
        }
    }

    NeuralNetworkBackPropagation(NeuralNetwork* new_neural_network_pointer)
    {
        neural_network_pointer = new_neural_network_pointer;
    }

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        const Index trainable_layers_number = trainable_layers_pointers.size();

        layers.resize(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            switch (trainable_layers_pointers(i)->get_type())
            {
            case Layer::Type::Perceptron:
            {
                layers(i) = new PerceptronLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Type::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Type::Recurrent:
            {
                layers(i) = new RecurrentLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Type::LongShortTermMemory:
            {
                layers(i) = new LongShortTermMemoryLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Type::Convolutional:
            {
            //    layers(i) = new ConvolutionalLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Type::Pooling:
            {
                layers(i) = new PoolingLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            case Layer::Type::Flatten:
            {
                layers(i) = new FlattenLayerBackPropagation(batch_samples_number, trainable_layers_pointers(i));
            }
            break;

            default: break;
            }
        }
    }

    void print() const
    {
        cout << "Neural network back-propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<LayerBackPropagation*, 1> layers;
};


struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM() {}

    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network_pointer)
    {
        neural_network_pointer = new_neural_network_pointer;
    }

    virtual ~NeuralNetworkBackPropagationLM()
    {
        const Index layers_number = layers.size();

        for(Index i = 0; i < layers_number; i++)
        {
            delete layers(i);
        }
    }

    void set(const Index new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network_pointer = new_neural_network_pointer;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        const Index trainable_layers_number = trainable_layers_pointers.size();

        layers.resize(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            switch (trainable_layers_pointers(i)->get_type())
            {
            case Layer::Type::Perceptron:

            layers(i) = new PerceptronLayerBackPropagationLM(batch_samples_number, trainable_layers_pointers(i));

            break;

            case Layer::Type::Probabilistic:

            layers(i) = new ProbabilisticLayerBackPropagationLM(batch_samples_number, trainable_layers_pointers(i));

            break;

            default:
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

                throw invalid_argument(buffer.str());
            }
            }
        }
    }

    void print()
    {
        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<LayerBackPropagationLM*, 1> layers;
};

}



#endif

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

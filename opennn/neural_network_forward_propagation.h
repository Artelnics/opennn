#ifndef NEURALNETWORKFORWARDPROPAGATION_H
#define NEURALNETWORKFORWARDPROPAGATION_H

#include <string>


#include "neural_network.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct ForwardPropagation
{
    /// Default constructor.

    ForwardPropagation() {}

    ForwardPropagation(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network_pointer)
    {
        set(new_batch_samples_number, new_neural_network_pointer);
    }

    /// Destructor.

    virtual ~ForwardPropagation()
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
            switch(layers_pointers(i)->get_type())
            {
            case Layer::Type::Perceptron:
            {
                layers(i) = new PerceptronLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::Perceptron3D:
            {
                layers(i) = new PerceptronLayer3DForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::Probabilistic3D:
            {
                layers(i) = new ProbabilisticLayer3DForwardPropagation(batch_samples_number, layers_pointers(i));

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

            case Layer::Type::Scaling2D:
            {
                layers(i) = new ScalingLayer2DForwardPropagation(batch_samples_number, layers_pointers(i));
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

            case Layer::Type::Embedding:
            {
                layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::MultiheadAttention:
            {
                layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, layers_pointers(i));

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


}
#endif

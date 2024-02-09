#ifndef NEURALNETWORKBACKPROPAGATION_H
#define NEURALNETWORKBACKPROPAGATION_H

#include <string>

#include "neural_network.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

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

        const Tensor<Layer*, 1> layers_pointers = neural_network_pointer->get_layers_pointers();

        const Index layers_number = layers_pointers.size();

        layers.resize(layers_number);
        layers.setConstant(nullptr);

        for(Index i = 0; i < layers_number; i++)
        {
            switch(layers_pointers(i)->get_type())
            {
            case Layer::Type::Perceptron:
            {
                layers(i) = new PerceptronLayerBackPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerBackPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Recurrent:
            {
                layers(i) = new RecurrentLayerBackPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::LongShortTermMemory:
            {
                layers(i) = new LongShortTermMemoryLayerBackPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Convolutional:
            {
                layers(i) = new ConvolutionalLayerBackPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Pooling:
            {
                layers(i) = new PoolingLayerBackPropagation(batch_samples_number, layers_pointers(i));
            }
            break;

            case Layer::Type::Flatten:
            {
                layers(i) = new FlattenLayerBackPropagation(batch_samples_number, layers_pointers(i));
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


}
#endif

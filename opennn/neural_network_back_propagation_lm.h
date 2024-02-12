#ifndef NEURALNETWORKBACKPROPAGATIONLM_H
#define NEURALNETWORKBACKPROPAGATIONLM_H

#include <string>

#include "neural_network.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM() {}

    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network)
    {
        neural_network = new_neural_network;
    }

    virtual ~NeuralNetworkBackPropagationLM()
    {
        const Index layers_number = layers.size();

        for(Index i = 0; i < layers_number; i++)
        {
            delete layers(i);
        }
    }


    void set(const Index new_batch_samples_number, NeuralNetwork* new_neural_network)
    {
        batch_samples_number = new_batch_samples_number;

        neural_network = new_neural_network;

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network->get_trainable_layers();

        const Index trainable_layers_number = trainable_layers_pointers.size();

        layers.resize(trainable_layers_number);

        for(Index i = 0; i < trainable_layers_number; i++)
        {
            switch(trainable_layers_pointers(i)->get_type())
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

    NeuralNetwork* neural_network = nullptr;

    Tensor<LayerBackPropagationLM*, 1> layers;
};

}
#endif

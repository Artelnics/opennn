//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        cout << "Hello OpenNN!" << endl;

        DataSet data_set("C:/Users/rodrigo ingelmo/Downloads/sum.csv", ';', false);

        //El problema era que el get_input_variables_number es un float y no un Index, por eso no iba
        type input_variables_number = data_set.get_input_variables_number();
        cout << input_variables_number<<endl;
        type target_variables_number = data_set.get_target_variables_number();
        cout << target_variables_number<< endl;

        Index hidden_neurons_number=3;

        cout << Index(target_variables_number);

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, { Index(input_variables_number), hidden_neurons_number,Index(target_variables_number) });


        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_default();
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        GeneticAlgorithm gen_alg(&training_strategy);

        gen_alg.set_default();
        gen_alg.set_initialization_method(GeneticAlgorithm::InitializationMethod::Random);
        gen_alg.initialize_population();
        Tensor<bool, 2 > population= gen_alg.get_population();
        Index genes = gen_alg.get_genes_number();
        Index individuals = gen_alg.get_individuals_number();
        //Presento los individuos
        for (Index i = 0; i < individuals; i++)
        {
            cout << "\n Individuo num: " << i + 1<<"\n";
            for (Index j = 0; j < genes; j++)

            {
                if (population(i,j))
                {
                    cout << "X_" <<j<< " esta activado\n";
                }
            }
        }
        //No se como hacer esto para que no falle

        //Hagamos una prueba de evaluar la población


        TrainingStrategy* training_strategy_pointer = gen_alg.get_training_strategy_pointer();

        TrainingResults training_results;

        const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer(); 
        
        DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

        NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();



        


        cout << "Bye OpenNN!" << endl;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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


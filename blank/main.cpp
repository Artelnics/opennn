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
        cout << "Hello OpenNN" << endl;

        srand(time(nullptr));

        DataSet data_set ("C:/Users/rodrigo ingelmo/Documents/.csv",';',false);

        const Index input_variables_number = data_set.get_input_variables_number();

        const Index target_variables_number = data_set.get_target_variables_number();

        const Index hidden_neurons_number = 1;

        data_set.set_missing_values_method(DataSet::MissingValuesMethod::Mean);

        data_set.impute_missing_values_mean();

        data_set.split_samples_sequential();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, {input_variables_number, target_variables_number});

        // Training strategy

       TrainingStrategy training_strategy(&neural_network, &data_set);

       training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

       GeneticAlgorithm genetic_algorithm(&training_strategy);

       genetic_algorithm.set_initialization_method(GeneticAlgorithm::InitializationMethod::Random);

       genetic_algorithm.set_individuals_number(4);

       genetic_algorithm.set_elitism_size(1);

       genetic_algorithm.set_maximum_epochs_number(1);

       genetic_algorithm.initialize_population();

       cout << "Initialize population" << endl;

       Tensor<bool,2> population = genetic_algorithm.get_population();

       cout << population << endl;

       cout<< "Inputs number: " << count(population.data(),population.data()+population.size(),1) << endl;

       genetic_algorithm.set_display(false);
       
       genetic_algorithm.evaluate_population();
       
       cout << "Training errors" << endl;
       
       cout<<genetic_algorithm.get_training_errors() << endl;
       
       cout << "Selection errors" << endl;
       
       cout << genetic_algorithm.get_selection_errors() << endl;
       
       cout << "Performing fitness assignment" << endl;
       
       genetic_algorithm.perform_fitness_assignment();
       
       cout << genetic_algorithm.get_fitness() << endl;
       
       cout << "Performing selection" << endl;
       
       genetic_algorithm.perform_selection();
       
       cout << genetic_algorithm.get_selection() << endl;
       
       cout<< "Performing Crossover" << endl;
       
       genetic_algorithm.perform_crossover();
       
       population = genetic_algorithm.get_population();
       
       cout << "New Population generated " << endl;
       
       cout << population << endl;

       cout << "Bye OpenNN" << endl;
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


//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem.

// System includes

#include <iostream>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        DataSet data_set("../breast_cancer/data/sum.csv", ';', false);

        const Index input_variables_number = data_set.get_input_variables_number();

        const Index target_variables_number = data_set.get_target_variables_number();

        const Index hidden_neurons_number = 1;

        // Neural Network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation,
            { input_variables_number, hidden_neurons_number, target_variables_number });

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);

        training_strategy.set_display_period(1000);
        // training_strategy.perform_training();

        ModelSelection model_selection(&training_strategy);

        model_selection.set_inputs_selection_method(ModelSelection::InputsSelectionMethod::GENETIC_ALGORITHM);

        GeneticAlgorithm* genetic_algorithm_pointer = model_selection.get_genetic_algorithm_pointer();

        genetic_algorithm_pointer->set_individuals_number(80);

        genetic_algorithm_pointer->set_initialization_method(GeneticAlgorithm::InitializationMethod::Random);

        genetic_algorithm_pointer->initialize_population();
        genetic_algorithm_pointer->print_population();

        //genetic_algorithm_pointer->set_maximum_epochs_number(1);

        //genetic_algorithm_pointer->set_default();

        /*for (Index i = 0; i < genetic_algorithm_pointer->get_maximum_iterations_number(); i++)
        {
            genetic_algorithm_pointer->initialize_population();

            genetic_algorithm_pointer->print_population();

            genetic_algorithm_pointer->evaluate_population();

            cout << i + 1<<endl;
        }*/
        

        //InputsSelectionResults input_selection_results = genetic_algorithm_pointer->perform_inputs_selection();

       /* std::ofstream MyFile("../blank/data/MeanSEH.csv");
        for (Index i = 0; i < input_selection_results.mean_selection_error_history.size(); i++)
        {
            MyFile << i + 1 << ";" << input_selection_results.mean_selection_error_history(i) << "\n";

        }
        MyFile.close();

        std::ofstream MyFileMean_training_error_history("../blank/data/MeanTEH.csv");
        for (Index i = 0; i < input_selection_results.mean_training_error_history.size(); i++)
        {
            MyFileMean_training_error_history << i + 1 << ";" << input_selection_results.mean_training_error_history(i) << "\n";

        }
        MyFileMean_training_error_history.close();


*/      system("pause");
       
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
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

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


        DataSet data_set ("C:/Users/rodrigo ingelmo/Documents/5_years_mortality.csv",';',true);

        //cout<<data_set.get_input_columns_names();
        //system("pause");
        //cout<<data_set.get_input_variables_names();

        system("pause");
        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();
        const Index hidden_neurons_number = 2;
        data_set.set_missing_values_method(DataSet::MissingValuesMethod::Mean);

        data_set.impute_missing_values_mean();
       //data_set.split_samples_sequential();


        // Neural network

        //data_set.print_data_preview();

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});


        // Training strategy

       TrainingStrategy training_strategy(&neural_network, &data_set);

       training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

       GeneticAlgorithm genetic_algorithm(&training_strategy);

       genetic_algorithm.set_initialization_method(GeneticAlgorithm::InitializationMethod::Random);

       genetic_algorithm.set_individuals_number(200);
       genetic_algorithm.set_maximum_epochs_number(10);


       //genetic_algorithm.initialize_population_random();



        /*Tensor<bool,1> pruebas(6);
        pruebas.setConstant(false);
        //cout<<pruebas;
        if(is_false(pruebas))
        {
            cout<<"Hola"<<endl;
        }*/


       //cout<<genetic_algorithm.get_population()<<endl;

       genetic_algorithm.set_mutation_rate(0.01);
       genetic_algorithm.set_elitism_size(2);

       genetic_algorithm.initialize_population();
       //Tensor<bool,1> individual=genetic_algorithm.get_population().chip(0,0);
       //cout<<individual.size()<<endl;
       //system("pause");
       //Tensor<bool,1> individual_columns=genetic_algorithm.get_individual_as_columns_from_variables(individual);
       //cout<<individual_columns.size();
       //system("pause");
       //
       Tensor<Index,1> target_original_columns=data_set.get_target_columns_indices();


       //for(Index i=0;i<genetic_algorithm.get_individuals_number();i++)
       //{
       //    Tensor<bool,1> individual=genetic_algorithm.get_population().chip(i,0);
       //    cout<<count(individual.data(),individual.data()+individual.size(),1);
       //    system("pause");
       //    Tensor<Index,1> individual_indexes=genetic_algorithm.get_individual_as_columns_indexes_from_variables(individual);
       //    cout<<individual_indexes.size()<<endl;
       //    system("pause");

       //    cout<<"Los names bien puestos"<<endl;
       //    for(Index j=0;j<individual_indexes.size();j++)
       //    {
       //        cout<<data_set.get_columns()(individual_indexes(j)).name<<endl;


       //    }
       //    //system("pause");
       //}

       genetic_algorithm.evaluate_population();
       genetic_algorithm.perform_fitness_assignment();
       genetic_algorithm.perform_selection();
       genetic_algorithm.perform_crossover();
       genetic_algorithm.perform_mutation();



       ///*Tensor<bool,1> individual=genetic_algorithm.get_population().chip(0,0);
       //genetic_algorithm.transform_individual_to_columns(individual);*/

       // //genetic_algorithm.evaluate_population();

       // //cout<<genetic_algorithm.get_fitness().cumsum(0)<<endl;

       ////cout<<genetic_algorithm.get_population().dimension(1);

       InputsSelectionResults inputs_selection_results=genetic_algorithm.perform_inputs_selection();

       //cout<<inputs_selection_results.mean_selection_error_history<<endl;
       // Testing analysis

      // training_strategy.perform_training();

      ofstream mean_selection_error_csv("C:/Users/rodrigo ingelmo/Documents/MSEH.csv");

      mean_selection_error_csv<<inputs_selection_results.mean_selection_error_history;

      mean_selection_error_csv.close();

      ofstream optimum_selection_error_csv("C:/Users/rodrigo ingelmo/Documents/OPSEH.csv");
      optimum_selection_error_csv<<inputs_selection_results.selection_error_history;
      optimum_selection_error_csv.close();


      TestingAnalysis testing_analysis(&neural_network, &data_set);

      TestingAnalysis::RocAnalysisResults roc_analysis_results=testing_analysis.perform_roc_analysis();

       cout<<"AUC: "<< roc_analysis_results.area_under_curve<<endl;

       system("pause");

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


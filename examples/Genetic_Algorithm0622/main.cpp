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

        srand(static_cast<unsigned>(time(nullptr)));

        //Para construir el dataset el último parámetro es para ver si tenemos o no una fila de cabecera 

        DataSet data_set("../data/sum.csv", ';', false);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        /*Prueba de que lo he hecho bien */
        const Index hidden_neurons_number = 3;

        //Neural network
        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, { input_variables_number,hidden_neurons_number, target_variables_number });

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        //Algoritmo Genético

        GeneticAlgorithm genetic_algorithm(&training_strategy);

        DataSet* data_set_pointer = &data_set;

        Tensor<Correlation, 2> correlations_matrix = data_set_pointer->calculate_input_target_columns_correlations();

        Tensor<type, 1> correlations = get_correlation_values(correlations_matrix).chip(0, 1);

        Tensor<type, 1> correlations_fitness(correlations.size());

        const Tensor<Index, 1> rank = calculate_rank_greater(correlations);

        /* Index aciertos = 0;
         Index fallos = 0;

         for (Index i = 0; i < correlations.size(); i++)
         {


             cout << "La variable: " << rank(i) << "tiene una correlacion de: " << correlations(rank(i));
             cout << "\n" ;

             if ((i <= 49 && rank(i) <= 49) || (i>49 && rank(i)> 49))
             {
                 aciertos++;
             }
             else
             {
                 fallos++;
             }



         }
         cout << "Num de aciertos=  " << aciertos << "\n";
         cout << "Num de fallos= " << fallos << "\n";*/

         //Ahora vamos a hacer la generación de la población con la correlación 

        Tensor <type, 1> probabilidades(correlations.size());

        type sum = ((input_variables_number) * (input_variables_number + static_cast<opennn::type>(1))) / 2;
        type sumprob = 0;

        for (Index i = 0; i < probabilidades.size(); i++)
        {
            type prob = (input_variables_number - (rank(i))) / sum;

            cout << "La probabilidad de la input " << rank(i) + 1 << " es de : " << prob << "\n";

            probabilidades(i) = type(prob);

            cout << probabilidades(i) << "\n";

            sumprob += static_cast<opennn::type>(prob);


        }
        cout << "La suma de las probabilidades es de: " << sumprob;

        //Creación de un individuo con presión selectiva máxima 

        //genetic_algorithm.set_genes_number(input_variables_number);
        //Tengo que ver por qué falla esta linea


        genetic_algorithm.set_individuals_number(100);

        Tensor<bool, 2> population(100,100);

        
        //Caso de máxima presión 
        /*for (Index i = 0; i < genetic_algorithm.get_individuals_number(); i++)
        {


            do {
                bool is_repeated = false;
                Tensor <bool, 1> individual(100);
                individual.setConstant(false);

                Index num_activated_genes = rand() % 100 + 1;
                for (Index j = 0; j < num_activated_genes; j++)
                {
                    individual(rank(j)) = true;
                }

                for (Index j = 0; j < i; j++)
                {
                    Tensor<bool, 1> row = population.chip(j, 0);

                    if (are_equal(individual, row))
                    {
                        is_repeated = true;
                        break;
                    }
                }
            } while (is_repeated);
        }*/
        bool is_repeated;
        Tensor <bool, 1> individual(input_variables_number);
        for (Index i = 0; i < 100; i++)
        {
            do 
            {
                is_repeated = false;

                //Generacion del individuo

                individual.setConstant(false);

                //Creamos un número aleatorio de genes que se van a activar
                Index num_activated_genes = static_cast<Eigen::Index>(rand() % input_variables_number+1) ;

                for (Index j = 0; j < num_activated_genes; j++)
                {
                    individual(rank(j)) = true;
                }

                for (Index j = 0; j < i; j++)
                {
                    Tensor<bool, 1> row = population.chip(j, 0);

                    if (are_equal(individual, row))
                    {
                        is_repeated = true;
                        break;
                    }
                }


            } while (is_repeated);

            for (Index j = 0; j < 100 /*Numero de genes*/ ; j++)
            {
                population(i, j) = individual(j);
            }
        }

        //Printeamos las soluciones

        for (Index i = 0; i < 100; i++)
        {
            cout << "El individuo num "<< i+1 <<"tiene las inputs siguientes" << "\n";
            for (Index j = 0; j < 100; j++)
            {
                if (population(i, j))
                {
                    cout << "X_" << j;
                }

            }

        }
        /*Vamos a establecer los parámetros
         genetic_algorithm.set_initialization_method(GeneticAlgorithm::InitializationMethod::Random);

         genetic_algorithm.set_individuals_number(100);
         genetic_algorithm.set_elitism_size(2);
         genetic_algorithm.initialize_population();
         const Tensor <bool, 2> population = genetic_algorithm.get_population();*/

         /*Hacemos un print de la población que toma
         for (Index i = 0; i < genetic_algorithm.get_individuals_number(); i++)
         {
             cout << "Solucion num: " << i;
             cout << "\n";
             for (Index j = 0; j < genetic_algorithm.get_genes_number(); j++)
             {
                 if (population(i, j))
                 {
                     cout << "x_" << j << "  ";
                 }

             }

             cout << "\n";

         }*/
    }
        catch (const exception& e)
        {
            cerr << e.what() << endl;

            return 1;
        }
    }

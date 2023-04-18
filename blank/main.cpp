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
using namespace std;

    DataSet data_set("../data/Dataset_prueba.csv", ',', true);




    Tensor<bool,1 > get_individual_columns(Tensor<bool,1>& individual)
    {
        DataSet data_set_pointer("../data/Dataset_prueba.csv", ',', true);

        const Index columns_number = data_set_pointer.get_input_columns_number();

//        const Index original_input_number = initial_input_columns_indices.size();

        Tensor<bool, 1> columns_indexes(4);

        columns_indexes.setConstant(false);

        Index genes_count = 0;

        Index categories_number;

        for(Index i = 0; i < 4; i++)
        {
            if(data_set_pointer.get_column_type(i) == DataSet::ColumnType::Categorical)
            {
                categories_number = data_set_pointer.get_columns()(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                {
                    if(individual(genes_count+j))
                    {
                        columns_indexes(i) = true;
                    }
                }

                genes_count += categories_number;
            }
            else
            {
                columns_indexes(i) = individual(genes_count);

                genes_count++;
            }
        }

        return columns_indexes;
    }

    Tensor<bool, 1> get_individual_variables(Tensor <bool, 1>& individual_columns)
    {
        DataSet data_set_pointer("../data/Dataset_prueba.csv", ',', true);

        const Index columns_number = data_set_pointer.get_input_columns_number();

//        const Index original_input_number = initial_input_columns_indices.size();

        const Index columns_size = individual_columns.size();

//        const Index genes_number = get_genes_number();

        Tensor <bool, 1> individual_columns_to_variables(4);

        const Tensor<Index, 1> input_columns_indices = data_set_pointer.get_input_columns_indices();

        individual_columns_to_variables.setConstant(false);

    //    cout << "individual_columns_to_variables: " << individual_columns_to_variables << endl;

        Index input_index = 0;

        Index column_index = 0;

    //    cout << "llego a 2." << endl;

    //    for(Index i = 0; i < columns_size; i++)
    //    {

    //        cout << "individual_columns_to_variables(dentro del for): \n" << individual_columns_to_variables << endl;

    //        cout << "llego a 3." << endl;

    //        column_index = input_columns_indices(i);

    //        cout << "llego a 4." << endl;

    //        if(data_set_pointer->get_column_type(column_index) == DataSet::ColumnType::Categorical)
    //        {
    //            if(individual_columns(i))
    //            {
    //                for(Index j = 0; j < data_set_pointer->get_columns()(column_index).get_categories_number(); j++)
    //                {
    //                    individual_columns_to_variables(input_index + j) = true;
    //                }
    //            }
    //            input_index += data_set_pointer->get_columns()(column_index).get_categories_number(); //salta el numero de categoricas que tengamos y la pone en true
    //        }
    //        else if(individual_columns(i))
    //        {
    //            cout << "llego a 5." << endl;
    //            individual_columns_to_variables(input_index) = true;

    //            input_index++;
    //        }
    //        else
    //        {
    //            cout << "llego a 6." << endl;
    //            input_index++;
    //        }

    //        cout << "individual_columns_to_variables(al final del for): \n" << individual_columns_to_variables << endl;
    //    }

    //    cout << "llego a fin." << endl;

     // LLENA ESTO DE COUTS PARECE QUE SALE BIEN, VUELVE AL CROSSOVER.

        const Tensor<DataSet::Column, 1> columns = data_set_pointer.get_columns();

        Index variable_index = 0;

        for (Index i = 0; i < columns_size; i++)
        {

            if(individual_columns(i) == true)
            {
                if(columns(i).column_use != DataSet::VariableUse::Target && columns(i).type == DataSet::ColumnType::Categorical)
                {
                    Index categories_number = data_set_pointer.get_columns()(i).get_categories_number();

    //                cout << "categories_number: " << categories_number << endl;

                    for(Index j = 0; j < categories_number; j++)
                    {
                        individual_columns_to_variables(variable_index + j) = true;
                    }

                    variable_index += categories_number;
                }
                else if (columns(i).column_use != DataSet::VariableUse::Target)
                {
                    individual_columns_to_variables(variable_index) = true;
                    variable_index++;
                }
            }
            else
            {
                variable_index++;
            }

    //        if(data_set_pointer->get_column_type(i) == DataSet::ColumnType::Categorical)
    //        {
    //            Index categories_number = data_set_pointer->get_columns()(i).get_categories_number();

    //            if(individual_columns(i) == true)
    //            {
    //                for(Index j = 0; j < categories_number; j++)
    //                {
    //                    individual_columns_to_variables(variable_index + j) = true;
    //                }
    //            }

    //            variable_index += categories_number;
    //        }
    //        else if(individual_columns(i) == true)
    //        {
    //            individual_columns(variable_index) = true;
    //            variable_index++;
    //        }
    //        else
    //        {
    //            variable_index++;
    //        }
        }

        return individual_columns_to_variables;
    }

    Tensor<Index,1> get_individual_as_columns_indexes_from_variables(Tensor<bool, 1>& individual) // esta daba problemas de dimension al principio
    {
        Tensor <bool, 1> individual_columns = get_individual_columns(individual); // esta función funciona bien

    //    cout << "de get_inidividual_columns me da esto: " << individual_columns << endl;

    //        DataSet data_set("../data/Dataset_prueba.csv", ',', true);  // si lo pongo asi no necesito el dataset


//        Tensor<Index, 1> input_columns_indices = training_strategy_pointer->get_data_set_pointer()->get_input_columns_indices(); // esto me lo esta haciendo mal

        // Sirve cuando todas las columnas estan en input

//        Index columns_number = training_strategy_pointer->get_data_set_pointer()->get_columns_number();
//        Index target_number = training_strategy_pointer->get_data_set_pointer()->get_target_columns_number();
    //    Index constant_number = training_strategy_pointer->get_data_set_pointer()->get_constant_columns_number();

        Index inputs_number =  4 /*columns_number - target_number*/;

    //    cout << "original_inputs_number: " << inputs_number << endl;

        Tensor<Index, 1> input_columns_indices_first(inputs_number);

        for(Index i = 0; i < inputs_number ; i++)
        {
            input_columns_indices_first(i) = i;
        }

    //    cout << "input_columns_indices_first_try: " << input_columns_indices_first_try << endl;

    //    cout << "input_columns_indices: " << input_columns_indices << endl;

        Index active_columns = 0;

        for(Index i = 0; i < individual_columns.size(); i++)
        {
            if(individual_columns(i) == true)
            {
                active_columns++;
            }
        }

    //    cout << "numero de inputs activos (dimension del vector de salida): " << active_columns << endl; // bien

        Tensor <Index, 1> individual_columns_indexes (active_columns);

        Index active_column_index = 0;

        for(Index i = 0; i < individual_columns.size(); i++) // menor o igual ahora
        {
            if(individual_columns(i) == true)
            {
                individual_columns_indexes(active_column_index) = input_columns_indices_first(i);

                active_column_index++;

            }
        }

        return individual_columns_indexes;
    }



int main(int argc, char *argv[])
{
    try
    {
        cout << "Hello OpenNN" << endl;

        Tensor <bool, 1> individual_columns(4);

        individual_columns.setValues({0,1,0,0});

        Tensor<bool, 1> individual_variables(4);

//        individual_variables.setValues({0,1,0,1});

        individual_variables = get_individual_variables(individual_columns);

        cout << "Individual_columns (inputs): \n" << individual_columns << endl;
        cout << "Individual_variables (output): \n" << individual_variables << endl;

        Tensor <bool, 1> new_inidividual_columns(4);

        new_inidividual_columns = get_individual_columns(individual_variables);

        cout << "new_individual_columns: " << new_inidividual_columns << endl;

        Tensor <Index, 1> individual_as_columns_indexes_from_variables = get_individual_as_columns_indexes_from_variables(individual_variables);

        cout << "individual_as_columns_indexes_from_variables: " << individual_as_columns_indexes_from_variables << endl;


        cout << "Good bye!" << endl;

        return 0;
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

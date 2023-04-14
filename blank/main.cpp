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

Tensor<bool, 1> get_individual_variables(Tensor <bool, 1>& individual_columns)
{
    DataSet data_set("../data/Dataset_prueba.csv", ',', true);

    const Index columns_number = data_set.get_input_columns_number();

    cout << "columns_number: " << columns_number << endl;

    const Index genes_number = data_set.get_input_variables_number();

    cout << "genes_number: " << genes_number << endl ;

    Tensor <bool, 1> individual_columns_to_variables(genes_number);

    const Tensor<Index, 1> input_columns_indices = data_set.get_input_columns_indices();

    cout << "input_columns_indices: " << input_columns_indices << endl;
    cout << "input_columns_indices: " << data_set.get_input_columns_names() << endl;

    cout << "input_columns_indices: " << data_set.get_input_variables_indices() << endl;
    cout << "input_columns_indices: " << data_set.get_input_variables_names() << endl;

//    cout << "target_indice: " << data_set.get_target_columns_indices() << endl;

    individual_columns_to_variables.setConstant(false);

    Index input_index = 0;

    Index column_index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        column_index = input_columns_indices(i);

        if(data_set.get_column_type(column_index) == DataSet::ColumnType::Categorical)
        {
            if(individual_columns(i))
            {
                for(Index j = 0; j < data_set.get_columns()(column_index).get_categories_number(); j++)
                {
                    individual_columns_to_variables(input_index + j) = true;
                }
            }
            input_index += data_set.get_columns()(column_index).get_categories_number();
        }
        else if(individual_columns(i))
        {
            individual_columns_to_variables(input_index) = true;

            input_index++;
        }
        else
        {
            input_index++;
        }
    }

    return individual_columns_to_variables;
}




Tensor<bool,1 > get_individual_columns(Tensor<bool,1>& individual)
{
    DataSet data_set("../data/Dataset_prueba.csv", ',', true);

    const Index columns_number = data_set.get_input_columns_number();

    Tensor<bool, 1> columns_indexes(columns_number);

    columns_indexes.setConstant(false);

    Index genes_count = 0;

    Index categories_number;

    for(Index i = 0; i < columns_number; i++)
    {
        if(data_set.get_column_type(i) == DataSet::ColumnType::Categorical)
        {
            categories_number = data_set.get_columns()(i).get_categories_number();

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




Tensor<Index,1> get_individual_as_columns_indexes_from_variables(Tensor<bool,1>& individual)
{
    Tensor <bool, 1> individual_columns = get_individual_columns(individual);

    DataSet data_set("../data/Dataset_prueba.csv", ',', true);

    Tensor<Index, 1> input_columns_indices = data_set.get_input_columns_indices();

//    cout << "input_columns_indices: " << input_columns_indices << endl;

    Index active_columns = 0;

    for(Index i=0; i < individual_columns.size(); i++)
    {
        if(individual_columns(i) == true)
        {
            active_columns++;
        }
    }

    Tensor <Index, 1> individual_columns_indexes (active_columns);

    Index active_column_index = 0;

    for(Index i=0; i < individual_columns.size(); i++)
    {
        if(individual_columns(i) == true)
        {
            individual_columns_indexes(active_column_index) = input_columns_indices(i);

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

        Tensor <bool, 1> individual_columns(5);

        individual_columns.setValues({0,1,1,0,1});

        Tensor<bool, 1> individual_variables(7);

        individual_variables = get_individual_variables(individual_columns);

        cout << "Individual_columns (inputs): \n" << individual_columns << endl;
        cout << "Individual_variables (output): \n" << individual_variables << endl;

        Tensor <bool, 1> new_inidividual_columns(5);

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

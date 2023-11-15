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
#include <chrono>

// OpenNN includes

#include "../opennn/opennn.h"

#include <iostream>

using namespace std;
using namespace OpenNN;

int main()
{
   try
   {
        cout << "Blank\n";      

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("/Users/alvaros/test_sample.csv", ',', true);

        Tensor<string, 1> variables_names = data_set.get_variables_names();
        Tensor<string, 1>  columns_names = data_set.get_columns_names();
        Tensor<DataSet::VariableUse, 1> columns_uses = data_set.get_columns_uses();
        Tensor<string, 1> column_types = data_set.get_columns_types();
        Index columns_number = data_set.get_columns_number();

        Tensor<DataSet::Column, 1> old_columns = data_set.get_columns();

        map<string, DataSet> groupedData = data_set.group_by(data_set, "block_id");

       for(auto& pair : groupedData)
       {
           DataSet& subset = pair.second;

           subset.set_columns_number(columns_number);
           subset.set_columns_names(columns_names);
           subset.set_columns_uses(columns_uses);
           subset.set_columns_types(column_types);
           subset.set_variables_names_from_columns(variables_names, old_columns);
           subset.set_lags_number(1);
           subset.set_steps_ahead_number(1);
           subset.transform_time_series();

           cout << pair.first << ": " << subset.get_data() << endl;
       }

       cout << "Bye!" << endl;

       Tensor<type, 2> merged_data;
       bool is_first_iteration = true;

       for(const auto& pair : groupedData)
       {
           const DataSet& value = pair.second;

           if(is_first_iteration)
           {
               merged_data = value.get_data();
               is_first_iteration = false;
           }
           else
           {
               Tensor<type, 2> new_merged_data = merged_data.concatenate(value.get_data(), 0);
               merged_data = new_merged_data;
           }
       }

       quicksort_by_column(merged_data, 0);

       cout << "Ordered and merged_data: " << merged_data << endl;

        cout << "Bye!" << endl;

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

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E U K E M I A   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)
//   artelnics@artelnics.com

// This is a classical pattern recognition problem.

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"


#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

using namespace OpenNN;

vector<int> get_row_indices_sorted(Tensor<type,1>& x)
{
    vector<type> y(x.size());

    vector<int> index;

    size_t n(0);

    generate(begin(y), end(y), [&]{ return n++; });

    sort(begin(y), end(y), [&](int i1, int i2) { return x[i1] < x[i2]; } );

    for (auto v : y) index.push_back(v);

    return index;
}


Tensor<type, 2> get_subtensor_sorted(Tensor<type, 2>& data)
{
    Tensor<type, 1> shrink_data_dimension(data.dimension(0));

    memcpy(shrink_data_dimension.data(), data.data(), static_cast<size_t>(shrink_data_dimension.size())*sizeof(type));

    vector<int> indices_sorted = get_row_indices_sorted(shrink_data_dimension);

    Tensor<type, 2> sorted_output(data.dimension(0),data.dimension(1));

    for(Index i =0; i<data.dimension(0); i++)
    {
        sorted_output(i,0) = data(indices_sorted[i], 0);
        sorted_output(i,1) = data(indices_sorted[i], 1);
    }
    return sorted_output;

}

int main(void)
{
    try
    {
        cout<<"Hello"<<endl;

        DataSet data_set("C:/opennn/examples/leukemia/data/leukemia.csv",';',false);

        data_set.set_training();

        Tensor<type, 2> data = data_set.get_data();

        const Index rows_number = data.dimension(0);
        const Index columns_number = data.dimension(1);

        Tensor<Index, 1> rows_indices(rows_number);
        rows_indices = data_set.get_training_instances_indices();

        Tensor<Index, 1> columns_indices(2);
        columns_indices[1] = static_cast<Index>(columns_number-1);

        for(Index j =0; j<columns_number-1; j++)
        {
            columns_indices[0] = static_cast<Index>(j);

            Tensor<type, 2> subtensor = data_set.get_subtensor_data(rows_indices, columns_indices);

            Tensor<type, 2> sorted_output = get_subtensor_sorted(subtensor);

            int counter = 0;

            for(Index i=0; i<rows_number-1;i++)
            {
                if(sorted_output(i,1) != sorted_output(i+1,1))
                {
                    counter++;
                }
            }

            if (counter == 1)
            {
                cout<<"gen: "<<j<<endl;

            }
        }

        return 0;
    }
    catch(exception& e)
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

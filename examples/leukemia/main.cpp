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

//#include <iostream>
//#include <vector>
//#include <numeric>      // std::iota
//#include <algorithm>    // std::sort, std::stable_sort

using namespace OpenNN;

//vector<int> get_row_indices_sorted(Tensor<type,1>& x)
//{
//    vector<type> y(x.size());

//    vector<int> index;

//    size_t n(0);

//    generate(begin(y), end(y), [&]{ return n++; });

//    sort(begin(y), end(y), [&](int i1, int i2) { return x[i1] < x[i2]; } );

//    for (auto v : y) index.push_back(v);

//    return index;
//}


//Tensor<type, 2> get_subtensor_sorted(Tensor<type, 2>& data)
//{
//    Tensor<type, 1> shrink_data_dimension(data.dimension(0));

//    memcpy(shrink_data_dimension.data(), data.data(), static_cast<size_t>(shrink_data_dimension.size())*sizeof(type));

//    vector<int> indices_sorted = get_row_indices_sorted(shrink_data_dimension);

//    Tensor<type, 2> sorted_output(data.dimension(0),data.dimension(1));

//    for(Index i =0; i<data.dimension(0); i++)
//    {
//        sorted_output(i,0) = data(indices_sorted[i], 0);
//        sorted_output(i,1) = data(indices_sorted[i], 1);
//    }
//    return sorted_output;

//}

int main(void)
{
    try
    {
        cout << "OpenNN. Leukemia Example." << endl;

        // Device

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        DataSet data_set("../data/leukemia.csv",';',false);

        data_set.set_thread_pool_device(thread_pool_device);
        data_set.set_training();

        Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
        Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        #pragma omp parallel for

        for(int i=0; i<input_variables_indices.dimension(0); i++)
        {

            CorrelationResults logistic_correlation = logistic_correlations(thread_pool_device,
                                                      data_set.get_data().chip(input_variables_indices(i),1),
                                                      data_set.get_data().chip(target_variables_indices(0),1));

            CorrelationResults gauss_correlation = gauss_correlations(thread_pool_device,
                                                   data_set.get_data().chip(input_variables_indices(i),1),
                                                   data_set.get_data().chip(target_variables_indices(0),1));


            if(abs(logistic_correlation.correlation) > abs(gauss_correlation.correlation) &&
                    abs(logistic_correlation.correlation) > 0.9)
            {
                cout << "Gen: " << i << endl;
                cout << "Logistic correlation: " << logistic_correlation.correlation << endl;
            }

            if(abs(gauss_correlation.correlation) > abs(logistic_correlation.correlation) &&
                    abs(gauss_correlation.correlation) > 0.9)
            {
                cout<<"Gen: "<<i<<endl;
                cout<<"Gauss correlation: "<<gauss_correlation.correlation<<endl;
            }

            if(i%250 == 0)
            {
                cout<<static_cast<float>(i)/static_cast<float>(input_variables_indices.dimension(0))*100
                   <<"% dataset evaluated"<<endl;
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

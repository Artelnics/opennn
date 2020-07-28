//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;


int main(void)
{
    try
    {
        cout << "Hello Normal correlation" << endl;

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        Tensor<type,1> x(7);
        Tensor<type,1> y(7);

        y.setValues({0,0,0,1,0,0,0});
        x.setValues({1,2,3,4,5,6,7});

        DataSet data_set("C:/opennn/examples/leukemia/data/leukemia.csv",';',false);

        for(Index i=0;i<data_set.get_data().dimension(1)-1;i++)
        {
            CorrelationResults gauss = gauss_correlations(thread_pool_device,
                                                          data_set.get_data().chip(i,1),
                                                          data_set.get_data().chip(data_set.get_data().dimension(1)-1,1));


            if(gauss.correlation > 0.9)
            {
                cout<<"column: " << i <<endl;
                cout<<"gauss: " << gauss.correlation <<endl;
            }

//            CorrelationResults logistic = logistic_correlations(thread_pool_device, x, y);

        }












        CorrelationResults gauss = gauss_correlations(thread_pool_device, x, y);
        CorrelationResults logistic = logistic_correlations(thread_pool_device, x, y);

        cout<<"Gauss correlation: "<<gauss.correlation<<endl;
        cout<<"Logistic correlation: "<<logistic.correlation<<endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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

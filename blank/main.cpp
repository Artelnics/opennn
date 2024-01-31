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
#include <algorithm>
#include <execution>

// OpenNN includes

#include "../opennn/opennn.h"
#include <iostream>

using namespace std;
using namespace opennn;
//using namespace Eigen;


int main()
{
   try
   {
        cout << "Blank\n";

        Tensor<float, 2> inputs(1000, 1000);
        Tensor<float, 2> synaptic_weights(1000, 1000);
        Tensor<float, 2> combinations(1000, 1000);

        inputs.setRandom();
        synaptic_weights.setRandom();
        combinations.setRandom();

        auto start = std::chrono::system_clock::now();

        for(Index i = 0; i < 1000; i++)
        cblas_sgemm(CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            inputs.dimension(0),
            synaptic_weights.dimension(1),
            inputs.dimension(1),
            type(1),
            (float*)inputs.data(),
            inputs.dimension(0),
            (float*)synaptic_weights.data(),
            synaptic_weights.dimension(0),
            type(1),
            (float*)combinations.data(),
            inputs.dimension(0));

        auto end = std::chrono::system_clock::now();
        auto time = end - start;

        cout << "MKL Time: " << time.count() << endl;

        cout << combinations.dimension(0) << endl;

        const int n = omp_get_max_threads();

        ThreadPool* thread_pool = new ThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, n);

        const Eigen::array<IndexPair<Index>, 1> A_B = { IndexPair<Index>(1, 0) };

        start = std::chrono::system_clock::now();
        
        for (Index i = 0; i < 1000; i++)
            combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, A_B);
        end = std::chrono::system_clock::now();
        time = end - start;

        cout << "Eigen Time: " << time.count() << endl;

        cout << combinations.dimension(0) << endl;

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

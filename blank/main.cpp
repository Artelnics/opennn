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

// OneDNN
#include "oneapi/dnnl/dnnl.hpp"
#include "../mkl/mkl.h"

using namespace std;
using namespace opennn;
using namespace std::chrono;
//using namespace Eigen;


int main()
{
   try
   {
        cout << "Blank\n";
        /*
        const int M = 10000;
        const int K = 10000;
        const int N = 10000;

        const int n = omp_get_max_threads();
        ThreadPool* thread_pool = new ThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(thread_pool, n); 

        Tensor<float, 2> t_A(M, K);
        Tensor<float, 2> t_B(K, N);
        Tensor<float, 2> AB(M, M);
        t_A.setRandom();
        t_B.setRandom();


        //ONEDNN
        auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
        auto stream = dnnl::stream(engine);

        auto A_md = dnnl::memory::desc({ M, K }, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        auto B_md = dnnl::memory::desc({ K, N }, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        auto AB_md = dnnl::memory::desc({ M, N }, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

        auto matmul_prim_desc = dnnl::matmul::primitive_desc(engine, A_md, B_md, AB_md);
        auto matmul_prim = dnnl::matmul(matmul_prim_desc);

        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, dnnl::memory(A_md, engine, t_A.data())},
            {DNNL_ARG_WEIGHTS, dnnl::memory(B_md, engine, t_B.data())},
            {DNNL_ARG_DST, dnnl::memory(AB_md, engine)}
        };

        auto start_time_onednn = high_resolution_clock::now();

        for(Index i = 0; i < 10; i++)
            matmul_prim.execute(stream, args);
        stream.wait();

        auto end_time_onednn = high_resolution_clock::now();
        auto duration_onednn = duration_cast<milliseconds>(end_time_onednn - start_time_onednn);
        auto seconds_onednn = duration_onednn.count() / 1000;
        auto milliseconds_onednn = duration_onednn.count() % 1000;

        std::cout << "Execution time oneDNN:" << std::endl;
        std::cout << seconds_onednn << " seconds " << milliseconds_onednn << " milliseconds" << std::endl;
        std::cout << "Total Milliseconds: " << duration_onednn.count() << std::endl;


        //MKL
        auto start_time_mkl = high_resolution_clock::now();
        
        for(Index i = 0; i < 10; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, t_A.data(), K, t_B.data(), N, 0.0f, AB.data(), N); //MKL

        auto end_time_mkl = high_resolution_clock::now();
        auto duration_mkl = duration_cast<milliseconds>(end_time_mkl - start_time_mkl);
        auto seconds_mkl = duration_mkl.count() / 1000;
        auto milliseconds_mkl = duration_mkl.count() % 1000;

        std::cout << "Execution time MKL:" << std::endl;
        std::cout << seconds_mkl << " seconds " << milliseconds_mkl << " milliseconds" << std::endl;
        std::cout << "Total Milliseconds: " << duration_mkl.count() << std::endl;

        //EIGEN
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };

        auto start_time_eigen = high_resolution_clock::now();

        for(Index i = 0; i < 10; i++)
            AB.device(*thread_pool_device) = t_A.contract(t_B, product_dims);

        auto end_time_eigen = high_resolution_clock::now();
        auto duration_eigen = duration_cast<milliseconds>(end_time_eigen - start_time_eigen);
        auto seconds_eigen = duration_eigen.count() / 1000;
        auto milliseconds_eigen = duration_eigen.count() % 1000;

        std::cout << "Execution time Eigen:" << std::endl;
        std::cout << seconds_eigen << " seconds " << milliseconds_eigen << " milliseconds" << std::endl;
        std::cout << "Total Milliseconds: " << duration_eigen.count() << std::endl;
        */

        Tensor<type, 3> A(2, 2, 2);
        A.setValues({ {{1, 2},
                       {3, 4}},
            
                      {{5, 6},
                       {7, 8}} });

        Tensor<type, 3> B(2, 2, 3);
        B.setValues({ {{1, 2, 3},
                       {4, 5, 6}},

                      {{7, 8, 9},
                       {10, 11, 12}} });

        const Eigen::array<IndexPair<Index>, 2> contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };

        Tensor<type, 2> C = A.contract(B, contraction_indices);

        cout << C << endl;

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

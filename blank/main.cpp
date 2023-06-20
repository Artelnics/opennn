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

using namespace opennn;
using namespace std;


int main(int argc, char *argv[])
{
   try
   {
        cout << "OpenNN. Average Pooling Example." << endl;

        const Index batch_samples_number = 2;
        const Index channels_number = 3;
        const Index input_rows_number = 5;
        const Index input_columns_number = 5;

        Tensor<type, 4> inputs(batch_samples_number,
                               channels_number,
                               input_rows_number,
                               input_columns_number);

        inputs.setValues({
                {
                    {
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0}
                    },
                    {
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0}
                    },
                    {
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0},
                        {0.0, 1.0, 2.0, 2.0, 2.0}
                    }
                },

                {
                    {
                      {0.0, 5.0, 2.0, 2.0, 2.0},
                      {0.0, 5.0, 2.0, 2.0, 2.0},
                      {0.0, 5.0, 2.0, 2.0, 2.0},
                      {0.0, 5.0, 2.0, 2.0, 2.0},
                      {0.0, 5.0, 2.0, 2.0, 2.0}
                    },
                    {
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0}
                    },
                    {
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0},
                      {0.0, 1.0, 2.0, 2.0, 2.0}
                    }
                }

            });

        const Eigen::array<ptrdiff_t, 3> mean_dimensions = {0, 2, 3};
        const Eigen::array<ptrdiff_t, 4> reshape_dims = {1, channels_number, 1, 1};

        Tensor<type, 1> means = inputs.mean(mean_dimensions);

        cout << "means_dimensions: " << means.dimensions() << endl;
        cout << "means: " << means << endl;

        cout << "means_reshape_dimensions: " << means << endl;

        cout << "inputs_dimensions: " << inputs.dimensions() << endl;
//        cout << "inputs: " << inputs << endl;

//        Tensor<type, 4> outputs = inputs - means.reshape(reshape_dims);

        const Eigen::array<ptrdiff_t, 4> broadcast_dims = {batch_samples_number,
                                                           1,
                                                           input_rows_number,
                                                           input_columns_number};

        Tensor<type, 4> reshaped_means = means.reshape(reshape_dims).broadcast(broadcast_dims);
        //means.reshape(reshape_dims).broadcast(broadcast_dims);
        Tensor<type, 4> means_matrix(inputs);
        means_matrix.setConstant(5.0);
        cout << "------------" << endl;
        cout << "reshaped_means_dimensions: " << reshaped_means.dimensions() << endl;
        cout << "reshaped_means: " << reshaped_means << endl;
        cout << "reshaped_means chip: " << reshaped_means.chip(1,1) << endl;
cout << "------------" << endl;
        Tensor<type, 4> outputs = inputs - reshaped_means + means_matrix;

//        Tensor<type, 4> broadcasted_tensor = scalar_tensor.broadcast(broadcast_dims);

//        cout << "test: " << broadcasted_tensor << endl;

        cout << "outputs: " << outputs << endl;
//        normalized_inputs = (inputs - mean.reshape(reshape_dims)) /
//                            (variance.reshape(reshape_dims).sqrt() + epsilon);



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

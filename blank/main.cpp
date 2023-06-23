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

        const Index batch_samples_number = 1;
        const Index channels_number = 3;
        const Index input_rows_number = 3;
        const Index input_columns_number = 2;

        Tensor<string, 4> inputs(batch_samples_number,
                                   channels_number,
                                   input_rows_number,
                                   input_columns_number);

        inputs.setValues({
            {  // Batch 1
                {  // Channel 1
                    {"r1", "r1"},  // Row 1
                    {"r1", "r1"},  // Row 2
                    {"r1", "r1"}   // Row 3
                },
                {  // Channel 2
                   {"g1", "g1"},  // Row 1
                   {"g1", "g1"},  // Row 2
                   {"g1", "g1"}   // Row 3
                },
                {  // Channel 3
                   {"b1", "b1"},  // Row 1
                   {"b1", "b1"},  // Row 2
                   {"b1", "b1"}   // Row 3
                }
            },
        });

        string* inputs_pointer = const_cast<string*>(inputs.data());

        const Index single_channel_size = input_columns_number * input_rows_number;
        const Index image_size = channels_number * input_columns_number * input_rows_number;

        const Index next_image = input_rows_number*input_columns_number*channels_number;

    #pragma omp parallel for
        for(int i = 0; i < batch_samples_number ;i++)
        {
            const TensorMap<Tensor<string, 3>> single_image(inputs_pointer+i*batch_samples_number,
                                                          channels_number,
                                                          input_rows_number,
                                                          input_columns_number);

            cout << "single_image: " << single_image << endl;
        }

        cout << "chip" << inputs.chip(1, 1) << endl;

        cout << "--------------------------" << endl;

        for(Index i = 0; i < inputs.size(); i++)
        {
            cout << inputs(i) << " ";
        }

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

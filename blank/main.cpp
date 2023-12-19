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
/*
        srand(static_cast<unsigned>(time(nullptr)));

        DataSet data_set;

        data_set.set_data_file_name("../data/example2.txt");
        data_set.set_text_separator(DataSet::Separator::Tab);

        data_set.read_txt_language_model();

        Tensor<type, 2> inputs_data = data_set.get_input_data();

        Index batch_size = inputs_data.dimension(0);

        Index input_length;

        for(Index i = 1; i < inputs_data.dimension(1); i++)
        {
            if(inputs_data(0, i) == 1) input_length = i;
        }

        Index context_length = inputs_data.dimension(1) - input_length;

        TensorMap<Tensor<type, 2>> input(inputs_data.data(), batch_size, input_length);
        TensorMap<Tensor<type, 2>> context(inputs_data.data() + batch_size*(input_length), batch_size, context_length);

        cout << "Input:" << endl << input << endl;
        cout << "Context:" << endl << context << endl;

        Tensor<type, 0> max_input_value = input.maximum();
        Index input_dim = Index(max_input_value(0)) + 1;

        Tensor<type, 0> max_context_value = context.maximum();
        Index context_dim = Index(max_context_value(0)) + 1;

        Tensor<DynamicTensor<type>, 1> inputs(2);
        inputs(0) = DynamicTensor<type>((Tensor<type, 2>) context);
        inputs(1) = DynamicTensor<type>((Tensor<type, 2>) input);

        DataSetBatch dataset_batch;
        dataset_batch.set_inputs(inputs);

        Index embedding_depth = 5;
        Index perceptron_depth = 7;
        Index number_of_heads = 4;
        Index number_of_layers = 1;

        Transformer transformer({input_length, context_length, input_dim, context_dim,
                                embedding_depth, perceptron_depth, number_of_heads, number_of_layers});

        NeuralNetworkForwardPropagation forward_propagation(batch_size, &transformer);

        auto start = std::chrono::system_clock::now();
        transformer.forward_propagate(dataset_batch, forward_propagation, true);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<float,std::milli> duration = end - start;

        std::cout << duration.count()/1000 << "s" << std::endl;
*/
//        Index total_number_of_layers = forward_propagation.layers();

//        cout << forward_propagation.layers()

        /// @todo encajar embedding con attention

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

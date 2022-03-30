//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/layer.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

#include "data_set.h"

int main()
{
    try
    {
        cout<<"Blank script! "<<endl;

        DataSet data_set;

        data_set.set_data_file_name("E:/opennn/blank/test-6px-python-bmp/");

        data_set.read_bmp();

        const Index samples_number = data_set.get_training_samples_number();

        const Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        DataSetBatch batch(samples_number, &data_set);

        batch.fill(samples_indices, input_variables_indices, target_variables_indices);

        Eigen::array<Eigen::Index, 4> extents = {0, 0, 0, 0};
        Eigen::array<Eigen::Index, 4> offsets = {batch.inputs_4d.dimension(0),
                                                 batch.inputs_4d.dimension(1)-1, //padding
                                                 batch.inputs_4d.dimension(2),
                                                 batch.inputs_4d.dimension(3)};

       // remove padding
        Tensor<float, 4> new_batch = batch.inputs_4d.slice(extents, offsets);
        batch.inputs_4d = new_batch;

        cout<<"Batch dimensions "<<batch.inputs_4d.dimensions()<<"\n"<<endl;

        cout<<"------"<<endl;

        cout<<"Blue image and Blue channel all values should be set to 255. \n Current Tensor: \n "<<
               batch.inputs_4d.chip(0,3).chip(0,2);

        cout<<"------"<<endl;
        cout<<"Blue image and Green channel all values should be set to 0. \n Current Tensor: \n "<<
               batch.inputs_4d.chip(0,3).chip(1,2)<<endl;

        cout<<"------"<<endl;
        cout<<"Red image and Red channel all values should be set to 255. \n Current Tensor: \n "<<
               batch.inputs_4d.chip(2,3).chip(2,2)<<endl;

        cout<<"Bye!"<<endl;
        return 0;
    }
    catch(const exception& e)
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

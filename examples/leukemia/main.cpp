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

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Leukemia Example." << endl;

        // Device

        DataSet data_set("../data/leukemia.csv",';',false);

        data_set.set_training();

        data_set.calculate_input_target_columns_correlations();
/*
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
*/
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

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
#include "../opennn/text_analytics.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "Hello OpenNN!" << endl;

        Layer layer;

        Tensor<type, 2> x(5,5);
        x.setConstant(3);
        Tensor<Index, 1> x_dimensions(2);
        x_dimensions.setValues({5,5});

        Tensor<type, 2> y(5,5);
        Tensor<Index, 1> y_dimensions(2);
        y_dimensions.setValues({y.dimension(0),y.dimension(1)});

        layer.logistic(x.data(), x_dimensions, y.data(), y_dimensions);

        cout << "x:" << endl << x << endl;
        cout << "y:" << endl << y << endl;

        getchar();

        PerceptronLayer* pl = new PerceptronLayer(3,3,PerceptronLayer::ActivationFunction::HyperbolicTangent);

        pl->set_parameters_constant(2);

        Tensor<type, 2> inputs(3,3);
        inputs.setConstant(2);

        Tensor<Index, 1> dims(2);
        dims.setValues({inputs.dimension(0),inputs.dimension(1)});

        cout << "INPUTS: " << endl << inputs << endl;
        cout << "OUTPUTS: " << endl << pl->calculate_outputs(inputs) << endl;
        cout << "--------------------------" << endl;

        Tensor<type, 2> outputs(3,3);

        Tensor<Index, 1> output_dims(2);
        output_dims.setValues({outputs.dimension(0),outputs.dimension(1)});

        pl->calculate_outputs(inputs.data(), dims, outputs.data(), dims);

        cout << "INPUTS: " << endl << inputs << endl;
        cout << "OUTPUTS: " << endl << outputs << endl;
        cout << "--------------------------" << endl;


        cout << "Goodbye!" << endl;

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

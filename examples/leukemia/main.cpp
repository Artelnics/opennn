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

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main()
{
    try
    {
        cout << "OpenNN. Leukemia Example." << endl;

        // Device

        DataSet data_set("../data/leukemia.csv",';',false);

        data_set.set_training();

        const Tensor<Correlation, 2> correlation_results = data_set.calculate_input_target_columns_correlations();

        cout << "Separable genes: " << endl;

        for(Index i = 0; i < correlation_results.size(); i++)
        {
            if(abs(correlation_results(i).r - 1) < numeric_limits<type>::min()
            || abs(correlation_results(i).r + 1) < numeric_limits<type>::min())

            cout << "Gene " << i << " correlation: " << correlation_results(i).r << endl;
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

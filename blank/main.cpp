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


//Tensor<type, 2> box_plots_to_tensor(const Tensor<BoxPlot, 1>& box_plots)
//{
//    const Index columns_number = box_plots.dimension(0);

//    Tensor<type, 2> summary(5, columns_number);

//    for(Index i = 0; i < columns_number; i++)
//    {
//        const BoxPlot& box_plot = box_plots(i);
//        summary(0, i) = box_plot.minimum;
//        summary(1, i) = box_plot.first_quartile;
//        summary(2, i) = box_plot.median;
//        summary(3, i) = box_plot.third_quartile;
//        summary(4, i) = box_plot.maximum;
//    }

//    //todo
//    Eigen::array<Index, 2> new_shape = {1, 5 * columns_number};
//    Tensor<type, 2> reshaped_summary = summary.reshape(new_shape);

//    return reshaped_summary;
//}


int main()
{
   try
   {
        cout << "Blank\n";

        srand(static_cast<unsigned>(time(nullptr)));

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

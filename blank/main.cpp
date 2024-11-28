//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
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
#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Blank project." << endl;

        const unsigned int threads_number = thread::hardware_concurrency();
        unique_ptr<ThreadPool> thread_pool = make_unique<ThreadPool>(threads_number);
        unique_ptr<ThreadPoolDevice> thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

        Tensor<type, 1> x(10);
        x.setValues({0,1,2,3,4,5,6,7,8,9,10});

        Tensor<type, 1> y(10);
        y.setValues({0,1,2,3,4,5,6,7,8,9,10});

        Correlation correlation = linear_correlation(thread_pool_device.get(), x, y);

        correlation.print();

        cout << "Bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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

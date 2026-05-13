//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   T E M P L A T E
//
//   Empty starting point for new OpenNN experiments. Copy this file into a
//   new example folder, add your dataset to ./data, and start writing code
//   in main(). The CMakeLists already wires this target up the same way as
//   the other examples.

#include <iostream>

#include "../opennn/opennn.h"

using namespace opennn;


int main()
{
    try
    {
        // Write your experiment here.
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
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
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

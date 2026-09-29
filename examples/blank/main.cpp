//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   E X A M P L E
//
//   Empty starting point for a new OpenNN application.
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/network/network.h"

using namespace opennn;

int main()
{
    try
    {
        std::cout << "This is a blank example" << std::endl;

        // Write your application here.

        std::cout << "Bye!" << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

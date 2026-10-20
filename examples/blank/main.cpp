// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

// Empty starting point for a new OpenNN application.

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

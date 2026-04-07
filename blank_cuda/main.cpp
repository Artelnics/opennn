//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/opennn.h"
#include <iostream>

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

#ifdef OPENNN_CUDA


#endif

        cout << "Bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <ctime>

#include "../opennn/pch.h"
#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/vgg16.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/image_dataset.h"
#include "../opennn/scaling_layer_4d.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/dense_layer.h"

using namespace std;
using namespace chrono;
using namespace Eigen;
using namespace opennn;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

#ifdef OPENNN_CUDA
        cout << "Enable CUDA in pch.h" << endl;
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

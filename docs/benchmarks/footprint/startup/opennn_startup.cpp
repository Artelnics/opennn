//   OpenNN startup-latency benchmark: construct a small MLP, run one forward
//   pass, print the result, exit. Measures time-to-first-prediction.

#include "../../../opennn/standard_networks.h"
#include "../../../opennn/neural_network.h"
#include "../../../opennn/configuration.h"

using namespace opennn;

int main()
{
    Configuration::instance().set(Device::Auto, Type::FP32);

    // Small MLP: 10 inputs -> 64 hidden -> 1 output
    ApproximationNetwork network({10}, {64}, {1});

    MatrixR input(1, 10);
    input.setOnes();

    const MatrixR output = network.calculate_outputs(input);

    cout << "prediction " << output(0, 0) << endl;
    return 0;
}

// The smallest program that exercises the installed package: build a network,
// run it on the CPU, and print one number. If this compiles, links and runs,
// the exported target, its include directories and its transitive
// dependencies are all in place.
#include "opennn/core/configuration.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/dense_layer.h"

#include <iostream>
#include <cstdio>

extern "C"
{
#include <jpeglib.h>
}

int main()
{
    using namespace opennn;

    Configuration::instance().set(Device::CPU, Type::FP32);

    Network network;
    network.add_layer(std::make_unique<opennn::Dense>(Shape{4}, Shape{3}, "ReLU"), {-1});
    network.add_layer(std::make_unique<opennn::Dense>(Shape{3}, Shape{1}, "Identity"), {0});
    network.compile();
    network.set_parameters_random();

    MatrixR inputs(2, 4);
    inputs.setOnes();
    const MatrixR outputs = network.calculate_outputs(inputs);

    // Exercise the bundled JPEG archive as well as its imported target. A
    // dense-only executable can link without pulling any JPEG symbols in.
    jpeg_decompress_struct decoder{};
    jpeg_error_mgr errors{};
    decoder.err = jpeg_std_error(&errors);
    jpeg_create_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);

    std::cout << "opennn package smoke: " << outputs.rows() << "x" << outputs.cols()
              << " outputs, first " << outputs(0, 0) << "\n";
    return outputs.rows() == 2 && outputs.cols() == 1 && outputs.allFinite() ? 0 : 1;
}

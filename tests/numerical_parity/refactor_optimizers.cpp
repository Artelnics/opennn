// Optimizer parity test — DEV-REFACTOR branch
#include "../opennn/pch.h"
#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/dense_layer.h"
#include "../opennn/loss.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/quasi_newton_method.h"
#include "../opennn/levenberg_marquardt_algorithm.h"

#include <fstream>
#include <iomanip>

using namespace opennn;
using namespace std;

void make_dataset(Dataset& dataset)
{
    MatrixR data(20, 3);
    for(Index i = 0; i < 20; i++)
    {
        data(i, 0) = type(i) * type(0.1);
        data(i, 1) = type(20 - i) * type(0.05);
        data(i, 2) = type(0.3) * data(i, 0) + type(0.7) * data(i, 1) + type(0.1);
    }
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
}

int main()
{
    ofstream out("refactor_optimizers.txt");
    out << fixed << setprecision(10);

    // Test 1: Adam — 10 epochs
    {
        Dataset dataset(20, {2}, {1});
        make_dataset(dataset);

        NeuralNetwork nn;
        nn.add_layer(make_unique<opennn::Dense<2>>(Shape{2}, Shape{1}, "Linear"));
        nn.compile();
        nn.get_parameters().setConstant(type(0.1));

        Loss loss(&nn, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        loss.set_regularization("None");

        AdaptiveMomentEstimation adam(&loss);
        adam.set_display(false);
        adam.set_maximum_epochs(10);
        adam.set_batch_size(20);

        TrainingResults results = adam.train();

        out << "# Adam_10epochs" << endl;
        out << "final_error " << results.get_training_error() << endl;
        out << "epochs " << results.get_epochs_number() << endl;
    }

    // Test 2: SGD — 10 epochs
    {
        Dataset dataset(20, {2}, {1});
        make_dataset(dataset);

        NeuralNetwork nn;
        nn.add_layer(make_unique<opennn::Dense<2>>(Shape{2}, Shape{1}, "Linear"));
        nn.compile();
        nn.get_parameters().setConstant(type(0.1));

        Loss loss(&nn, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        loss.set_regularization("None");

        StochasticGradientDescent sgd(&loss);
        sgd.set_display(false);
        sgd.set_maximum_epochs(10);
        sgd.set_batch_size(20);

        TrainingResults results = sgd.train();

        out << "# SGD_10epochs" << endl;
        out << "final_error " << results.get_training_error() << endl;
        out << "epochs " << results.get_epochs_number() << endl;
    }

    // Test 3: QuasiNewton — 10 epochs
    {
        Dataset dataset(20, {2}, {1});
        make_dataset(dataset);

        NeuralNetwork nn;
        nn.add_layer(make_unique<opennn::Dense<2>>(Shape{2}, Shape{1}, "Linear"));
        nn.compile();
        nn.get_parameters().setConstant(type(0.1));

        Loss loss(&nn, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        loss.set_regularization("None");

        QuasiNewtonMethod qn(&loss);
        qn.set_display(false);
        qn.set_maximum_epochs(10);

        TrainingResults results = qn.train();

        out << "# QuasiNewton_10epochs" << endl;
        out << "final_error " << results.get_training_error() << endl;
        out << "epochs " << results.get_epochs_number() << endl;
    }

    // Test 4: LevenbergMarquardt — 10 epochs
    {
        Dataset dataset(20, {2}, {1});
        make_dataset(dataset);

        NeuralNetwork nn;
        nn.add_layer(make_unique<opennn::Dense<2>>(Shape{2}, Shape{1}, "Linear"));
        nn.compile();
        nn.get_parameters().setConstant(type(0.1));

        Loss loss(&nn, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        loss.set_regularization("None");

        LevenbergMarquardtAlgorithm lm(&loss);
        lm.set_display(false);
        lm.set_maximum_epochs(10);

        TrainingResults results = lm.train();

        out << "# LM_10epochs" << endl;
        out << "final_error " << results.get_training_error() << endl;
        out << "epochs " << results.get_epochs_number() << endl;
    }

    // Test 5: Final outputs after training (Adam)
    {
        Dataset dataset(20, {2}, {1});
        make_dataset(dataset);

        NeuralNetwork nn;
        nn.add_layer(make_unique<opennn::Dense<2>>(Shape{2}, Shape{1}, "Linear"));
        nn.compile();
        nn.get_parameters().setConstant(type(0.1));

        Loss loss(&nn, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        loss.set_regularization("None");

        AdaptiveMomentEstimation adam(&loss);
        adam.set_display(false);
        adam.set_maximum_epochs(50);
        adam.set_batch_size(20);
        adam.train();

        MatrixR input(3, 2);
        input << 0.0f, 1.0f, 0.5f, 0.75f, 1.0f, 0.5f;
        MatrixR outputs = nn.calculate_outputs(input);

        out << "# Adam_Outputs" << endl;
        for(Index i = 0; i < 3; i++)
            out << "output " << i << " " << outputs(i, 0) << endl;
    }

    out.close();
    cout << "Refactor optimizer outputs written." << endl;
    return 0;
}

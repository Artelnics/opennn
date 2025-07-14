//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   P R O J E C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/time_series_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "adaptive_moment_estimation.h"
#include "testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Forecasting Example." << endl;

        // Data set

        //TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting.csv", ",", true, false);
        //TimeSeriesDataset time_series_dataset("../data/Pendulum.csv", ",", false, false);
        TimeSeriesDataset time_series_dataset("../data/twopendulum.csv", ";", false, false);

        time_series_dataset.set_sample_uses("Training");

        time_series_dataset.print();

        const Index neurons_number = 4;
        const Index time_steps = 1;
        const Index input_size = time_series_dataset.get_raw_variables_number("Input");
        const Index batch_size = 1000;
        const Index epoch = 100;

        const vector<Index> input_variable_indices = time_series_dataset.get_variable_indices("Input");
        const vector<Index> target_variable_indices = time_series_dataset.get_variable_indices("Target");
        const vector<Index> decoder_variable_indices = time_series_dataset.get_variable_indices("Decoder");

        const Index training_batches_number = time_series_dataset.get_samples_number() / batch_size;

        const vector<Index> training_samples_indices = time_series_dataset.get_sample_indices("Training");

        // NeuralNetwork neural_network;

        ForecastingNetwork neural_network({time_series_dataset.get_variables_number("Input")},
                                          {neurons_number},
                                          {time_series_dataset.get_variables_number("Target")});


        neural_network.print();

        // Recurrent recurrent_layer({input_size, time_steps}, {neurons_number});
        // recurrent_layer.set_activation_function("HyperbolicTangent");

        // recurrent_layer.print();

        TrainingStrategy training_strategy(&neural_network, &time_series_dataset);
        training_strategy.set_loss_index("MeanSquaredError");
        training_strategy.get_loss_index()->set_regularization_method("NoRegularization");
        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());

        AdaptiveMomentEstimationData optimization_data(adam);

        LossIndex* loss_index = training_strategy.get_loss_index();
        loss_index->set(&neural_network, &time_series_dataset);

        Batch training_batch(batch_size, &time_series_dataset);
        ForwardPropagation training_forward_propagation(batch_size, &neural_network);
        // unique_ptr<LayerBackPropagation> back_propagation = make_unique<RecurrentBackPropagation>(batch_size, &recurrent_layer);

        BackPropagation training_back_propagation(batch_size, loss_index);

        vector<vector<Index>> training_batches(training_batches_number);
        type training_error = type(0);

        optimization_data.iteration = 1;

        for (int i = 0; i < epoch; ++i) {
            cout << "epoca: " << i << endl;


            training_batches = time_series_dataset.get_batches(training_samples_indices, batch_size, false);

            training_error = type(0);

            for (int iteration = 0; iteration < training_batches_number; ++iteration) {

                training_batch.fill(training_batches[iteration],
                                    input_variable_indices,
                                    decoder_variable_indices,
                                    target_variable_indices);

                //crea forward propagate
                // unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<RecurrentForwardPropagation>(batch_size, &recurrent_layer);
                // recurrent_layer.forward_propagate(training_batch.get_input_pairs(), forward_propagation, false);

                neural_network.forward_propagate(training_batch.get_input_pairs(),
                                                  training_forward_propagation,
                                                  true);

                loss_index->back_propagate(training_batch,
                                           training_forward_propagation,
                                           training_back_propagation);

                training_error += training_back_propagation.error();

                adam->update_parameters(training_back_propagation, optimization_data);

            }

            training_error /= type(training_batches_number);
            cout << "training error: " << training_error << endl;
        }


        //forward_propagate
        // cout << endl << "FORWARD_PROPAGATE" << endl;
        // const Index input_size = 7;
        // const Index time_steps = 1;
        // const Index batch_size = 20;
        // const Index neurons_number =3;

        // Recurrent recurrent_layer({input_size, time_steps}, {neurons_number});
        // recurrent_layer.set_timesteps(time_steps);
        // recurrent_layer.set_activation_function("HyperbolicTangent");

        // recurrent_layer.print();

        // Tensor<type, 3> input_tensor(batch_size, time_steps, input_size); // [batch, time, input]
        // for(Index b = 0; b < batch_size; ++b)
        //     for(Index t = 0; t < time_steps; ++t)
        //         for(Index k = 0; k < input_size; ++k)
        //             input_tensor(b, t, k) = static_cast<type>(0.1 * b + 0.05 * t + 0.01 * k);

        // //crear inputs pair
        // vector<std::pair<type*, dimensions>> input_pairs = {{input_tensor.data(), {batch_size, time_steps, input_size}}};

        // //crea forward propagate
        // unique_ptr<LayerForwardPropagation> forward_propagation = make_unique<RecurrentForwardPropagation>(batch_size, &recurrent_layer);
        // recurrent_layer.forward_propagate(input_pairs, forward_propagation, false);

        // // Extraer datos relevantes
        // auto* recurrent_forward = static_cast<RecurrentForwardPropagation*>(forward_propagation.get());

        // const auto& hidden_states = recurrent_forward->hidden_states;
        // const auto& activation_derivatives = recurrent_forward->activation_derivatives;

        // // Mostrar resultados
        // cout << endl << "== Salida paso a paso ==" << endl;
        // cout << "batch, time_step, inputs[inputs_size], hidden_states[neurons], activation_functions[neurons]" << endl;
        // for(Index t = 0; t < time_steps; ++t)
        // {
        //     cout << std::fixed << std::setprecision(6);

        //     for(Index b = 0; b < batch_size; ++b)
        //     {
        //         cout << "batch " << b << ", t=" << t;
        //         cout << ", x(inputs)=[";
        //         for(Index k = 0; k < input_size; ++k)
        //         {
        //             cout << input_tensor(b, t, k);
        //             if(k < input_size-1) cout << ", ";
        //         }

        //         cout << "], h(hidden_states)=[";
        //         for(Index n = 0; n < neurons_number; ++n){
        //             cout << "h(" << n << ")=" << hidden_states(b, t, n);
        //             if(n < neurons_number-1) cout << "; ";
        //         }
        //         cout << "], d_act=[";
        //         for(Index n = 0; n < neurons_number; ++n)
        //         {
        //             cout << activation_derivatives(b, t, n);
        //             if(n < neurons_number-1) cout << ", ";
        //         }
        //         cout << "]\n";
        //     }
        // }

        // //back_propagate
        // cout << endl << "BACK_PROPAGATE" << endl;

        // Tensor<type,2> deltas(batch_size, neurons_number);
        // deltas.setConstant(1.0);

        // vector<std::pair<type*,dimensions>> delta_pairs = {{deltas.data(), {batch_size, neurons_number} }};

        // //crear back_propagation
        // unique_ptr<LayerBackPropagation> back_propagation = make_unique<RecurrentBackPropagation>(batch_size, &recurrent_layer);

        // recurrent_layer.back_propagate(input_pairs, delta_pairs, forward_propagation, back_propagation);

        // //mostrar resultados
        // cout << "back_propagation calculado" << endl;
        // auto* back = static_cast<RecurrentBackPropagation*>(back_propagation.get());

        // cout << "input_deltas     = " << back->input_deltas(0,0,0)      << '\n';

        // const auto& dW_in  = back->input_weight_deltas;
        // const auto& dU     = back->recurrent_weight_deltas;
        // const auto& db     = back->bias_deltas;
        // const auto& dX     = back->input_deltas;

        // std::cout << "\n== Gradientes acumulados ==" << endl;
        // std::cout << "Biases: " << endl << db << endl << endl;
        // std::cout << "Inputs_weight: " << endl << dW_in << endl << endl;
        // std::cout << "recurrent_weight: " << endl << dU << endl << endl;

        // std::cout << "\n== dX(inputs_deltas) por paso de tiempo ==" << std::endl;
        // for(Index t=0; t<time_steps; ++t)
        // {
        //     cout << "-- t=" << t << " --" << endl;
        //     for (int i = 0; i < batch_size; ++i){
        //         cout << "Batch " << i;
        //         for (int j = 0; j < input_size; ++j)
        //             cout << ", d" << j << "="<< dX(i,t,j);
        //         cout << endl;
        //     }
        //     cout << endl;
        // }

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
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

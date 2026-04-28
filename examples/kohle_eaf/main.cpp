//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   K O H L E   E A F   A P P L I C A T I O N
//
//   Predict specific electrical energy consumption (kWh/t) of an
//   Electric Arc Furnace from process variables and scrap composition.
//
//   Dataset: data/per_ton/dataset_per_ton.csv (3,736 heats)
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "Kohle EAF - Per-tonne energy prediction" << endl;

        const Index neurons_number = 8;
        const type regularization_weight = type(0.001);

        // Dataset

        Dataset dataset("../data/per_ton/dataset_per_ton.csv", ",", true, false);

        const vector<string> non_target_outputs = {
            "NivelEscoria_output", "EnergiaVuelco_output",
            "PromedioEscoriaEspumante_output", "TiempoEscoriaEspumante_output"
        };

        for(const string& col : non_target_outputs)
            dataset.set_variable_role(col, "None");

        dataset.set_variable_role("PotenciaActiva_per_ton_output", "Target");
        dataset.set_shape("Input", {dataset.get_features_number("Input")});

        dataset.split_samples_random(type(0.6), type(0.2), type(0.2));

        // Neural Network

        ApproximationNetwork neural_network(dataset.get_input_shape(), {neurons_number}, dataset.get_target_shape());

        Bounding* bounding = (Bounding*)neural_network.get_first("Bounding");

        if(bounding)
            bounding->set_bounding_method("NoBounding");

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &dataset);

        training_strategy.get_loss()->set_regularization_method("L2");
        training_strategy.get_loss()->set_regularization_weight(regularization_weight);
        training_strategy.get_optimization_algorithm()->set_display_period(100);

        TrainingResults training_results = training_strategy.train();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &dataset);
        testing_analysis.print_goodness_of_fit_analysis();

        // Export

        const vector<Variable>& variables = dataset.get_variables();
        ModelExpression model_expression(&neural_network);

        model_expression.save_javascript("../data/per_ton/process_and_scraps_to_potencia_per_ton.html", variables);
        model_expression.save_python("../data/per_ton/process_and_scraps_to_potencia_per_ton.py", variables);
        model_expression.save_c("../data/per_ton/process_and_scraps_to_potencia_per_ton.c", variables);

        cout << "Model exported to data/per_ton/" << endl;

        // Response optimization

        ResponseOptimization response_optimization(&neural_network);

        response_optimization.set_iterations(15);
        response_optimization.set_evaluations_number(2000);
        response_optimization.set_zoom_factor(type(0.7));

        response_optimization.set_condition("PotenciaActiva_per_ton_output",
                                            ResponseOptimization::ConditionType::Minimize);

        const MatrixR result = response_optimization.perform_response_optimization();

        if(result.rows() > 0)
        {
            const vector<Variable> input_vars = neural_network.get_input_variables();

            cout << "\nOptimal operating point:" << endl;

            for(Index i = 0; i < Index(input_vars.size()); i++)
                cout << "  " << input_vars[i].name << " = " << result(0, i) << endl;

            cout << "  PotenciaActiva_per_ton = " << result(0, Index(input_vars.size())) << " kWh/t" << endl;
        }

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

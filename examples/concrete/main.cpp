//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   R E S P O N S E   O P T I M I Z A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Asks a network trained on the UCI concrete data for the strongest mix it can find, and
// holds that mix to constraints that read the response as well as the ingredients.
//
// Other problems set on this same network -- unconstrained search, a closed mix mass, a
// strength against cement front, a fixed strength target, and a tightly constrained
// multiobjective mix -- are covered by tests/response_optimization/concrete_scenarios_test.cpp,
// which reports what each one returns.

#include <filesystem>
#include <iostream>

#include "opennn/response_optimization/response_optimization.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"
#include "opennn/neural_network/neural_network.h"

using namespace opennn;

#ifndef CONCRETE_EXAMPLE_DIR
#define CONCRETE_EXAMPLE_DIR "."
#endif

int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        const auto set_problem = [](ResponseOptimization& response_optimization)
        {
            response_optimization.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

            // Strength per kilogram of binder: a ratio of the output to a group of inputs.

            response_optimization.add_constraint("strength / (cement + slag + fly_ash)",
                                                 ResponseOptimization::Constraint::Condition::GreaterEqual,
                                                 {0.10f});

            // Strength per unit of mix cost, with indicative prices per kilogram.

            response_optimization.add_constraint("strength / (0.10 * cement + 0.05 * slag + 0.04 * fly_ash"
                                                 " + 1.20 * sp + 0.02 * coarse_agg + 0.02 * fine_agg)",
                                                 ResponseOptimization::Constraint::Condition::GreaterEqual,
                                                 {0.55f});

            // The water to cement ratio is nonlinear too, but it only reads inputs.

            response_optimization.add_constraint("water / cement",
                                                 ResponseOptimization::Constraint::Condition::Between,
                                                 {0.35f, 0.60f});

            response_optimization.add_constraint("age",
                                                 ResponseOptimization::Constraint::Condition::Equal,
                                                 {28.0f});
        };

        DomainContraction domain_contraction(&concrete_network);

        set_problem(domain_contraction);

        GeneticResponse genetic_response(&concrete_network);

        set_problem(genetic_response);

        cout << "Iterative domain contraction:" << endl;

        cout << domain_contraction.perform_response_optimization() << endl;

        cout << "Non dominated genetic:" << endl;

        cout << genetic_response.perform_response_optimization() << endl;

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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

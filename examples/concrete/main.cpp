//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   R E S P O N S E   O P T I M I Z A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   Optimizes a concrete mix with a pretrained response model.

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

namespace
{

void configure_problem(ResponseOptimization& optimization)
{
    optimization.add_objective(
        "strength", ResponseOptimization::Objective::Sense::Maximize);
    optimization.add_constraint(
        "strength / (cement + slag + fly_ash)",
        ResponseOptimization::Constraint::Condition::GreaterEqual, {0.10f});
    optimization.add_constraint(
        "strength / (0.10 * cement + 0.05 * slag + 0.04 * fly_ash"
        " + 1.20 * sp + 0.02 * coarse_agg + 0.02 * fine_agg)",
        ResponseOptimization::Constraint::Condition::GreaterEqual, {0.55f});
    optimization.add_constraint(
        "water / cement",
        ResponseOptimization::Constraint::Condition::Between, {0.35f, 0.60f});
    optimization.add_constraint(
        "age", ResponseOptimization::Constraint::Condition::Equal, {28.0f});
}

}

int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork network(
            filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&network);

        configure_problem(domain_contraction);

        GeneticResponse genetic_response(&network);

        configure_problem(genetic_response);

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

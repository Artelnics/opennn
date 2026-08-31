//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   R E S P O N S E   O P T I M I Z A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opennn/response_optimization/response_optimization.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/dataset/tabular_dataset.h"

using namespace opennn;

#ifndef CONCRETE_EXAMPLE_DIR
#define CONCRETE_EXAMPLE_DIR "."
#endif

/*
// BASIC EXAMPLE
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl; 
  
        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&concrete_network);

        domain_contraction.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        //domain_contraction.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        cout << domain_contraction.perform_response_optimization() << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/
// FORMULA EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl; 
  
        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&concrete_network);

        domain_contraction.add_constraint("water - 0.30 * cement",
                                                    ResponseOptimization::Constraint::Condition::GreaterEqual, 
                                                    {0.0f});

        domain_contraction.add_constraint("water - 0.70 * cement",
                                                    ResponseOptimization::Constraint::Condition::LessEqual, 
                                                    {0.0f});

        domain_contraction.add_constraint("age", ResponseOptimization::Constraint::Condition::Equal, {28.0f});

        domain_contraction.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        //domain_contraction.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        cout << domain_contraction.perform_response_optimization() << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/



// NSGA-II EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        GeneticResponse genetic_response(&concrete_network);

        genetic_response.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        genetic_response.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        //genetic_response.add_constraint("age", ResponseOptimization::Constraint::Condition::Equal, {28.0f});

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
*/


// SIMPLEX EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl; 
  
        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&concrete_network);

        domain_contraction.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                                    ResponseOptimization::Constraint::Condition::Equal,
                                                    {2325.012558f});

        domain_contraction.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        //domain_contraction.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        cout << domain_contraction.perform_response_optimization() << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/



// IDC MULTIOBJECTIVE EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&concrete_network);

        domain_contraction.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        domain_contraction.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        cout << domain_contraction.perform_response_optimization() << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/



// CONSTRAINED PARETO FRONT EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        GeneticResponse genetic_response(&concrete_network);

        genetic_response.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        genetic_response.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        genetic_response.add_constraint("age", ResponseOptimization::Constraint::Condition::Equal, {28.0f});

        genetic_response.add_constraint("water - 0.70 * cement",
                                             ResponseOptimization::Constraint::Condition::LessEqual,
                                             {0.0f});

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
*/



// FIXED EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&concrete_network);

        domain_contraction.add_objective("strength", ResponseOptimization::Objective::Sense::Fixed, 50.0f);

        domain_contraction.add_constraint("cement",
                                                    ResponseOptimization::Constraint::Condition::Between,
                                                    {150.0f, 350.0f});

        cout << domain_contraction.perform_response_optimization() << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/

// BOTH ALGORITHMS EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        DomainContraction domain_contraction(&concrete_network);

        domain_contraction.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        domain_contraction.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

        GeneticResponse genetic_response(&concrete_network);

        genetic_response.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

        genetic_response.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

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
    */


// TIGHT MULTIOBJECTIVE EXAMPLE
/*
int main()
{
    try
    {
        cout << "OpenNN. Concrete Response Optimization Example." << endl;

        NeuralNetwork concrete_network(filesystem::path(CONCRETE_EXAMPLE_DIR) / "nn" / "concrete_uci.json");

        const auto set_problem = [](ResponseOptimization& response_optimization)
        {
            response_optimization.add_objective("strength", ResponseOptimization::Objective::Sense::Maximize);

            response_optimization.add_objective("cement", ResponseOptimization::Objective::Sense::Minimize);

            response_optimization.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                                 ResponseOptimization::Constraint::Condition::Equal,
                                                 {2325.012558f});

            response_optimization.add_constraint("age",
                                                 ResponseOptimization::Constraint::Condition::Equal,
                                                 {28.0f});

            response_optimization.add_constraint("water",
                                                 ResponseOptimization::Constraint::Condition::Between,
                                                 {175.0f, 185.0f});

            response_optimization.add_constraint("sp",
                                                 ResponseOptimization::Constraint::Condition::Between,
                                                 {4.0f, 8.0f});

            response_optimization.add_constraint("water - 0.40 * cement",
                                                 ResponseOptimization::Constraint::Condition::GreaterEqual,
                                                 {0.0f});

            response_optimization.add_constraint("water - 0.55 * cement",
                                                 ResponseOptimization::Constraint::Condition::LessEqual,
                                                 {0.0f});

            response_optimization.add_constraint("strength",
                                                 ResponseOptimization::Constraint::Condition::GreaterEqual,
                                                 {40.0f});
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
*/


// NONLINEAR OUTPUT CONSTRAINT EXAMPLE
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
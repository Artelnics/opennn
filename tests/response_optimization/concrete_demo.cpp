//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   R E S P O N S E   D E M O
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A plain main, written the way a user writes one: load a network, state a problem, run it,
// read the numbers. No fixtures, no test macros, no helper types. Every problem below is a
// self contained block that can be copied out of here into a program of your own.
//
// The problems are ordered so that each one adds a kind of constraint to the previous, and
// together they use every condition the response optimization understands today:
//
//     Equal   Between   GreaterEqual   LessEqual   Greater   Less   AllowedSet   Integer   Cardinality
//
// and every objective sense: Minimize, Maximize, Fixed, plus more than one objective at
// once, which returns a front instead of a point.
//
// The network is the trained UCI concrete network the concrete example ships. Its inputs
// are the seven ingredients of a mix and its age; its output is the strength the mix
// reaches. Nothing here writes to it.
//
// Build and run:
//
//     cmake --build build-ninja --target opennn_response_demo
//     build-ninja/bin/opennn_response_demo

#include <filesystem>
#include <iomanip>
#include <iostream>

#include "opennn/network/network.h"
#include "opennn/response_optimization/response_optimization.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"
#include "opennn/core/random_utilities.h"

using namespace opennn;

// Short names for the two enums, so a problem reads as the sentence it is.

using Sense = ResponseOptimization::Objective::Sense;
using Condition = ResponseOptimization::Constraint::Condition;

#ifndef CONCRETE_NETWORK_DIR
#define CONCRETE_NETWORK_DIR "examples/concrete"
#endif

// The columns of every result: the eight inputs the network takes, then its response.

const char* const columns[] =
    {"cement", "slag", "fly_ash", "water", "sp", "coarse_agg", "fine_agg", "age", "strength"};

const int columns_number = 9;


// Prints a banner, so the output of a long run can be read problem by problem.

void title(const string& text)
{
    cout << "\n================================================================\n"
         << text << "\n"
         << "================================================================\n";
}


// Prints what came back. One row is a mix; a multiobjective run returns many.

void show(const MatrixR& results)
{
    const Index rows_shown = min(results.rows(), Index(6));

    cout << fixed << setprecision(2);

    for (int j = 0; j < columns_number; j++)
        cout << setw(11) << columns[j];

    cout << "\n";

    for (Index i = 0; i < rows_shown; i++)
    {
        for (Index j = 0; j < results.cols(); j++)
            cout << setw(11) << results(i, j);

        cout << "\n";
    }

    if (results.rows() > rows_shown)
        cout << "... " << results.rows() - rows_shown << " more row(s), "
             << results.rows() << " in all\n";
}


int main()
{
    try
    {
        Network network(filesystem::path(CONCRETE_NETWORK_DIR) / "nn" / "concrete_uci.json");

        // Fixing the seed makes every run below repeat exactly. Drop it to see the spread.

        set_seed(1234);


        // 1. The simplest problem there is: one objective, nothing to satisfy.
        //
        // Both solvers take the same problem and are used the same way. DomainContraction
        // shrinks a box around the best point it has seen; GeneticResponse evolves a
        // population. Everywhere below either one can be swapped in for the other.

        title("1. Maximize strength, unconstrained");

        DomainContraction problem_1(&network);

        problem_1.add_objective("strength", Sense::Maximize);

        show(problem_1.perform_response_optimization());

        cout << "\nThe same problem, given to the other solver:\n\n";

        GeneticResponse problem_1_genetic(&network);

        problem_1_genetic.add_objective("strength", Sense::Maximize);

        show(problem_1_genetic.perform_response_optimization());

        cout << "\nWith nothing holding it back the search walks to a corner of the box the\n"
                "network was trained on, and reports a strength above anything in the data.\n"
                "The solver is right; the network is extrapolating. Hence the constraints.\n";


        // 2. Equal, on one input. The age is not something to search over: the mix is
        // specified at 28 days, so it is pinned and the search happens in what is left.

        title("2. Maximize strength at a fixed age (Equal)");

        DomainContraction problem_2(&network);

        problem_2.add_objective("strength", Sense::Maximize);
        problem_2.add_constraint("age", Condition::Equal, {28.0f});

        show(problem_2.perform_response_optimization());


        // 3. GreaterEqual and LessEqual, on expressions rather than variables. The water to
        // cement ratio is written here as two linear bounds, which is how a specification
        // usually states it.

        title("3. A water to cement band, as two linear bounds (GreaterEqual, LessEqual)");

        DomainContraction problem_3(&network);

        problem_3.add_objective("strength", Sense::Maximize);

        problem_3.add_constraint("water - 0.30 * cement", Condition::GreaterEqual, {0.0f});
        problem_3.add_constraint("water - 0.70 * cement", Condition::LessEqual, {0.0f});
        problem_3.add_constraint("age", Condition::Equal, {28.0f});

        const MatrixR results_3 = problem_3.perform_response_optimization();

        show(results_3);

        cout << "\nwater/cement = " << results_3(0, 3)/results_3(0, 0) << ", asked for [0.30, 0.70]\n";


        // 4. Between, which says the same thing in one line, and on a ratio, which is what
        // the quantity actually is. The expression is nonlinear, and it reads inputs only.

        title("4. The same band written as a ratio (Between)");

        DomainContraction problem_4(&network);

        problem_4.add_objective("strength", Sense::Maximize);

        problem_4.add_constraint("water / cement", Condition::Between, {0.35f, 0.60f});
        problem_4.add_constraint("age", Condition::Equal, {28.0f});

        const MatrixR results_4 = problem_4.perform_response_optimization();

        show(results_4);

        cout << "\nwater/cement = " << results_4(0, 3)/results_4(0, 0) << ", asked for [0.35, 0.60]\n";


        // 5. Greater and Less: the strict pair. They ask for the same side of the bound as
        // GreaterEqual and LessEqual, but they will not settle on the bound itself.

        title("5. Strictly inside the bounds (Greater, Less)");

        DomainContraction problem_5(&network);

        problem_5.add_objective("strength", Sense::Maximize);

        problem_5.add_constraint("cement", Condition::Less, {400.0f});
        problem_5.add_constraint("slag", Condition::Greater, {50.0f});
        problem_5.add_constraint("age", Condition::Equal, {28.0f});

        const MatrixR results_5 = problem_5.perform_response_optimization();

        show(results_5);

        cout << "\ncement = " << results_5(0, 0) << " (below 400), slag = "
             << results_5(0, 1) << " (above 50)\n";


        // 6. One equality over seven variables. A cubic metre of concrete weighs what it
        // weighs, so the ingredients have to add up: nothing can be added to the mix
        // without taking something else out of it.

        title("6. A closed batch: one equality over seven variables (Equal)");

        DomainContraction problem_6(&network);

        problem_6.add_objective("strength", Sense::Maximize);

        problem_6.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                 Condition::Equal,
                                 {2325.012558f});

        const MatrixR results_6 = problem_6.perform_response_optimization();

        show(results_6);

        cout << "\nbatch mass = " << results_6.row(0).head(7).sum() << ", asked for 2325.01\n";


        // 7. A constraint that reads the response. Nothing about the mix alone says whether
        // it holds: the network has to be asked, and the answer moves as the mix moves.

        title("7. A floor on the response itself (GreaterEqual on an output)");

        GeneticResponse problem_7(&network);

        problem_7.add_objective("cement", Sense::Minimize);
        problem_7.add_constraint("strength", Condition::GreaterEqual, {40.0f});

        const MatrixR results_7 = problem_7.perform_response_optimization();

        show(results_7);

        cout << "\nThe cheapest mix that still reaches 40 MPa: " << results_7(0, 0)
             << " kg of cement for " << results_7(0, 8) << " MPa\n";


        // 8. Constraints that mix the response and the inputs in one expression. These are
        // the reason the constraints are solved rather than checked: the quantity is a
        // ratio of what came out to what went in.

        title("8. Ratios of the response to the mix (nonlinear, output coupled)");

        DomainContraction problem_8(&network);

        problem_8.add_objective("strength", Sense::Maximize);

        // Strength per kilogram of binder.

        problem_8.add_constraint("strength / (cement + slag + fly_ash)",
                                 Condition::GreaterEqual,
                                 {0.10f});

        // Strength per unit of mix cost, at indicative prices per kilogram.

        problem_8.add_constraint("strength / (0.10 * cement + 0.05 * slag + 0.04 * fly_ash"
                                 " + 1.20 * sp + 0.02 * coarse_agg + 0.02 * fine_agg)",
                                 Condition::GreaterEqual,
                                 {0.55f});

        problem_8.add_constraint("age", Condition::Equal, {28.0f});

        const MatrixR results_8 = problem_8.perform_response_optimization();

        show(results_8);

        const float binder_8 = results_8(0, 0) + results_8(0, 1) + results_8(0, 2);

        const float cost_8 = 0.10f*results_8(0, 0) + 0.05f*results_8(0, 1) + 0.04f*results_8(0, 2)
                           + 1.20f*results_8(0, 4) + 0.02f*results_8(0, 5) + 0.02f*results_8(0, 6);

        cout << "\nMPa per kg of binder = " << results_8(0, 8)/binder_8 << ", asked for 0.10 or more\n"
             << "MPa per unit of cost  = " << results_8(0, 8)/cost_8 << ", asked for 0.55 or more\n";


        // 9. AllowedSet: a variable that may only take one of a few listed values. Ages are
        // tested at 7, 28 and 90 days, and nothing in between is a thing you can order.

        title("9. Only the standard test ages (AllowedSet)");

        DomainContraction problem_9(&network);

        problem_9.add_objective("strength", Sense::Maximize);

        problem_9.add_constraint("age", Condition::AllowedSet, {7.0f, 28.0f, 90.0f});
        problem_9.add_constraint("water / cement", Condition::Between, {0.35f, 0.60f});

        const MatrixR results_9 = problem_9.perform_response_optimization();

        show(results_9);

        cout << "\nage = " << results_9(0, 7) << ", one of {7, 28, 90}\n";


        // 10. Integer: a variable that has to come out whole. It applies to a single input
        // variable, not to an expression, because it is a property of the variable itself.

        title("10. Whole kilograms and whole days (Integer)");

        DomainContraction problem_10(&network);

        problem_10.add_objective("strength", Sense::Maximize);

        problem_10.add_constraint("cement", Condition::Integer);
        problem_10.add_constraint("age", Condition::Integer);
        problem_10.add_constraint("age", Condition::LessEqual, {90.0f});

        const MatrixR results_10 = problem_10.perform_response_optimization();

        show(results_10);

        // Printed with more decimals than the table above, which would round anything into
        // looking whole.

        cout << "\ncement = " << setprecision(4) << results_10(0, 0)
             << " kg, age = " << results_10(0, 7) << " days\n" << setprecision(2);


        // 11. Cardinality: how many of a group of variables may be in play at once. Here
        // one of the three admixtures is allowed, because a plant with one silo cannot
        // dose three. The variables it counts must be able to reach zero, and the ones
        // switched off come back at exactly zero.
        //
        // Expect warnings about local domains running short of feasible points: switching
        // variables off carves the box up, so random draws land outside it far more often.

        title("11. One admixture in the mix (Cardinality)");

        DomainContraction problem_11(&network);

        problem_11.add_objective("strength", Sense::Maximize);

        problem_11.add_constraint("slag; fly_ash; sp", Condition::Cardinality, {1.0f});
        problem_11.add_constraint("age", Condition::Equal, {28.0f});

        const MatrixR results_11 = problem_11.perform_response_optimization();

        show(results_11);

        cout << "\nslag = " << results_11(0, 1) << ", fly_ash = " << results_11(0, 2)
             << ", sp = " << results_11(0, 4)
             << ": one of them is dosed, the other two are off\n";


        // 12. A target rather than a direction. Fixed asks the search to land on a value
        // instead of pushing it as far as it goes, which is what a specification asks for.

        title("12. Hit a strength of 50 MPa exactly (Fixed)");

        DomainContraction problem_12(&network);

        problem_12.add_objective("strength", Sense::Fixed, 50.0f);
        problem_12.add_constraint("cement", Condition::Between, {150.0f, 350.0f});

        const MatrixR results_12 = problem_12.perform_response_optimization();

        show(results_12);

        cout << "\nstrength = " << results_12(0, 8) << ", asked for 50.00\n";


        // 13. Two objectives that pull against each other. There is no single best mix, so
        // what comes back is a front: every row is a mix that no other row beats on both
        // counts, and choosing between them is yours to do.

        title("13. Strength against cement: two objectives, so a front (Maximize, Minimize)");

        GeneticResponse problem_13(&network);

        problem_13.add_objective("strength", Sense::Maximize);
        problem_13.add_objective("cement", Sense::Minimize);

        const MatrixR results_13 = problem_13.perform_response_optimization();

        show(results_13);

        cout << "\ncement runs from " << results_13.col(0).minCoeff() << " to "
             << results_13.col(0).maxCoeff() << " kg, and the strength with it from "
             << results_13.col(8).minCoeff() << " to " << results_13.col(8).maxCoeff() << " MPa\n";


        // 14. A mix design as it is really written down, with every kind of constraint at
        // once: a closed batch, three nonlinear bands on ratios, one bound that reads the
        // response, a listed set of ages, and two objectives over all of it.

        title("14. A full mix design: everything at once");

        GeneticResponse problem_14(&network);

        problem_14.add_objective("strength", Sense::Maximize);
        problem_14.add_objective("cement", Sense::Minimize);

        problem_14.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                  Condition::Equal,
                                  {2325.012558f});

        // Water to binder: the ratio that governs strength and durability.

        problem_14.add_constraint("water / (cement + slag + fly_ash)",
                                  Condition::Between,
                                  {0.35f, 0.50f});

        // How much of the binder is slag and fly ash rather than clinker.

        problem_14.add_constraint("(slag + fly_ash) / (cement + slag + fly_ash)",
                                  Condition::Between,
                                  {0.20f, 0.50f});

        // The fine share of the aggregate, which sets how the mix handles.

        problem_14.add_constraint("fine_agg / (coarse_agg + fine_agg)",
                                  Condition::Between,
                                  {0.35f, 0.45f});

        // The response has to earn its binder, and the mix has to be strong enough to use.

        problem_14.add_constraint("strength / (cement + slag + fly_ash)",
                                  Condition::GreaterEqual,
                                  {0.10f});

        problem_14.add_constraint("strength", Condition::GreaterEqual, {30.0f});

        problem_14.add_constraint("age", Condition::AllowedSet, {7.0f, 28.0f, 90.0f});

        const MatrixR results_14 = problem_14.perform_response_optimization();

        show(results_14);

        cout << "\nrow       mass   w/binder   scm share   sand ratio   MPa per kg\n";

        for (Index i = 0; i < min(results_14.rows(), Index(6)); i++)
        {
            const float binder = results_14(i, 0) + results_14(i, 1) + results_14(i, 2);

            cout << setw(3) << i
                 << setw(11) << results_14.row(i).head(7).sum()
                 << setw(11) << results_14(i, 3)/binder
                 << setw(12) << (results_14(i, 1) + results_14(i, 2))/binder
                 << setw(13) << results_14(i, 6)/(results_14(i, 5) + results_14(i, 6))
                 << setw(13) << results_14(i, 8)/binder << "\n";
        }

        cout << "\nasked for       2325.01   [0.35, 0.50]   [0.20, 0.50]   [0.35, 0.45]   0.10 or more\n";


        // 15. What a problem that cannot be solved does. It throws, with a message, rather
        // than handing back a mix that does not meet what was asked. The network was
        // trained on strengths up to about 82 MPa, so 150 is not on the table.

        title("15. Asking for the impossible");

        try
        {
            DomainContraction problem_15(&network);

            problem_15.add_objective("cement", Sense::Minimize);
            problem_15.add_constraint("strength", Condition::GreaterEqual, {150.0f});

            show(problem_15.perform_response_optimization());

            cout << "\nThis line should not be reached.\n";
        }
        catch (const exception& error)
        {
            cout << "\ncaught, as it should be: " << error.what() << "\n";
        }


        // 16. And what a problem that was written down wrongly does. These are caught as
        // the constraint is added, before anything is run.

        title("16. Problems that are stated wrongly");

        try
        {
            DomainContraction problem_16(&network);

            problem_16.add_constraint("cement", Condition::Between, {400.0f, 200.0f});
        }
        catch (const exception& error)
        {
            cout << "\nan empty interval:     " << error.what() << "\n";
        }

        try
        {
            DomainContraction problem_16(&network);

            problem_16.add_constraint("2 * cement", Condition::Integer);
        }
        catch (const exception& error)
        {
            cout << "one input read through an expression: " << error.what() << "\n";
        }

        try
        {
            DomainContraction problem_16(&network);

            problem_16.add_constraint("cement", Condition::Cardinality, {1.0f});
        }
        catch (const exception& error)
        {
            cout << "counting one variable: " << error.what() << "\n";
        }

        try
        {
            DomainContraction problem_16(&network);

            problem_16.add_constraint("cement; slag", Condition::Cardinality, {1.0f});

            problem_16.add_objective("strength", Sense::Maximize);

            show(problem_16.perform_response_optimization());
        }
        catch (const exception& error)
        {
            cout << "counting a variable that cannot be switched off: " << error.what() << "\n";
        }

        try
        {
            DomainContraction problem_16(&network);

            problem_16.add_constraint("age", Condition::Equal, {28.0f});

            problem_16.perform_response_optimization();
        }
        catch (const exception& error)
        {
            cout << "constraints but no objective: " << error.what() << "\n";
        }

        cout << "\nDone.\n";

        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

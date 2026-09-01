//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   R E S P O N S E   S C E N A R I O S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// These scenarios run the response optimizers against the trained UCI concrete network
// that the concrete example ships, so the numbers they report mean something: a mix, its
// water to cement ratio, and the strength the network predicts for it.
//
// Every check below re-derives its quantity from the returned columns with plain
// arithmetic, never by asking the optimizer for the residual it just used. A fault in
// the residual would otherwise agree with itself and pass.
//
// What the repair returns on its own, before any search runs over it, is studied
// separately in feasibility_test.cpp.

#include "tests/pch.h"

#include <iomanip>

#include "tests/response_optimization/concrete_fixture.h"

#include "opennn/registry.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/core/random_utilities.h"

namespace
{

// Reads a bound with the slack the fixture defines, under the name the scenarios use.

float slack(const float bound) { return bound_slack(bound); }


enum class Driver { Contraction, Genetic };


const char* driver_name(const Driver driver)
{
    return (driver == Driver::Genetic) ? "genetic" : "contraction";
}


unique_ptr<ResponseOptimization> make_driver(const Driver driver)
{
    if (driver == Driver::Genetic)
        return make_unique<GeneticResponse>(&concrete_network());

    return make_unique<DomainContraction>(&concrete_network());
}


// What the run produced, printed rather than asserted. Reading these side by side is how
// a scenario tells you whether the front spread out, collapsed, or hugged a bound.

void report(const string& scenario, const Driver driver, const MatrixR& results)
{
    cout << "\n[ " << scenario << " | " << driver_name(driver) << " ] "
         << results.rows() << " point(s)\n";

    cout << left << setw(12) << "column" << right
         << setw(12) << "min" << setw(12) << "max" << setw(12) << "spread" << "\n";

    for (Index j = 0; j < ColumnsNumber; j++)
    {
        const float smallest = results.col(j).minCoeff();
        const float largest = results.col(j).maxCoeff();

        cout << left << setw(12) << column_names[j] << right << fixed << setprecision(3)
             << setw(12) << smallest << setw(12) << largest << setw(12) << (largest - smallest) << "\n";
    }
}


void expect_shape(const MatrixR& results)
{
    ASSERT_GT(results.rows(), 0);
    ASSERT_EQ(results.cols(), Index(ColumnsNumber));
}


// No mix may leave the box the scaling layer was trained on, whatever the constraints ask.

void expect_inside_the_box(const MatrixR& results)
{
    const Scaling* scaling_layer =
        static_cast<const Scaling*>(concrete_network().get_first(LayerType::Scaling));

    const VectorR minimums = scaling_layer->get_minimums();
    const VectorR maximums = scaling_layer->get_maximums();

    for (Index i = 0; i < results.rows(); i++)
        for (Index j = 0; j <= Age; j++)
        {
            EXPECT_GE(results(i, j), minimums(j) - slack(minimums(j)))
                << "row " << i << " " << column_names[j] << " = " << results(i, j);

            EXPECT_LE(results(i, j), maximums(j) + slack(maximums(j)))
                << "row " << i << " " << column_names[j] << " = " << results(i, j);
        }
}


// The strength the network actually predicts for the mix that came back, recomputed from
// the input columns. It catches a result whose response column drifted from its inputs.

VectorR predict_strength(const MatrixR& results)
{
    return concrete_network().calculate_outputs(MatrixR(results.leftCols(Age + 1))).col(0);
}


void expect_response_matches_the_mix(const MatrixR& results)
{
    const VectorR predicted = predict_strength(results);

    for (Index i = 0; i < results.rows(); i++)
        EXPECT_NEAR(results(i, Strength), predicted(i), 1e-2f)
            << "row " << i << " reports a strength its own mix does not produce";
}


MatrixR run(const string& scenario,
            const Driver driver,
            const function<void(ResponseOptimization&)>& set_problem)
{
    set_seed(1234);

    const unique_ptr<ResponseOptimization> optimization = make_driver(driver);

    set_problem(*optimization);

    const MatrixR results = optimization->perform_response_optimization();

    expect_shape(results);
    expect_inside_the_box(results);
    expect_response_matches_the_mix(results);

    report(scenario, driver, results);

    return results;
}


float best_strength(const MatrixR& results) { return results.col(Strength).maxCoeff(); }

}


// The scenarios below were carried over from the concrete example, where each one lived
// as a commented out main that had to be uncommented to be run.


TEST(ConcreteScenario, MaximizeStrengthWithoutConstraints)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("unconstrained strength", driver, set_problem);

        // Worth reading the report for this one: with nothing to hold it back the search
        // walks every variable to a corner of the training box and reports around 94 MPa,
        // above the 82.6 MPa the network was trained against. The optimizer is doing its
        // job; the network is extrapolating. It is the reason the scenarios that follow
        // constrain the mix rather than trusting an unconstrained optimum.

        EXPECT_GT(best_strength(results), 60.0f);
    }
}


TEST(ConcreteScenario, WaterCementBandAndFixedAge)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_constraint("water - 0.30 * cement", Condition::GreaterEqual, {0.0f});
        optimization.add_constraint("water - 0.70 * cement", Condition::LessEqual, {0.0f});
        optimization.add_constraint("age", Condition::Equal, {28.0f});

        optimization.add_objective("strength", Sense::Maximize);
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("water/cement band, age 28", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float cement = results(i, Cement);
            const float water = results(i, Water);

            EXPECT_GE(water, 0.30f*cement - slack(water)) << "row " << i;
            EXPECT_LE(water, 0.70f*cement + slack(water)) << "row " << i;

            EXPECT_NEAR(results(i, Age), 28.0f, slack(28.0f)) << "row " << i;
        }
    }
}


TEST(ConcreteScenario, MixMassIsClosed)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                    Condition::Equal,
                                    {mix_mass});

        optimization.add_objective("strength", Sense::Maximize);
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("closed mix mass", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float mass = results.row(i).segment(Cement, FineAgg - Cement + 1).sum();

            EXPECT_NEAR(mass, mix_mass, slack(mix_mass)) << "row " << i;
        }
    }
}


TEST(ConcreteScenario, StrengthAgainstCementFront)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
        optimization.add_objective("cement", Sense::Minimize);
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("strength against cement", driver, set_problem);

        // Two objectives that genuinely conflict should return a front, not a point, and
        // it should spread over cement rather than pile up at one mix.

        EXPECT_GT(results.rows(), 1);

        EXPECT_GT(results.col(Cement).maxCoeff() - results.col(Cement).minCoeff(), 1.0f)
            << "the front collapsed onto a single cement content";
    }
}


TEST(ConcreteScenario, ConstrainedStrengthAgainstCementFront)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
        optimization.add_objective("cement", Sense::Minimize);

        optimization.add_constraint("age", Condition::Equal, {28.0f});
        optimization.add_constraint("water - 0.70 * cement", Condition::LessEqual, {0.0f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("constrained front", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            EXPECT_NEAR(results(i, Age), 28.0f, slack(28.0f)) << "row " << i;

            EXPECT_LE(results(i, Water), 0.70f*results(i, Cement) + slack(results(i, Water)))
                << "row " << i;
        }
    }
}


TEST(ConcreteScenario, FixedStrengthTargetIsReached)
{
    constexpr float target = 50.0f;

    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Fixed, target);
        optimization.add_constraint("cement", Condition::Between, {150.0f, 350.0f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("fixed strength 50", driver, set_problem);

        EXPECT_NEAR(results(0, Strength), target, 2.0f)
            << "the search did not settle on a reachable target";

        for (Index i = 0; i < results.rows(); i++)
        {
            EXPECT_GE(results(i, Cement), 150.0f - slack(150.0f)) << "row " << i;
            EXPECT_LE(results(i, Cement), 350.0f + slack(350.0f)) << "row " << i;
        }
    }
}


TEST(ConcreteScenario, TightMultiobjectiveMixStaysFeasible)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
        optimization.add_objective("cement", Sense::Minimize);

        optimization.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                    Condition::Equal,
                                    {mix_mass});

        optimization.add_constraint("age", Condition::Equal, {28.0f});
        optimization.add_constraint("water", Condition::Between, {175.0f, 185.0f});
        optimization.add_constraint("sp", Condition::Between, {4.0f, 8.0f});
        optimization.add_constraint("water - 0.40 * cement", Condition::GreaterEqual, {0.0f});
        optimization.add_constraint("water - 0.55 * cement", Condition::LessEqual, {0.0f});
        optimization.add_constraint("strength", Condition::GreaterEqual, {40.0f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("tight multiobjective", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float cement = results(i, Cement);
            const float water = results(i, Water);

            const float mass = results.row(i).segment(Cement, FineAgg - Cement + 1).sum();

            EXPECT_NEAR(mass, mix_mass, slack(mix_mass)) << "row " << i;
            EXPECT_NEAR(results(i, Age), 28.0f, slack(28.0f)) << "row " << i;

            EXPECT_GE(water, 175.0f - slack(175.0f)) << "row " << i;
            EXPECT_LE(water, 185.0f + slack(185.0f)) << "row " << i;

            EXPECT_GE(results(i, Sp), 4.0f - slack(4.0f)) << "row " << i;
            EXPECT_LE(results(i, Sp), 8.0f + slack(8.0f)) << "row " << i;

            EXPECT_GE(water, 0.40f*cement - slack(water)) << "row " << i;
            EXPECT_LE(water, 0.55f*cement + slack(water)) << "row " << i;

            EXPECT_GE(results(i, Strength), 40.0f - slack(40.0f)) << "row " << i;
        }
    }
}


// The scenario that motivated the unified constraint solve: the constrained quantities
// are ratios of the response to the inputs, so they cannot be repaired by looking at the
// input alone.

TEST(ConcreteScenario, NonlinearOutputConstraintsHold)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);

        // Strength per kilogram of binder.

        optimization.add_constraint("strength / (cement + slag + fly_ash)",
                                    Condition::GreaterEqual,
                                    {0.10f});

        // Strength per unit of mix cost, with indicative prices per kilogram.

        optimization.add_constraint("strength / (0.10 * cement + 0.05 * slag + 0.04 * fly_ash"
                                    " + 1.20 * sp + 0.02 * coarse_agg + 0.02 * fine_agg)",
                                    Condition::GreaterEqual,
                                    {0.55f});

        // The water to cement ratio is nonlinear too, but it only reads inputs.

        optimization.add_constraint("water / cement", Condition::Between, {0.35f, 0.60f});

        optimization.add_constraint("age", Condition::Equal, {28.0f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("nonlinear output constraints", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float strength = results(i, Strength);

            const float binder = results(i, Cement) + results(i, Slag) + results(i, FlyAsh);

            const float cost = 0.10f*results(i, Cement) + 0.05f*results(i, Slag)
                             + 0.04f*results(i, FlyAsh) + 1.20f*results(i, Sp)
                             + 0.02f*results(i, CoarseAgg) + 0.02f*results(i, FineAgg);

            const float water_cement = results(i, Water)/results(i, Cement);

            EXPECT_GE(strength/binder, 0.10f - slack(0.10f)) << "row " << i;
            EXPECT_GE(strength/cost, 0.55f - slack(0.55f)) << "row " << i;

            EXPECT_GE(water_cement, 0.35f - slack(0.35f)) << "row " << i;
            EXPECT_LE(water_cement, 0.60f + slack(0.60f)) << "row " << i;

            EXPECT_NEAR(results(i, Age), 28.0f, slack(28.0f)) << "row " << i;
        }
    }
}


// A mix design as it is actually written down: the batch has to weigh what a cubic metre
// weighs, and the quantities that a specifier reads are ratios rather than masses. Four
// of the five constraints are nonlinear, and the last of them reads the response, so the
// only way to satisfy it is to move the mix and see what comes back.

TEST(ConcreteScenario, MixDesignRatiosOnAClosedBatch)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);

        // The batch closes on the mass of one cubic metre: a simplex over the seven
        // ingredients, so nothing can be added without taking something else out.

        optimization.add_constraint("cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
                                    Condition::Equal,
                                    {mix_mass});

        // Water to binder ratio, the ratio that governs strength and durability.

        optimization.add_constraint("water / (cement + slag + fly_ash)",
                                    Condition::Between,
                                    {0.35f, 0.50f});

        // How much of the binder is slag and fly ash rather than clinker.

        optimization.add_constraint("(slag + fly_ash) / (cement + slag + fly_ash)",
                                    Condition::Between,
                                    {0.20f, 0.50f});

        // Sand ratio: the fine share of the total aggregate, which sets workability.

        optimization.add_constraint("fine_agg / (coarse_agg + fine_agg)",
                                    Condition::Between,
                                    {0.35f, 0.45f});

        // Binder efficiency, in strength per kilogram of binder. This one reads the
        // response, so no amount of looking at the mix alone can tell whether it holds.

        optimization.add_constraint("strength / (cement + slag + fly_ash)",
                                    Condition::GreaterEqual,
                                    {0.10f});

        optimization.add_constraint("age", Condition::Equal, {28.0f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR results = run("mix design ratios", driver, set_problem);

        // The ratios are what the constraints are written in, and none of them is a
        // column, so print them next to their bounds. Reading which ones come back sitting
        // on a bound is reading which ones actually shaped the mix.

        cout << left << setw(14) << "ratio" << right
             << setw(10) << "value" << setw(10) << "lower" << setw(10) << "upper" << "\n";

        for (Index i = 0; i < results.rows(); i++)
        {
            const float binder = results(i, Cement) + results(i, Slag) + results(i, FlyAsh);

            const float aggregate = results(i, CoarseAgg) + results(i, FineAgg);

            const float mass = results.row(i).segment(Cement, FineAgg - Cement + 1).sum();

            const float water_binder = results(i, Water)/binder;
            const float replacement = (results(i, Slag) + results(i, FlyAsh))/binder;
            const float sand_ratio = results(i, FineAgg)/aggregate;
            const float efficiency = results(i, Strength)/binder;

            if (i == 0)
                cout << fixed << setprecision(4)
                     << left << setw(14) << "water/binder" << right
                     << setw(10) << water_binder << setw(10) << 0.35f << setw(10) << 0.50f << "\n"
                     << left << setw(14) << "scm share" << right
                     << setw(10) << replacement << setw(10) << 0.20f << setw(10) << 0.50f << "\n"
                     << left << setw(14) << "sand ratio" << right
                     << setw(10) << sand_ratio << setw(10) << 0.35f << setw(10) << 0.45f << "\n"
                     << left << setw(14) << "MPa per kg" << right
                     << setw(10) << efficiency << setw(10) << 0.10f << setw(10) << 0.0f << "\n";

            EXPECT_NEAR(mass, mix_mass, slack(mix_mass)) << "row " << i;

            EXPECT_GE(water_binder, 0.35f - slack(0.35f)) << "row " << i;
            EXPECT_LE(water_binder, 0.50f + slack(0.50f)) << "row " << i;

            EXPECT_GE(replacement, 0.20f - slack(0.20f)) << "row " << i;
            EXPECT_LE(replacement, 0.50f + slack(0.50f)) << "row " << i;

            EXPECT_GE(sand_ratio, 0.35f - slack(0.35f)) << "row " << i;
            EXPECT_LE(sand_ratio, 0.45f + slack(0.45f)) << "row " << i;

            EXPECT_GE(efficiency, 0.10f - slack(0.10f)) << "row " << i;

            EXPECT_NEAR(results(i, Age), 28.0f, slack(28.0f)) << "row " << i;
        }
    }
}


// The scenarios below are not from the example. They ask what the results look like when
// the problem is varied, rather than whether one problem is solved.


TEST(ConcreteScenario, BothDriversAgreeOnTheBestStrength)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
        optimization.add_constraint("age", Condition::Equal, {28.0f});
    };

    const float contraction_best = best_strength(run("agreement", Driver::Contraction, set_problem));
    const float genetic_best = best_strength(run("agreement", Driver::Genetic, set_problem));

    cout << "\ncontraction " << contraction_best << " vs genetic " << genetic_best
         << " (" << 100.0f*abs(contraction_best - genetic_best)/max(contraction_best, genetic_best)
         << "% apart)\n";

    // The two searches share nothing but the problem, and on this surface they land
    // within a few hundredths of a percent of each other. The bound is set far wider than
    // that, at 5%, so it reports a driver that stopped early rather than ordinary drift.

    EXPECT_NEAR(contraction_best, genetic_best, 0.05f*max(contraction_best, genetic_best));
}


TEST(ConcreteScenario, TighteningTheOutputConstraintKeepsResultsFeasible)
{
    // Raising the floor shrinks the feasible set. Every level must still be honoured, and
    // the best strength found must not fall below the floor that was asked for.

    for (const float floor_strength : {40.0f, 50.0f, 60.0f})
    {
        const auto set_problem = [floor_strength](ResponseOptimization& optimization)
        {
            optimization.add_objective("cement", Sense::Minimize);
            optimization.add_constraint("strength", Condition::GreaterEqual, {floor_strength});
        };

        const MatrixR results =
            run("strength floor " + to_string(int(floor_strength)), Driver::Genetic, set_problem);

        for (Index i = 0; i < results.rows(); i++)
            EXPECT_GE(results(i, Strength), floor_strength - slack(floor_strength))
                << "row " << i << " at floor " << floor_strength;
    }
}


TEST(ConcreteScenario, ImpossibleStrengthIsReportedNotReturned)
{
    // The network was trained on strengths up to about 82 MPa. Asking for 150 cannot be
    // met, and the run has to say so rather than hand back an infeasible mix.

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        set_seed(1234);

        const unique_ptr<ResponseOptimization> optimization = make_driver(driver);

        optimization->add_objective("cement", Sense::Minimize);
        optimization->add_constraint("strength", Condition::GreaterEqual, {150.0f});

        EXPECT_THROW(optimization->perform_response_optimization(), runtime_error)
            << driver_name(driver) << " returned a result for an unreachable strength";
    }
}


TEST(ConcreteScenario, TheSameSeedGivesTheSameResult)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
        optimization.add_constraint("water / cement", Condition::Between, {0.35f, 0.60f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        const MatrixR first = run("repeatability", driver, set_problem);
        const MatrixR second = run("repeatability", driver, set_problem);

        ASSERT_EQ(first.rows(), second.rows()) << driver_name(driver);

        EXPECT_LE((first - second).cwiseAbs().maxCoeff(), 1e-3f)
            << driver_name(driver) << " is not repeatable under a fixed seed";
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

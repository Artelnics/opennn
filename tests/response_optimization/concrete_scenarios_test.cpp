//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   R E S P O N S E   S C E N A R I O S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Optimizer and repair checks against the trained UCI concrete network.
// Constraint assertions use independent arithmetic on the returned columns.

#include "tests/pch.h"

#include <filesystem>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/registry.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"

using namespace opennn;

namespace
{

using Sense = ResponseOptimization::Objective::Sense;
using Condition = ResponseOptimization::Constraint::Condition;

// Result columns: the eight mix variables the network takes, then the response it gives.

enum Column { Cement, Slag, FlyAsh, Water, Sp, CoarseAgg, FineAgg, Age, Strength, ColumnsNumber };

const char* const column_names[ColumnsNumber] =
    {"cement", "slag", "fly_ash", "water", "sp", "coarse_agg", "fine_agg", "age", "strength"};

// The mass of one cubic metre of mix, used by every case that closes the batch.

constexpr float mix_mass = 2325.012558f;

// Constraints are met to a relative slack inside the optimizer, and a repaired point is
// placed a little inside the bound rather than on it. These checks only have to catch a
// point that is actually outside, so they read the bound with a wider tolerance.

float slack(const float bound) { return max(1e-2f, abs(bound)*1e-3f); }


// One shared network for every test in the binary. Nothing writes to it.

Network& concrete_network()
{
    static Network network = []
    {
        // These scenarios exercise the optimizers with a small 8-52-1 model.
        // Pin it to CPU so their runtime does not depend on GPU launch overhead.
        Configuration::instance().set(Device::CPU, Type::FP32);
        return Network(std::filesystem::path(CONCRETE_NETWORK_DIR) / "nn" / "concrete_uci.json");
    }();

    return network;
}


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
    SCOPED_TRACE(scenario);
    SCOPED_TRACE(driver_name(driver));
    set_seed(1234);

    const unique_ptr<ResponseOptimization> optimization = make_driver(driver);

    set_problem(*optimization);

    const MatrixR results = optimization->perform_response_optimization();

    expect_shape(results);
    expect_inside_the_box(results);
    expect_response_matches_the_mix(results);

    return results;
}


float best_strength(const MatrixR& results) { return results.col(Strength).maxCoeff(); }


constexpr Index draws_number = 200;


struct ConstraintSpec
{
    string expression;

    Condition condition = Condition::Equal;

    vector<float> values;
};


struct FeasibilityCase
{
    string name;

    vector<ConstraintSpec> constraints;

    // The fewest variables the repaired set is expected to move in. A case whose
    // constraints pin most of the mix cannot spread over as many as an open one.

    Index least_variables_moved = 0;
};


// Cases progress from an open input box to a constrained mix design.

vector<FeasibilityCase> feasibility_cases()
{
    return
    {
        {
            "open box",
            {},
            0
        },
        {
            "cement band",
            {{"cement", Condition::Between, {200.0f, 400.0f}}},
            6
        },
        {
            "standard test ages",
            {{"age", Condition::AllowedSet, {7.0f, 28.0f, 90.0f}}},
            6
        },
        {
            "closed batch",
            {{"cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
              Condition::Equal, {mix_mass}}},
            6
        },
        {
            "two ratio bands",
            {{"water / (cement + slag + fly_ash)", Condition::Between, {0.35f, 0.50f}},
             {"fine_agg / (coarse_agg + fine_agg)", Condition::Between, {0.35f, 0.45f}}},
            4
        },
        {
            "water binder floor",
            {{"water / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.45f}}},
            4
        },
        {
            "strength floor",
            {{"strength", Condition::GreaterEqual, {50.0f}}},
            4
        },
        {
            "binder efficiency",
            {{"strength / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.10f}}},
            4
        },
        {
            "durability class",
            {{"water / (cement + slag + fly_ash)", Condition::LessEqual, {0.45f}},
             {"cement + slag + fly_ash", Condition::GreaterEqual, {320.0f}},
             {"strength", Condition::GreaterEqual, {40.0f}}},
            4
        },
        {
            "low carbon binder",
            {{"(slag + fly_ash) / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.50f}},
             {"strength", Condition::GreaterEqual, {30.0f}},
             {"age", Condition::Equal, {28.0f}}},
            4
        },
        {
            "cost ceiling",
            {{"0.10 * cement + 0.05 * slag + 0.04 * fly_ash + 1.20 * sp"
              " + 0.02 * coarse_agg + 0.02 * fine_agg", Condition::LessEqual, {80.0f}},
             {"strength", Condition::GreaterEqual, {40.0f}}},
            4
        },
        {
            "mix design",
            {{"cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
              Condition::Equal, {mix_mass}},
             {"water / (cement + slag + fly_ash)", Condition::Between, {0.35f, 0.50f}},
             {"(slag + fly_ash) / (cement + slag + fly_ash)", Condition::Between, {0.20f, 0.50f}},
             {"fine_agg / (coarse_agg + fine_agg)", Condition::Between, {0.35f, 0.45f}},
             {"strength / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.10f}},
             {"age", Condition::Equal, {28.0f}}},
            4
        }
    };
}


// Reaches the repair without a solver around it.

class FeasibleSetProbe : public ResponseOptimization
{
public:

    explicit FeasibleSetProbe(const FeasibilityCase& feasibility_case)
        : ResponseOptimization(&concrete_network())
    {
        for (const ConstraintSpec& constraint : feasibility_case.constraints)
            add_constraint(constraint.expression, constraint.condition, constraint.values);
    }

    using ResponseOptimization::calculate_domain;
    using ResponseOptimization::calculate_random_input;
    using ResponseOptimization::solve;

private:

    MatrixR single_optimization() override { return {}; }
    MatrixR multi_optimization() override { return {}; }
};


// What a run of repairs produced. Starting points that were already feasible are counted
// apart from the rest: they come back untouched, so they say nothing about the repair and
// would flatter any measure of spread they were mixed into.

struct RepairedCloud
{
    MatrixR points;
    Index already_feasible = 0;

    Index repaired() const { return points.rows(); }
};


RepairedCloud repair_from_random_starts(FeasibleSetProbe& probe,
                                        const pair<VectorR, VectorR>& domain,
                                        const Index draws)
{
    RepairedCloud cloud;

    cloud.points = MatrixR(draws, domain.first.size());

    Index kept = 0;

    for (Index i = 0; i < draws; i++)
    {
        const VectorR start = probe.calculate_random_input(domain);

        const VectorR input = probe.solve(start).first;

        if (input.size() == 0) continue;

        // An untouched return is the early out: the draw already satisfied everything.

        if ((input - start).cwiseAbs().maxCoeff() <= 0.0f)
        {
            cloud.already_feasible++;
            continue;
        }

        cloud.points.row(kept) = input.transpose();

        kept++;
    }

    // Preserve retained rows before shrinking: assigning an unevaluated view
    // of the same matrix can read storage invalidated by the resize.
    cloud.points.conservativeResize(kept, Eigen::NoChange);

    return cloud;
}


// Two points count as one when every variable agrees to a thousandth of its own range,
// far below anything a search would treat as a different mix.

Index count_distinct(const MatrixR& points, const VectorR& span)
{
    const VectorR guarded_span = span.cwiseMax(EPSILON);

    Index distinct = 0;

    for (Index i = 0; i < points.rows(); i++)
    {
        bool seen = false;

        for (Index j = 0; j < i && !seen; j++)
            seen = ((points.row(i) - points.row(j)).cwiseAbs().array()
                    / guarded_span.transpose().array() < 1e-3f).all();

        if (!seen) distinct++;
    }

    return distinct;
}


// How many variables the set actually moves in. A repair that always walked to the same
// corner would score one, or none.

Index count_variables_moved(const MatrixR& points, const VectorR& span)
{
    Index variables_moved = 0;

    for (Index j = 0; j < points.cols(); j++)
    {
        if (span(j) <= EPSILON) continue;

        if ((points.col(j).maxCoeff() - points.col(j).minCoeff())/span(j) > 0.05f)
            variables_moved++;
    }

    return variables_moved;
}

}


TEST(ConcreteScenario, MaximizeStrengthWithoutConstraints)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.add_objective("strength", Sense::Maximize);
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        SCOPED_TRACE(driver_name(driver));
        const MatrixR results = run("unconstrained strength", driver, set_problem);

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
        SCOPED_TRACE(driver_name(driver));
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
        SCOPED_TRACE(driver_name(driver));
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
        SCOPED_TRACE(driver_name(driver));
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
        SCOPED_TRACE(driver_name(driver));
        const MatrixR results = run("constrained front", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            EXPECT_NEAR(results(i, Age), 28.0f, slack(28.0f)) << "row " << i;

            EXPECT_LE(results(i, Water), 0.70f*results(i, Cement) + slack(results(i, Water)))
                << "row " << i;
        }
    }
}


TEST(ConcreteScenario, ConstrainedFrontKeepsAllowedAges)
{
    const auto set_problem = [](ResponseOptimization& optimization)
    {
        optimization.set_points_number(32);
        optimization.set_iterations_number(3);
        optimization.add_objective("strength", Sense::Maximize);
        optimization.add_objective("cement", Sense::Minimize);
        optimization.add_constraint("age", Condition::AllowedSet, {7.0f, 28.0f, 90.0f});
        optimization.add_constraint("water - 0.70 * cement", Condition::LessEqual, {0.0f});
    };

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        SCOPED_TRACE(driver_name(driver));
        const MatrixR results = run("constrained front with allowed ages", driver, set_problem);

        EXPECT_GT(results.rows(), 1);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float age = results(i, Age);
            EXPECT_TRUE(age == 7.0f || age == 28.0f || age == 90.0f) << "row " << i;
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
        SCOPED_TRACE(driver_name(driver));
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
        SCOPED_TRACE(driver_name(driver));
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
        SCOPED_TRACE(driver_name(driver));
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
        SCOPED_TRACE(driver_name(driver));
        const MatrixR results = run("mix design ratios", driver, set_problem);

        for (Index i = 0; i < results.rows(); i++)
        {
            const float binder = results(i, Cement) + results(i, Slag) + results(i, FlyAsh);

            const float aggregate = results(i, CoarseAgg) + results(i, FineAgg);

            const float mass = results.row(i).segment(Cement, FineAgg - Cement + 1).sum();

            const float water_binder = results(i, Water)/binder;
            const float replacement = (results(i, Slag) + results(i, FlyAsh))/binder;
            const float sand_ratio = results(i, FineAgg)/aggregate;
            const float efficiency = results(i, Strength)/binder;

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
    // met, and the run has to say so rather than hand back an infeasible mix. An impossible
    // run spends its whole sampling budget before giving up, so the budget is kept small.

    for (const Driver driver : {Driver::Contraction, Driver::Genetic})
    {
        SCOPED_TRACE(driver_name(driver));
        set_seed(1234);

        const unique_ptr<ResponseOptimization> optimization = make_driver(driver);
        optimization->set_points_number(50);
        optimization->set_iterations_number(3);

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
        SCOPED_TRACE(driver_name(driver));
        const MatrixR first = run("repeatability", driver, set_problem);
        const MatrixR second = run("repeatability", driver, set_problem);

        ASSERT_EQ(first.rows(), second.rows()) << driver_name(driver);

        EXPECT_LE((first - second).cwiseAbs().maxCoeff(), 1e-3f)
            << driver_name(driver) << " is not repeatable under a fixed seed";
    }
}


class FeasibilityStudy : public testing::TestWithParam<FeasibilityCase> {};


// The one thing every case has to show: different starting points must give different
// feasible points. If they do not, the repair has thrown away the spread the solvers
// depend on, whatever else it got right.

TEST_P(FeasibilityStudy, DifferentStartsGiveDifferentPoints)
{
    set_seed(1234);

    const FeasibilityCase feasibility_case = GetParam();

    FeasibleSetProbe probe(feasibility_case);

    const pair<VectorR, VectorR> domain = probe.calculate_domain();

    const VectorR span = domain.second - domain.first;

    const RepairedCloud cloud = repair_from_random_starts(probe, domain, draws_number);

    // A case has to leave something behind, whether the repair placed it or the draw
    // already satisfied everything.

    ASSERT_GT(cloud.already_feasible + cloud.repaired(), 0)
        << "no starting point survived, so the case says nothing about the repair";

    if (cloud.repaired() == 0) return;

    EXPECT_EQ(count_distinct(cloud.points, span), cloud.repaired())
        << "different starting points collapsed onto shared repaired points";

    EXPECT_GE(count_variables_moved(cloud.points, span), feasibility_case.least_variables_moved)
        << "the repaired set moves in too few variables to be a set rather than a point";
}


INSTANTIATE_TEST_SUITE_P(
    Cases,
    FeasibilityStudy,
    testing::ValuesIn(feasibility_cases()),
    [](const testing::TestParamInfo<FeasibilityCase>& info)
    {
        string name = info.param.name;

        for (char& character : name)
            if (character == ' ') character = '_';

        return name;
    });


// One bound, crossed by every point that needed repair, so where they land is visible.
// Solving the violation to zero would leave them all on the surface; the repair aims past
// it by a share of the violation instead.

TEST(Feasibility, RepairedPointsDoNotPileOntoTheBound)
{
    set_seed(1234);

    const FeasibilityCase feasibility_case =
    {
        "water binder floor",
        {{"water / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.45f}}},
        4
    };

    FeasibleSetProbe probe(feasibility_case);

    const pair<VectorR, VectorR> domain = probe.calculate_domain();

    const RepairedCloud cloud = repair_from_random_starts(probe, domain, draws_number);

    ASSERT_GT(cloud.repaired(), 0);

    Index on_the_bound = 0;

    for (Index i = 0; i < cloud.repaired(); i++)
    {
        const float binder = cloud.points(i, Cement) + cloud.points(i, Slag) + cloud.points(i, FlyAsh);

        const float water_binder = cloud.points(i, Water)/binder;

        EXPECT_GE(water_binder, 0.45f - slack(0.45f)) << "row " << i;

        if (water_binder <= 0.45f + 1e-4f) on_the_bound++;
    }

    // Almost all of them land clear of the surface, so a majority is a wide bound. It
    // catches the repair reverting to solving the violation to exactly zero.

    EXPECT_LT(on_the_bound, cloud.repaired()/2)
        << "the repaired points piled onto the constraint surface";
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

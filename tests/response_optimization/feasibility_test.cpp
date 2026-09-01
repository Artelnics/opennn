//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F E A S I B I L I T Y   S T U D Y
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Studies the repair on its own, with no search around it.
//
// Both solvers are built on get_feasible_point: they draw a point, hand it to the repair,
// and work with whatever comes back. So the repair decides what they have to search. A
// repair that answered every starting point with the same mix, or that put every mix on a
// constraint surface, would satisfy every feasibility check made of it and still leave the
// solvers nothing to explore.
//
// Each case below states its constraints, runs many random starting points through the
// repair, and reports what came back: how many were already feasible and passed through
// untouched, how many the repair placed, how many of those are distinct, which variables
// the resulting set moves in, and how close to their bounds the points sit.
//
// To add a case, add an entry to feasibility_cases(). Nothing else needs to change.

#include "tests/pch.h"

#include <iomanip>

#include "tests/response_optimization/concrete_fixture.h"

#include "opennn/core/random_utilities.h"

namespace
{

// How many starting points each case draws. Enough for the counts below to mean
// something, and small enough that the whole study runs in a second.

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

    // What the case is here to find out, in one line.

    string intent;

    vector<ConstraintSpec> constraints;

    // The fewest variables the repaired set is expected to move in. A case whose
    // constraints pin most of the mix cannot spread over as many as an open one.

    Index least_variables_moved = 0;
};


// Every case is a mix a concrete engineer might actually ask for. They are ordered from
// the least constrained to the most, so reading the reports in order shows the feasible
// set closing in.

vector<FeasibilityCase> feasibility_cases()
{
    return
    {
        {
            "open box",
            "with nothing to satisfy, every draw should pass straight through the repair",
            {},
            0
        },
        {
            "cement band",
            "a single linear band on one variable, folded into the box before the repair sees it",
            {{"cement", Condition::Between, {200.0f, 400.0f}}},
            6
        },
        {
            "standard test ages",
            "an allowed set, where the repair has to land on one of three exact values",
            {{"age", Condition::AllowedSet, {7.0f, 28.0f, 90.0f}}},
            6
        },
        {
            "closed batch",
            "one equality over seven variables: a simplex the repair has to stay on",
            {{"cement + slag + fly_ash + water + sp + coarse_agg + fine_agg",
              Condition::Equal, {mix_mass}}},
            6
        },
        {
            "two ratio bands",
            "two nonlinear bands on ratios of the mix, with no response involved",
            {{"water / (cement + slag + fly_ash)", Condition::Between, {0.35f, 0.50f}},
             {"fine_agg / (coarse_agg + fine_agg)", Condition::Between, {0.35f, 0.45f}}},
            4
        },
        {
            "water binder floor",
            "one lower bound every repaired point has to cross, so it shows where they land",
            {{"water / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.45f}}},
            4
        },
        {
            "strength floor",
            "a bound on the response alone, which no amount of reading the mix can check",
            {{"strength", Condition::GreaterEqual, {50.0f}}},
            4
        },
        {
            "binder efficiency",
            "a ratio of the response to the mix that produced it, the case the solve exists for",
            {{"strength / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.10f}}},
            4
        },
        {
            "durability class",
            "an exposure class as a specifier writes it: capped water, minimum binder, minimum strength",
            {{"water / (cement + slag + fly_ash)", Condition::LessEqual, {0.45f}},
             {"cement + slag + fly_ash", Condition::GreaterEqual, {320.0f}},
             {"strength", Condition::GreaterEqual, {40.0f}}},
            4
        },
        {
            "low carbon binder",
            "half the binder replaced by slag and fly ash, still asked to reach a working strength",
            {{"(slag + fly_ash) / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.50f}},
             {"strength", Condition::GreaterEqual, {30.0f}},
             {"age", Condition::Equal, {28.0f}}},
            4
        },
        {
            "cost ceiling",
            "the usual trade: a strength to reach and a budget per cubic metre to reach it in",
            {{"0.10 * cement + 0.05 * slag + 0.04 * fly_ash + 1.20 * sp"
              " + 0.02 * coarse_agg + 0.02 * fine_agg", Condition::LessEqual, {80.0f}},
             {"strength", Condition::GreaterEqual, {40.0f}}},
            4
        },
        {
            "mix design",
            "a closed batch, three ratio bands, a response earning its binder, and a fixed age",
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
    using ResponseOptimization::get_feasible_point;

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
    MatrixR responses;

    Index drawn = 0;
    Index already_feasible = 0;
    Index failed = 0;

    Index repaired() const { return points.rows(); }
};


RepairedCloud repair_from_random_starts(FeasibleSetProbe& probe,
                                        const pair<VectorR, VectorR>& domain,
                                        const Index draws)
{
    RepairedCloud cloud;

    cloud.drawn = draws;
    cloud.points = MatrixR(draws, domain.first.size());
    cloud.responses = MatrixR(draws, concrete_network().get_outputs_number());

    Index kept = 0;

    for (Index i = 0; i < draws; i++)
    {
        const VectorR start = probe.calculate_random_input(domain);

        const auto [input, output] = probe.get_feasible_point(start, domain);

        if (input.size() == 0)
        {
            cloud.failed++;
            continue;
        }

        // An untouched return is the early out: the draw already satisfied everything.

        if ((input - start).cwiseAbs().maxCoeff() <= 0.0f)
        {
            cloud.already_feasible++;
            continue;
        }

        cloud.points.row(kept) = input.transpose();
        cloud.responses.row(kept) = output.transpose();

        kept++;
    }

    cloud.points = cloud.points.topRows(kept);
    cloud.responses = cloud.responses.topRows(kept);

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


Index count_free_variables(const VectorR& span)
{
    Index free_variables = 0;

    for (Index j = 0; j < span.size(); j++)
        if (span(j) > EPSILON) free_variables++;

    return free_variables;
}


void report_case(const FeasibilityCase& feasibility_case, const pair<VectorR, VectorR>& domain)
{
    cout << "\n================================================================\n"
         << feasibility_case.name << "\n"
         << "  " << feasibility_case.intent << "\n";

    cout << "\nconstraints\n";

    if (feasibility_case.constraints.empty())
        cout << "  (none)\n";

    for (const ConstraintSpec& constraint : feasibility_case.constraints)
    {
        cout << "  " << constraint.expression << " " << condition_name(constraint.condition);

        for (const float value : constraint.values)
            cout << " " << value;

        cout << "\n";
    }

    // What calculate_domain made of them. A single variable linear constraint is folded
    // into the box here and never reaches the repair, which is why some cases show a
    // narrower range than the network was trained on, or none at all.

    cout << "\ndomain the repair works in\n"
         << left << setw(14) << "  column" << right
         << setw(12) << "lower" << setw(12) << "upper" << "\n";

    for (Index j = 0; j <= Age; j++)
        cout << left << setw(14) << string("  ") + column_names[j] << right
             << fixed << setprecision(3)
             << setw(12) << domain.first(j) << setw(12) << domain.second(j)
             << (domain.second(j) - domain.first(j) <= EPSILON ? "   (pinned)" : "") << "\n";
}


void report_cloud(const RepairedCloud& cloud, const VectorR& span)
{
    cout << "\nstarting points\n"
         << "  drawn            " << cloud.drawn << "\n"
         << "  already feasible " << cloud.already_feasible << "\n"
         << "  repaired         " << cloud.repaired() << "\n"
         << "  failed           " << cloud.failed << "\n";

    if (cloud.repaired() == 0)
    {
        cout << "\n  (the repair placed nothing, so there is no set to describe)\n";

        return;
    }

    cout << "\nthe set the repair returned\n"
         << "  distinct         " << count_distinct(cloud.points, span)
         << " of " << cloud.repaired() << "\n"
         << "  variables moved  " << count_variables_moved(cloud.points, span)
         << " of " << count_free_variables(span) << " free\n";

    cout << left << setw(14) << "  column" << right
         << setw(12) << "min" << setw(12) << "max" << setw(14) << "spread/range" << "\n";

    for (Index j = 0; j <= Age; j++)
    {
        if (span(j) <= EPSILON) continue;

        const float smallest = cloud.points.col(j).minCoeff();
        const float largest = cloud.points.col(j).maxCoeff();

        cout << left << setw(14) << string("  ") + column_names[j] << right
             << fixed << setprecision(3)
             << setw(12) << smallest << setw(12) << largest
             << setw(14) << (largest - smallest)/span(j) << "\n";
    }

    cout << left << setw(14) << "  strength" << right
         << setw(12) << cloud.responses.col(0).minCoeff()
         << setw(12) << cloud.responses.col(0).maxCoeff() << "\n";
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

    report_case(feasibility_case, domain);

    const RepairedCloud cloud = repair_from_random_starts(probe, domain, draws_number);

    report_cloud(cloud, span);

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
        "where points land once they have crossed a bound",
        {{"water / (cement + slag + fly_ash)", Condition::GreaterEqual, {0.45f}}},
        4
    };

    FeasibleSetProbe probe(feasibility_case);

    const pair<VectorR, VectorR> domain = probe.calculate_domain();

    const RepairedCloud cloud = repair_from_random_starts(probe, domain, draws_number);

    ASSERT_GT(cloud.repaired(), 0);

    Index on_the_bound = 0;

    float deepest = 0.0f;

    for (Index i = 0; i < cloud.repaired(); i++)
    {
        const float binder = cloud.points(i, Cement) + cloud.points(i, Slag) + cloud.points(i, FlyAsh);

        const float water_binder = cloud.points(i, Water)/binder;

        EXPECT_GE(water_binder, 0.45f - bound_slack(0.45f)) << "row " << i;

        if (water_binder <= 0.45f + 1e-4f) on_the_bound++;

        deepest = max(deepest, water_binder - 0.45f);
    }

    cout << "\nwhere the repaired points sit against the bound they crossed\n"
         << "  on the bound     " << on_the_bound << " of " << cloud.repaired() << "\n"
         << "  deepest inside   " << deepest << "\n";

    // Almost all of them land clear of the surface, so a majority is a wide bound. It
    // catches the repair reverting to solving the violation to exactly zero.

    EXPECT_LT(on_the_bound, cloud.repaired()/2)
        << "the repaired points piled onto the constraint surface";
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

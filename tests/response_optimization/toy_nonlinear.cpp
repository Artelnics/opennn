//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O Y   N O N L I N E A R   S Y S T E M   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tests/pch.h"

#include <cmath>

#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/NumericalDiff>

#include "opennn/core/string_utilities.h"
#include "opennn/response_optimization/response_optimization.h"

using namespace opennn;

namespace
{

enum Mix { Cement, Water, BinderA, BinderB, BinderC, VariablesNumber };

const vector<pair<string, Index>> mix_columns = {{"cement", Cement}, {"water", Water},
                                                 {"binder_a", BinderA},
                                                 {"binder_b", BinderB},
                                                 {"binder_c", BinderC}};

const vector<pair<string, Index>> response_columns = {{"strength", 0}};

constexpr float tolerance = 1e-3f;

constexpr float margin_factor = 0.1f;

constexpr float binder_span = 60.0f;


VectorR fake_output(const VectorR& point)
{
    VectorR strength(1);

    strength(0) = 100.0f/pow(4.0f, point(Water)/point(Cement));

    return strength;
}


vector<Index> get_group_members(const string& text)
{
    vector<Index> members;

    for (const string_view entry : get_token_views(text, ';'))
    {
        const CompiledExpression member =
            compile_expression(string(trim_view(entry)), mix_columns, response_columns);

        members.push_back(member.linear_input_terms.front().first);
    }

    return members;
}


VectorR seed_switches(const VectorR& point, const vector<Index>& members, const Index budget)
{
    VectorR unknowns = VectorR::Zero(VariablesNumber + Index(members.size()));

    unknowns.head(VariablesNumber) = point;

    vector<Index> positions(members.size());

    iota(positions.begin(), positions.end(), Index(0));

    ranges::sort(positions, {}, [&](const Index position) { return -abs(point(members[size_t(position)])); });

    for (Index i = 0; i < budget; i++)
        unknowns(VariablesNumber + positions[size_t(i)]) = 1.0f;

    return unknowns;
}


struct ToySystem : Eigen::DenseFunctor<float>
{
    ToySystem(vector<CompiledExpression> new_equations,
              vector<pair<float, float>> new_bands,
              const Index unknowns_number)
        : Eigen::DenseFunctor<float>(int(unknowns_number), int(new_equations.size())),
          equations(move(new_equations)),
          bands(move(new_bands)) {}

    int operator()(const VectorR& point, VectorR& residuals) const
    {
        const VectorR strength = fake_output(point);

        residuals.resize(Index(equations.size()));

        for (Index i = 0; i < Index(equations.size()); i++)
        {
            const auto [lower, upper] = bands[size_t(i)];

            const float value = equations[size_t(i)].evaluate(point, strength);

            const float residual = (value < lower) ? value - lower
                                 : (value > upper) ? value - upper
                                                   : 0.0f;

            const float inset = min(margin_factor*max(abs(residual),
                                                      margin_factor*((residual < 0.0f) ? abs(lower) : abs(upper))),
                                    0.5f*(upper - lower));

            residuals(i) = residual + ((residual > 0.0f) ? inset : (residual < 0.0f) ? -inset : 0.0f);
        }

        return 0;
    }

    vector<CompiledExpression> equations;

    vector<pair<float, float>> bands;
};

}


TEST(ToyNonlinearSystem, LevenbergMarquardtReachesTheRoot)
{
    const float unbounded = numeric_limits<float>::infinity();

    const vector<Index> members = get_group_members("binder_a; binder_b; binder_c");

    constexpr Index budget = 2;

    vector<CompiledExpression> equations;
    vector<pair<float, float>> bands;

    const auto add = [&](CompiledExpression equation, const float lower, const float upper)
    {
        equations.push_back(move(equation));
        bands.emplace_back(lower, upper);
    };

    add(compile_expression("cement + water", mix_columns, response_columns), 400.0f, 400.0f);
    add(compile_expression("water / cement", mix_columns, response_columns), 0.35f, 0.60f);
    add(compile_expression("strength - 0.20 * cement", mix_columns, response_columns), 0.0f, unbounded);
    add(compile_expression("binder_a + binder_b + binder_c", mix_columns, response_columns), 60.0f, 60.0f);

    vector<Index> switches;

    for (const Index member : members)
    {
        const Index switch_column = VariablesNumber + Index(switches.size());

        switches.push_back(switch_column);

        add(compile_coupling(member, switch_column, binder_span), -tolerance, tolerance);
        add(compile_binarity(switch_column), -0.5f*tolerance, 0.5f*tolerance);
    }

    add(compile_sum(switches), float(budget), float(budget));

    VectorR point(VariablesNumber);
    point << 200.0f, 260.0f, 30.0f, 25.0f, 15.0f;

    VectorR unknowns = seed_switches(point, members, budget);

    ToySystem system(move(equations), move(bands), unknowns.size());

    Eigen::NumericalDiff<ToySystem, Eigen::Central> numerical_diff(system);

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ToySystem, Eigen::Central>>
        levenberg_marquardt(numerical_diff);

    VectorR start_residuals;

    system(unknowns, start_residuals);

    ASSERT_NE(levenberg_marquardt.minimize(unknowns),
              Eigen::LevenbergMarquardtSpace::ImproperInputParameters);

    VectorR residuals;

    system(unknowns, residuals);

    EXPECT_LT(residuals.norm(), 1e-3f);

    const float cement = unknowns(Cement);
    const float water = unknowns(Water);

    const float binder_a = unknowns(BinderA);
    const float binder_b = unknowns(BinderB);
    const float binder_c = unknowns(BinderC);

    const float switch_a = unknowns(VariablesNumber);
    const float switch_b = unknowns(VariablesNumber + 1);
    const float switch_c = unknowns(VariablesNumber + 2);

    const float water_cement = water/cement;
    const float strength = fake_output(unknowns)(0);

    const Index binders_used = Index(switch_a > 0.5f)
                             + Index(switch_b > 0.5f)
                             + Index(switch_c > 0.5f);

    EXPECT_NEAR(cement + water, 400.0f, 1e-2f);

    EXPECT_GE(water_cement, 0.35f);
    EXPECT_LE(water_cement, 0.60f);

    EXPECT_GE(strength, 0.20f*cement);

    EXPECT_NEAR(binder_a + binder_b + binder_c, 60.0f, 1e-2f);

    EXPECT_EQ(binders_used, 2);

    EXPECT_LE(min({binder_a, binder_b, binder_c}), 2.0f*tolerance*binder_span);

    cout << fixed << setprecision(4)
         << "\ncement        " << cement << "\n"
         << "water         " << water << "\n"
         << "binder_a      " << binder_a << "   switch " << switch_a << "\n"
         << "binder_b      " << binder_b << "   switch " << switch_b << "\n"
         << "binder_c      " << binder_c << "   switch " << switch_c << "\n"
         << "batch         " << cement + water << "   = 400\n"
         << "water/cement  " << water_cement << "   in [0.35, 0.60]\n"
         << "strength      " << strength << "   >= " << 0.20f*cement << "\n"
         << "binders       " << binder_a + binder_b + binder_c << "   = 60, "
                             << binders_used << " of 3 in play\n"
         << "residual norm " << start_residuals.norm() << "  ->  " << residuals.norm() << "\n"
         << "iterations    " << levenberg_marquardt.iterations()
         << ",  evaluations " << levenberg_marquardt.nfev() << "\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

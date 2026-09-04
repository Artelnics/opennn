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

using Constraint = ResponseOptimization::Constraint;
using Condition = Constraint::Condition;

enum Mix { Cement, Water, BinderA, BinderB, BinderC, VariablesNumber };

const vector<pair<string, Index>> mix_columns = {{"cement", Cement}, {"water", Water},
                                                 {"binder_a", BinderA},
                                                 {"binder_b", BinderB},
                                                 {"binder_c", BinderC}};

const vector<pair<string, Index>> response_columns = {{"strength", 0}};

constexpr float tolerance = 1e-3f;

constexpr float margin_factor = 0.1f;

constexpr float binder_span = 60.0f;


VectorR fake_output(const VectorR& mix)
{
    VectorR strength(1);

    strength(0) = 100.0f/pow(4.0f, mix(Water)/mix(Cement));

    return strength;
}


CompiledExpression compile_group(const string& text)
{
    vector<Index> members;

    for (const string_view entry : get_token_views(text, ';'))
    {
        const CompiledExpression member =
            compile_expression(string(trim_view(entry)), mix_columns, response_columns);

        members.push_back(member.linear_input_terms.front().first);
    }

    CompiledExpression group = compile_sum(members);

    group.text = text;

    return group;
}


vector<Constraint> expand_cardinality(const Constraint& cardinality, const Index first_activation)
{
    vector<Constraint> expanded;

    vector<Index> activations;

    for (const auto& [member, coefficient] : cardinality.expression.linear_input_terms)
    {
        const Index activation = first_activation + Index(activations.size());

        activations.push_back(activation);

        expanded.push_back(Constraint{compile_coupling(member, activation, binder_span),
                                      Condition::Between, {-tolerance, tolerance}});

        expanded.push_back(Constraint{compile_binarity(activation),
                                      Condition::Between, {-0.5f*tolerance, 0.5f*tolerance}});
    }

    expanded.push_back(Constraint{compile_sum(activations),
                                  Condition::Equal, {cardinality.values[0]}});

    return expanded;
}


VectorR seed_activations(const VectorR& mix, const Constraint& cardinality)
{
    const auto& counted = cardinality.expression.linear_input_terms;

    VectorR point = VectorR::Zero(VariablesNumber + Index(counted.size()));

    point.head(VariablesNumber) = mix;

    vector<Index> positions(counted.size());

    iota(positions.begin(), positions.end(), Index(0));

    ranges::sort(positions, {},
                 [&](const Index position) { return -abs(mix(counted[size_t(position)].first)); });

    for (Index i = 0; i < Index(cardinality.values[0]); i++)
        point(VariablesNumber + positions[size_t(i)]) = 1.0f;

    return point;
}


struct ToySystem : Eigen::DenseFunctor<float>
{
    ToySystem(const vector<Constraint>& new_constraints, const Index unknowns_number)
        : Eigen::DenseFunctor<float>(int(unknowns_number), int(new_constraints.size())),
          constraints(new_constraints) {}

    int operator()(const VectorR& mix, VectorR& residuals) const
    {
        const VectorR strength = fake_output(mix);

        residuals.resize(Index(constraints.size()));

        for (Index i = 0; i < Index(constraints.size()); i++)
        {
            const Constraint& constraint = constraints[size_t(i)];

            const float value = constraint.expression.evaluate(mix, strength);

            const float residual = constraint.calculate_residual(value, tolerance, margin_factor);

            residuals(i) = isfinite(residual) ? residual : 0.0f;
        }

        return 0;
    }

    vector<Constraint> constraints;
};

}


TEST(ToyNonlinearSystem, LevenbergMarquardtReachesTheRoot)
{
    const Constraint expression_1{compile_expression("cement + water", mix_columns, response_columns),
                                  Condition::Equal, {400.0f}};

    const Constraint expression_2{compile_expression("water / cement", mix_columns, response_columns),
                                  Condition::Between, {0.35f, 0.60f}};

    const Constraint expression_3{compile_expression("strength - 0.20 * cement", mix_columns, response_columns),
                                  Condition::GreaterEqual, {0.0f}};

    const Constraint expression_4{compile_group("binder_a; binder_b; binder_c"),
                                  Condition::Cardinality, {2.0f}};

    const Constraint expression_5{compile_expression("binder_a + binder_b + binder_c", mix_columns, response_columns),
                                  Condition::Equal, {60.0f}};

    vector<Constraint> constraints_expressions =
        {expression_1, expression_2, expression_3, expression_5};

    for (const Constraint& expanded : expand_cardinality(expression_4, VariablesNumber))
        constraints_expressions.push_back(expanded);

    VectorR mix(VariablesNumber);
    mix << 200.0f, 260.0f, 30.0f, 25.0f, 15.0f;

    VectorR point = seed_activations(mix, expression_4);

    ToySystem system(constraints_expressions, point.size());

    Eigen::NumericalDiff<ToySystem, Eigen::Central> numerical_diff(system);

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ToySystem, Eigen::Central>>
        levenberg_marquardt(numerical_diff);

    VectorR start_residuals;

    system(point, start_residuals);

    ASSERT_NE(levenberg_marquardt.minimize(point),
              Eigen::LevenbergMarquardtSpace::ImproperInputParameters);

    VectorR residuals;

    system(point, residuals);

    EXPECT_LT(residuals.norm(), 1e-3f);

    const float cement = point(Cement);
    const float water = point(Water);

    const float binder_a = point(BinderA);
    const float binder_b = point(BinderB);
    const float binder_c = point(BinderC);

    const float switch_a = point(VariablesNumber);
    const float switch_b = point(VariablesNumber + 1);
    const float switch_c = point(VariablesNumber + 2);

    const float water_cement = water/cement;
    const float strength = fake_output(point)(0);

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

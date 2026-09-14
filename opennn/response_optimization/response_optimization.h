// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/pch.h"
#include "opennn/response_optimization/expression_evaluator.h"

namespace opennn
{

class Network;

class ResponseOptimization
{
public:

    void set(Network* = nullptr);

    struct Objective
    {
        enum class Sense { Minimize, Maximize, Fixed };

        CompiledExpression expression;

        Sense sense = Sense::Minimize;

        float value = 0.0f;
    };

    struct Constraint
    {
        enum class Condition
        {
            Equal, Between, GreaterEqual, LessEqual, Greater, Less, AllowedSet, Integer, Cardinality
        };

        CompiledExpression expression;

        Condition condition = Condition::Equal;

        vector<float> values;

        float calculate_residual(const VectorR&, const VectorR&, float margin = 0.0f) const;

        pair<float, float> calculate_bounds() const;
    };

    explicit ResponseOptimization(Network* = nullptr);

    virtual ~ResponseOptimization();

    void add_objective(const string&, Objective::Sense, float value = 0.0f);
    void add_constraint(const string&, Constraint::Condition, const vector<float>& values = {});

    MatrixR perform_response_optimization();

protected:

    Network* network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    virtual MatrixR single_optimization() = 0;

    virtual MatrixR multi_optimization() = 0;

    pair<VectorR, VectorR> calculate_domain() const;

    VectorR calculate_random_input(const pair<VectorR, VectorR>&) const;

    pair<VectorR, VectorR> get_feasible_point(VectorR, const pair<VectorR, VectorR>&) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    Index iterations_number = 20;
    Index points_number = 1000;

    Index requested_front_size = 100;

private:

    VectorR assign_categories(const VectorR&) const;

    MatrixR estimate_jacobian(const VectorR&, const VectorR&, const pair<VectorR, VectorR>&) const;

    static float bound_tolerance(float bound) { return max(EPSILON, abs(bound) * bound_tolerance_factor); }

    static constexpr float bound_tolerance_factor = 1e-4f;

    static constexpr float difference_step = 1e-3f;

    Index repair_passes = 8;

    float feasibility_margin = 0.1f;

    float diversity_factor = 0.2f;
};

}

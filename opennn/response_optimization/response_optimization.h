// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

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

        string string_expression;

        Condition condition = Condition::Equal;

        vector<float> values;

        vector<CompiledExpression> equations;

        vector<pair<float, float>> equation_limits;

        vector<pair<Index, Index>> involved_variables;

        void compile_equations(const Network*, const VectorR& spans, Index first_switch, float tolerance);
    };

    struct FeasibilitySystem
    {
        void initialize();

        void reshape_borders(const pair<VectorR, VectorR>&);

        VectorR force_into_borders(const VectorR&) const;

        VectorR evaluate(const VectorR&, VectorR&, VectorR&, const VectorR& = {}) const;

        MatrixR calculate_jacobian(const VectorR&, const VectorR&, const VectorR&, const VectorR& = {}) const;

        pair<VectorR, VectorR> solve(VectorR) const;

        const ResponseOptimization* problem = nullptr;

        vector<pair<const Constraint*, Index>> rows;

        pair<VectorR, VectorR> borders;
    };

    explicit ResponseOptimization(Network* = nullptr);

    virtual ~ResponseOptimization();

    void add_objective(const string&, Objective::Sense, float value = 0.0f);
    void add_constraint(const string&, Constraint::Condition, const vector<float>& values = {});

    void set_iterations_number(Index);
    void set_points_number(Index);

    void set_feasibility_margin_factor(float);

    MatrixR perform_response_optimization();

protected:

    Network* network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    FeasibilitySystem feasibility_system;

    virtual MatrixR single_optimization() = 0;

    virtual MatrixR multi_optimization() = 0;

    pair<VectorR, VectorR> calculate_domain();

    VectorR calculate_random_input(const pair<VectorR, VectorR>&) const;

    void assign_random_categories(VectorR&, float probability = 1.0f) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    Index iterations_number = 20;
    Index points_number = 1000;

    Index requested_front_size = 100;

private:

    pair<VectorR, VectorR> get_unconstrained_domain() const;

    float constraint_tolerance = 1e-3f;

    float feasibility_margin_factor = 0.1f;

    Index feasibility_evaluations = 50;

    float diversity_factor = 0.2f;
};

}

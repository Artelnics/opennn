// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class FeasibilityRepairSystem
{
public:

    using Constraint = ResponseOptimization::Constraint;

    enum class Solver { LevenbergMarquardt, GaussNewton = 2 };

    FeasibilityRepairSystem(Network*, span<const Constraint>, pair<VectorR, VectorR> domain,
                            Index rounds = 1, Index evaluations = 50,
                            float tolerance = 1e-6f, float margin = 0.1f, float constraint_tolerance = 1e-3f);

    void set_domain(pair<VectorR, VectorR>);
    void set_solver(Solver);

    pair<VectorR, VectorR> solve(VectorR);

private:

    using Condition = Constraint::Condition;

    struct LevenbergMarquardtFunctor;

    void solve_levenberg_marquardt(VectorR&, MatrixR);
    void solve_gauss_newton(VectorR&, MatrixR);

    bool round_discrete(VectorR&) const;
    const VectorR& evaluate_outputs(const VectorR&);
    void evaluate_constraints(const VectorR&);
    float calculate_constraint_value(Index) const;
    pair<float, float> calculate_constraint_residual(Index) const;
    float get_bound_tolerance(float) const;

    MatrixR calculate_network_jacobian(const VectorR&, const VectorR&) const;
    MatrixR calculate_jacobian(const VectorR&, const VectorR&, const VectorR&, const VectorR& = {}) const;
    void calculate_numerical_jacobian(const VectorR&, const VectorR&, const VectorR&,
                                      const vector<Index>&, MatrixR&) const;

    Network* network;
    span<const Constraint> constraints;
    pair<VectorR, VectorR> input_bounds;
    vector<pair<Index, Index>> categorical_blocks;
    bool constraints_read_output = false;

    Solver solver = Solver::LevenbergMarquardt;
    Index feasibility_rounds;
    Index feasibility_evaluations;
    float numeric_tolerance;
    float feasibility_margin_factor;
    float constraint_tolerance;

    static constexpr float difference_step = 1e-3f;
    static constexpr float discrete_difference_step = 0.25f;
    static constexpr float box_weight = 100.0f;

    VectorR box_scales;
    VectorR row_scales;
    VectorR row_values;
    VectorR row_residuals;
    VectorR evaluated_point;
    VectorR evaluated_output;
};

}

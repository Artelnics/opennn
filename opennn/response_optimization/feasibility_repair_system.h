// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class FeasibilityRepairSystem
{
public:

    using Constraint = ResponseOptimization::Constraint;
    using Solver = ResponseOptimization::FeasibilitySolver;

    FeasibilityRepairSystem(Network*, span<const Constraint>, pair<VectorR, VectorR> domain,
                            Index rounds = 1, Index evaluations = 50,
                            float tolerance = 1e-6f, float margin = 0.1f, float constraint_tolerance = 1e-3f);

    void set_domain(pair<VectorR, VectorR>);
    void set_solver(Solver);

    pair<VectorR, VectorR> solve(VectorR);
    pair<Eigen::VectorXd, VectorR> solve_double(Eigen::VectorXd);

    const pair<double, string>& get_closest_failure() const { return closest_failure; }

private:

    using Condition = Constraint::Condition;
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Layout>;

    struct LevenbergMarquardtFunctor;

    pair<Vector, VectorR> repair(Vector);
    bool round_discrete(Vector&) const;
    void set_rounding_errors(const Vector&, const Matrix&);

    const VectorR& evaluate_outputs(const Vector&);
    void evaluate_constraints(const Vector&);
    pair<double, double> calculate_constraint_residual(Index) const;
    float get_bound_tolerance(float) const;
    void update_closest_failure();

    MatrixR calculate_network_jacobian(const VectorR&, const VectorR&) const;
    Matrix calculate_jacobian(const Vector&, const Vector&, const VectorR&, const Vector& = {}) const;
    void calculate_numerical_jacobian(const Vector&, const Vector&, const Vector&, const vector<Index>&,
                                      Matrix&) const;

    void solve_levenberg_marquardt(Vector&, Matrix);
    void solve_gauss_newton(Vector&, Matrix, bool broyden);
    double calculate_error() const;
    Vector calculate_step(const Matrix&, bool hold_equalities, Vector& step_rows) const;
    bool try_step(const Vector&, const Vector&, Index trials, Index& evaluations, Vector&);
    void update_jacobian(Matrix&, const Vector& step, const Vector& value_change) const;

    Network* network;
    span<const Constraint> constraints;
    pair<Vector, Vector> bounds;
    Vector spans;
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
    static constexpr double solver_tolerance = 1e-10;

    bool float_points = false;

    Vector box_scales;
    Vector row_scales;
    Vector rounding_errors;
    Vector row_values;
    Vector row_residuals;
    Vector evaluated_point;
    VectorR evaluated_output;

    pair<double, string> closest_failure{numeric_limits<double>::infinity(), {}};
};

}

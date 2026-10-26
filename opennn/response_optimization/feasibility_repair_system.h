// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

// Repairs a point into the constraints: each round projects it onto the box and the discrete conditions, and
// a solver drives the residuals of the violated constraints to zero. The repair runs in double; the network
// evaluates in float.
class FeasibilityRepairSystem
{
public:

    using Constraint = ResponseOptimization::Constraint;

    // Broyden takes Gauss-Newton's projected steps, with met equalities kept in them, but evaluates the Jacobian
    // only at the start of a round and after a failed step; each accepted step corrects it with the secant update
    // of least change in span units.
    enum class Solver { LevenbergMarquardt, GaussNewton = 2, Broyden };

    FeasibilityRepairSystem(Network*, span<const Constraint>, pair<VectorR, VectorR> domain,
                            Index rounds = 1, Index evaluations = 50,
                            float tolerance = 1e-6f, float margin = 0.1f, float constraint_tolerance = 1e-3f);

    void set_domain(pair<VectorR, VectorR>);
    void set_solver(Solver);

    // For a caller that keeps float points: the constraints are evaluated, in double, at the float point the
    // repair will return, and a violated bound is aimed a float step inside, so the point returned meets them.
    pair<VectorR, VectorR> solve(VectorR);

    // For a caller that keeps double points: the constraints are evaluated at the double point. It reaches points
    // that float cannot represent, such as gaps below a float step; rounding them to float can lose feasibility.
    pair<Eigen::VectorXd, VectorR> solve_double(Eigen::VectorXd);

    const pair<double, string>& get_closest_failure() const { return closest_failure; }

private:

    using Condition = Constraint::Condition;
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Layout>;

    // The rows of the repair in progress at the point last evaluated, and the scales of its round.
    struct Workspace
    {
        Vector box_scales;
        Vector row_scales;
        Vector row_margins;
        Vector row_values;
        Vector row_residuals;
        Vector evaluated_point;
        VectorR evaluated_output;
    };

    struct LevenbergMarquardtFunctor;

    pair<Vector, VectorR> repair(Vector, bool float_points);
    bool round_discrete(Vector&) const;
    void set_float_margins(const Vector&, const Matrix&);

    const VectorR& evaluate_outputs(const Vector&);
    void evaluate_constraints(const Vector&);
    double calculate_constraint_value(Index) const;
    pair<double, double> calculate_constraint_residual(Index) const;
    void update_closest_failure();
    float get_bound_tolerance(float) const;

    MatrixR calculate_network_jacobian(const VectorR&, const VectorR&) const;
    Matrix calculate_jacobian(const Vector&, const Vector&, const VectorR&, const Vector& = {}) const;
    void calculate_numerical_jacobian(const Vector&, const Vector&, const Vector&, const vector<Index>&,
                                      Matrix&) const;

    // The solvers, in feasibility_repair_solvers.cpp.
    void solve_levenberg_marquardt(Vector&, Matrix);
    void solve_gauss_newton(Vector&, Matrix, bool broyden = false);
    double calculate_merit() const;
    Vector calculate_step(const Matrix&, const Vector&, Vector&) const;
    bool search_step(const Vector&, const Vector&, Index, Index&, Vector&);
    void evaluate_step_rows(const Vector&, const Vector&, Matrix&) const;
    Vector get_equality_targets() const;
    static void correct_jacobian(Matrix&, const Vector&, const Vector&, const Vector&);

    Network* network;
    span<const Constraint> constraints;
    pair<Vector, Vector> bounds;
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
    Workspace work;
    pair<double, string> closest_failure{numeric_limits<double>::infinity(), {}};
};

}

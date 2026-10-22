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

    enum class Solver { LevenbergMarquardt, GaussNewton = 2 };

    FeasibilityRepairSystem(Network*, span<const Constraint>, pair<VectorR, VectorR> domain,
                            Index rounds = 1, Index evaluations = 50,
                            float tolerance = 1e-6f, float margin = 0.1f, float constraint_tolerance = 1e-3f);

    void set_domain(pair<VectorR, VectorR>);
    void set_solver(Solver);

    pair<VectorR, VectorR> solve(VectorR);

    // The same repair carried out in double precision: the point, the constraint values, their residuals
    // and derivatives, and the solver's state. The network still evaluates in float. It reaches points
    // that float cannot resolve, such as long sums of nearly cancelling terms or gaps below a float step;
    // rounding its point to float can lose that feasibility.
    pair<Eigen::VectorXd, VectorR> solve_double(Eigen::VectorXd);

    const pair<double, string>& get_closest_failure() const { return closest_failure; }

private:

    using Condition = Constraint::Condition;

    template<typename Scalar> using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar> using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Layout>;

    template<typename Scalar>
    struct Workspace
    {
        Vector<Scalar> box_scales;
        Vector<Scalar> row_scales;
        Vector<Scalar> row_values;
        Vector<Scalar> row_residuals;
        Vector<Scalar> evaluated_point;
        VectorR evaluated_output;
    };

    template<typename Scalar> struct LevenbergMarquardtFunctor;

    template<typename Scalar> Workspace<Scalar>& get_workspace();
    template<typename Scalar> const Workspace<Scalar>& get_workspace() const;
    template<typename Scalar> const pair<Vector<Scalar>, Vector<Scalar>>& get_bounds() const;

    template<typename Scalar> pair<Vector<Scalar>, VectorR> repair(Vector<Scalar>);
    template<typename Scalar> void solve_levenberg_marquardt(Vector<Scalar>&, Matrix<Scalar>);
    template<typename Scalar> void solve_gauss_newton(Vector<Scalar>&, Matrix<Scalar>);

    template<typename Scalar> bool round_discrete(Vector<Scalar>&) const;
    template<typename Scalar> const VectorR& evaluate_outputs(const Vector<Scalar>&);
    template<typename Scalar> void evaluate_constraints(const Vector<Scalar>&);
    template<typename Scalar> void update_closest_failure();
    template<typename Scalar> Scalar calculate_constraint_value(Index) const;
    template<typename Scalar> pair<Scalar, Scalar> calculate_constraint_residual(Index) const;
    float get_bound_tolerance(float) const;

    MatrixR calculate_network_jacobian(const VectorR&, const VectorR&) const;
    template<typename Scalar>
    Matrix<Scalar> calculate_jacobian(const Vector<Scalar>&, const Vector<Scalar>&, const VectorR&,
                                      const Vector<Scalar>& = {}) const;
    template<typename Scalar>
    void calculate_numerical_jacobian(const Vector<Scalar>&, const Vector<Scalar>&, const Vector<Scalar>&,
                                      const vector<Index>&, Matrix<Scalar>&) const;

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
    static constexpr double double_solver_tolerance = 1e-10;

    pair<Eigen::VectorXd, Eigen::VectorXd> double_bounds;
    Workspace<float> float_workspace;
    Workspace<double> double_workspace;
    pair<double, string> closest_failure{numeric_limits<double>::infinity(), {}};
};

}

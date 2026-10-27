// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/response_optimization/feasibility_repair_system.h"

#include <Eigen/QR>
#include <unsupported/Eigen/LevenbergMarquardt>

namespace opennn
{

struct FeasibilityRepairSystem::LevenbergMarquardtFunctor : Eigen::DenseFunctor<double>
{
    LevenbergMarquardtFunctor(FeasibilityRepairSystem& new_system, Matrix jacobian)
        : Eigen::DenseFunctor<double>(int(new_system.spans.size()),
                                      int(new_system.constraints.size() + new_system.spans.size())),
          system(new_system), initial_jacobian(move(jacobian)) {}

    int operator()(const Vector& point, Vector& violations)
    {
        system.evaluate_constraints(point);

        const Vector& values = system.row_values;
        const auto& [lower, upper] = system.bounds;

        violations.resize(values.size() + point.size());
        violations.head(values.size()) = values.array().isFinite()
            .select(system.row_residuals.array()/system.row_scales.array(), 1.0/double(system.numeric_tolerance));
        violations.tail(point.size()) =
            ((point - upper).cwiseMax(0.0) + (point - lower).cwiseMin(0.0)).cwiseQuotient(system.box_scales);

        return 0;
    }

    int df(const Vector& point, JacobianType& jacobian)
    {
        system.evaluate_constraints(point);

        const Vector& values = system.row_values;
        const Vector& residuals = system.row_residuals;
        const auto& [lower, upper] = system.bounds;

        const Matrix value_jacobian = initial_jacobian.size() != 0
            ? exchange(initial_jacobian, Matrix{})
            : system.calculate_jacobian(point, values, system.evaluated_output, residuals);

        jacobian.setZero(values.size() + point.size(), point.size());

        for (Index i = 0; i < values.size(); i++)
            if (residuals(i) != 0.0 && isfinite(residuals(i)))
                jacobian.row(i) = value_jacobian.row(i)
                                * (system.calculate_constraint_residual(i).second/system.row_scales(i));

        for (Index j = 0; j < point.size(); j++)
            if (point(j) > upper(j) || point(j) < lower(j))
                jacobian(values.size() + j, j) = 1.0/system.box_scales(j);

        return 0;
    }

    FeasibilityRepairSystem& system;
    Matrix initial_jacobian;
};


void FeasibilityRepairSystem::solve_levenberg_marquardt(Vector& point, Matrix initial_jacobian)
{
    LevenbergMarquardtFunctor functor(*this, move(initial_jacobian));
    Eigen::LevenbergMarquardt<LevenbergMarquardtFunctor> levenberg_marquardt(functor);
    levenberg_marquardt.setMaxfev(feasibility_evaluations);
    levenberg_marquardt.setFtol(solver_tolerance);
    levenberg_marquardt.setXtol(solver_tolerance);
    levenberg_marquardt.setGtol(0.0);
    levenberg_marquardt.minimize(point);
}


double FeasibilityRepairSystem::calculate_error() const
{
    return row_residuals.cwiseQuotient(row_scales).squaredNorm();
}


FeasibilityRepairSystem::Vector FeasibilityRepairSystem::calculate_step(const Matrix& jacobian,
                                                                        const bool hold_equalities,
                                                                        Vector& step_rows) const
{
    Matrix system(jacobian.rows(), jacobian.cols());
    Vector residuals(jacobian.rows());
    Index count = 0;

    for (Index i = 0; i < jacobian.rows(); i++)
    {
        const auto [lower, upper] = constraints[size_t(i)].get_bounds(numeric_tolerance, constraint_tolerance);
        const bool held = hold_equalities && row_residuals(i) == 0.0 && lower == upper;

        step_rows(i) = double(held || row_residuals(i) != 0.0);
        if (step_rows(i) == 0.0) continue;

        const double slope = held ? 1.0 : calculate_constraint_residual(i).second;
        system.row(count) = jacobian.row(i)*(slope/row_scales(i));
        residuals(count++) = (held ? row_values(i) - double(lower) : row_residuals(i))/row_scales(i);
    }

    auto rows = system.topRows(count);
    rows.array().rowwise() *= spans.transpose().array();

    if (!rows.allFinite() || !residuals.head(count).allFinite())
        return Vector::Constant(jacobian.cols(), numeric_limits<double>::quiet_NaN());

    const double scale = max(1.0, rows.cwiseAbs().maxCoeff());
    rows /= scale;
    residuals.head(count) /= scale;

    return rows.completeOrthogonalDecomposition().solve(-residuals.head(count));
}


bool FeasibilityRepairSystem::try_step(const Vector& point, const Vector& step, const Index trials,
                                       Index& evaluations, Vector& candidate)
{
    const auto& [lower, upper] = bounds;
    const tuple start(row_values, row_residuals, evaluated_point, evaluated_output);
    const double error = calculate_error();
    double fraction = 1.0;

    for (Index j = 0; j < trials && evaluations < feasibility_evaluations; j++, fraction *= 0.5)
    {
        evaluations++;
        candidate = point + (fraction*spans).cwiseProduct(step);
        if (!candidate.allFinite()) continue;
        candidate = candidate.cwiseMax(lower).cwiseMin(upper);
        if (candidate == point) break;
        evaluate_constraints(candidate);
        if (row_values.allFinite() && row_residuals.allFinite() && calculate_error() < error) return true;
    }

    tie(row_values, row_residuals, evaluated_point, evaluated_output) = start;
    return false;
}


void FeasibilityRepairSystem::update_jacobian(Matrix& jacobian, const Vector& step, const Vector& value_change) const
{
    const Eigen::Array<bool, Eigen::Dynamic, 1> movable = spans.array() > 0.0;
    const Vector span_step = movable.select(step.array()/spans.array(), 0.0);
    const double length = span_step.squaredNorm();

    if (!(length > 0.0) || !isfinite(length)) return;

    const Vector correction = (value_change - jacobian*step)/length;
    const Vector weights = movable.select(span_step.array()/spans.array(), 0.0);

    if (correction.allFinite()) jacobian += correction*weights.transpose();
}


void FeasibilityRepairSystem::solve_gauss_newton(Vector& point, Matrix jacobian, const bool broyden)
{
    Vector step_rows(row_values.size());
    bool updated = false;

    for (Index i = 0; i < feasibility_evaluations;)
    {
        if (!row_values.allFinite() || !row_residuals.allFinite())
        {
            point.setConstant(numeric_limits<double>::quiet_NaN());
            return;
        }

        if ((row_residuals.array() == 0.0).all()) return;

        if (i > 0 && !broyden)
            jacobian = calculate_jacobian(point, row_values, evaluated_output, row_residuals);

        const Vector step = calculate_step(jacobian, broyden, step_rows);
        const Vector values = row_values;
        Vector candidate;

        if (step.allFinite() && try_step(point, step, updated ? 1 : 8, i, candidate))
        {
            if (broyden) update_jacobian(jacobian, candidate - point, row_values - values);
            updated = broyden;
            point = move(candidate);
            continue;
        }

        if (!updated)
        {
            if (!step.allFinite()) point.setConstant(numeric_limits<double>::quiet_NaN());
            return;
        }

        const Matrix evaluated = calculate_jacobian(point, row_values, evaluated_output, step_rows);

        for (Index j = 0; j < step_rows.size(); j++)
            if (step_rows(j) != 0.0) jacobian.row(j) = evaluated.row(j);

        updated = false;
    }
}

}

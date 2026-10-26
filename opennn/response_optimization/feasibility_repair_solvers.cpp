// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

// The solvers of the feasibility repair. Levenberg-Marquardt minimizes the scaled residuals together with rows
// that penalize leaving the box. Gauss-Newton and Broyden take projected minimum-norm steps in span units and
// backtrack until the merit, the squared scaled residuals, falls.

#include "opennn/response_optimization/feasibility_repair_system.h"

#include <Eigen/QR>
#include <unsupported/Eigen/LevenbergMarquardt>

namespace opennn
{

struct FeasibilityRepairSystem::LevenbergMarquardtFunctor : Eigen::DenseFunctor<double>
{
    using Base = Eigen::DenseFunctor<double>;

    LevenbergMarquardtFunctor(FeasibilityRepairSystem& system, Matrix jacobian)
        : Base(int(system.bounds.first.size()), int(system.constraints.size() + system.bounds.first.size())),
          system(system), initial_jacobian(move(jacobian)) {}

    int operator()(const Vector& point, Vector& violations)
    {
        const Workspace& rows = system.work;
        system.evaluate_constraints(point);
        violations.resize(rows.row_values.size() + point.size());
        violations.head(rows.row_values.size()) = rows.row_values.array().isFinite()
            .select(rows.row_residuals.array()/rows.row_scales.array(), 1.0/double(system.numeric_tolerance));
        const auto& [lower, upper] = system.bounds;
        violations.tail(point.size()) =
            ((point - upper).cwiseMax(0.0) + (point - lower).cwiseMin(0.0)).cwiseQuotient(rows.box_scales);
        return 0;
    }

    int df(const Vector& point, Base::JacobianType& jacobian)
    {
        const Workspace& rows = system.work;
        system.evaluate_constraints(point);
        const Matrix value_jacobian = initial_jacobian.size() != 0 ? exchange(initial_jacobian, Matrix{})
            : system.calculate_jacobian(point, rows.row_values, rows.evaluated_output, rows.row_residuals);
        jacobian.setZero(rows.row_values.size() + point.size(), point.size());
        for (Index i = 0; i < rows.row_values.size(); i++)
            if (rows.row_residuals(i) != 0.0 && isfinite(rows.row_residuals(i)))
                jacobian.row(i) = value_jacobian.row(i)*(system.calculate_constraint_residual(i).second/rows.row_scales(i));
        const auto& [lower, upper] = system.bounds;
        for (Index j = 0; j < point.size(); j++)
            if (point(j) > upper(j) || point(j) < lower(j))
                jacobian(rows.row_values.size() + j, j) = 1.0/rows.box_scales(j);
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


double FeasibilityRepairSystem::calculate_merit() const
{
    return work.row_residuals.cwiseQuotient(work.row_scales).squaredNorm();
}


// The minimum-norm step in span units for the rows of the step: the rows not met and, for Broyden, the met
// equalities, aimed at their values. It is not finite when the system or its solution is not.
FeasibilityRepairSystem::Vector FeasibilityRepairSystem::calculate_step(const Matrix& jacobian, const Vector& targets,
                                                                        Vector& stepped) const
{
    const Vector spans = bounds.second - bounds.first;
    Matrix system(jacobian.rows(), jacobian.cols());
    Vector residuals(jacobian.rows());
    Index rows = 0;
    for (Index j = 0; j < jacobian.rows(); j++)
    {
        const bool held = work.row_residuals(j) == 0.0 && !isnan(targets(j));
        stepped(j) = double(held || work.row_residuals(j) != 0.0);
        if (stepped(j) == 0.0) continue;
        const double slope = held ? 1.0 : calculate_constraint_residual(j).second;
        system.row(rows) = jacobian.row(j)*(slope/work.row_scales(j));
        residuals(rows++) = (held ? work.row_values(j) - targets(j) : work.row_residuals(j))/work.row_scales(j);
    }
    auto active_jacobian = system.topRows(rows);
    active_jacobian.array().rowwise() *= spans.transpose().array();
    if (!active_jacobian.allFinite() || !residuals.head(rows).allFinite())
        return Vector::Constant(jacobian.cols(), numeric_limits<double>::quiet_NaN());

    const double scale = max(1.0, active_jacobian.cwiseAbs().maxCoeff());
    if (scale > 1.0)
    {
        active_jacobian /= scale;
        residuals.head(rows) /= scale;
    }
    return active_jacobian.completeOrthogonalDecomposition().solve(-residuals.head(rows));
}


// Halves the step from the point, for at most the given trials and within the evaluation budget, until a candidate
// lowers the merit. When one does, the workspace holds its rows.
bool FeasibilityRepairSystem::search_step(const Vector& point, const Vector& step, const Index trials,
                                          Index& evaluations, Vector& candidate)
{
    const auto& [lower, upper] = bounds;
    const Vector spans = upper - lower;
    const double merit = calculate_merit();
    double fraction = 1.0;
    for (Index j = 0; j < trials && evaluations < feasibility_evaluations; j++, fraction *= 0.5)
    {
        evaluations++;
        candidate = point + (fraction*spans).cwiseProduct(step);
        if (!candidate.allFinite()) continue;
        candidate = candidate.cwiseMax(lower).cwiseMin(upper);
        if ((candidate.array() == point.array()).all()) return false;
        evaluate_constraints(candidate);
        if (work.row_values.allFinite() && work.row_residuals.allFinite() && calculate_merit() < merit)
            return true;
    }
    return false;
}


void FeasibilityRepairSystem::evaluate_step_rows(const Vector& point, const Vector& stepped, Matrix& jacobian) const
{
    const Matrix evaluated = calculate_jacobian(point, work.row_values, work.evaluated_output, stepped);
    for (Index j = 0; j < stepped.size(); j++)
        if (stepped(j) != 0.0) jacobian.row(j) = evaluated.row(j);
}


FeasibilityRepairSystem::Vector FeasibilityRepairSystem::get_equality_targets() const
{
    Vector targets = Vector::Constant(Index(constraints.size()), numeric_limits<double>::quiet_NaN());
    for (Index i = 0; i < targets.size(); i++)
        if (const auto [lower, upper] = constraints[size_t(i)].get_bounds(numeric_tolerance, constraint_tolerance);
            lower == upper)
            targets(i) = double(lower);
    return targets;
}


// Broyden's update of least change in span units: the corrected Jacobian maps the step onto the change it
// produced in the constraint values, and acts as before on every direction orthogonal to the step.
void FeasibilityRepairSystem::correct_jacobian(Matrix& jacobian, const Vector& step, const Vector& value_change,
                                               const Vector& spans)
{
    const Eigen::Array<bool, Eigen::Dynamic, 1> open = spans.array() > 0.0;
    const Vector span_step = open.select(step.array()/spans.array(), 0.0);
    const double length = span_step.squaredNorm();
    if (!(length > 0.0) || !isfinite(length)) return;
    const Vector correction = (value_change - jacobian*step)/length;
    const Vector weights = open.select(span_step.array()/spans.array(), 0.0);
    if (correction.allFinite()) jacobian += correction*weights.transpose();
}


void FeasibilityRepairSystem::solve_gauss_newton(Vector& point, Matrix jacobian, const bool broyden)
{
    const Vector spans = bounds.second - bounds.first;
    Vector stepped(work.row_residuals.size());

    // Broyden keeps a met equality in the step, aimed at its value. Left out, it would drift back out while the
    // step met the other rows, and with steps less exact than Gauss-Newton's the rows would take turns.
    const Vector targets = broyden ? get_equality_targets()
                                   : Vector::Constant(stepped.size(), numeric_limits<double>::quiet_NaN());

    // Broyden corrects its Jacobian after each accepted step and tries a step from the corrected Jacobian at full
    // length only. When that step fails, the rows of the step are evaluated again; a failed step from an evaluated
    // Jacobian ends the solve, as in Gauss-Newton.
    bool corrected = false;

    for (Index i = 0; i < feasibility_evaluations;)
    {
        if (!work.row_values.allFinite() || !work.row_residuals.allFinite())
        {
            point.setConstant(numeric_limits<double>::quiet_NaN());
            return;
        }
        if ((work.row_residuals.array() == 0.0).all()) return;

        if (i > 0 && !broyden)
            jacobian = calculate_jacobian(point, work.row_values, work.evaluated_output, work.row_residuals);
        const Vector step = calculate_step(jacobian, targets, stepped);
        const Workspace start = broyden ? work : Workspace{};
        Vector candidate;
        if (step.allFinite() && search_step(point, step, corrected ? 1 : 8, i, candidate))
        {
            if (broyden) correct_jacobian(jacobian, candidate - point, work.row_values - start.row_values, spans);
            corrected = broyden;
            point = move(candidate);
            continue;
        }
        if (!corrected)
        {
            if (!step.allFinite()) point.setConstant(numeric_limits<double>::quiet_NaN());
            return;
        }
        work = start;
        evaluate_step_rows(point, stepped, jacobian);
        corrected = false;
    }
}

}

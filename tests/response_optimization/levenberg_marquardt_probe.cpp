//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G   M A R Q U A R D T   P R O B E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A plain main that asks one question: can Eigen's own Levenberg Marquardt solve a system
// of equations written as a functor, including one whose residuals read a network output?
//
// Nothing here touches the response optimization. It is a probe, kept next to the response
// tests because the shape it tries out is the shape FeasibilitySystem already has: a set of
// residuals, some analytical in the inputs, some coupled through the network response, and
// a numerical jacobian over all of them. If the probe holds, the same functor can later be
// filled with the assembled constraint rows instead of the hand written ones below.
//
// Two systems are solved:
//
//     1. The textbook one, in double precision and with an exact answer to check against.
//     2. Three concrete mix equations, one of which calls the trained network.
//
// The second is where the arithmetic gets interesting. The network answers in float, so its
// response carries about seven digits; a central difference taken with the default step of
// sqrt(machine epsilon in double), around 1e-8, would divide that noise by nothing and hand
// the solver a jacobian of garbage. The epsfcn argument of NumericalDiff is what sets that
// step, and it is passed here rather than left at zero. The tolerances follow: a residual
// built on a float response cannot be driven to 1e-10, so the run is asked for what it can
// actually reach.
//
// Build and run:
//
//     cmake --build build-ninja --target opennn_lm_probe
//     build-ninja/bin/opennn_lm_probe
//
// It returns 0 when every system was solved and every check passed, 1 otherwise.

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/NumericalDiff>

#include "opennn/network/network.h"

using namespace std;
using namespace opennn;

#ifndef CONCRETE_NETWORK_DIR
#define CONCRETE_NETWORK_DIR "examples/concrete"
#endif


// What the solver says it stopped on, in words. Anything but ImproperInputParameters and
// TooManyFunctionEvaluation is a converged run; which criterion tripped first says how it
// converged, not whether it did.

const char* status_name(const LevenbergMarquardtSpace::Status status)
{
    switch (status)
    {
    case LevenbergMarquardtSpace::NotStarted:                       return "not started";
    case LevenbergMarquardtSpace::Running:                          return "running";
    case LevenbergMarquardtSpace::ImproperInputParameters:          return "improper input parameters";
    case LevenbergMarquardtSpace::RelativeReductionTooSmall:        return "relative reduction too small";
    case LevenbergMarquardtSpace::RelativeErrorTooSmall:            return "relative error too small";
    case LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall:return "relative error and reduction too small";
    case LevenbergMarquardtSpace::CosinusTooSmall:                  return "cosinus too small";
    case LevenbergMarquardtSpace::TooManyFunctionEvaluation:        return "too many function evaluations";
    case LevenbergMarquardtSpace::FtolTooSmall:                     return "ftol too small";
    case LevenbergMarquardtSpace::XtolTooSmall:                     return "xtol too small";
    case LevenbergMarquardtSpace::GtolTooSmall:                     return "gtol too small";
    case LevenbergMarquardtSpace::UserAsked:                        return "user asked";
    }

    return "?";
}


void title(const string& text)
{
    cout << "\n================================================================\n"
         << text << "\n"
         << "================================================================\n";
}


// 1. Two equations, two unknowns, no network in sight.
//
//     x0^2 + x1     - 4 = 0
//     x0   + sin(x1) - 1 = 0

struct SystemFunctor : Eigen::DenseFunctor<double>
{
    SystemFunctor(const int variables_number, const int residuals_number)
        : Eigen::DenseFunctor<double>(variables_number, residuals_number) {}

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& f) const
    {
        f(0) = x(0)*x(0) + x(1) - 4.0;

        f(1) = x(0) + sin(x(1)) - 1.0;

        return 0;
    }
};


// The first equation gives x1 = 4 - x0^2, so the pair collapses to one function of x0. It
// is solved here by bisection, which cannot land anywhere the solver did unless the root is
// really there. This is the check: an answer with a small residual is not yet an answer at
// the right place, and only an independent solve says which.

double bisect_reference_root(const double lower, const double upper, const int steps = 200)
{
    auto g = [](const double t) { return t + sin(4.0 - t*t) - 1.0; };

    double a = lower;
    double b = upper;

    for (int i = 0; i < steps; i++)
    {
        const double m = 0.5*(a + b);

        (g(a)*g(m) <= 0.0 ? b : a) = m;
    }

    return 0.5*(a + b);
}


bool solve_analytical_system()
{
    title("1. Two analytical equations in two unknowns");

    SystemFunctor functor(2, 2);

    // Numerical jacobian.

    Eigen::NumericalDiff<SystemFunctor, Eigen::Central> numerical_diff(functor);

    // Levenberg Marquardt.

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<SystemFunctor, Eigen::Central>>
        levenberg_marquardt(numerical_diff);

    levenberg_marquardt.setMaxfev(500);
    levenberg_marquardt.setFtol(1e-10);
    levenberg_marquardt.setXtol(1e-10);

    Eigen::VectorXd x(2);
    x << 1.0, 1.0;

    const LevenbergMarquardtSpace::Status status = levenberg_marquardt.minimize(x);

    Eigen::VectorXd residuals(2);
    functor(x, residuals);

    const double reference_x0 = bisect_reference_root(0.0, 2.0);
    const double reference_x1 = 4.0 - reference_x0*reference_x0;

    cout << scientific << setprecision(10)
         << "status        " << status_name(status) << "\n"
         << "iterations    " << levenberg_marquardt.iterations()
         << ",  evaluations " << levenberg_marquardt.nfev() << "\n"
         << "x             " << x(0) << "  " << x(1) << "\n"
         << "bisection     " << reference_x0 << "  " << reference_x1 << "\n"
         << "residuals     " << residuals(0) << "  " << residuals(1) << "\n"
         << "norm          " << residuals.norm() << "\n";

    const bool solved = residuals.norm() < 1e-10
                     && abs(x(0) - reference_x0) < 1e-6
                     && abs(x(1) - reference_x1) < 1e-6;

    cout << (solved
             ? "\nSolved, and at the same root the bisection found.\n"
             : "\nNOT solved.\n");

    return solved;
}


// 2. The same machinery, with the trained concrete network inside one of the residuals.
//
// The unknowns are three of the eight mix variables; the other five are held at the values
// of a reference mix. The three equations are the three kinds a real problem mixes:
//
//     cement + water        - total        = 0     analytical, linear
//     water/cement          - ratio        = 0     analytical, nonlinear
//     strength(mix)         - target       = 0     coupled through the network
//
// The right hand sides are read off a reference mix, so the system is known to have a root
// and it is known where: the solver has to walk back to a point it was never told. The
// first two equations pin cement and water between them, and the network equation is left
// to place the age, which is the part that no analytical expression could have done.

struct MixSystem : Eigen::DenseFunctor<double>
{
    MixSystem(Network& new_network,
              const Eigen::VectorXd& new_mix,
              const double new_total,
              const double new_ratio,
              const double new_strength)
        : Eigen::DenseFunctor<double>(3, 3),
          network(new_network),
          mix(new_mix),
          total(new_total),
          ratio(new_ratio),
          strength(new_strength) {}

    // The three unknowns are cement, water and age, in that order.

    enum Unknown { Cement = 0, Water = 1, Age = 2 };

    enum Input { CementColumn = 0, WaterColumn = 3, AgeColumn = 7, InputsNumber = 8 };

    double predict(const Eigen::VectorXd& x) const
    {
        MatrixR inputs(1, Index(InputsNumber));

        for (int i = 0; i < InputsNumber; i++)
            inputs(0, Index(i)) = float(mix(i));

        inputs(0, Index(CementColumn)) = float(x(Cement));
        inputs(0, Index(WaterColumn)) = float(x(Water));
        inputs(0, Index(AgeColumn)) = float(x(Age));

        return double(network.calculate_outputs(inputs)(0, 0));
    }

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& f) const
    {
        // Analytical constraint.

        f(0) = x(Cement) + x(Water) - total;

        // Another analytical constraint, this one nonlinear.

        f(1) = x(Water)/x(Cement) - ratio;

        // Coupled constraint containing the network output.

        const double response = predict(x);

        f(2) = response - strength;

        return 0;
    }

    Network& network;

    Eigen::VectorXd mix;

    double total = 0.0;
    double ratio = 0.0;
    double strength = 0.0;
};


bool solve_coupled_system(Network& network)
{
    title("2. Three mix equations, one of them through the network");

    // A mix from the middle of the data, taken as the answer to walk back to.

    Eigen::VectorXd reference(MixSystem::InputsNumber);
    reference << 332.5, 142.5, 0.0, 228.0, 0.0, 932.0, 594.0, 90.0;

    Eigen::VectorXd reference_unknowns(3);
    reference_unknowns << reference(MixSystem::CementColumn),
                          reference(MixSystem::WaterColumn),
                          reference(MixSystem::AgeColumn);

    MixSystem functor(network,
                      reference,
                      reference_unknowns(0) + reference_unknowns(1),
                      reference_unknowns(1)/reference_unknowns(0),
                      0.0);

    functor.strength = functor.predict(reference_unknowns);

    // The step of the central difference. Left at zero it would be about 1e-8 of each
    // variable, far below what a float response can resolve, and the third row of the
    // jacobian would be noise. 1e-8 here asks for sqrt(1e-8) = 1e-4 instead.

    Eigen::NumericalDiff<MixSystem, Eigen::Central> numerical_diff(functor, 1e-8);

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<MixSystem, Eigen::Central>>
        levenberg_marquardt(numerical_diff);

    levenberg_marquardt.setMaxfev(500);
    levenberg_marquardt.setFtol(1e-8);
    levenberg_marquardt.setXtol(1e-8);

    // Somewhere else entirely: more cement, less water, a fifth of the age.

    Eigen::VectorXd x(3);
    x << 400.0, 190.0, 18.0;

    Eigen::VectorXd start_residuals(3);
    functor(x, start_residuals);

    const LevenbergMarquardtSpace::Status status = levenberg_marquardt.minimize(x);

    Eigen::VectorXd residuals(3);
    functor(x, residuals);

    cout << fixed << setprecision(4)
         << "target        cement + water = " << functor.total
         << ",  water/cement = " << functor.ratio
         << ",  strength = " << functor.strength << "\n"
         << "reference     " << reference_unknowns(0) << "  "
                             << reference_unknowns(1) << "  "
                             << reference_unknowns(2) << "\n"
         << "solution      " << x(0) << "  " << x(1) << "  " << x(2) << "\n"
         << "response      " << functor.predict(x) << "\n"
         << scientific << setprecision(4)
         << "gap           " << abs(x(0) - reference_unknowns(0)) << "  "
                             << abs(x(1) - reference_unknowns(1)) << "  "
                             << abs(x(2) - reference_unknowns(2)) << "\n"
         << "residuals     " << residuals(0) << "  "
                             << residuals(1) << "  "
                             << residuals(2) << "\n"
         << "norm          " << start_residuals.norm() << "  ->  " << residuals.norm() << "\n"
         << fixed << setprecision(0)
         << "status        " << status_name(status)
         << ",  iterations " << levenberg_marquardt.iterations()
         << ",  evaluations " << levenberg_marquardt.nfev() << "\n";

    // The two analytical rows are solved in double and hold to their own precision. The
    // coupled row is only ever as sharp as the float response behind it, so it is asked for
    // a strength within a hundredth of a megapascal, which is far below the error of the
    // network itself. The unknowns have to come back to the reference mix, the age loosely,
    // since the response flattens with age and a whole day of it moves the strength little.

    const bool solved = abs(residuals(0)) < 1e-6
                     && abs(residuals(1)) < 1e-8
                     && abs(residuals(2)) < 1e-2
                     && abs(x(0) - reference_unknowns(0)) < 1e-2
                     && abs(x(1) - reference_unknowns(1)) < 1e-2
                     && abs(x(2) - reference_unknowns(2)) < 1.0;

    cout << (solved
             ? "\nSolved. The network row behaves like any other equation: give the residual\n"
               "a step it can see, and the numerical jacobian carries the response along.\n"
             : "\nNOT solved.\n");

    return solved;
}


int main()
{
    try
    {
        const bool analytical_solved = solve_analytical_system();

        Network network(filesystem::path(CONCRETE_NETWORK_DIR) / "nn" / "concrete_uci.json");

        const bool coupled_solved = solve_coupled_system(network);

        title(analytical_solved && coupled_solved ? "All systems solved" : "Some system failed");

        return analytical_solved && coupled_solved ? 0 : 1;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "tests/pch.h"
#include "opennn/training/quasi_newton_update.h"
#include <Eigen/Cholesky>

using namespace opennn;

namespace
{
Eigen::MatrixXd product_form(const MatrixR& previous, const VectorR& step, const VectorR& gradient)
{
    const Eigen::VectorXd s = step.cast<double>(), y = gradient.cast<double>();
    const double rho = 1.0 / s.dot(y);
    const Eigen::MatrixXd projection = Eigen::MatrixXd::Identity(s.size(), s.size()) - rho * s * y.transpose();
    return projection * previous.cast<double>() * projection.transpose() + rho * s * s.transpose();
}
}

TEST(QuasiNewtonUpdate, MatchesProductFormAndSecantEquation)
{
    std::mt19937 generator(20260921);
    std::normal_distribution<float> normal;
    for (const Index n : {1, 2, 7, 32})
        for (int trial = 0; trial < 20; ++trial)
        {
            SCOPED_TRACE(::testing::Message() << "n=" << n << ", trial=" << trial);
            const MatrixR seed = MatrixR::NullaryExpr(n, n, [&] { return normal(generator); });
            MatrixR previous = seed * seed.transpose() / float(n);
            previous.diagonal().array() += 0.5f;
            VectorR step = VectorR::NullaryExpr(n, [&] { return normal(generator); });
            VectorR gradient = step;
            for (Index i = 0; i < n; ++i) gradient(i) *= 0.1f + float(i + 1) / float(n);
            const float scale = std::pow(10.0f, float(trial % 5 - 2));
            step *= scale;
            gradient *= scale;
            MatrixR updated(n, n);
            VectorR work(n);
            ASSERT_TRUE(detail::bfgs_inverse_update(previous, step, gradient, updated, work));
            const Eigen::MatrixXd expected = product_form(previous, step, gradient);
            EXPECT_LE((updated.cast<double>() - expected).norm(), 2.0e-5 * std::max(1.0, expected.norm()));
            EXPECT_LE((updated * gradient - step).norm(), 2.0e-4f * step.norm());
            EXPECT_TRUE((updated.array() == updated.transpose().array()).all());
            Eigen::LLT<MatrixR> factorization(updated);
            EXPECT_EQ(factorization.info(), Eigen::Success);
        }
}

TEST(QuasiNewtonUpdate, RemainsStableAcrossRepeatedUpdates)
{
    std::mt19937 generator(17);
    std::normal_distribution<float> normal;
    VectorR diagonal(6);
    diagonal << 0.1f, 0.3f, 1.0f, 3.0f, 10.0f, 30.0f;
    MatrixR previous = MatrixR::Identity(6, 6), updated(6, 6);
    VectorR work(6);
    for (int iteration = 0; iteration < 60; ++iteration)
    {
        SCOPED_TRACE(iteration);
        const VectorR step = VectorR::NullaryExpr(6, [&] { return normal(generator); });
        const VectorR gradient = diagonal.array() * step.array();
        const Eigen::MatrixXd expected = product_form(previous, step, gradient);
        ASSERT_TRUE(detail::bfgs_inverse_update(previous, step, gradient, updated, work));
        EXPECT_LE((updated.cast<double>() - expected).norm(), 5.0e-5 * expected.norm());
        EXPECT_LE((updated * gradient - step).norm(), 2.0e-4f * step.norm());
        Eigen::LLT<MatrixR> factorization(updated);
        EXPECT_EQ(factorization.info(), Eigen::Success);
        previous.swap(updated);
    }
}

TEST(QuasiNewtonUpdate, KeepsPreviousMatrixWhenCurvatureIsTooSmall)
{
    const MatrixR previous = MatrixR::Identity(2, 2);
    VectorR step(2), gradient(2), work(2);
    step << 1.0f, 0.0f;
    for (float curvature : {-1.0f, 0.0f, EPSILON / 2, EPSILON})
    {
        gradient << curvature, 1.0f;
        MatrixR updated = MatrixR::Constant(2, 2, 99.0f);
        EXPECT_FALSE(detail::bfgs_inverse_update(previous, step, gradient, updated, work));
        EXPECT_TRUE((updated.array() == previous.array()).all());
    }
    gradient << 2 * EPSILON, 1.0f;
    MatrixR updated(2, 2);
    EXPECT_TRUE(detail::bfgs_inverse_update(previous, step, gradient, updated, work));
    EXPECT_TRUE(updated.allFinite());
}

TEST(QuasiNewtonUpdate, KeepsPreviousMatrixWhenHessianCurvatureIsUnusable)
{
    const VectorR step = VectorR::Ones(2), gradient = step;
    VectorR work(2);
    for (float diagonal : {-1.0f, 0.0f, EPSILON / 4})
    {
        const MatrixR previous = diagonal * MatrixR::Identity(2, 2);
        MatrixR updated(2, 2);
        EXPECT_FALSE(detail::bfgs_inverse_update(previous, step, gradient, updated, work));
        EXPECT_TRUE((updated.array() == previous.array()).all());
    }
}

TEST(QuasiNewtonUpdate, RejectsNonfiniteInputsAndRestoresOverflowingOutput)
{
    const MatrixR previous = MatrixR::Identity(2, 2);
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        SCOPED_TRACE(scenario);
        VectorR step = VectorR::Ones(2), gradient = step, work(2);
        MatrixR updated(2, 2);
        if (scenario == 0) step(0) = std::numeric_limits<float>::quiet_NaN();
        if (scenario == 1) gradient(0) = std::numeric_limits<float>::infinity();
        if (scenario == 2) gradient.setConstant(std::numeric_limits<float>::max());
        if (scenario == 3)
        {
            step << 1.0e30f, 1.0f;
            gradient << 0.0f, 1.0f;
        }
        EXPECT_FALSE(detail::bfgs_inverse_update(previous, step, gradient, updated, work));
        EXPECT_TRUE((updated.array() == previous.array()).all());
    }
}

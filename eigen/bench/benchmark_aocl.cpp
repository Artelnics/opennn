/*
 * benchmark_aocl.cpp - AOCL Performance Benchmark Suite for Eigen
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Description:
 * ------------
 * This benchmark suite evaluates the performance of Eigen mathematical
 * operations when integrated with AMD Optimizing CPU Libraries (AOCL). It
 * tests:
 *
 * 1. Vector Math Operations: Transcendental functions (exp, sin, cos, sqrt,
 * log, etc.) using AOCL Vector Math Library (VML) for optimized
 * double-precision operations
 *
 * 2. Matrix Operations: BLAS Level-3 operations (DGEMM) using AOCL BLAS library
 *    with support for both single-threaded and multithreaded execution
 *
 * 3. Linear Algebra: LAPACK operations (eigenvalue decomposition) using
 * libflame
 *
 * 4. Real-world Scenarios: Financial risk computation simulating covariance
 * matrix calculations and eigenvalue analysis for portfolio optimization
 *
 * The benchmark automatically detects AOCL configuration and adjusts test
 * execution accordingly, providing performance comparisons between standard
 * Eigen operations and AOCL-accelerated implementations.
 *
 * Compilation:
 * ------------
 * # Using AOCC compiler (recommended for best AOCL compatibility):
 * clang++ -O3 -g -DEIGEN_USE_AOCL_ALL -I<PATH_TO_EIGEN_INCLUDE>
 * -I${AOCL_ROOT}/include \
 *         -Wno-parentheses src/benchmark_aocl.cpp -L${AOCL_ROOT}/lib \
 *         -lamdlibm -lm -lblis -lflame -lpthread -lrt -pthread \
 *         -o build/eigen_aocl_benchmark
 *
 * # Alternative: Using GCC with proper library paths:
 * g++ -O3 -g -DEIGEN_USE_AOCL_ALL -I<PATH_TO_EIGEN_INCLUDE>
 * -I${AOCL_ROOT}/include \
 *     -Wno-parentheses src/benchmark_aocl.cpp -L${AOCL_ROOT}/lib \
 *     -lamdlibm -lm -lblis -lflame -lpthread -lrt \
 *     -o build/eigen_aocl_benchmark
 *
 * # For multithreaded BLIS support:
 * clang++ -O3 -g -fopenmp -DEIGEN_USE_AOCL_MT -I<PATH_TO_EIGEN_INCLUDE> \
 *         -I${AOCL_ROOT}/include -Wno-parentheses src/benchmark_aocl.cpp \
 *         -L${AOCL_ROOT}/lib -lamdlibm -lm -lblis-mt -lflame -lpthread -lrt \
 *         -o build/eigen_aocl_benchmark_mt
 *
 * Usage:
 * ------
 * export AOCL_ROOT=/path/to/aocl/installation
 * export LD_LIBRARY_PATH=$AOCL_ROOT/lib:$LD_LIBRARY_PATH
 * ./build/eigen_aocl_benchmark
 *
 * Developer:
 * ----------
 * Name: Sharad Saurabh Bhaskar
 * Email: shbhaska@amd.com
 * Organization: Advanced Micro Devices, Inc.
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

// Simple - just include Eigen headers
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Only include CBLAS if AOCL BLIS is available
#ifdef EIGEN_USE_AOCL_ALL
#include <cblas.h>
#endif

using namespace std;
using namespace std::chrono;
using namespace Eigen;

void benchmarkVectorMath(int size) {
  VectorXd v = VectorXd::LinSpaced(size, 0.1, 10.0);
  VectorXd result(size);
  double elapsed_ms = 0;

  cout << "\n--- Vector Math Benchmark (size = " << size << ") ---" << endl;

  auto start = high_resolution_clock::now();
  result = v.array().exp();
  auto end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "exp() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().sin();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "sin() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().cos();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "cos() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().sqrt();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "sqrt() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().cbrt();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "cbrt() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().abs();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "abs() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().log();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "log() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().log10();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "log10() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().exp2();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "exp2() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().asin();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "asin() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().sinh();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "sinh() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().acos();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "acos() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().cosh();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "cosh() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().tan();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "tan() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().atan();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "atan() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().tanh();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "tanh() time: " << elapsed_ms << " ms" << endl;

  VectorXd v2 = VectorXd::Random(size);
  start = high_resolution_clock::now();
  result = v.array() + v2.array();
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "add() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().pow(2.0);
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "pow() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().max(v2.array());
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "max() time: " << elapsed_ms << " ms" << endl;

  start = high_resolution_clock::now();
  result = v.array().min(v2.array());
  end = high_resolution_clock::now();
  elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "min() time: " << elapsed_ms << " ms" << endl;
}

// Function to benchmark BLAS operation: Matrix multiplication.
void benchmarkMatrixMultiplication(int matSize) {
  cout << "\n--- BLIS-st DGEMM Benchmark (" << matSize << " x " << matSize
       << ") ---" << endl;

  MatrixXd A = MatrixXd::Random(matSize, matSize);
  MatrixXd B = MatrixXd::Random(matSize, matSize);
  MatrixXd C(matSize, matSize);

  auto start = high_resolution_clock::now();
  C = A * B;
  auto end = high_resolution_clock::now();
  double elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "Matrix multiplication time: " << elapsed_ms << " ms" << endl;
}

// Benchmark BLIS directly using its CBLAS interface if available.
void benchmarkBlisMultithreaded(int matSize, int numThreads) {
#if defined(EIGEN_AOCL_USE_BLIS_MT)
  cout << "\n--- BLIS-mt DGEMM Benchmark (" << matSize << " x " << matSize
       << ", threads=" << numThreads << ") ---" << endl;
  vector<double> A(matSize * matSize);
  vector<double> B(matSize * matSize);
  vector<double> C(matSize * matSize);
  for (auto &v : A)
    v = static_cast<double>(rand()) / RAND_MAX;
  for (auto &v : B)
    v = static_cast<double>(rand()) / RAND_MAX;
  double alpha = 1.0, beta = 0.0;
  string th = to_string(numThreads);
  setenv("BLIS_NUM_THREADS", th.c_str(), 1);
  auto start = high_resolution_clock::now();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matSize, matSize,
              matSize, alpha, A.data(), matSize, B.data(), matSize, beta,
              C.data(), matSize);
  auto end = high_resolution_clock::now();
  double elapsed_ms = duration_cast<milliseconds>(end - start).count();
  cout << "BLIS dgemm time: " << elapsed_ms << " ms" << endl;
#else
  (void)matSize;
  (void)numThreads;
  cout << "\nBLIS multithreaded support not enabled." << endl;
#endif
}

// Function to benchmark LAPACK operation: Eigenvalue decomposition.
void benchmarkEigenDecomposition(int matSize) {
  cout << "\n--- Eigenvalue Decomposition Benchmark (Matrix Size: " << matSize
       << " x " << matSize << ") ---" << endl;
  MatrixXd M = MatrixXd::Random(matSize, matSize);
  // Make matrix symmetric (necessary for eigenvalue decomposition of
  // self-adjoint matrices)
  M = (M + M.transpose()) * 0.5;

  SelfAdjointEigenSolver<MatrixXd> eigensolver;
  auto start = high_resolution_clock::now();
  eigensolver.compute(M);
  auto end = high_resolution_clock::now();
  double elapsed_ms = duration_cast<milliseconds>(end - start).count();
  if (eigensolver.info() == Success) {
    cout << "Eigenvalue decomposition time: " << elapsed_ms << " ms" << endl;
  } else {
    cout << "Eigenvalue decomposition failed." << endl;
  }
}

// Function simulating a real-world FSI risk computation scenario.
// Example: Compute covariance matrix from simulated asset returns, then perform
// eigenvalue decomposition.
void benchmarkFSIRiskComputation(int numPeriods, int numAssets) {
  cout << "\n--- FSI Risk Computation Benchmark ---" << endl;
  cout << "Simulating " << numPeriods << " periods for " << numAssets
       << " assets." << endl;

  // Simulate asset returns: each column represents an asset's returns.
  MatrixXd returns = MatrixXd::Random(numPeriods, numAssets);

  // Compute covariance matrix: cov = (returns^T * returns) / (numPeriods - 1)
  auto start = high_resolution_clock::now();
  MatrixXd cov = (returns.transpose() * returns) / (numPeriods - 1);
  auto end = high_resolution_clock::now();
  double cov_time = duration_cast<milliseconds>(end - start).count();
  cout << "Covariance matrix computation time: " << cov_time << " ms" << endl;

  // Eigenvalue decomposition on covariance matrix.
  SelfAdjointEigenSolver<MatrixXd> eigensolver;
  start = high_resolution_clock::now();
  eigensolver.compute(cov);
  end = high_resolution_clock::now();
  double eig_time = duration_cast<milliseconds>(end - start).count();
  if (eigensolver.info() == Success) {
    cout << "Eigenvalue decomposition (covariance) time: " << eig_time << " ms"
         << endl;
    cout << "Top 3 Eigenvalues: "
         << eigensolver.eigenvalues().tail(3).transpose() << endl;
  } else {
    cout << "Eigenvalue decomposition failed." << endl;
  }
}

int main() {
  cout << "=== AOCL Benchmark for Eigen on AMD Platforms ===" << endl;
  cout << "Developer: Sharad Saurabh Bhaskar (shbhaska@amd.com)" << endl;
  cout << "Organization: Advanced Micro Devices, Inc." << endl;
  cout << "License: Mozilla Public License 2.0" << endl << endl;

  // Print AOCL configuration
#ifdef EIGEN_USE_AOCL_MT
  cout << "AOCL Mode: MULTITHREADED (MT)" << endl;
  cout << "Features: Multithreaded BLIS, AOCL VML, LAPACK" << endl;
#elif defined(EIGEN_USE_AOCL_ALL)
  cout << "AOCL Mode: SINGLE-THREADED (ALL)" << endl;
  cout << "Features: Single-threaded BLIS, AOCL VML, LAPACK" << endl;
#else
  cout << "AOCL Mode: DISABLED" << endl;
  cout << "Using standard Eigen implementation" << endl;
#endif
  cout << "Hardware threads available: " << thread::hardware_concurrency() << endl << endl;

  // Benchmark vector math functions with varying vector sizes.
  vector<int> vectorSizes = {5000000, 10000000, 50000000};
  for (int size : vectorSizes) {
    benchmarkVectorMath(size);
  }

  // Benchmark matrix multiplication for varying sizes.
  vector<int> matrixSizes = {1024};
  for (int msize : matrixSizes) {
    benchmarkMatrixMultiplication(msize);
#if defined(EIGEN_AOCL_USE_BLIS_MT)
    benchmarkBlisMultithreaded(msize, thread::hardware_concurrency());
#endif
  }

  // Benchmark LAPACK: Eigenvalue Decomposition.
  for (int msize : matrixSizes) {
    benchmarkEigenDecomposition(msize);
  }

  // Benchmark a complex FSI risk computation scenario.
  // For example, simulate 10,000 time periods (days) for 500 assets.
  benchmarkFSIRiskComputation(10000, 500);

  cout << "\n=== Benchmark Complete ===" << endl;
  return 0;
}

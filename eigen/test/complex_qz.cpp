// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"

#include <Eigen/Eigenvalues>

/* this test covers the following files:
   ComplexQZ.h
*/

template <typename MatrixType>
void generate_random_matrix_pair(const Index dim, MatrixType& A, MatrixType& B) {
  A.resize(dim, dim);
  B.resize(dim, dim);
  A.setRandom();
  B.setRandom();
  // Set each row of B with a probability of 10% to 0
  for (int i = 0; i < dim; i++) {
    if (internal::random<int>(0, 10) == 0) B.row(i).setZero();
  }
}

template <typename MatrixType>
void complex_qz(const MatrixType& A, const MatrixType& B) {
  using std::abs;
  const Index dim = A.rows();
  ComplexQZ<MatrixType> qz(A, B);
  VERIFY_IS_EQUAL(qz.info(), Success);
  auto T = qz.matrixT(), S = qz.matrixS();
  bool is_all_zero_T = true, is_all_zero_S = true;
  using RealScalar = typename MatrixType::RealScalar;
  RealScalar tol = dim * 10 * NumTraits<RealScalar>::epsilon();
  for (Index j = 0; j < dim; j++) {
    for (Index i = j + 1; i < dim; i++) {
      if (std::abs(T(i, j)) > tol) {
        std::cerr << std::abs(T(i, j)) << std::endl;
        is_all_zero_T = false;
      }
      if (std::abs(S(i, j)) > tol) {
        std::cerr << std::abs(S(i, j)) << std::endl;
        is_all_zero_S = false;
      }
    }
  }
  VERIFY_IS_EQUAL(is_all_zero_T, true);
  VERIFY_IS_EQUAL(is_all_zero_S, true);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixS() * qz.matrixZ(), A);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixT() * qz.matrixZ(), B);
  VERIFY_IS_APPROX(qz.matrixQ() * qz.matrixQ().adjoint(), MatrixType::Identity(dim, dim));
  VERIFY_IS_APPROX(qz.matrixZ() * qz.matrixZ().adjoint(), MatrixType::Identity(dim, dim));
}

EIGEN_DECLARE_TEST(complex_qz) {
  // const Index dim1 = 15;
  // const Index dim2 = 80;
  for (int i = 0; i < g_repeat; i++) {
    // Check for very small, fixed-sized double- and float complex matrices
    Eigen::Matrix2cd A_2x2, B_2x2;
    A_2x2.setRandom();
    B_2x2.setRandom();
    B_2x2.row(1).setZero();
    Eigen::Matrix3cf A_3x3, B_3x3;
    A_3x3.setRandom();
    B_3x3.setRandom();
    B_3x3.col(i % 3).setRandom();
    // Test for small float complex matrices
    Eigen::MatrixXcf A_float, B_float;
    const Index dim1 = internal::random<Index>(15, 80), dim2 = internal::random<Index>(15, 80);
    generate_random_matrix_pair(dim1, A_float, B_float);
    // Test for a bit larger double complex matrices
    Eigen::MatrixXcd A_double, B_double;
    generate_random_matrix_pair(dim2, A_double, B_double);
    CALL_SUBTEST_1(complex_qz(A_2x2, B_2x2));
    CALL_SUBTEST_2(complex_qz(A_3x3, B_3x3));
    CALL_SUBTEST_3(complex_qz(A_float, B_float));
    CALL_SUBTEST_4(complex_qz(A_double, B_double));
  }
}

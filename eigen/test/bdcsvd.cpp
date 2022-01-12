// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
#include <Eigen/SVD>
#include <iostream>
#include <Eigen/LU>


#define SVD_DEFAULT(M) BDCSVD<M>
#define SVD_FOR_MIN_NORM(M) BDCSVD<M>
#include "svd_common.h"

// Check all variants of JacobiSVD
template<typename MatrixType>
void bdcsvd(const MatrixType& a = MatrixType(), bool pickrandom = true)
{
  MatrixType m;
  if(pickrandom) {
    m.resizeLike(a);
    svd_fill_random(m);
  }
  else
    m = a;

  CALL_SUBTEST(( svd_test_all_computation_options<BDCSVD<MatrixType> >(m, false)  ));
}

template<typename MatrixType>
void bdcsvd_method()
{
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.bdcSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixV());
  VERIFY_IS_APPROX(m.bdcSvd(ComputeFullU|ComputeFullV).solve(m), m);
  VERIFY_IS_APPROX(m.bdcSvd(ComputeFullU|ComputeFullV).transpose().solve(m), m);
  VERIFY_IS_APPROX(m.bdcSvd(ComputeFullU|ComputeFullV).adjoint().solve(m), m);
}

// Compare the Singular values returned with Jacobi and Bdc.
template<typename MatrixType>
void compare_bdc_jacobi(const MatrixType& a = MatrixType(), unsigned int computationOptions = 0, int algoswap = 16, bool random = true)
{
  MatrixType m = random ? MatrixType::Random(a.rows(), a.cols()) : a;

  BDCSVD<MatrixType> bdc_svd(m.rows(), m.cols(), computationOptions);
  bdc_svd.setSwitchSize(algoswap);
  bdc_svd.compute(m);

  JacobiSVD<MatrixType> jacobi_svd(m);
  VERIFY_IS_APPROX(bdc_svd.singularValues(), jacobi_svd.singularValues());

  if(computationOptions & ComputeFullU) VERIFY_IS_APPROX(bdc_svd.matrixU(), jacobi_svd.matrixU());
  if(computationOptions & ComputeThinU) VERIFY_IS_APPROX(bdc_svd.matrixU(), jacobi_svd.matrixU());
  if(computationOptions & ComputeFullV) VERIFY_IS_APPROX(bdc_svd.matrixV(), jacobi_svd.matrixV());
  if(computationOptions & ComputeThinV) VERIFY_IS_APPROX(bdc_svd.matrixV(), jacobi_svd.matrixV());
}

// Verifies total deflation is **not** triggered.
void compare_bdc_jacobi_instance(bool structure_as_m, int algoswap = 16)
{
  MatrixXd m(4, 3);
  if (structure_as_m) {
    // The first 3 rows are the reduced form of Matrix 1 as shown below, and it
    // has nonzero elements in the first column and diagonals only.
    m << 1.056293, 0, 0,
         -0.336468, 0.907359, 0,
         -1.566245, 0, 0.149150,
         -0.1, 0, 0;
  } else {
    // Matrix 1.
    m << 0.882336, 18.3914, -26.7921,
         -5.58135, 17.1931, -24.0892,
         -20.794, 8.68496, -4.83103,
         -8.4981, -10.5451, 23.9072;
  }
  compare_bdc_jacobi(m, 0, algoswap, false);
}

EIGEN_DECLARE_TEST(bdcsvd)
{
  CALL_SUBTEST_3(( svd_verify_assert<BDCSVD<Matrix3f>  >(Matrix3f()) ));
  CALL_SUBTEST_4(( svd_verify_assert<BDCSVD<Matrix4d>  >(Matrix4d()) ));
  CALL_SUBTEST_7(( svd_verify_assert<BDCSVD<MatrixXf>  >(MatrixXf(10,12)) ));
  CALL_SUBTEST_8(( svd_verify_assert<BDCSVD<MatrixXcd> >(MatrixXcd(7,5)) ));

  CALL_SUBTEST_101(( svd_all_trivial_2x2(bdcsvd<Matrix2cd>) ));
  CALL_SUBTEST_102(( svd_all_trivial_2x2(bdcsvd<Matrix2d>) ));

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_3(( bdcsvd<Matrix3f>() ));
    CALL_SUBTEST_4(( bdcsvd<Matrix4d>() ));
    CALL_SUBTEST_5(( bdcsvd<Matrix<float,3,5> >() ));

    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE/2),
        c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE/2);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    CALL_SUBTEST_6((  bdcsvd(Matrix<double,Dynamic,2>(r,2)) ));
    CALL_SUBTEST_7((  bdcsvd(MatrixXf(r,c)) ));
    CALL_SUBTEST_7((  compare_bdc_jacobi(MatrixXf(r,c)) ));
    CALL_SUBTEST_10(( bdcsvd(MatrixXd(r,c)) ));
    CALL_SUBTEST_10(( compare_bdc_jacobi(MatrixXd(r,c)) ));
    CALL_SUBTEST_8((  bdcsvd(MatrixXcd(r,c)) ));
    CALL_SUBTEST_8((  compare_bdc_jacobi(MatrixXcd(r,c)) ));

    // Test on inf/nan matrix
    CALL_SUBTEST_7(  (svd_inf_nan<BDCSVD<MatrixXf>, MatrixXf>()) );
    CALL_SUBTEST_10( (svd_inf_nan<BDCSVD<MatrixXd>, MatrixXd>()) );
  }

  // test matrixbase method
  CALL_SUBTEST_1(( bdcsvd_method<Matrix2cd>() ));
  CALL_SUBTEST_3(( bdcsvd_method<Matrix3f>() ));

  // Test problem size constructors
  CALL_SUBTEST_7( BDCSVD<MatrixXf>(10,10) );

  // Check that preallocation avoids subsequent mallocs
  // Disabled because not supported by BDCSVD
  // CALL_SUBTEST_9( svd_preallocate<void>() );

  CALL_SUBTEST_2( svd_underoverflow<void>() );

  // Without total deflation issues.
  CALL_SUBTEST_11((  compare_bdc_jacobi_instance(true) ));
  CALL_SUBTEST_12((  compare_bdc_jacobi_instance(false) ));

  // With total deflation issues before, when it shouldn't be triggered.
  CALL_SUBTEST_13((  compare_bdc_jacobi_instance(true, 3) ));
  CALL_SUBTEST_14((  compare_bdc_jacobi_instance(false, 3) ));
}

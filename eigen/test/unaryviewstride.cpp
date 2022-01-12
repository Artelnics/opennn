// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Andrew Johnson <andrew.johnson@arjohnsonau.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<int OuterStride,int InnerStride,typename VectorType> void unaryview_stride(const VectorType& m)
{
  typedef typename VectorType::Scalar Scalar;
  Index rows = m.rows();
  Index cols = m.cols();
  VectorType vec = VectorType::Random(rows, cols);

  struct view_op {
    EIGEN_EMPTY_STRUCT_CTOR(view_op)
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar& 
      operator()(const Scalar& v) const { return v; }
  };

  CwiseUnaryView<view_op, VectorType, Stride<OuterStride,InnerStride>> vec_view(vec);
  VERIFY(vec_view.outerStride() == (OuterStride == 0 ? 0 : OuterStride));
  VERIFY(vec_view.innerStride() == (InnerStride == 0 ? 1 : InnerStride));
}

EIGEN_DECLARE_TEST(unaryviewstride)
{
    CALL_SUBTEST_1(( unaryview_stride<1,2>(MatrixXf()) ));
    CALL_SUBTEST_1(( unaryview_stride<0,0>(MatrixXf()) ));
    CALL_SUBTEST_2(( unaryview_stride<1,2>(VectorXf()) ));
    CALL_SUBTEST_2(( unaryview_stride<0,0>(VectorXf()) ));
    CALL_SUBTEST_3(( unaryview_stride<1,2>(RowVectorXf()) ));
    CALL_SUBTEST_3(( unaryview_stride<0,0>(RowVectorXf()) ));
}

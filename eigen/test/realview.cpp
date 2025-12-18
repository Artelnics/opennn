// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

// wrapper that disables array-oriented access to real and imaginary components
struct TestComplex : public std::complex<float> {
  TestComplex() = default;
  TestComplex(const TestComplex&) = default;
  TestComplex(std::complex<float> x) : std::complex<float>(x){};
  TestComplex(float x) : std::complex<float>(x){};
};
template <>
struct NumTraits<TestComplex> : NumTraits<std::complex<float>> {};
template <>
struct internal::random_impl<TestComplex> : internal::random_impl<std::complex<float>> {};

template <typename T>
void test_realview_readonly(const T&) {
  using Scalar = typename T::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  constexpr Index minRows = T::RowsAtCompileTime == Dynamic ? 1 : T::RowsAtCompileTime;
  constexpr Index maxRows = T::MaxRowsAtCompileTime == Dynamic ? (EIGEN_TEST_MAX_SIZE / 2) : T::MaxRowsAtCompileTime;
  constexpr Index minCols = T::ColsAtCompileTime == Dynamic ? 1 : T::ColsAtCompileTime;
  constexpr Index maxCols = T::MaxColsAtCompileTime == Dynamic ? (EIGEN_TEST_MAX_SIZE / 2) : T::MaxColsAtCompileTime;

  constexpr Index rowFactor = (NumTraits<Scalar>::IsComplex && !T::IsRowMajor) ? 2 : 1;
  constexpr Index colFactor = (NumTraits<Scalar>::IsComplex && T::IsRowMajor) ? 2 : 1;
  constexpr Index sizeFactor = NumTraits<Scalar>::IsComplex ? 2 : 1;

  Index rows = internal::random<Index>(minRows, maxRows);
  Index cols = internal::random<Index>(minCols, maxCols);

  T A(rows, cols), B(rows, cols);

  VERIFY(A.realView().rows() == rowFactor * A.rows());
  VERIFY(A.realView().cols() == colFactor * A.cols());
  VERIFY(A.realView().size() == sizeFactor * A.size());

  A.setRandom();
  VERIFY_IS_APPROX(A.matrix().cwiseAbs2().sum(), A.realView().matrix().cwiseAbs2().sum());

  RealScalar alpha = internal::random(RealScalar(1), RealScalar(2));

  // B = A * alpha
  for (Index r = 0; r < rows; r++) {
    for (Index c = 0; c < cols; c++) {
      B.coeffRef(r, c) = A.coeff(r, c) * Scalar(alpha);
    }
  }
  VERIFY_IS_CWISE_APPROX(B.realView(), A.realView() * alpha);

  // B = A / alpha
  for (Index r = 0; r < rows; r++) {
    for (Index c = 0; c < cols; c++) {
      B.coeffRef(r, c) = A.coeff(r, c) / Scalar(alpha);
    }
  }
  VERIFY_IS_CWISE_APPROX(B.realView(), A.realView() / alpha);
}

template <typename T>
void test_realview(const T&) {
  using Scalar = typename T::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  constexpr Index minRows = T::RowsAtCompileTime == Dynamic ? 1 : T::RowsAtCompileTime;
  constexpr Index maxRows = T::MaxRowsAtCompileTime == Dynamic ? (EIGEN_TEST_MAX_SIZE / 2) : T::MaxRowsAtCompileTime;
  constexpr Index minCols = T::ColsAtCompileTime == Dynamic ? 1 : T::ColsAtCompileTime;
  constexpr Index maxCols = T::MaxColsAtCompileTime == Dynamic ? (EIGEN_TEST_MAX_SIZE / 2) : T::MaxColsAtCompileTime;

  constexpr Index rowFactor = (NumTraits<Scalar>::IsComplex && !T::IsRowMajor) ? 2 : 1;
  constexpr Index colFactor = (NumTraits<Scalar>::IsComplex && T::IsRowMajor) ? 2 : 1;
  constexpr Index sizeFactor = NumTraits<Scalar>::IsComplex ? 2 : 1;

  const Index rows = internal::random<Index>(minRows, maxRows);
  const Index cols = internal::random<Index>(minCols, maxCols);
  const Index realViewRows = rowFactor * rows;
  const Index realViewCols = colFactor * cols;

  const T A = T::Random(rows, cols);
  T B;

  VERIFY_IS_EQUAL(A.realView().rows(), rowFactor * A.rows());
  VERIFY_IS_EQUAL(A.realView().cols(), colFactor * A.cols());
  VERIFY_IS_EQUAL(A.realView().size(), sizeFactor * A.size());

  VERIFY_IS_APPROX(A.matrix().cwiseAbs2().sum(), A.realView().matrix().cwiseAbs2().sum());

  // test re-sizing realView during assignment
  B.realView() = A.realView();
  VERIFY_IS_APPROX(A, B);
  VERIFY_IS_APPROX(A.realView(), B.realView());

  const RealScalar alpha = internal::random(RealScalar(1), RealScalar(2));

  // B = A * alpha
  for (Index r = 0; r < rows; r++) {
    for (Index c = 0; c < cols; c++) {
      B.coeffRef(r, c) = A.coeff(r, c) * Scalar(alpha);
    }
  }
  VERIFY_IS_APPROX(B.realView(), A.realView() * alpha);

  B = A;
  B.realView() *= alpha;
  VERIFY_IS_APPROX(B.realView(), A.realView() * alpha);

  // B = A / alpha
  for (Index r = 0; r < rows; r++) {
    for (Index c = 0; c < cols; c++) {
      B.coeffRef(r, c) = A.coeff(r, c) / Scalar(alpha);
    }
  }
  VERIFY_IS_APPROX(B.realView(), A.realView() / alpha);

  B = A;
  B.realView() /= alpha;
  VERIFY_IS_APPROX(B.realView(), A.realView() / alpha);

  // force some usual access patterns
  Index malloc_size = (rows * cols * sizeof(Scalar)) + sizeof(RealScalar);
  void* data1 = internal::aligned_malloc(malloc_size);
  void* data2 = internal::aligned_malloc(malloc_size);
  Scalar* ptr1 = reinterpret_cast<Scalar*>(reinterpret_cast<uint8_t*>(data1) + sizeof(RealScalar));
  Scalar* ptr2 = reinterpret_cast<Scalar*>(reinterpret_cast<uint8_t*>(data2) + sizeof(RealScalar));
  Map<T> C(ptr1, rows, cols), D(ptr2, rows, cols);

  C.setRandom();
  D.setRandom();
  for (Index r = 0; r < realViewRows; r++) {
    for (Index c = 0; c < realViewCols; c++) {
      C.realView().coeffRef(r, c) = D.realView().coeff(r, c);
    }
  }
  VERIFY_IS_CWISE_EQUAL(C, D);

  C = A;

  for (Index c = 0; c < realViewCols - 1; c++) {
    B.realView().row(0).coeffRef(realViewCols - 1 - c) = C.realView().row(0).coeff(c + 1);
  }
  D.realView().row(0).tail(realViewCols - 1) = C.realView().row(0).tail(realViewCols - 1).reverse();
  VERIFY_IS_CWISE_EQUAL(B.realView().row(0).tail(realViewCols - 1), D.realView().row(0).tail(realViewCols - 1));

  for (Index r = 0; r < realViewRows - 1; r++) {
    B.realView().col(0).coeffRef(realViewRows - 1 - r) = C.realView().col(0).coeff(r + 1);
  }
  D.realView().col(0).tail(realViewRows - 1) = C.realView().col(0).tail(realViewRows - 1).reverse();
  VERIFY_IS_CWISE_EQUAL(B.realView().col(0).tail(realViewRows - 1), D.realView().col(0).tail(realViewRows - 1));
}

template <typename ComplexScalar, bool Enable = internal::packet_traits<ComplexScalar>::Vectorizable>
struct test_edge_cases_impl {
  static void run() {
    using namespace internal;
    using RealScalar = typename NumTraits<ComplexScalar>::Real;
    using ComplexPacket = typename packet_traits<ComplexScalar>::type;
    using RealPacket = typename unpacket_traits<ComplexPacket>::as_real;
    constexpr int ComplexSize = unpacket_traits<ComplexPacket>::size;
    constexpr int RealSize = 2 * ComplexSize;
    VectorX<ComplexScalar> a_data(2 * ComplexSize);
    Map<const VectorX<RealScalar>> a_data_asreal(reinterpret_cast<const RealScalar*>(a_data.data()), 2 * a_data.size());
    VectorX<RealScalar> b_data(RealSize);

    a_data.setRandom();
    evaluator<RealView<VectorX<ComplexScalar>>> eval(a_data.realView());

    for (Index offset = 0; offset < RealSize; offset++) {
      for (Index begin = 0; offset + begin < RealSize; begin++) {
        for (Index count = 0; begin + count < RealSize; count++) {
          b_data.setRandom();
          RealPacket res = eval.template packetSegment<Unaligned, RealPacket>(offset, begin, count);
          pstoreSegment(b_data.data(), res, begin, count);
          VERIFY_IS_CWISE_EQUAL(a_data_asreal.segment(offset + begin, count), b_data.segment(begin, count));
        }
      }
    }
  }
};

template <typename ComplexScalar>
struct test_edge_cases_impl<ComplexScalar, false> {
  static void run() {}
};

template <typename ComplexScalar>
void test_edge_cases(const ComplexScalar&) {
  test_edge_cases_impl<ComplexScalar>::run();
}

template <typename Scalar, int Rows, int Cols, int MaxRows = Rows, int MaxCols = Cols>
void test_realview_readonly() {
  // if Rows == 1, don't test ColMajor as it is not a valid array
  using ColMajorMatrixType = Matrix<Scalar, Rows, Cols, Rows == 1 ? RowMajor : ColMajor, MaxRows, MaxCols>;
  using ColMajorArrayType = Array<Scalar, Rows, Cols, Rows == 1 ? RowMajor : ColMajor, MaxRows, MaxCols>;
  // if Cols == 1, don't test RowMajor as it is not a valid array
  using RowMajorMatrixType = Matrix<Scalar, Rows, Cols, Cols == 1 ? ColMajor : RowMajor, MaxRows, MaxCols>;
  using RowMajorArrayType = Array<Scalar, Rows, Cols, Cols == 1 ? ColMajor : RowMajor, MaxRows, MaxCols>;
  test_realview_readonly(ColMajorMatrixType());
  test_realview_readonly(ColMajorArrayType());
  test_realview_readonly(RowMajorMatrixType());
  test_realview_readonly(RowMajorArrayType());
}

template <typename Scalar, int Rows, int Cols, int MaxRows = Rows, int MaxCols = Cols>
void test_realview_readwrite() {
  // if Rows == 1, don't test ColMajor as it is not a valid array
  using ColMajorMatrixType = Matrix<Scalar, Rows, Cols, Rows == 1 ? RowMajor : ColMajor, MaxRows, MaxCols>;
  using ColMajorArrayType = Array<Scalar, Rows, Cols, Rows == 1 ? RowMajor : ColMajor, MaxRows, MaxCols>;
  // if Cols == 1, don't test RowMajor as it is not a valid array
  using RowMajorMatrixType = Matrix<Scalar, Rows, Cols, Cols == 1 ? ColMajor : RowMajor, MaxRows, MaxCols>;
  using RowMajorArrayType = Array<Scalar, Rows, Cols, Cols == 1 ? ColMajor : RowMajor, MaxRows, MaxCols>;
  test_realview(ColMajorMatrixType());
  test_realview(ColMajorArrayType());
  test_realview(RowMajorMatrixType());
  test_realview(RowMajorArrayType());
}

template <int Rows, int Cols, int MaxRows = Rows, int MaxCols = Cols>
void test_realview() {
  test_realview_readwrite<float, Rows, Cols, MaxRows, MaxCols>();
  test_realview_readwrite<std::complex<float>, Rows, Cols, MaxRows, MaxCols>();
  test_realview_readwrite<double, Rows, Cols, MaxRows, MaxCols>();
  test_realview_readwrite<std::complex<double>, Rows, Cols, MaxRows, MaxCols>();
  test_realview_readwrite<long double, Rows, Cols, MaxRows, MaxCols>();
  test_realview_readwrite<std::complex<long double>, Rows, Cols, MaxRows, MaxCols>();
  test_realview_readonly<TestComplex, Rows, Cols, MaxRows, MaxCols>();
}

EIGEN_DECLARE_TEST(realview) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((test_realview<Dynamic, Dynamic, Dynamic, Dynamic>()));
    CALL_SUBTEST_2((test_realview<Dynamic, Dynamic, 17, Dynamic>()));
    CALL_SUBTEST_3((test_realview<Dynamic, Dynamic, Dynamic, 19>()));
    CALL_SUBTEST_4((test_realview<Dynamic, Dynamic, 17, 19>()));
    CALL_SUBTEST_5((test_realview<17, Dynamic, 17, Dynamic>()));
    CALL_SUBTEST_6((test_realview<Dynamic, 19, Dynamic, 19>()));
    CALL_SUBTEST_7((test_realview<17, 19, 17, 19>()));
    CALL_SUBTEST_8((test_realview<Dynamic, 1>()));
    CALL_SUBTEST_9((test_realview<1, Dynamic>()));
    CALL_SUBTEST_10((test_realview<1, 1>()));
    CALL_SUBTEST_11(test_edge_cases(std::complex<float>()));
    CALL_SUBTEST_12(test_edge_cases(std::complex<double>()));
  }
}

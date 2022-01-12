// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <vector>
#include <Eigen/Core>

struct MyPodType {
  double x;
  int y;
  float z;
};

// Plain-old-data serialization.
void test_pod_type() {
  MyPodType initial = {1.3, 17, 1.9f};
  MyPodType clone = {-1, -1, -1};
  
  Eigen::Serializer<MyPodType> serializer;
  
  // Determine required size.
  size_t buffer_size = serializer.size(initial);
  VERIFY_IS_EQUAL(buffer_size, sizeof(MyPodType));
  
  // Serialize.
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = serializer.serialize(begin, end, initial);
  VERIFY(dest != nullptr);
  VERIFY_IS_EQUAL(dest - begin, buffer_size);
  
  // Deserialize.
  const uint8_t* src = serializer.deserialize(begin, end, clone);
  VERIFY(src != nullptr);
  VERIFY_IS_EQUAL(src - begin, buffer_size);
  VERIFY_IS_EQUAL(clone.x, initial.x);
  VERIFY_IS_EQUAL(clone.y, initial.y);
  VERIFY_IS_EQUAL(clone.z, initial.z);

  // Serialize with bounds checking errors.
  dest = serializer.serialize(begin, end - 1, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(begin, begin, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(nullptr, nullptr, initial);
  VERIFY(dest == nullptr);

  // Deserialize with bounds checking errors.
  src = serializer.deserialize(begin, end - 1, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(begin, begin, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(nullptr, nullptr, clone);
  VERIFY(src == nullptr);
}

// Matrix, Vector, Array
template<typename T>
void test_eigen_type(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  const T initial = T::Random(rows, cols);
  
  // Serialize.
  Eigen::Serializer<T> serializer;
  size_t buffer_size = serializer.size(initial);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = serializer.serialize(begin, end, initial);
  VERIFY(dest != nullptr);
  VERIFY_IS_EQUAL(dest - begin, buffer_size);
  
  // Deserialize.
  T clone;
  const uint8_t* src = serializer.deserialize(begin, end, clone);
  VERIFY(src != nullptr);
  VERIFY_IS_EQUAL(src - begin, buffer_size);
  VERIFY_IS_CWISE_EQUAL(clone, initial);

  // Serialize with bounds checking errors.
  dest = serializer.serialize(begin, end - 1, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(begin, begin, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(nullptr, nullptr, initial);
  VERIFY(dest == nullptr);

  // Deserialize with bounds checking errors.
  src = serializer.deserialize(begin, end - 1, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(begin, begin, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(nullptr, nullptr, clone);
  VERIFY(src == nullptr);
}

// Test a collection of dense types.
template<typename T1, typename T2, typename T3>
void test_dense_types(const T1& type1, const T2& type2, const T3& type3) {
  
  // Make random inputs.
  const T1 x1 = T1::Random(type1.rows(), type1.cols());
  const T2 x2 = T2::Random(type2.rows(), type2.cols());
  const T3 x3 = T3::Random(type3.rows(), type3.cols());
  
  // Allocate buffer and serialize.
  size_t buffer_size = Eigen::serialize_size(x1, x2, x3);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = Eigen::serialize(begin, end, x1, x2, x3);
  VERIFY(dest != nullptr);
  
  // Clone everything.
  T1 y1;
  T2 y2;
  T3 y3;
  const uint8_t* src = Eigen::deserialize(begin, end, y1, y2, y3);
  VERIFY(src != nullptr);

  // Verify they equal.
  VERIFY_IS_CWISE_EQUAL(y1, x1);
  VERIFY_IS_CWISE_EQUAL(y2, x2);
  VERIFY_IS_CWISE_EQUAL(y3, x3);

  // Serialize everything with bounds checking errors.
  dest = Eigen::serialize(begin, end - 1, y1, y2, y3);
  VERIFY(dest == nullptr);
  dest = Eigen::serialize(begin, begin, y1, y2, y3);
  VERIFY(dest == nullptr);
  dest = Eigen::serialize(nullptr, nullptr, y1, y2, y3);
  VERIFY(dest == nullptr);

  // Deserialize everything with bounds checking errors.
  src = Eigen::deserialize(begin, end - 1, y1, y2, y3);
  VERIFY(src == nullptr);
  src = Eigen::deserialize(begin, begin, y1, y2, y3);
  VERIFY(src == nullptr);
  src = Eigen::deserialize(nullptr, nullptr, y1, y2, y3);
  VERIFY(src == nullptr);
}

EIGEN_DECLARE_TEST(serializer)
{
  CALL_SUBTEST( test_pod_type() );

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( test_eigen_type(Eigen::Array33f()) );
    CALL_SUBTEST( test_eigen_type(Eigen::ArrayXd(10)) );
    CALL_SUBTEST( test_eigen_type(Eigen::Vector3f()) );
    CALL_SUBTEST( test_eigen_type(Eigen::Matrix4d()) );
    CALL_SUBTEST( test_eigen_type(Eigen::MatrixXd(15, 17)) );
    
    CALL_SUBTEST( test_dense_types( Eigen::Array33f(),
                                    Eigen::ArrayXd(10),
                                    Eigen::MatrixXd(15, 17)) );
  }
}

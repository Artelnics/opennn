// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATH_FUNCTIONS_CLANG_H
#define EIGEN_MATH_FUNCTIONS_CLANG_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <>
EIGEN_STRONG_INLINE Packet16f pfrexp<Packet16f>(const Packet16f& a, Packet16f& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE Packet8d pfrexp<Packet8d>(const Packet8d& a, Packet8d& exponent) {
  return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE Packet16f pldexp<Packet16f>(const Packet16f& a, const Packet16f& exponent) {
  return pldexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE Packet8d pldexp<Packet8d>(const Packet8d& a, const Packet8d& exponent) {
  return pldexp_generic(a, exponent);
}

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet16f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet8d)

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_CLANG_H

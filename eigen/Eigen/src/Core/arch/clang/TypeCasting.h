// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_CLANG_H
#define EIGEN_TYPE_CASTING_CLANG_H

namespace Eigen {
namespace internal {

//==============================================================================
// preinterpret
//==============================================================================
template <>
EIGEN_STRONG_INLINE Packet16f preinterpret<Packet16f, Packet16i>(const Packet16i& a) {
  return reinterpret_cast<Packet16f>(a);
}
template <>
EIGEN_STRONG_INLINE Packet16i preinterpret<Packet16i, Packet16f>(const Packet16f& a) {
  return reinterpret_cast<Packet16i>(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d preinterpret<Packet8d, Packet8l>(const Packet8l& a) {
  return reinterpret_cast<Packet8d>(a);
}
template <>
EIGEN_STRONG_INLINE Packet8l preinterpret<Packet8l, Packet8d>(const Packet8d& a) {
  return reinterpret_cast<Packet8l>(a);
}

//==============================================================================
// pcast
//==============================================================================
#if EIGEN_HAS_BUILTIN(__builtin_convertvector)
template <>
EIGEN_STRONG_INLINE Packet16i pcast<Packet16f, Packet16i>(const Packet16f& a) {
  return __builtin_convertvector(a, Packet16i);
}
template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16i, Packet16f>(const Packet16i& a) {
  return __builtin_convertvector(a, Packet16f);
}

template <>
EIGEN_STRONG_INLINE Packet8l pcast<Packet8d, Packet8l>(const Packet8d& a) {
  return __builtin_convertvector(a, Packet8l);
}
template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet8l, Packet8d>(const Packet8l& a) {
  return __builtin_convertvector(a, Packet8d);
}
#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_TYPE_CASTING_CLANG_H

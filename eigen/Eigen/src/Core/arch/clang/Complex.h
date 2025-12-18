// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_CLANG_H
#define EIGEN_COMPLEX_CLANG_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <typename RealScalar, int N>
struct complex_packet_wrapper {
  using RealPacketT = detail::VectorType<RealScalar, 2 * N>;
  EIGEN_STRONG_INLINE complex_packet_wrapper() = default;
  EIGEN_STRONG_INLINE explicit complex_packet_wrapper(const RealPacketT& a) : v(a) {}
  EIGEN_STRONG_INLINE constexpr std::complex<RealScalar> operator[](Index i) const {
    return std::complex<RealScalar>(v[2 * i], v[2 * i + 1]);
  }
  RealPacketT v;
};

using Packet8cf = complex_packet_wrapper<float, 8>;
using Packet4cf = complex_packet_wrapper<float, 4>;
using Packet2cf = complex_packet_wrapper<float, 2>;
using Packet4cd = complex_packet_wrapper<double, 4>;
using Packet2cd = complex_packet_wrapper<double, 2>;

struct generic_complex_packet_traits : default_packet_traits {
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasArg = 0,
    HasSetLinear = 0,
    HasConj = 1,
    // Math functions
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
  };
};

template <>
struct packet_traits<std::complex<float>> : generic_complex_packet_traits {
  using type = Packet8cf;
  using half = Packet8cf;
  enum {
    size = 8,
  };
};

template <>
struct unpacket_traits<Packet8cf> : generic_unpacket_traits {
  using type = std::complex<float>;
  using half = Packet8cf;
  using as_real = Packet16f;
  enum {
    size = 8,
  };
};

template <>
struct packet_traits<std::complex<double>> : generic_complex_packet_traits {
  using type = Packet4cd;
  using half = Packet4cd;
  enum {
    size = 4,
    HasExp = 0,  // FIXME(rmlarsen): pexp_complex is broken for double.
  };
};

template <>
struct unpacket_traits<Packet4cd> : generic_unpacket_traits {
  using type = std::complex<double>;
  using half = Packet4cd;
  using as_real = Packet8d;
  enum {
    size = 4,
  };
};

// ------------ Load and store ops ----------
#define EIGEN_CLANG_COMPLEX_LOAD_STORE(PACKET_TYPE)                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE ploadu<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type* from) {   \
    return PACKET_TYPE(ploadu<typename unpacket_traits<PACKET_TYPE>::as_real>(&numext::real_ref(*from))); \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE pload<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type* from) {    \
    return PACKET_TYPE(pload<typename unpacket_traits<PACKET_TYPE>::as_real>(&numext::real_ref(*from)));  \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE void pstoreu<typename unpacket_traits<PACKET_TYPE>::type, PACKET_TYPE>(             \
      typename unpacket_traits<PACKET_TYPE>::type * to, const PACKET_TYPE& from) {                        \
    pstoreu(&numext::real_ref(*to), from.v);                                                              \
  }                                                                                                       \
  template <>                                                                                             \
  EIGEN_STRONG_INLINE void pstore<typename unpacket_traits<PACKET_TYPE>::type, PACKET_TYPE>(              \
      typename unpacket_traits<PACKET_TYPE>::type * to, const PACKET_TYPE& from) {                        \
    pstore(&numext::real_ref(*to), from.v);                                                               \
  }

EIGEN_CLANG_COMPLEX_LOAD_STORE(Packet8cf);
EIGEN_CLANG_COMPLEX_LOAD_STORE(Packet4cd);
#undef EIGEN_CLANG_COMPLEX_LOAD_STORE

template <>
EIGEN_STRONG_INLINE Packet8cf pset1<Packet8cf>(const std::complex<float>& from) {
  const float re = numext::real(from);
  const float im = numext::imag(from);
  return Packet8cf(Packet16f{re, im, re, im, re, im, re, im, re, im, re, im, re, im, re, im});
}

template <>
EIGEN_STRONG_INLINE Packet4cd pset1<Packet4cd>(const std::complex<double>& from) {
  const double re = numext::real(from);
  const double im = numext::imag(from);
  return Packet4cd(Packet8d{re, im, re, im, re, im, re, im});
}

// ----------- Unary ops ------------------
#define DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, OP)                        \
  template <>                                                             \
  EIGEN_STRONG_INLINE PACKET_TYPE OP<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return PACKET_TYPE(OP(a.v));                                          \
  }

#define EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(PACKET_TYPE)                                             \
  DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, pnegate)                                                    \
  DELEGATE_UNARY_TO_REAL_OP(PACKET_TYPE, pzero)                                                      \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type pfirst<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return a[0];                                                                                     \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pexp<PACKET_TYPE>(const PACKET_TYPE& a) {                          \
    return pexp_complex(a);                                                                          \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE plog<PACKET_TYPE>(const PACKET_TYPE& a) {                          \
    return plog_complex(a);                                                                          \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE psqrt<PACKET_TYPE>(const PACKET_TYPE& a) {                         \
    return psqrt_complex(a);                                                                         \
  }

EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(Packet8cf);
EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS(Packet4cd);

template <>
EIGEN_STRONG_INLINE Packet8cf pconj<Packet8cf>(const Packet8cf& a) {
  return Packet8cf(__builtin_shufflevector(a.v, -a.v, 0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31));
}
template <>
EIGEN_STRONG_INLINE Packet4cf pconj<Packet4cf>(const Packet4cf& a) {
  return Packet4cf(__builtin_shufflevector(a.v, -a.v, 0, 9, 2, 11, 4, 13, 6, 15));
}
template <>
EIGEN_STRONG_INLINE Packet2cf pconj<Packet2cf>(const Packet2cf& a) {
  return Packet2cf(__builtin_shufflevector(a.v, -a.v, 0, 5, 2, 7));
}

template <>
EIGEN_STRONG_INLINE Packet4cd pconj<Packet4cd>(const Packet4cd& a) {
  return Packet4cd(__builtin_shufflevector(a.v, -a.v, 0, 9, 2, 11, 4, 13, 6, 15));
}
template <>
EIGEN_STRONG_INLINE Packet2cd pconj<Packet2cd>(const Packet2cd& a) {
  return Packet2cd(__builtin_shufflevector(a.v, -a.v, 0, 5, 2, 7));
}

#undef DELEGATE_UNARY_TO_REAL_OP
#undef EIGEN_CLANG_COMPLEX_UNARY_CWISE_OPS

// Flip real and imaginary parts, i.e.  {re(a), im(a)} -> {im(a), re(a)}.
template <>
EIGEN_STRONG_INLINE Packet8cf pcplxflip<Packet8cf>(const Packet8cf& a) {
  return Packet8cf(__builtin_shufflevector(a.v, a.v, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
}
template <>
EIGEN_STRONG_INLINE Packet4cf pcplxflip<Packet4cf>(const Packet4cf& a) {
  return Packet4cf(__builtin_shufflevector(a.v, a.v, 1, 0, 3, 2, 5, 4, 7, 6));
}
template <>
EIGEN_STRONG_INLINE Packet2cf pcplxflip<Packet2cf>(const Packet2cf& a) {
  return Packet2cf(__builtin_shufflevector(a.v, a.v, 1, 0, 3, 2));
}
template <>
EIGEN_STRONG_INLINE Packet4cd pcplxflip<Packet4cd>(const Packet4cd& a) {
  return Packet4cd(__builtin_shufflevector(a.v, a.v, 1, 0, 3, 2, 5, 4, 7, 6));
}
template <>
EIGEN_STRONG_INLINE Packet2cd pcplxflip<Packet2cd>(const Packet2cd& a) {
  return Packet2cd(__builtin_shufflevector(a.v, a.v, 1, 0, 3, 2));
}

// Copy real to imaginary part, i.e. {re(a), im(a)} -> {re(a), re(a)}.
template <>
EIGEN_STRONG_INLINE Packet8cf pdupreal<Packet8cf>(const Packet8cf& a) {
  return Packet8cf(__builtin_shufflevector(a.v, a.v, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14));
}
template <>
EIGEN_STRONG_INLINE Packet4cf pdupreal<Packet4cf>(const Packet4cf& a) {
  return Packet4cf(__builtin_shufflevector(a.v, a.v, 0, 0, 2, 2, 4, 4, 6, 6));
}
template <>
EIGEN_STRONG_INLINE Packet2cf pdupreal<Packet2cf>(const Packet2cf& a) {
  return Packet2cf(__builtin_shufflevector(a.v, a.v, 0, 0, 2, 2));
}
template <>
EIGEN_STRONG_INLINE Packet4cd pdupreal<Packet4cd>(const Packet4cd& a) {
  return Packet4cd(__builtin_shufflevector(a.v, a.v, 0, 0, 2, 2, 4, 4, 6, 6));
}
template <>
EIGEN_STRONG_INLINE Packet2cd pdupreal<Packet2cd>(const Packet2cd& a) {
  return Packet2cd(__builtin_shufflevector(a.v, a.v, 0, 0, 2, 2));
}

// Copy imaginary to real part, i.e. {re(a), im(a)} -> {im(a), im(a)}.
template <>
EIGEN_STRONG_INLINE Packet8cf pdupimag<Packet8cf>(const Packet8cf& a) {
  return Packet8cf(__builtin_shufflevector(a.v, a.v, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15));
}
template <>
EIGEN_STRONG_INLINE Packet4cf pdupimag<Packet4cf>(const Packet4cf& a) {
  return Packet4cf(__builtin_shufflevector(a.v, a.v, 1, 1, 3, 3, 5, 5, 7, 7));
}
template <>
EIGEN_STRONG_INLINE Packet2cf pdupimag<Packet2cf>(const Packet2cf& a) {
  return Packet2cf(__builtin_shufflevector(a.v, a.v, 1, 1, 3, 3));
}
template <>
EIGEN_STRONG_INLINE Packet4cd pdupimag<Packet4cd>(const Packet4cd& a) {
  return Packet4cd(__builtin_shufflevector(a.v, a.v, 1, 1, 3, 3, 5, 5, 7, 7));
}
template <>
EIGEN_STRONG_INLINE Packet2cd pdupimag<Packet2cd>(const Packet2cd& a) {
  return Packet2cd(__builtin_shufflevector(a.v, a.v, 1, 1, 3, 3));
}

template <>
EIGEN_STRONG_INLINE Packet8cf ploaddup<Packet8cf>(const std::complex<float>* from) {
  return Packet8cf(Packet16f{std::real(from[0]), std::imag(from[0]), std::real(from[0]), std::imag(from[0]),
                             std::real(from[1]), std::imag(from[1]), std::real(from[1]), std::imag(from[1]),
                             std::real(from[2]), std::imag(from[2]), std::real(from[2]), std::imag(from[2]),
                             std::real(from[3]), std::imag(from[3]), std::real(from[3]), std::imag(from[3])});
}
template <>
EIGEN_STRONG_INLINE Packet4cd ploaddup<Packet4cd>(const std::complex<double>* from) {
  return Packet4cd(Packet8d{std::real(from[0]), std::imag(from[0]), std::real(from[0]), std::imag(from[0]),
                            std::real(from[1]), std::imag(from[1]), std::real(from[1]), std::imag(from[1])});
}

template <>
EIGEN_STRONG_INLINE Packet8cf ploadquad<Packet8cf>(const std::complex<float>* from) {
  return Packet8cf(Packet16f{std::real(from[0]), std::imag(from[0]), std::real(from[0]), std::imag(from[0]),
                             std::real(from[0]), std::imag(from[0]), std::real(from[0]), std::imag(from[0]),
                             std::real(from[1]), std::imag(from[1]), std::real(from[1]), std::imag(from[1]),
                             std::real(from[1]), std::imag(from[1]), std::real(from[1]), std::imag(from[1])});
}
template <>
EIGEN_STRONG_INLINE Packet4cd ploadquad<Packet4cd>(const std::complex<double>* from) {
  return pset1<Packet4cd>(*from);
}

template <>
EIGEN_STRONG_INLINE Packet8cf preverse<Packet8cf>(const Packet8cf& a) {
  return Packet8cf(reinterpret_cast<Packet16f>(preverse(reinterpret_cast<Packet8d>(a.v))));
}
template <>
EIGEN_STRONG_INLINE Packet4cd preverse<Packet4cd>(const Packet4cd& a) {
  return Packet4cd(__builtin_shufflevector(a.v, a.v, 6, 7, 4, 5, 2, 3, 0, 1));
}

// ----------- Binary ops ------------------
#define DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, OP)                                             \
  template <>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE OP<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return PACKET_TYPE(OP(a.v, b.v));                                                           \
  }

#define EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(PACKET_TYPE)                                            \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, psub)                                                      \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pand)                                                      \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, por)                                                       \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pxor)                                                      \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, pandnot)                                                   \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pdiv<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return pdiv_complex(a, b);                                                                       \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pcmp_eq<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    const PACKET_TYPE t = PACKET_TYPE(pcmp_eq(a.v, b.v));                                            \
    return PACKET_TYPE(pand(pdupreal(t).v, pdupimag(t).v));                                          \
  }

EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(Packet8cf);
EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS(Packet4cd);

// Binary ops that are needed on sub-packets for predux and predux_mul.
#define EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(PACKET_TYPE)                                 \
  DELEGATE_BINARY_TO_REAL_OP(PACKET_TYPE, padd)                                                   \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmul<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return pmul_complex(a, b);                                                                    \
  }

EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet8cf);
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet4cf);
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet2cf);
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet4cd);
EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS(Packet2cd);

#define EIGEN_CLANG_PACKET_SCATTER_GATHER(PACKET_TYPE)                                                               \
  template <>                                                                                                        \
  EIGEN_STRONG_INLINE void pscatter(unpacket_traits<PACKET_TYPE>::type* to, const PACKET_TYPE& from, Index stride) { \
    constexpr int size = unpacket_traits<PACKET_TYPE>::size;                                                         \
    for (int i = 0; i < size; ++i) {                                                                                 \
      to[i * stride] = from[i];                                                                                      \
    }                                                                                                                \
  }                                                                                                                  \
  template <>                                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pgather<typename unpacket_traits<PACKET_TYPE>::type, PACKET_TYPE>(                 \
      const unpacket_traits<PACKET_TYPE>::type* from, Index stride) {                                                \
    constexpr int size = unpacket_traits<PACKET_TYPE>::size;                                                         \
    PACKET_TYPE result;                                                                                              \
    for (int i = 0; i < size; ++i) {                                                                                 \
      const unpacket_traits<PACKET_TYPE>::type from_i = from[i * stride];                                            \
      result.v[2 * i] = numext::real(from_i);                                                                        \
      result.v[2 * i + 1] = numext::imag(from_i);                                                                    \
    }                                                                                                                \
    return result;                                                                                                   \
  }

EIGEN_CLANG_PACKET_SCATTER_GATHER(Packet8cf);
EIGEN_CLANG_PACKET_SCATTER_GATHER(Packet4cd);
#undef EIGEN_CLANG_PACKET_SCATTER_GATHER

#undef DELEGATE_BINARY_TO_REAL_OP
#undef EIGEN_CLANG_COMPLEX_BINARY_CWISE_OPS
#undef EIGEN_CLANG_COMPLEX_REDUCER_BINARY_CWISE_OPS

// ------------ ternary ops -------------
template <>
EIGEN_STRONG_INLINE Packet8cf pselect<Packet8cf>(const Packet8cf& mask, const Packet8cf& a, const Packet8cf& b) {
  return Packet8cf(reinterpret_cast<Packet16f>(
      pselect(reinterpret_cast<Packet8d>(mask.v), reinterpret_cast<Packet8d>(a.v), reinterpret_cast<Packet8d>(b.v))));
}

namespace detail {
template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet8cf>(Packet8cf& p1, Packet8cf& p2) {
  Packet16f tmp = __builtin_shufflevector(p1.v, p2.v, 0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
  p2.v = __builtin_shufflevector(p1.v, p2.v, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
  p1.v = tmp;
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet4cd>(Packet4cd& p1, Packet4cd& p2) {
  Packet8d tmp = __builtin_shufflevector(p1.v, p2.v, 0, 1, 8, 9, 2, 3, 10, 11);
  p2.v = __builtin_shufflevector(p1.v, p2.v, 4, 5, 12, 13, 6, 7, 14, 15);
  p1.v = tmp;
}
}  // namespace detail

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8cf, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8cf, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8cf, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4cd, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet4cd, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_CLANG_H

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Rasmus Munk Larsen
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_CLANG_H
#define EIGEN_PACKET_MATH_CLANG_H

namespace Eigen {
namespace internal {

namespace detail {
// namespace detail contains implementation details specific to this
// file, while namespace internal contains internal APIs used elsewhere
// in Eigen.
template <typename ScalarT, int n>
using VectorType = ScalarT __attribute__((ext_vector_type(n), aligned(n * sizeof(ScalarT))));
}  // namespace detail

// --- Primary packet type definitions (fixed at 64 bytes) ---

// TODO(rmlarsen): Generalize to other vector sizes.
static_assert(EIGEN_GENERIC_VECTOR_SIZE_BYTES == 64, "We currently assume the full vector size is 64 bytes");
using Packet16f = detail::VectorType<float, 16>;
using Packet8d = detail::VectorType<double, 8>;
using Packet16i = detail::VectorType<int32_t, 16>;
using Packet8l = detail::VectorType<int64_t, 8>;

// --- packet_traits specializations ---
struct generic_float_packet_traits : default_packet_traits {
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasRound = 1,
    HasMin = 1,
    HasMax = 1,
    HasCmp = 1,
    HasSet1 = 1,
    HasCast = 1,
    HasBitwise = 1,
    HasRedux = 1,
    HasSign = 1,
    HasArg = 0,
    HasConj = 1,
    // Math functions
    HasReciprocal = 1,
    HasSin = 1,
    HasCos = 1,
    HasTan = 1,
    HasACos = 1,
    HasASin = 1,
    HasATan = 1,
    HasATanh = 1,
    HasLog = 1,
    HasLog1p = 1,
    HasExpm1 = 1,
    HasExp = 1,
    HasPow = 1,
    HasNdtri = 1,
    HasBessel = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasCbrt = 1,
    HasTanh = 1,
    HasErf = 1,
    HasErfc = 1
  };
};

template <>
struct packet_traits<float> : generic_float_packet_traits {
  using type = Packet16f;
  using half = Packet16f;
  enum {
    size = 16,
  };
};

template <>
struct packet_traits<double> : generic_float_packet_traits {
  using type = Packet8d;
  using half = Packet8d;
  enum { size = 8, HasACos = 0, HasASin = 0 };
};

struct generic_integer_packet_traits : default_packet_traits {
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasMin = 1,
    HasMax = 1,
    HasCmp = 1,
    HasSet1 = 1,
    HasCast = 1,
    HasBitwise = 1,
    HasRedux = 1,
    // Set remaining to 0
    HasRound = 1,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasReciprocal = 0,
    HasArg = 0,
    HasConj = 1,
    HasExp = 0,
    HasLog = 0,
    HasSin = 0,
    HasCos = 0,
  };
};

template <>
struct packet_traits<int32_t> : generic_integer_packet_traits {
  using type = Packet16i;
  using half = Packet16i;
  enum {
    size = 16,
  };
};

template <>
struct packet_traits<int64_t> : generic_integer_packet_traits {
  using type = Packet8l;
  using half = Packet8l;
  enum {
    size = 8,
  };
};

// --- unpacket_traits specializations ---
struct generic_unpacket_traits : default_unpacket_traits {
  enum {
    alignment = EIGEN_GENERIC_VECTOR_SIZE_BYTES,
    vectorizable = true,
  };
};

template <>
struct unpacket_traits<Packet16f> : generic_unpacket_traits {
  using type = float;
  using half = Packet16f;
  using integer_packet = Packet16i;
  enum {
    size = 16,
  };
};
template <>
struct unpacket_traits<Packet8d> : generic_unpacket_traits {
  using type = double;
  using half = Packet8d;
  using integer_packet = Packet8l;
  enum {
    size = 8,
  };
};
template <>
struct unpacket_traits<Packet16i> : generic_unpacket_traits {
  using type = int32_t;
  using half = Packet16i;
  enum {
    size = 16,
  };
};
template <>
struct unpacket_traits<Packet8l> : generic_unpacket_traits {
  using type = int64_t;
  using half = Packet8l;
  enum {
    size = 8,
  };
};

namespace detail {
// --- vector type helpers ---
template <typename VectorT>
struct ScalarTypeOfVector {
  using type = std::remove_all_extents_t<std::remove_reference_t<decltype(VectorT()[0])>>;
};

template <typename VectorT>
using scalar_type_of_vector_t = typename ScalarTypeOfVector<VectorT>::type;

template <typename VectorType>
struct UnsignedVectorHelpter {
  static VectorType v;
  static constexpr int n = __builtin_vectorelements(v);
  using UnsignedScalar = std::make_unsigned_t<scalar_type_of_vector_t<VectorType>>;
  using type = UnsignedScalar __attribute__((ext_vector_type(n), aligned(n * sizeof(UnsignedScalar))));
};

template <typename VectorT>
using unsigned_vector_t = typename UnsignedVectorHelpter<VectorT>::type;

template <typename VectorT>
using HalfPacket = VectorType<typename unpacket_traits<VectorT>::type, unpacket_traits<VectorT>::size / 2>;

template <typename VectorT>
using QuarterPacket = VectorType<typename unpacket_traits<VectorT>::type, unpacket_traits<VectorT>::size / 4>;

// load and store helpers.
template <typename VectorT>
EIGEN_STRONG_INLINE VectorT load_vector_unaligned(const scalar_type_of_vector_t<VectorT>* from) {
  VectorT to;
  constexpr int n = __builtin_vectorelements(to);
  for (int i = 0; i < n; ++i) {
    to[i] = from[i];
  }
  return to;
}

template <typename VectorT>
EIGEN_STRONG_INLINE VectorT load_vector_aligned(const scalar_type_of_vector_t<VectorT>* from) {
  return *reinterpret_cast<const VectorT*>(assume_aligned<EIGEN_GENERIC_VECTOR_SIZE_BYTES>(from));
}

template <typename VectorT>
EIGEN_STRONG_INLINE void store_vector_unaligned(scalar_type_of_vector_t<VectorT>* to, const VectorT& from) {
  constexpr int n = __builtin_vectorelements(from);
  for (int i = 0; i < n; ++i) {
    *to++ = from[i];
  }
}

template <typename VectorT>
EIGEN_STRONG_INLINE void store_vector_aligned(scalar_type_of_vector_t<VectorT>* to, const VectorT& from) {
  *reinterpret_cast<VectorT*>(assume_aligned<EIGEN_GENERIC_VECTOR_SIZE_BYTES>(to)) = from;
}

}  // namespace detail

// --- Intrinsic-like specializations ---

// --- Load/Store operations ---
#define EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(PACKET_TYPE)                                                         \
  template <>                                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE ploadu<PACKET_TYPE>(const detail::scalar_type_of_vector_t<PACKET_TYPE>* from) { \
    return detail::load_vector_unaligned<PACKET_TYPE>(from);                                                      \
  }                                                                                                               \
  template <>                                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pload<PACKET_TYPE>(const detail::scalar_type_of_vector_t<PACKET_TYPE>* from) {  \
    return detail::load_vector_aligned<PACKET_TYPE>(from);                                                        \
  }                                                                                                               \
  template <>                                                                                                     \
  EIGEN_STRONG_INLINE void pstoreu<detail::scalar_type_of_vector_t<PACKET_TYPE>, PACKET_TYPE>(                    \
      detail::scalar_type_of_vector_t<PACKET_TYPE> * to, const PACKET_TYPE& from) {                               \
    detail::store_vector_unaligned<PACKET_TYPE>(to, from);                                                        \
  }                                                                                                               \
  template <>                                                                                                     \
  EIGEN_STRONG_INLINE void pstore<detail::scalar_type_of_vector_t<PACKET_TYPE>, PACKET_TYPE>(                     \
      detail::scalar_type_of_vector_t<PACKET_TYPE> * to, const PACKET_TYPE& from) {                               \
    detail::store_vector_aligned<PACKET_TYPE>(to, from);                                                          \
  }

EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet16f)
EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet8d)
EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet16i)
EIGEN_CLANG_PACKET_LOAD_STORE_PACKET(Packet8l)
#undef EIGEN_CLANG_PACKET_LOAD_STORE_PACKET

// --- Broadcast operation ---
template <>
EIGEN_STRONG_INLINE Packet16f pset1frombits<Packet16f>(uint32_t from) {
  return Packet16f(numext::bit_cast<float>(from));
}

template <>
EIGEN_STRONG_INLINE Packet8d pset1frombits<Packet8d>(uint64_t from) {
  return Packet8d(numext::bit_cast<double>(from));
}

#define EIGEN_CLANG_PACKET_SET1(PACKET_TYPE)                                                            \
  template <>                                                                                           \
  EIGEN_STRONG_INLINE PACKET_TYPE pset1<PACKET_TYPE>(const unpacket_traits<PACKET_TYPE>::type& from) {  \
    return PACKET_TYPE(from);                                                                           \
  }                                                                                                     \
  template <>                                                                                           \
  EIGEN_STRONG_INLINE unpacket_traits<PACKET_TYPE>::type pfirst<PACKET_TYPE>(const PACKET_TYPE& from) { \
    return from[0];                                                                                     \
  }

EIGEN_CLANG_PACKET_SET1(Packet16f)
EIGEN_CLANG_PACKET_SET1(Packet8d)
EIGEN_CLANG_PACKET_SET1(Packet16i)
EIGEN_CLANG_PACKET_SET1(Packet8l)
#undef EIGEN_CLANG_PACKET_SET1

// --- Arithmetic operations ---
#define EIGEN_CLANG_PACKET_ARITHMETIC(PACKET_TYPE)                             \
  template <>                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE pisnan<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return reinterpret_cast<PACKET_TYPE>(a != a);                              \
  }                                                                            \
  template <>                                                                  \
  EIGEN_STRONG_INLINE PACKET_TYPE pnegate<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return -a;                                                                 \
  }

EIGEN_CLANG_PACKET_ARITHMETIC(Packet16f)
EIGEN_CLANG_PACKET_ARITHMETIC(Packet8d)
EIGEN_CLANG_PACKET_ARITHMETIC(Packet16i)
EIGEN_CLANG_PACKET_ARITHMETIC(Packet8l)
#undef EIGEN_CLANG_PACKET_ARITHMETIC

// --- Bitwise operations (via casting) ---

namespace detail {

// Note: pcast functions are not template specializations, just helpers
// identical to preinterpret. We duplicate them here to avoid a circular
// dependence with TypeCasting.h.
EIGEN_STRONG_INLINE Packet16i pcast_float_to_int(const Packet16f& a) { return reinterpret_cast<Packet16i>(a); }
EIGEN_STRONG_INLINE Packet16f pcast_int_to_float(const Packet16i& a) { return reinterpret_cast<Packet16f>(a); }
EIGEN_STRONG_INLINE Packet8l pcast_double_to_long(const Packet8d& a) { return reinterpret_cast<Packet8l>(a); }
EIGEN_STRONG_INLINE Packet8d pcast_long_to_double(const Packet8l& a) { return reinterpret_cast<Packet8d>(a); }

}  // namespace detail

// Bitwise ops for integer packets
#define EIGEN_CLANG_PACKET_BITWISE_INT(PACKET_TYPE)                                                  \
  template <>                                                                                        \
  constexpr EIGEN_STRONG_INLINE PACKET_TYPE pzero<PACKET_TYPE>(const PACKET_TYPE& /*unused*/) {      \
    return PACKET_TYPE(0);                                                                           \
  }                                                                                                  \
  template <>                                                                                        \
  constexpr EIGEN_STRONG_INLINE PACKET_TYPE ptrue<PACKET_TYPE>(const PACKET_TYPE& /*unused*/) {      \
    return PACKET_TYPE(0) == PACKET_TYPE(0);                                                         \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pand<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return a & b;                                                                                    \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE por<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {     \
    return a | b;                                                                                    \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pxor<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return a ^ b;                                                                                    \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pandnot<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return a & ~b;                                                                                   \
  }                                                                                                  \
  template <int N>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE parithmetic_shift_right(const PACKET_TYPE& a) {                    \
    return a >> N;                                                                                   \
  }                                                                                                  \
  template <int N>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE plogical_shift_right(const PACKET_TYPE& a) {                       \
    using UnsignedT = detail::unsigned_vector_t<PACKET_TYPE>;                                        \
    return reinterpret_cast<PACKET_TYPE>(reinterpret_cast<UnsignedT>(a) >> N);                       \
  }                                                                                                  \
  template <int N>                                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE plogical_shift_left(const PACKET_TYPE& a) {                        \
    return a << N;                                                                                   \
  }

EIGEN_CLANG_PACKET_BITWISE_INT(Packet16i)
EIGEN_CLANG_PACKET_BITWISE_INT(Packet8l)
#undef EIGEN_CLANG_PACKET_BITWISE_INT

// Bitwise ops for floating point packets
#define EIGEN_CLANG_PACKET_BITWISE_FLOAT(PACKET_TYPE, CAST_TO_INT, CAST_FROM_INT)                    \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE ptrue<PACKET_TYPE>(const PACKET_TYPE& /* unused */) {              \
    using Scalar = detail::scalar_type_of_vector_t<PACKET_TYPE>;                                     \
    return CAST_FROM_INT(PACKET_TYPE(Scalar(0)) == PACKET_TYPE(Scalar(0)));                          \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pand<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return CAST_FROM_INT(CAST_TO_INT(a) & CAST_TO_INT(b));                                           \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE por<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {     \
    return CAST_FROM_INT(CAST_TO_INT(a) | CAST_TO_INT(b));                                           \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pxor<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {    \
    return CAST_FROM_INT(CAST_TO_INT(a) ^ CAST_TO_INT(b));                                           \
  }                                                                                                  \
  template <>                                                                                        \
  EIGEN_STRONG_INLINE PACKET_TYPE pandnot<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return CAST_FROM_INT(CAST_TO_INT(a) & ~CAST_TO_INT(b));                                          \
  }

EIGEN_CLANG_PACKET_BITWISE_FLOAT(Packet16f, detail::pcast_float_to_int, detail::pcast_int_to_float)
EIGEN_CLANG_PACKET_BITWISE_FLOAT(Packet8d, detail::pcast_double_to_long, detail::pcast_long_to_double)
#undef EIGEN_CLANG_PACKET_BITWISE_FLOAT

// --- Min/Max operations ---
#if EIGEN_HAS_BUILTIN(__builtin_elementwise_min) && EIGEN_HAS_BUILTIN(__builtin_elementwise_max) && \
    EIGEN_HAS_BUILTIN(__builtin_elementwise_abs)
#define EIGEN_CLANG_PACKET_ELEMENTWISE(PACKET_TYPE)                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pmin<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {                   \
    /* Match NaN propagation of std::min. */                                                                        \
    return a == a ? __builtin_elementwise_min(a, b) : a;                                                            \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pmax<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {                   \
    /* Match NaN propagation of std::max. */                                                                        \
    return a == a ? __builtin_elementwise_max(a, b) : a;                                                            \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pmin<PropagateNumbers, PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return __builtin_elementwise_min(a, b);                                                                         \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pmax<PropagateNumbers, PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) { \
    return __builtin_elementwise_max(a, b);                                                                         \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pmin<PropagateNaN, PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {     \
    return a != a ? a : (b != b ? b : __builtin_elementwise_min(a, b));                                             \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pmax<PropagateNaN, PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b) {     \
    return a != a ? a : (b != b ? b : __builtin_elementwise_max(a, b));                                             \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pabs<PACKET_TYPE>(const PACKET_TYPE& a) {                                         \
    return __builtin_elementwise_abs(a);                                                                            \
  }                                                                                                                 \
  template <>                                                                                                       \
  EIGEN_STRONG_INLINE PACKET_TYPE pselect<PACKET_TYPE>(const PACKET_TYPE& mask, const PACKET_TYPE& a,               \
                                                       const PACKET_TYPE& b) {                                      \
    return mask != 0 ? a : b;                                                                                       \
  }

EIGEN_CLANG_PACKET_ELEMENTWISE(Packet16f)
EIGEN_CLANG_PACKET_ELEMENTWISE(Packet8d)
EIGEN_CLANG_PACKET_ELEMENTWISE(Packet16i)
EIGEN_CLANG_PACKET_ELEMENTWISE(Packet8l)
#undef EIGEN_CLANG_PACKET_ELEMENTWISE
#endif

// --- Math functions (float/double only) ---

#if EIGEN_HAS_BUILTIN(__builtin_elementwise_floor) && EIGEN_HAS_BUILTIN(__builtin_elementwise_ceil) &&      \
    EIGEN_HAS_BUILTIN(__builtin_elementwise_round) && EIGEN_HAS_BUILTIN(__builtin_elementwise_roundeven) && \
    EIGEN_HAS_BUILTIN(__builtin_elementwise_trunc) && EIGEN_HAS_BUILTIN(__builtin_elementwise_sqrt)
#define EIGEN_CLANG_PACKET_MATH_FLOAT(PACKET_TYPE)                            \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE pfloor<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_elementwise_floor(a);                                    \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE pceil<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return __builtin_elementwise_ceil(a);                                     \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE pround<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_elementwise_round(a);                                    \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE print<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return __builtin_elementwise_roundeven(a);                                \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE ptrunc<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return __builtin_elementwise_trunc(a);                                    \
  }                                                                           \
  template <>                                                                 \
  EIGEN_STRONG_INLINE PACKET_TYPE psqrt<PACKET_TYPE>(const PACKET_TYPE& a) {  \
    return __builtin_elementwise_sqrt(a);                                     \
  }

EIGEN_CLANG_PACKET_MATH_FLOAT(Packet16f)
EIGEN_CLANG_PACKET_MATH_FLOAT(Packet8d)
#undef EIGEN_CLANG_PACKET_MATH_FLOAT
#endif

// --- Fused Multiply-Add (MADD) ---
#if defined(__FMA__) && EIGEN_HAS_BUILTIN(__builtin_elementwise_fma)
#define EIGEN_CLANG_PACKET_MADD(PACKET_TYPE)                                                      \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmadd<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b,  \
                                                     const PACKET_TYPE& c) {                      \
    return __builtin_elementwise_fma(a, b, c);                                                    \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pmsub<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b,  \
                                                     const PACKET_TYPE& c) {                      \
    return __builtin_elementwise_fma(a, b, -c);                                                   \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pnmadd<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b, \
                                                      const PACKET_TYPE& c) {                     \
    return __builtin_elementwise_fma(-a, b, c);                                                   \
  }                                                                                               \
  template <>                                                                                     \
  EIGEN_STRONG_INLINE PACKET_TYPE pnmsub<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b, \
                                                      const PACKET_TYPE& c) {                     \
    return -(__builtin_elementwise_fma(a, b, c));                                                 \
  }
#else
// Fallback if FMA builtin is not available
#define EIGEN_CLANG_PACKET_MADD(PACKET_TYPE)                                                     \
  template <>                                                                                    \
  EIGEN_STRONG_INLINE PACKET_TYPE pmadd<PACKET_TYPE>(const PACKET_TYPE& a, const PACKET_TYPE& b, \
                                                     const PACKET_TYPE& c) {                     \
    return (a * b) + c;                                                                          \
  }
#endif

EIGEN_CLANG_PACKET_MADD(Packet16f)
EIGEN_CLANG_PACKET_MADD(Packet8d)
#undef EIGEN_CLANG_PACKET_MADD

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
      result[i] = from[i * stride];                                                                                  \
    }                                                                                                                \
    return result;                                                                                                   \
  }

EIGEN_CLANG_PACKET_SCATTER_GATHER(Packet16f)
EIGEN_CLANG_PACKET_SCATTER_GATHER(Packet8d)
EIGEN_CLANG_PACKET_SCATTER_GATHER(Packet16i)
EIGEN_CLANG_PACKET_SCATTER_GATHER(Packet8l)

#undef EIGEN_CLANG_PACKET_SCATTER_GATHER

// ---- Various operations that depend on __builtin_shufflevector.
#if EIGEN_HAS_BUILTIN(__builtin_shufflevector)
namespace detail {
template <typename Packet>
EIGEN_STRONG_INLINE Packet preverse_impl_8(const Packet& a) {
  return __builtin_shufflevector(a, a, 7, 6, 5, 4, 3, 2, 1, 0);
}
template <typename Packet>
EIGEN_STRONG_INLINE Packet preverse_impl_16(const Packet& a) {
  return __builtin_shufflevector(a, a, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}
}  // namespace detail

#define EIGEN_CLANG_PACKET_REVERSE(PACKET_TYPE, SIZE)                           \
  template <>                                                                   \
  EIGEN_STRONG_INLINE PACKET_TYPE preverse<PACKET_TYPE>(const PACKET_TYPE& a) { \
    return detail::preverse_impl_##SIZE(a);                                     \
  }

EIGEN_CLANG_PACKET_REVERSE(Packet16f, 16)
EIGEN_CLANG_PACKET_REVERSE(Packet8d, 8)
EIGEN_CLANG_PACKET_REVERSE(Packet16i, 16)
EIGEN_CLANG_PACKET_REVERSE(Packet8l, 8)
#undef EIGEN_CLANG_PACKET_REVERSE

namespace detail {
template <typename Packet>
EIGEN_STRONG_INLINE Packet ploaddup16(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 2 == 0, "Packet size must be a multiple of 2");
  using HalfPacket = HalfPacket<Packet>;
  HalfPacket a = load_vector_unaligned<HalfPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploadquad16(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 4 == 0, "Packet size must be a multiple of 4");
  using QuarterPacket = QuarterPacket<Packet>;
  QuarterPacket a = load_vector_unaligned<QuarterPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploaddup8(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 2 == 0, "Packet size must be a multiple of 2");
  using HalfPacket = HalfPacket<Packet>;
  HalfPacket a = load_vector_unaligned<HalfPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 1, 1, 2, 2, 3, 3);
}

template <typename Packet>
EIGEN_STRONG_INLINE Packet ploadquad8(const typename unpacket_traits<Packet>::type* from) {
  static_assert((unpacket_traits<Packet>::size) % 4 == 0, "Packet size must be a multiple of 4");
  using QuarterPacket = QuarterPacket<Packet>;
  QuarterPacket a = load_vector_unaligned<QuarterPacket>(from);
  return __builtin_shufflevector(a, a, 0, 0, 0, 0, 1, 1, 1, 1);
}

}  // namespace detail

template <>
EIGEN_STRONG_INLINE Packet16f ploaddup<Packet16f>(const float* from) {
  return detail::ploaddup16<Packet16f>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d ploaddup<Packet8d>(const double* from) {
  return detail::ploaddup8<Packet8d>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i ploaddup<Packet16i>(const int32_t* from) {
  return detail::ploaddup16<Packet16i>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8l ploaddup<Packet8l>(const int64_t* from) {
  return detail::ploaddup8<Packet8l>(from);
}

template <>
EIGEN_STRONG_INLINE Packet16f ploadquad<Packet16f>(const float* from) {
  return detail::ploadquad16<Packet16f>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8d ploadquad<Packet8d>(const double* from) {
  return detail::ploadquad8<Packet8d>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16i ploadquad<Packet16i>(const int32_t* from) {
  return detail::ploadquad16<Packet16i>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8l ploadquad<Packet8l>(const int64_t* from) {
  return detail::ploadquad8<Packet8l>(from);
}

template <>
EIGEN_STRONG_INLINE Packet16f plset<Packet16f>(const float& a) {
  Packet16f x{a + 0.0f, a + 1.0f, a + 2.0f,  a + 3.0f,  a + 4.0f,  a + 5.0f,  a + 6.0f,  a + 7.0f,
              a + 8.0f, a + 9.0f, a + 10.0f, a + 11.0f, a + 12.0f, a + 13.0f, a + 14.0f, a + 15.0f};
  return x;
}
template <>
EIGEN_STRONG_INLINE Packet8d plset<Packet8d>(const double& a) {
  return Packet8d{a + 0.0, a + 1.0, a + 2.0, a + 3.0, a + 4.0, a + 5.0, a + 6.0, a + 7.0};
}
template <>
EIGEN_STRONG_INLINE Packet16i plset<Packet16i>(const int32_t& a) {
  return Packet16i{a + 0, a + 1, a + 2,  a + 3,  a + 4,  a + 5,  a + 6,  a + 7,
                   a + 8, a + 9, a + 10, a + 11, a + 12, a + 13, a + 14, a + 15};
}
template <>
EIGEN_STRONG_INLINE Packet8l plset<Packet8l>(const int64_t& a) {
  return Packet8l{a + 0, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7};
}

template <>
EIGEN_STRONG_INLINE Packet16f peven_mask(const Packet16f& /* unused */) {
  float kTrue = numext::bit_cast<float>(int32_t(-1));
  float kFalse = 0.0f;
  return Packet16f{kTrue, kFalse, kTrue, kFalse, kTrue, kFalse, kTrue, kFalse,
                   kTrue, kFalse, kTrue, kFalse, kTrue, kFalse, kTrue, kFalse};
}

template <>
EIGEN_STRONG_INLINE Packet8d peven_mask(const Packet8d& /* unused */) {
  double kTrue = numext::bit_cast<double>(int64_t(-1l));
  double kFalse = 0.0;
  return Packet8d{kTrue, kFalse, kTrue, kFalse, kTrue, kFalse, kTrue, kFalse};
}

// Helpers for ptranspose.
namespace detail {

template <typename Packet>
EIGEN_ALWAYS_INLINE void zip_in_place16(Packet& p1, Packet& p2) {
  Packet tmp = __builtin_shufflevector(p1, p2, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  p2 = __builtin_shufflevector(p1, p2, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  p1 = tmp;
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void zip_in_place8(Packet& p1, Packet& p2) {
  Packet tmp = __builtin_shufflevector(p1, p2, 0, 8, 1, 9, 2, 10, 3, 11);
  p2 = __builtin_shufflevector(p1, p2, 4, 12, 5, 13, 6, 14, 7, 15);
  p1 = tmp;
}

template <typename Packet>
void zip_in_place(Packet& p1, Packet& p2);

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet16f>(Packet16f& p1, Packet16f& p2) {
  zip_in_place16(p1, p2);
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet8d>(Packet8d& p1, Packet8d& p2) {
  zip_in_place8(p1, p2);
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet16i>(Packet16i& p1, Packet16i& p2) {
  zip_in_place16(p1, p2);
}

template <>
EIGEN_ALWAYS_INLINE void zip_in_place<Packet8l>(Packet8l& p1, Packet8l& p2) {
  zip_in_place8(p1, p2);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 2>& kernel) {
  zip_in_place(kernel.packet[0], kernel.packet[1]);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 4>& kernel) {
  zip_in_place(kernel.packet[0], kernel.packet[2]);
  zip_in_place(kernel.packet[1], kernel.packet[3]);
  zip_in_place(kernel.packet[0], kernel.packet[1]);
  zip_in_place(kernel.packet[2], kernel.packet[3]);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 8>& kernel) {
  zip_in_place(kernel.packet[0], kernel.packet[4]);
  zip_in_place(kernel.packet[1], kernel.packet[5]);
  zip_in_place(kernel.packet[2], kernel.packet[6]);
  zip_in_place(kernel.packet[3], kernel.packet[7]);

  zip_in_place(kernel.packet[0], kernel.packet[2]);
  zip_in_place(kernel.packet[1], kernel.packet[3]);
  zip_in_place(kernel.packet[4], kernel.packet[6]);
  zip_in_place(kernel.packet[5], kernel.packet[7]);

  zip_in_place(kernel.packet[0], kernel.packet[1]);
  zip_in_place(kernel.packet[2], kernel.packet[3]);
  zip_in_place(kernel.packet[4], kernel.packet[5]);
  zip_in_place(kernel.packet[6], kernel.packet[7]);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void ptranspose_impl(PacketBlock<Packet, 16>& kernel) {
  EIGEN_UNROLL_LOOP
  for (int i = 0; i < 4; ++i) {
    const int m = (1 << i);
    EIGEN_UNROLL_LOOP
    for (int j = 0; j < m; ++j) {
      const int n = (1 << (3 - i));
      EIGEN_UNROLL_LOOP
      for (int k = 0; k < n; ++k) {
        const int idx = 2 * j * n + k;
        zip_in_place(kernel.packet[idx], kernel.packet[idx + n]);
      }
    }
  }
}

}  // namespace detail

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 16>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8d, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8d, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8d, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 16>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16i, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8l, 8>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8l, 4>& kernel) {
  detail::ptranspose_impl(kernel);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8l, 2>& kernel) {
  detail::ptranspose_impl(kernel);
}
#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_CLANG_H

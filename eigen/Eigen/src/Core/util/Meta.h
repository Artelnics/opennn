// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_META_H
#define EIGEN_META_H

#include "../InternalHeaderCheck.h"

#if defined(EIGEN_GPU_COMPILE_PHASE)

 #include <cfloat>

 #if defined(EIGEN_CUDA_ARCH)
  #include <math_constants.h>
 #endif

 #if defined(EIGEN_HIP_DEVICE_COMPILE)
  #include "Eigen/src/Core/arch/HIP/hcc/math_constants.h"
  #endif

#endif

// Recent versions of ICC require <cstdint> for pointer types below.
#define EIGEN_ICC_NEEDS_CSTDINT (EIGEN_COMP_ICC>=1600)

// Define portable (u)int{32,64} types
#include <cstdint>

namespace Eigen {
namespace numext {
typedef std::uint8_t  uint8_t;
typedef std::int8_t   int8_t;
typedef std::uint16_t uint16_t;
typedef std::int16_t  int16_t;
typedef std::uint32_t uint32_t;
typedef std::int32_t  int32_t;
typedef std::uint64_t uint64_t;
typedef std::int64_t  int64_t;
}
}

namespace Eigen {

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE DenseIndex;

/**
 * \brief The Index type as used for the API.
 * \details To change this, \c \#define the preprocessor symbol \c EIGEN_DEFAULT_DENSE_INDEX_TYPE.
 * \sa \blank \ref TopicPreprocessorDirectives, StorageIndex.
 */

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE Index;

namespace internal {

/** \internal
  * \file Meta.h
  * This file contains generic metaprogramming classes which are not specifically related to Eigen.
  * \note In case you wonder, yes we're aware that Boost already provides all these features,
  * we however don't want to add a dependency to Boost.
  */

// Only recent versions of ICC complain about using ptrdiff_t to hold pointers,
// and older versions do not provide *intptr_t types.
#if EIGEN_ICC_NEEDS_CSTDINT
typedef std::intptr_t  IntPtr;
typedef std::uintptr_t UIntPtr;
#else
typedef std::ptrdiff_t IntPtr;
typedef std::size_t UIntPtr;
#endif
#undef EIGEN_ICC_NEEDS_CSTDINT

struct true_type {  enum { value = 1 }; };
struct false_type { enum { value = 0 }; };

template<bool Condition>
struct bool_constant;

template<>
struct bool_constant<true> : true_type {};

template<>
struct bool_constant<false> : false_type {};

template<bool Condition, typename Then, typename Else>
struct conditional { typedef Then type; };

template<typename Then, typename Else>
struct conditional <false, Then, Else> { typedef Else type; };

template<typename T> struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };

template<typename T> struct remove_pointer { typedef T type; };
template<typename T> struct remove_pointer<T*> { typedef T type; };
template<typename T> struct remove_pointer<T*const> { typedef T type; };

template <class T> struct remove_const { typedef T type; };
template <class T> struct remove_const<const T> { typedef T type; };
template <class T> struct remove_const<const T[]> { typedef T type[]; };
template <class T, unsigned int Size> struct remove_const<const T[Size]> { typedef T type[Size]; };

template<typename T> struct remove_all { typedef T type; };
template<typename T> struct remove_all<const T>   { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T const&>  { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T&>        { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T const*>  { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T*>        { typedef typename remove_all<T>::type type; };

template<typename T> struct is_arithmetic      { enum { value = false }; };
template<> struct is_arithmetic<float>         { enum { value = true }; };
template<> struct is_arithmetic<double>        { enum { value = true }; };
template<> struct is_arithmetic<long double>   { enum { value = true }; };
template<> struct is_arithmetic<bool>          { enum { value = true }; };
template<> struct is_arithmetic<char>          { enum { value = true }; };
template<> struct is_arithmetic<signed char>   { enum { value = true }; };
template<> struct is_arithmetic<unsigned char> { enum { value = true }; };
template<> struct is_arithmetic<signed short>  { enum { value = true }; };
template<> struct is_arithmetic<unsigned short>{ enum { value = true }; };
template<> struct is_arithmetic<signed int>    { enum { value = true }; };
template<> struct is_arithmetic<unsigned int>  { enum { value = true }; };
template<> struct is_arithmetic<signed long>   { enum { value = true }; };
template<> struct is_arithmetic<unsigned long> { enum { value = true }; };

template<typename T, typename U> struct is_same { enum { value = 0 }; };
template<typename T> struct is_same<T,T> { enum { value = 1 }; };

template< class T >
struct is_void : is_same<void, typename remove_const<T>::type> {};

template<> struct is_arithmetic<signed long long>   { enum { value = true }; };
template<> struct is_arithmetic<unsigned long long> { enum { value = true }; };
using std::is_integral;

using std::make_unsigned;

template <typename T> struct add_const { typedef const T type; };
template <typename T> struct add_const<T&> { typedef T& type; };

template <typename T> struct is_const { enum { value = 0 }; };
template <typename T> struct is_const<T const> { enum { value = 1 }; };

template<typename T> struct add_const_on_value_type            { typedef const T type;  };
template<typename T> struct add_const_on_value_type<T&>        { typedef T const& type; };
template<typename T> struct add_const_on_value_type<T*>        { typedef T const* type; };
template<typename T> struct add_const_on_value_type<T* const>  { typedef T const* const type; };
template<typename T> struct add_const_on_value_type<T const* const>  { typedef T const* const type; };

using std::is_convertible;

/** \internal Allows to enable/disable an overload
  * according to a compile time condition.
  */
template<bool Condition, typename T=void> struct enable_if;

template<typename T> struct enable_if<true,T>
{ typedef T type; };

/** \internal
  * A base class do disable default copy ctor and copy assignment operator.
  */
class noncopyable
{
  EIGEN_DEVICE_FUNC noncopyable(const noncopyable&);
  EIGEN_DEVICE_FUNC const noncopyable& operator=(const noncopyable&);
protected:
  EIGEN_DEVICE_FUNC noncopyable() {}
  EIGEN_DEVICE_FUNC ~noncopyable() {}
};

/** \internal
  * Provides access to the number of elements in the object of as a compile-time constant expression.
  * It "returns" Eigen::Dynamic if the size cannot be resolved at compile-time (default).
  *
  * Similar to std::tuple_size, but more general.
  *
  * It currently supports:
  *  - any types T defining T::SizeAtCompileTime
  *  - plain C arrays as T[N]
  *  - std::array (c++11)
  *  - some internal types such as SingleRange and AllRange
  *
  * The second template parameter eases SFINAE-based specializations.
  */
template<typename T, typename EnableIf = void> struct array_size {
  enum { value = Dynamic };
};

template<typename T> struct array_size<T,typename internal::enable_if<((T::SizeAtCompileTime&0)==0)>::type> {
  enum { value = T::SizeAtCompileTime };
};

template<typename T, int N> struct array_size<const T (&)[N]> {
  enum { value = N };
};
template<typename T, int N> struct array_size<T (&)[N]> {
  enum { value = N };
};

template<typename T, std::size_t N> struct array_size<const std::array<T,N> > {
  enum { value = N };
};
template<typename T, std::size_t N> struct array_size<std::array<T,N> > {
  enum { value = N };
};


/** \internal
  * Analogue of the std::ssize free function.
  * It returns the signed size of the container or view \a x of type \c T
  *
  * It currently supports:
  *  - any types T defining a member T::size() const
  *  - plain C arrays as T[N]
  *
  * For C++20, this function just forwards to `std::ssize`, or any ADL discoverable `ssize` function.
  */
#if EIGEN_COMP_CXXVER < 20
template <typename T>
EIGEN_CONSTEXPR auto index_list_size(const T& x) {
  using R = std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(x.size())>>;
  return static_cast<R>(x.size());
}

template<typename T, std::ptrdiff_t N>
EIGEN_CONSTEXPR std::ptrdiff_t index_list_size(const T (&)[N]) { return N; }
#else
template <typename T>
EIGEN_CONSTEXPR auto index_list_size(T&& x) {
  using std::ssize;
  return ssize(std::forward<T>(x));
}
#endif // EIGEN_COMP_CXXVER

/** \internal
  * Convenient struct to get the result type of a nullary, unary, binary, or
  * ternary functor.
  * 
  * Pre C++11:
  * Supports both a Func::result_type member and templated
  * Func::result<Func(ArgTypes...)>::type member.
  * 
  * If none of these members is provided, then the type of the first
  * argument is returned.
  * 
  * Post C++11:
  * This uses std::result_of. However, note the `type` member removes
  * const and converts references/pointers to their corresponding value type.
  */
#if EIGEN_HAS_STD_INVOKE_RESULT
template<typename T> struct result_of;

template<typename F, typename... ArgTypes>
struct result_of<F(ArgTypes...)> {
  typedef typename std::invoke_result<F, ArgTypes...>::type type1;
  typedef typename remove_all<type1>::type type;
};
#elif EIGEN_HAS_STD_RESULT_OF
template<typename T> struct result_of {
  typedef typename std::result_of<T>::type type1;
  typedef typename remove_all<type1>::type type;
};
#else
template<typename T> struct result_of { };

struct has_none {int a[1];};
struct has_std_result_type {int a[2];};
struct has_tr1_result {int a[3];};

template<typename Func, int SizeOf>
struct nullary_result_of_select {};

template<typename Func>
struct nullary_result_of_select<Func, sizeof(has_std_result_type)> {typedef typename Func::result_type type;};

template<typename Func>
struct nullary_result_of_select<Func, sizeof(has_tr1_result)> {typedef typename Func::template result<Func()>::type type;};

template<typename Func>
struct result_of<Func()> {
    template<typename T>
    static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
    template<typename T>
    static has_tr1_result         testFunctor(T const *, typename T::template result<T()>::type const * = 0);
    static has_none               testFunctor(...);

    // note that the following indirection is needed for gcc-3.3
    enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
    typedef typename nullary_result_of_select<Func, FunctorType>::type type;
};

template<typename Func, typename ArgType, int SizeOf=sizeof(has_none)>
struct unary_result_of_select {typedef typename internal::remove_all<ArgType>::type type;};

template<typename Func, typename ArgType>
struct unary_result_of_select<Func, ArgType, sizeof(has_std_result_type)> {typedef typename Func::result_type type;};

template<typename Func, typename ArgType>
struct unary_result_of_select<Func, ArgType, sizeof(has_tr1_result)> {typedef typename Func::template result<Func(ArgType)>::type type;};

template<typename Func, typename ArgType>
struct result_of<Func(ArgType)> {
    template<typename T>
    static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
    template<typename T>
    static has_tr1_result         testFunctor(T const *, typename T::template result<T(ArgType)>::type const * = 0);
    static has_none               testFunctor(...);

    // note that the following indirection is needed for gcc-3.3
    enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
    typedef typename unary_result_of_select<Func, ArgType, FunctorType>::type type;
};

template<typename Func, typename ArgType0, typename ArgType1, int SizeOf=sizeof(has_none)>
struct binary_result_of_select {typedef typename internal::remove_all<ArgType0>::type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct binary_result_of_select<Func, ArgType0, ArgType1, sizeof(has_std_result_type)>
{typedef typename Func::result_type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct binary_result_of_select<Func, ArgType0, ArgType1, sizeof(has_tr1_result)>
{typedef typename Func::template result<Func(ArgType0,ArgType1)>::type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct result_of<Func(ArgType0,ArgType1)> {
    template<typename T>
    static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
    template<typename T>
    static has_tr1_result         testFunctor(T const *, typename T::template result<T(ArgType0,ArgType1)>::type const * = 0);
    static has_none               testFunctor(...);

    // note that the following indirection is needed for gcc-3.3
    enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
    typedef typename binary_result_of_select<Func, ArgType0, ArgType1, FunctorType>::type type;
};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2, int SizeOf=sizeof(has_none)>
struct ternary_result_of_select {typedef typename internal::remove_all<ArgType0>::type type;};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2>
struct ternary_result_of_select<Func, ArgType0, ArgType1, ArgType2, sizeof(has_std_result_type)>
{typedef typename Func::result_type type;};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2>
struct ternary_result_of_select<Func, ArgType0, ArgType1, ArgType2, sizeof(has_tr1_result)>
{typedef typename Func::template result<Func(ArgType0,ArgType1,ArgType2)>::type type;};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2>
struct result_of<Func(ArgType0,ArgType1,ArgType2)> {
    template<typename T>
    static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
    template<typename T>
    static has_tr1_result         testFunctor(T const *, typename T::template result<T(ArgType0,ArgType1,ArgType2)>::type const * = 0);
    static has_none               testFunctor(...);

    // note that the following indirection is needed for gcc-3.3
    enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
    typedef typename ternary_result_of_select<Func, ArgType0, ArgType1, ArgType2, FunctorType>::type type;
};

#endif

#if EIGEN_HAS_STD_INVOKE_RESULT
template<typename F, typename... ArgTypes>
struct invoke_result {
  typedef typename std::invoke_result<F, ArgTypes...>::type type1;
  typedef typename remove_all<type1>::type type;
};
#else
template<typename F, typename... ArgTypes>
struct invoke_result {
  typedef typename result_of<F(ArgTypes...)>::type type1;
  typedef typename remove_all<type1>::type type;
};
#endif

// C++14 integer/index_sequence.
#if defined(__cpp_lib_integer_sequence) && __cpp_lib_integer_sequence >= 201304L

using std::integer_sequence;
using std::make_integer_sequence;

using std::index_sequence;
using std::make_index_sequence;

#else 

template <typename T, T... Ints>
struct integer_sequence {
  static EIGEN_CONSTEXPR size_t size() EIGEN_NOEXCEPT { return sizeof...(Ints); }
};

template <typename T, typename Sequence, T N>
struct append_integer;

template<typename T, T... Ints, T N>
struct append_integer<T, integer_sequence<T, Ints...>, N> {
  using type = integer_sequence<T, Ints..., N>;
};

template<typename T, size_t N>
struct generate_integer_sequence {
  using type = typename append_integer<T, typename generate_integer_sequence<T, N-1>::type, N-1>::type;
};

template<typename T>
struct generate_integer_sequence<T, 0> {
  using type = integer_sequence<T>;
};

template <typename T, size_t N>
using make_integer_sequence = typename generate_integer_sequence<T, N>::type;

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template<size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

#endif

// Reduces a sequence of bools to true if all are true, false otherwise.
template<bool... values>
using reduce_all = std::is_same<integer_sequence<bool, values..., true>, integer_sequence<bool, true, values...> >;

// Reduces a sequence of bools to true if any are true, false if all false.
template<bool... values>
using reduce_any = std::integral_constant<bool,
    !std::is_same<integer_sequence<bool, values..., false>, integer_sequence<bool, false, values...> >::value>;

struct meta_yes { char a[1]; };
struct meta_no  { char a[2]; };

// Check whether T::ReturnType does exist
template <typename T>
struct has_ReturnType
{
  template <typename C> static meta_yes testFunctor(C const *, typename C::ReturnType const * = 0);
  template <typename C> static meta_no  testFunctor(...);

  enum { value = sizeof(testFunctor<T>(static_cast<T*>(0))) == sizeof(meta_yes) };
};

template<typename T> const T* return_ptr();

template <typename T, typename IndexType=Index>
struct has_nullary_operator
{
  template <typename C> static meta_yes testFunctor(C const *,typename enable_if<(sizeof(return_ptr<C>()->operator()())>0)>::type * = 0);
  static meta_no testFunctor(...);

  enum { value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};

template <typename T, typename IndexType=Index>
struct has_unary_operator
{
  template <typename C> static meta_yes testFunctor(C const *,typename enable_if<(sizeof(return_ptr<C>()->operator()(IndexType(0)))>0)>::type * = 0);
  static meta_no testFunctor(...);

  enum { value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};

template <typename T, typename IndexType=Index>
struct has_binary_operator
{
  template <typename C> static meta_yes testFunctor(C const *,typename enable_if<(sizeof(return_ptr<C>()->operator()(IndexType(0),IndexType(0)))>0)>::type * = 0);
  static meta_no testFunctor(...);

  enum { value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};

/** \internal In short, it computes int(sqrt(\a Y)) with \a Y an integer.
  * Usage example: \code meta_sqrt<1023>::ret \endcode
  */
template<int Y,
         int InfX = 0,
         int SupX = ((Y==1) ? 1 : Y/2),
         bool Done = ((SupX-InfX)<=1 ? true : ((SupX*SupX <= Y) && ((SupX+1)*(SupX+1) > Y))) >
                                // use ?: instead of || just to shut up a stupid gcc 4.3 warning
class meta_sqrt
{
    enum {
      MidX = (InfX+SupX)/2,
      TakeInf = MidX*MidX > Y ? 1 : 0,
      NewInf = int(TakeInf) ? InfX : int(MidX),
      NewSup = int(TakeInf) ? int(MidX) : SupX
    };
  public:
    enum { ret = meta_sqrt<Y,NewInf,NewSup>::ret };
};

template<int Y, int InfX, int SupX>
class meta_sqrt<Y, InfX, SupX, true> { public:  enum { ret = (SupX*SupX <= Y) ? SupX : InfX }; };


/** \internal Computes the least common multiple of two positive integer A and B
  * at compile-time. 
  */
template<int A, int B, int K=1, bool Done = ((A*K)%B)==0, bool Big=(A>=B)>
struct meta_least_common_multiple
{
  enum { ret = meta_least_common_multiple<A,B,K+1>::ret };
};
template<int A, int B, int K, bool Done>
struct meta_least_common_multiple<A,B,K,Done,false>
{
  enum { ret = meta_least_common_multiple<B,A,K>::ret };
};
template<int A, int B, int K>
struct meta_least_common_multiple<A,B,K,true,true>
{
  enum { ret = A*K };
};


/** \internal determines whether the product of two numeric types is allowed and what the return type is */
template<typename T, typename U> struct scalar_product_traits
{
  enum { Defined = 0 };
};

// FIXME quick workaround around current limitation of result_of
// template<typename Scalar, typename ArgType0, typename ArgType1>
// struct result_of<scalar_product_op<Scalar>(ArgType0,ArgType1)> {
// typedef typename scalar_product_traits<typename remove_all<ArgType0>::type, typename remove_all<ArgType1>::type>::ReturnType type;
// };

/** \internal Obtains a POD type suitable to use as storage for an object of a size
  * of at most Len bytes, aligned as specified by \c Align.
  */
template<unsigned Len, unsigned Align>
struct aligned_storage {
  struct type {
    EIGEN_ALIGN_TO_BOUNDARY(Align) unsigned char data[Len];
  };
};

} // end namespace internal

namespace numext {

#if defined(EIGEN_GPU_COMPILE_PHASE)
template<typename T> EIGEN_DEVICE_FUNC   void swap(T &a, T &b) { T tmp = b; b = a; a = tmp; }
#else
template<typename T> EIGEN_STRONG_INLINE void swap(T &a, T &b) { std::swap(a,b); }
#endif

using std::numeric_limits;

// Integer division with rounding up.
// T is assumed to be an integer type with a>=0, and b>0
template<typename T>
EIGEN_DEVICE_FUNC
T div_ceil(const T &a, const T &b)
{
  return (a+b-1) / b;
}

// The aim of the following functions is to bypass -Wfloat-equal warnings
// when we really want a strict equality comparison on floating points.
template<typename X, typename Y> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
bool equal_strict(const X& x,const Y& y) { return x == y; }

#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
bool equal_strict(const float& x,const float& y) { return std::equal_to<float>()(x,y); }

template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
bool equal_strict(const double& x,const double& y) { return std::equal_to<double>()(x,y); }
#endif

template<typename X, typename Y> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
bool not_equal_strict(const X& x,const Y& y) { return x != y; }

#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
bool not_equal_strict(const float& x,const float& y) { return std::not_equal_to<float>()(x,y); }

template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
bool not_equal_strict(const double& x,const double& y) { return std::not_equal_to<double>()(x,y); }
#endif

} // end namespace numext

namespace internal {
/// \internal Returns true if its argument is of integer or enum type.
/// FIXME this has the same purpose as `is_valid_index_type` in XprHelper.h
template<typename A>
constexpr bool is_int_or_enum_v = std::is_enum<A>::value || std::is_integral<A>::value;

/// \internal Gets the minimum of two values which may be integers or enums
template<typename A, typename B>
inline constexpr int plain_enum_min(A a, B b) {
  static_assert(is_int_or_enum_v<A>, "Argument a must be an integer or enum");
  static_assert(is_int_or_enum_v<B>, "Argument b must be an integer or enum");
  return ((int) a <= (int) b) ? (int) a : (int) b;
}

/// \internal Gets the maximum of two values which may be integers or enums
template<typename A, typename B>
inline constexpr int plain_enum_max(A a, B b) {
  static_assert(is_int_or_enum_v<A>, "Argument a must be an integer or enum");
  static_assert(is_int_or_enum_v<B>, "Argument b must be an integer or enum");
  return ((int) a >= (int) b) ? (int) a : (int) b;
}

/**
 * \internal
 *  `min_size_prefer_dynamic` gives the min between compile-time sizes. 0 has absolute priority, followed by 1,
 *  followed by Dynamic, followed by other finite values. The reason for giving Dynamic the priority over
 *  finite values is that min(3, Dynamic) should be Dynamic, since that could be anything between 0 and 3.
 */
template<typename A, typename B>
inline constexpr int min_size_prefer_dynamic(A a, B b) {
  static_assert(is_int_or_enum_v<A>, "Argument a must be an integer or enum");
  static_assert(is_int_or_enum_v<B>, "Argument b must be an integer or enum");
  if ((int) a == 0 || (int) b == 0) return 0;
  if ((int) a == 1 || (int) b == 1) return 1;
  if ((int) a == Dynamic || (int) b == Dynamic) return Dynamic;
  return plain_enum_min(a, b);
}

/**
 * \internal
 *  min_size_prefer_fixed is a variant of `min_size_prefer_dynamic` comparing MaxSizes. The difference is that finite values
 *  now have priority over Dynamic, so that min(3, Dynamic) gives 3. Indeed, whatever the actual value is
 *  (between 0 and 3), it is not more than 3.
 */
template<typename A, typename B>
inline constexpr int min_size_prefer_fixed(A a, B b) {
  static_assert(is_int_or_enum_v<A>, "Argument a must be an integer or enum");
  static_assert(is_int_or_enum_v<B>, "Argument b must be an integer or enum");
  if ((int) a == 0 || (int) b == 0) return 0;
  if ((int) a == 1 || (int) b == 1) return 1;
  if ((int) a == Dynamic && (int) b == Dynamic) return Dynamic;
  if ((int) a == Dynamic) return (int) b;
  if ((int) b == Dynamic) return (int) a;
  return plain_enum_min(a, b);
}

/// \internal see `min_size_prefer_fixed`. No need for a separate variant for MaxSizes here.
template<typename A, typename B>
inline constexpr int max_size_prefer_dynamic(A a, B b) {
  static_assert(is_int_or_enum_v<A>, "Argument a must be an integer or enum");
  static_assert(is_int_or_enum_v<B>, "Argument b must be an integer or enum");
  if ((int) a == Dynamic || (int) b == Dynamic) return Dynamic;
  return plain_enum_max(a, b);
}

/// \internal Calculate logical XOR at compile time
inline constexpr bool logical_xor(bool a, bool b) {
  return a != b;
}

/// \internal Calculate logical IMPLIES at compile time
inline constexpr bool check_implication(bool a, bool b) {
  return !a || b;
}
} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_META_H

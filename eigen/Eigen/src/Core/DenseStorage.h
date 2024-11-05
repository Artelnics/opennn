// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010-2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H

#ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
#define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(X) \
  X;                                                \
  EIGEN_DENSE_STORAGE_CTOR_PLUGIN;
#else
#define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(X)
#endif

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#if defined(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
#define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)
#else
#define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)                                              \
  eigen_assert((internal::is_constant_evaluated() || (std::uintptr_t(array) % Alignment == 0)) && \
               "this assertion is explained here: "                                               \
               "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html"       \
               " **** READ THIS WEB PAGE !!! ****");
#endif

#if EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_MAKE_STACK_ALLOCATION_ASSERT(X) \
  EIGEN_STATIC_ASSERT(X <= EIGEN_STACK_ALLOCATION_LIMIT, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG)
#else
#define EIGEN_MAKE_STACK_ALLOCATION_ASSERT(X)
#endif

/** \internal
 * Static array. If the MatrixOrArrayOptions require auto-alignment, the array will be automatically aligned:
 * to 16 bytes boundary if the total size is a multiple of 16 bytes.
 */
template <typename T, int Size, int MatrixOrArrayOptions,
          int Alignment = (MatrixOrArrayOptions & DontAlign) ? 0 : compute_default_alignment<T, Size>::value>
struct plain_array {
  EIGEN_ALIGN_TO_BOUNDARY(Alignment) T array[Size];
#if defined(EIGEN_NO_DEBUG) || defined(EIGEN_TESTING_PLAINOBJECT_CTOR)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() {
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(Alignment)
    EIGEN_MAKE_STACK_ALLOCATION_ASSERT(Size * sizeof(T))
  }
#endif
};

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 0> {
  T array[Size];
#if defined(EIGEN_NO_DEBUG) || defined(EIGEN_TESTING_PLAINOBJECT_CTOR)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() { EIGEN_MAKE_STACK_ALLOCATION_ASSERT(Size * sizeof(T)) }
#endif
};

template <typename T, int MatrixOrArrayOptions, int Alignment>
struct plain_array<T, 0, MatrixOrArrayOptions, Alignment> {
  T array[1];
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr plain_array() = default;
};

// this class is intended to be inherited by DenseStorage to take advantage of empty base optimization
template <int Rows, int Cols>
struct DenseStorageIndices {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index /*rows*/, Index /*cols*/) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& /*other*/) noexcept {}
};
template <int Rows>
struct DenseStorageIndices<Rows, Dynamic> {
  Index m_cols;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() : m_cols(0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index /*rows*/, Index cols) : m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return Rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return Rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index /*rows*/, Index cols) { m_cols = cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& other) noexcept {
    numext::swap(m_cols, other.m_cols);
  }
};
template <int Cols>
struct DenseStorageIndices<Dynamic, Cols> {
  Index m_rows;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() : m_rows(0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index rows, Index /*cols*/) : m_rows(rows) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * Cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index rows, Index /*cols*/) { m_rows = rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& other) noexcept {
    numext::swap(m_rows, other.m_rows);
  }
};
template <>
struct DenseStorageIndices<Dynamic, Dynamic> {
  Index m_rows;
  Index m_cols;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices() : m_rows(0), m_cols(0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(const DenseStorageIndices&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices& operator=(DenseStorageIndices&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorageIndices(Index rows, Index cols)
      : m_rows(rows), m_cols(cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index rows() const { return m_rows; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index cols() const { return m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr Index size() const { return m_rows * m_cols; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void set(Index rows, Index cols) {
    m_rows = rows;
    m_cols = cols;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void swap(DenseStorageIndices& other) noexcept {
    numext::swap(m_rows, other.m_rows);
    numext::swap(m_cols, other.m_cols);
  }
};

template <int Size, int Rows, int Cols>
struct use_trivial_ctors {
  static constexpr bool value = (Size >= 0) && (Rows >= 0) && (Cols >= 0) && (Size == Rows * Cols);
};

}  // end namespace internal

/** \internal
 *
 * \class DenseStorage
 * \ingroup Core_Module
 *
 * \brief Stores the data of a matrix
 *
 * This class stores the data of fixed-size, dynamic-size or mixed matrices
 * in a way as compact as possible.
 *
 * \sa Matrix
 */
template <typename T, int Size, int Rows, int Cols, int Options,
          bool Trivial = internal::use_trivial_ctors<Size, Rows, Cols>::value>
class DenseStorage;

// fixed-size storage with fixed dimensions
template <typename T, int Size, int Rows, int Cols, int Options>
class DenseStorage<T, Size, Rows, Cols, Options, true> : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;

  internal::plain_array<T, Size, Options> m_data;

 public:
  using Base::cols;
  using Base::rows;
#ifndef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
#else
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() { EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size) }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : Base(other), m_data(other.m_data) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = Size)
  }
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index cols) : Base(rows, cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    EIGEN_UNUSED_VARIABLE(size);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    numext::swap(m_data, other.m_data);
    Base::swap(other);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.array; }
};
// fixed-size storage with dynamic dimensions
template <typename T, int Size, int Rows, int Cols, int Options>
class DenseStorage<T, Size, Rows, Cols, Options, false> : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;

  internal::plain_array<T, Size, Options> m_data;

 public:
  using Base::cols;
  using Base::rows;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other) : Base(other), m_data() {
    Index size = other.size();
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data.array, other.m_data.array + size, m_data.array);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) : Base(other), m_data() {
    Index size = other.size();
    internal::smart_move(other.m_data.array, other.m_data.array + size, m_data.array);
    other.resize(Size, 0, 0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    Base::set(other.rows(), other.cols());
    Index size = other.size();
    internal::smart_copy(other.m_data.array, other.m_data.array + size, m_data.array);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) {
    Base::set(other.rows(), other.cols());
    Index size = other.size();
    internal::smart_move(other.m_data.array, other.m_data.array + size, m_data.array);
    other.resize(Size, 0, 0);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index cols) : Base(rows, cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    EIGEN_UNUSED_VARIABLE(size);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) {
    Index thisSize = this->size();
    Index otherSize = other.size();
    Index commonSize = numext::mini(thisSize, otherSize);
    std::swap_ranges(m_data.array, m_data.array + commonSize, other.m_data.array);
    if (thisSize > otherSize)
      internal::smart_move(m_data.array + commonSize, m_data.array + thisSize, other.m_data.array + commonSize);
    else if (otherSize > thisSize)
      internal::smart_move(other.m_data.array + commonSize, other.m_data.array + otherSize, m_data.array + commonSize);
    Base::swap(other);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data.array; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data.array; }
};
// null matrix specialization
template <typename T, int Rows, int Cols, int Options>
class DenseStorage<T, 0, Rows, Cols, Options, true> : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;

 public:
  using Base::cols;
  using Base::rows;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&&) = default;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index /*size*/, Index rows, Index cols)
      : Base(rows, cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) noexcept { Base::swap(other); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index /*size*/, Index rows, Index cols) {
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return nullptr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return nullptr; }
};
// dynamic matrix specialization
template <typename T, int Rows, int Cols, int Options>
class DenseStorage<T, Dynamic, Rows, Cols, Options, false> : internal::DenseStorageIndices<Rows, Cols> {
  using Base = internal::DenseStorageIndices<Rows, Cols>;
  static constexpr int Size = Dynamic;
  static constexpr bool Align = (Options & DontAlign) == 0;

  T* m_data;

 public:
  using Base::cols;
  using Base::rows;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage() : m_data(nullptr) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(const DenseStorage& other)
      : Base(other), m_data(internal::conditional_aligned_new_auto<T, Align>(other.size())) {
    Index size = other.size();
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(DenseStorage&& other) noexcept
      : Base(other), m_data(other.m_data) {
    other.set(0, 0);
    other.m_data = nullptr;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(const DenseStorage& other) {
    Base::set(other.rows(), other.cols());
    Index size = other.size();
    m_data = internal::conditional_aligned_new_auto<T, Align>(size);
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    internal::smart_copy(other.m_data, other.m_data + size, m_data);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage& operator=(DenseStorage&& other) noexcept {
    this->swap(other);
    return *this;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr DenseStorage(Index size, Index rows, Index cols)
      : Base(rows, cols), m_data(internal::conditional_aligned_new_auto<T, Align>(size)) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
  }
  EIGEN_DEVICE_FUNC ~DenseStorage() {
    Index size = this->size();
    internal::conditional_aligned_delete_auto<T, Align>(m_data, size);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void swap(DenseStorage& other) noexcept {
    numext::swap(m_data, other.m_data);
    Base::swap(other);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void conservativeResize(Index size, Index rows, Index cols) {
    Index oldSize = this->size();
    m_data = internal::conditional_aligned_realloc_new_auto<T, Align>(m_data, size, oldSize);
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr void resize(Index size, Index rows, Index cols) {
    Index oldSize = this->size();
    if (size != oldSize) {
      internal::conditional_aligned_delete_auto<T, Align>(m_data, oldSize);
      if (size > 0)  // >0 and not simply !=0 to let the compiler knows that size cannot be negative
      {
        m_data = internal::conditional_aligned_new_auto<T, Align>(size);
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
      } else
        m_data = nullptr;
    }
    Base::set(rows, cols);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr T* data() { return m_data; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr const T* data() const { return m_data; }
};
}  // end namespace Eigen

#endif  // EIGEN_MATRIX_H

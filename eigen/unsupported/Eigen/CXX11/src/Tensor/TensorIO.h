// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_IO_H
#define EIGEN_CXX11_TENSOR_TENSOR_IO_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

struct TensorIOFormat;

namespace internal {
template <typename Tensor, std::size_t rank>
struct TensorPrinter;
}

struct TensorIOFormat {
  TensorIOFormat(const std::vector<std::string>& _separator, const std::vector<std::string>& _prefix,
                 const std::vector<std::string>& _suffix, int _precision = StreamPrecision, int _flags = 0,
                 const std::string& _tenPrefix = "", const std::string& _tenSuffix = "", const char _fill = ' ')
      : tenPrefix(_tenPrefix),
        tenSuffix(_tenSuffix),
        prefix(_prefix),
        suffix(_suffix),
        separator(_separator),
        fill(_fill),
        precision(_precision),
        flags(_flags) {
    init_spacer();
  }

  TensorIOFormat(int _precision = StreamPrecision, int _flags = 0, const std::string& _tenPrefix = "",
                 const std::string& _tenSuffix = "", const char _fill = ' ')
      : tenPrefix(_tenPrefix), tenSuffix(_tenSuffix), fill(_fill), precision(_precision), flags(_flags) {
    // default values of prefix, suffix and separator
    prefix = {"", "["};
    suffix = {"", "]"};
    separator = {", ", "\n"};

    init_spacer();
  }

  void init_spacer() {
    if ((flags & DontAlignCols)) return;
    spacer.resize(prefix.size());
    spacer[0] = "";
    int i = int(tenPrefix.length()) - 1;
    while (i >= 0 && tenPrefix[i] != '\n') {
      spacer[0] += ' ';
      i--;
    }

    for (std::size_t k = 1; k < prefix.size(); k++) {
      int j = int(prefix[k].length()) - 1;
      while (j >= 0 && prefix[k][j] != '\n') {
        spacer[k] += ' ';
        j--;
      }
    }
  }

  static inline const TensorIOFormat Numpy() {
    std::vector<std::string> prefix = {"", "["};
    std::vector<std::string> suffix = {"", "]"};
    std::vector<std::string> separator = {" ", "\n"};
    return TensorIOFormat(separator, prefix, suffix, StreamPrecision, 0, "[", "]");
  }

  static inline const TensorIOFormat Plain() {
    std::vector<std::string> separator = {" ", "\n", "\n", ""};
    std::vector<std::string> prefix = {""};
    std::vector<std::string> suffix = {""};
    return TensorIOFormat(separator, prefix, suffix, StreamPrecision, 0, "", "", ' ');
  }

  static inline const TensorIOFormat Native() {
    std::vector<std::string> separator = {", ", ",\n", "\n"};
    std::vector<std::string> prefix = {"", "{"};
    std::vector<std::string> suffix = {"", "}"};
    return TensorIOFormat(separator, prefix, suffix, StreamPrecision, 0, "{", "}", ' ');
  }

  static inline const TensorIOFormat Legacy() {
    TensorIOFormat LegacyFormat(StreamPrecision, 0, "", "", ' ');
    LegacyFormat.legacy_bit = true;
    return LegacyFormat;
  }

  std::string tenPrefix;
  std::string tenSuffix;
  std::vector<std::string> prefix;
  std::vector<std::string> suffix;
  std::vector<std::string> separator;
  char fill;
  int precision;
  int flags;
  std::vector<std::string> spacer{};
  bool legacy_bit = false;
};

template <typename T, int Layout, int rank>
class TensorWithFormat;
// specialize for Layout=ColMajor, Layout=RowMajor and rank=0.
template <typename T, int rank>
class TensorWithFormat<T, RowMajor, rank> {
 public:
  TensorWithFormat(const T& tensor, const TensorIOFormat& format) : t_tensor(tensor), t_format(format) {}

  friend std::ostream& operator<<(std::ostream& os, const TensorWithFormat<T, RowMajor, rank>& wf) {
    // Evaluate the expression if needed
    typedef TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> Evaluator;
    TensorForcedEvalOp<const T> eval = wf.t_tensor.eval();
    Evaluator tensor(eval, DefaultDevice());
    tensor.evalSubExprsIfNeeded(NULL);
    internal::TensorPrinter<Evaluator, rank>::run(os, tensor, wf.t_format);
    // Cleanup.
    tensor.cleanup();
    return os;
  }

 protected:
  T t_tensor;
  TensorIOFormat t_format;
};

template <typename T, int rank>
class TensorWithFormat<T, ColMajor, rank> {
 public:
  TensorWithFormat(const T& tensor, const TensorIOFormat& format) : t_tensor(tensor), t_format(format) {}

  friend std::ostream& operator<<(std::ostream& os, const TensorWithFormat<T, ColMajor, rank>& wf) {
    // Switch to RowMajor storage and print afterwards
    typedef typename T::Index IndexType;
    std::array<IndexType, rank> shuffle;
    std::array<IndexType, rank> id;
    std::iota(id.begin(), id.end(), IndexType(0));
    std::copy(id.begin(), id.end(), shuffle.rbegin());
    auto tensor_row_major = wf.t_tensor.swap_layout().shuffle(shuffle);

    // Evaluate the expression if needed
    typedef TensorEvaluator<const TensorForcedEvalOp<const decltype(tensor_row_major)>, DefaultDevice> Evaluator;
    TensorForcedEvalOp<const decltype(tensor_row_major)> eval = tensor_row_major.eval();
    Evaluator tensor(eval, DefaultDevice());
    tensor.evalSubExprsIfNeeded(NULL);
    internal::TensorPrinter<Evaluator, rank>::run(os, tensor, wf.t_format);
    // Cleanup.
    tensor.cleanup();
    return os;
  }

 protected:
  T t_tensor;
  TensorIOFormat t_format;
};

template <typename T>
class TensorWithFormat<T, ColMajor, 0> {
 public:
  TensorWithFormat(const T& tensor, const TensorIOFormat& format) : t_tensor(tensor), t_format(format) {}

  friend std::ostream& operator<<(std::ostream& os, const TensorWithFormat<T, ColMajor, 0>& wf) {
    // Evaluate the expression if needed
    typedef TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> Evaluator;
    TensorForcedEvalOp<const T> eval = wf.t_tensor.eval();
    Evaluator tensor(eval, DefaultDevice());
    tensor.evalSubExprsIfNeeded(NULL);
    internal::TensorPrinter<Evaluator, 0>::run(os, tensor, wf.t_format);
    // Cleanup.
    tensor.cleanup();
    return os;
  }

 protected:
  T t_tensor;
  TensorIOFormat t_format;
};

namespace internal {
template <typename Tensor, std::size_t rank>
struct TensorPrinter {
  static void run(std::ostream& s, const Tensor& _t, const TensorIOFormat& fmt) {
    typedef typename internal::remove_const<typename Tensor::Scalar>::type Scalar;
    typedef typename Tensor::Index IndexType;
    static const int layout = Tensor::Layout;
    // backwards compatibility case: print tensor after reshaping to matrix of size dim(0) x
    // (dim(1)*dim(2)*...*dim(rank-1)).
    if (fmt.legacy_bit) {
      const IndexType total_size = internal::array_prod(_t.dimensions());
      if (total_size > 0) {
        const IndexType first_dim = Eigen::internal::array_get<0>(_t.dimensions());
        Map<const Array<Scalar, Dynamic, Dynamic, layout> > matrix(_t.data(), first_dim,
                                                                   total_size / first_dim);
        s << matrix;
        return;
      }
    }

    assert(layout == RowMajor);
    typedef typename conditional<is_same<Scalar, char>::value || is_same<Scalar, unsigned char>::value ||
                                     is_same<Scalar, numext::int8_t>::value || is_same<Scalar, numext::uint8_t>::value,
                                 int,
                                 typename conditional<is_same<Scalar, std::complex<char> >::value ||
                                                          is_same<Scalar, std::complex<unsigned char> >::value ||
                                                          is_same<Scalar, std::complex<numext::int8_t> >::value ||
                                                          is_same<Scalar, std::complex<numext::uint8_t> >::value,
                                                      std::complex<int>, const Scalar&>::type>::type PrintType;

    const IndexType total_size = array_prod(_t.dimensions());

    std::streamsize explicit_precision;
    if (fmt.precision == StreamPrecision) {
      explicit_precision = 0;
    } else if (fmt.precision == FullPrecision) {
      if (NumTraits<Scalar>::IsInteger) {
        explicit_precision = 0;
      } else {
        explicit_precision = significant_decimals_impl<Scalar>::run();
      }
    } else {
      explicit_precision = fmt.precision;
    }

    std::streamsize old_precision = 0;
    if (explicit_precision) old_precision = s.precision(explicit_precision);

    IndexType width = 0;

    bool align_cols = !(fmt.flags & DontAlignCols);
    if (align_cols) {
      // compute the largest width
      for (IndexType i = 0; i < total_size; i++) {
        std::stringstream sstr;
        sstr.copyfmt(s);
        sstr << static_cast<PrintType>(_t.data()[i]);
        width = std::max<IndexType>(width, IndexType(sstr.str().length()));
      }
    }
    std::streamsize old_width = s.width();
    char old_fill_character = s.fill();

    s << fmt.tenPrefix;
    for (IndexType i = 0; i < total_size; i++) {
      std::array<bool, rank> is_at_end{};
      std::array<bool, rank> is_at_begin{};

      // is the ith element the end of an coeff (always true), of a row, of a matrix, ...?
      for (std::size_t k = 0; k < rank; k++) {
        if ((i + 1) % (std::accumulate(_t.dimensions().rbegin(), _t.dimensions().rbegin() + k, 1,
                                       std::multiplies<IndexType>())) ==
            0) {
          is_at_end[k] = true;
        }
      }

      // is the ith element the begin of an coeff (always true), of a row, of a matrix, ...?
      for (std::size_t k = 0; k < rank; k++) {
        if (i % (std::accumulate(_t.dimensions().rbegin(), _t.dimensions().rbegin() + k, 1,
                                 std::multiplies<IndexType>())) ==
            0) {
          is_at_begin[k] = true;
        }
      }

      // do we have a line break?
      bool is_at_begin_after_newline = false;
      for (std::size_t k = 0; k < rank; k++) {
        if (is_at_begin[k]) {
          std::size_t separator_index = (k < fmt.separator.size()) ? k : fmt.separator.size() - 1;
          if (fmt.separator[separator_index].find('\n') != std::string::npos) {
            is_at_begin_after_newline = true;
          }
        }
      }

      bool is_at_end_before_newline = false;
      for (std::size_t k = 0; k < rank; k++) {
        if (is_at_end[k]) {
          std::size_t separator_index = (k < fmt.separator.size()) ? k : fmt.separator.size() - 1;
          if (fmt.separator[separator_index].find('\n') != std::string::npos) {
            is_at_end_before_newline = true;
          }
        }
      }

      std::stringstream suffix, prefix, separator;
      for (std::size_t k = 0; k < rank; k++) {
        std::size_t suffix_index = (k < fmt.suffix.size()) ? k : fmt.suffix.size() - 1;
        if (is_at_end[k]) {
          suffix << fmt.suffix[suffix_index];
        }
      }
      for (std::size_t k = 0; k < rank; k++) {
        std::size_t separator_index = (k < fmt.separator.size()) ? k : fmt.separator.size() - 1;
        if (is_at_end[k] &&
            (!is_at_end_before_newline || fmt.separator[separator_index].find('\n') != std::string::npos)) {
          separator << fmt.separator[separator_index];
        }
      }
      for (std::size_t k = 0; k < rank; k++) {
        std::size_t spacer_index = (k < fmt.spacer.size()) ? k : fmt.spacer.size() - 1;
        if (i != 0 && is_at_begin_after_newline && (!is_at_begin[k] || k == 0)) {
          prefix << fmt.spacer[spacer_index];
        }
      }
      for (int k = rank - 1; k >= 0; k--) {
        std::size_t prefix_index = (static_cast<std::size_t>(k) < fmt.prefix.size()) ? k : fmt.prefix.size() - 1;
        if (is_at_begin[k]) {
          prefix << fmt.prefix[prefix_index];
        }
      }

      s << prefix.str();
      if (width) {
        s.fill(fmt.fill);
        s.width(width);
        s << std::right;
      }
      s << _t.data()[i];
      s << suffix.str();
      if (i < total_size - 1) {
        s << separator.str();
      }
    }
    s << fmt.tenSuffix;
    if (explicit_precision) s.precision(old_precision);
    if (width) {
      s.fill(old_fill_character);
      s.width(old_width);
    }
  }
};

template <typename Tensor>
struct TensorPrinter<Tensor, 0> {
  static void run(std::ostream& s, const Tensor& _t, const TensorIOFormat& fmt) {
    typedef typename Tensor::Scalar Scalar;

    std::streamsize explicit_precision;
    if (fmt.precision == StreamPrecision) {
      explicit_precision = 0;
    } else if (fmt.precision == FullPrecision) {
      if (NumTraits<Scalar>::IsInteger) {
        explicit_precision = 0;
      } else {
        explicit_precision = significant_decimals_impl<Scalar>::run();
      }
    } else {
      explicit_precision = fmt.precision;
    }

    std::streamsize old_precision = 0;
    if (explicit_precision) old_precision = s.precision(explicit_precision);

    s << fmt.tenPrefix << _t.coeff(0) << fmt.tenSuffix;
    if (explicit_precision) s.precision(old_precision);
  }
};

}  // end namespace internal
template <typename T>
std::ostream& operator<<(std::ostream& s, const TensorBase<T, ReadOnlyAccessors>& t) {
  s << t.format(TensorIOFormat::Plain());
  return s;
}
}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_IO_H

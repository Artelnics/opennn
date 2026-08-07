//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P A R A L L E L   A L G O R I T H M S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once
#ifndef OPENNN_PARALLEL_ALGORITHMS_H_
#define OPENNN_PARALLEL_ALGORITHMS_H_

#include "opennn_types.h"

// Only pulled in when the toolchain can actually link the parallel
// backend; see the OPENNN_HAS_PARALLEL_ALGORITHMS block in CMakeLists.
#if defined(OPENNN_HAS_PARALLEL_ALGORITHMS)
#include <execution>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace opennn
{

inline constexpr ptrdiff_t parallel_algorithm_min_size = 65536;

inline bool use_parallel_algorithm(ptrdiff_t size) noexcept
{
    if (size < parallel_algorithm_min_size) return false;
#if defined(_OPENMP)
    if (omp_in_parallel()) return false;
#endif
    return true;
}

template<typename Iterator, typename Compare = less<>>
void sort_parallel_if_large(Iterator first, Iterator last, Compare compare = {})
{
#if defined(OPENNN_HAS_PARALLEL_ALGORITHMS) && defined(__cpp_lib_parallel_algorithm)
    if (use_parallel_algorithm(last - first))
        sort(execution::par, first, last, compare);
    else
#endif
        sort(first, last, compare);
}

template<typename Iterator, typename Compare = less<>>
void stable_sort_parallel_if_large(Iterator first, Iterator last, Compare compare = {})
{
#if defined(OPENNN_HAS_PARALLEL_ALGORITHMS) && defined(__cpp_lib_parallel_algorithm)
    if (use_parallel_algorithm(last - first))
        stable_sort(execution::par, first, last, compare);
    else
#endif
        stable_sort(first, last, compare);
}

template<typename Iterator, typename Compare = less<>>
void nth_element_parallel_if_large(Iterator first, Iterator nth, Iterator last,
                                   Compare compare = {})
{
#if defined(OPENNN_HAS_PARALLEL_ALGORITHMS) && defined(__cpp_lib_parallel_algorithm)
    if (use_parallel_algorithm(last - first))
        nth_element(execution::par, first, nth, last, compare);
    else
#endif
        nth_element(first, nth, last, compare);
}

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

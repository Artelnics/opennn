/*******************************************************************************
* Copyright 2020-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) helper for OpenMP offload
!******************************************************************************/

#ifndef _MKL_OMP_VARIANT_H_
#define _MKL_OMP_VARIANT_H_

#ifdef __cplusplus
#if __cplusplus > 199711L
#define NOTHROW noexcept
#else
#define NOTHROW throw()
#endif
#else
#define NOTHROW
#endif

#ifdef MKL_ILP64
#define MKL_VARIANT_NAME(domain, func) mkl_ ## domain ## _ ## func ## _omp_offload_ilp64
#else
#define MKL_VARIANT_NAME(domain, func) mkl_ ## domain ## _ ## func ## _omp_offload_lp64
#endif

#endif /* _MKL_OMP_VARIANT_H_ */

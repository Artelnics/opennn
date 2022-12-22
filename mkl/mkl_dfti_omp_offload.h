/*******************************************************************************
* Copyright 2019-2022 Intel Corporation.
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
!      Intel(R) oneAPI Math Kernel Library (oneMKL)
!      Discrete Fourier Transform Interface (DFTI) for OpenMP target (offload)
!******************************************************************************/


#ifndef MKL_DFTI_OMP_OFFLOAD_H_INCLUDED
#define MKL_DFTI_OMP_OFFLOAD_H_INCLUDED

#include "mkl_dfti.h"
#include "mkl_service.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

    DFTI_EXTERN MKL_LONG mkl_DftiCommitDescriptor_omp_offload(DFTI_DESCRIPTOR_HANDLE);

#pragma omp declare variant(mkl_DftiCommitDescriptor_omp_offload)             \
    match(construct = {target variant dispatch}, device = {arch(gen)})
    DFTI_EXTERN MKL_LONG DftiCommitDescriptor(DFTI_DESCRIPTOR_HANDLE);


    DFTI_EXTERN MKL_LONG mkl_DftiComputeForward_omp_offload(DFTI_DESCRIPTOR_HANDLE,
                                                            void *, ...);

#pragma omp declare variant(mkl_DftiComputeForward_omp_offload)               \
    match(construct = {target variant dispatch}, device = {arch(gen)})
    DFTI_EXTERN MKL_LONG DftiComputeForward(DFTI_DESCRIPTOR_HANDLE,
                                            void *, ...);


    DFTI_EXTERN MKL_LONG mkl_DftiComputeBackward_omp_offload(DFTI_DESCRIPTOR_HANDLE,
                                                             void *, ...);

#pragma omp declare variant(mkl_DftiComputeBackward_omp_offload)              \
    match(construct = {target variant dispatch}, device = {arch(gen)})
    DFTI_EXTERN MKL_LONG DftiComputeBackward(DFTI_DESCRIPTOR_HANDLE,
                                             void *, ...);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MKL_DFTI_OMP_OFFLOAD_H_INCLUDED

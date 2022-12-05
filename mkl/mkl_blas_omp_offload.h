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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) C/C++ OpenMP offload
!      interface
!******************************************************************************/

#ifndef _MKL_BLAS_OMP_OFFLOAD_H_
#define _MKL_BLAS_OMP_OFFLOAD_H_

#include "mkl_types.h"
#include "mkl_blas_omp_variant.h"

#if (_OPENMP >= 202011)
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

// Matrix transposition and copy API

#define mkl_simatcopy_batch_strided MKL_Simatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(simatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:AB)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(simatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Simatcopy_batch_strided(const char ordering, const char trans,
                                 size_t rows, size_t cols,
                                 const float alpha,
                                 float * AB, size_t lda, size_t ldb,
                                 size_t stride, size_t batch_size) NOTHROW;

#define mkl_dimatcopy_batch_strided MKL_Dimatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma  omp declare variant (MKL_CBLAS_VARIANT_NAME(dimatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dimatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Dimatcopy_batch_strided(const char ordering, const char trans,
                                 size_t rows, size_t cols,
                                 const double alpha,
                                 double * AB, size_t lda, size_t ldb,
                                 size_t stride, size_t batch_size) NOTHROW;

#define mkl_cimatcopy_batch_strided MKL_Cimatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cimatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cimatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Cimatcopy_batch_strided(const char ordering, const char trans,
                                 size_t rows, size_t cols,
                                 const MKL_Complex8 alpha,
                                 MKL_Complex8 * AB, size_t lda, size_t ldb,
                                 size_t stride, size_t batch_size) NOTHROW;

#define mkl_zimatcopy_batch_strided MKL_Zimatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zimatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zimatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Zimatcopy_batch_strided(const char ordering, const char trans,
                                 size_t rows, size_t cols,
                                 const MKL_Complex16 alpha,
                                 MKL_Complex16 * AB, size_t lda, size_t ldb,
                                 size_t stride, size_t batch_size) NOTHROW;

#define mkl_somatcopy_batch_strided MKL_Somatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(somatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(somatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Somatcopy_batch_strided(char ordering, char trans,
                                 size_t rows, size_t cols,
                                 const float alpha,
                                 const float * A, size_t lda, size_t stridea,
                                 float *B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

#define mkl_domatcopy_batch_strided MKL_Domatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(domatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(domatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Domatcopy_batch_strided(char ordering, char trans,
                                 size_t rows, size_t cols,
                                 const double alpha,
                                 const double * A, size_t lda, size_t stridea,
                                 double *B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

#define mkl_comatcopy_batch_strided MKL_Comatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(comatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(comatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Comatcopy_batch_strided(char ordering, char trans,
                                 size_t rows, size_t cols,
                                 const MKL_Complex8 alpha,
                                 const MKL_Complex8 * A, size_t lda, size_t stridea,
                                 MKL_Complex8 *B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

#define mkl_zomatcopy_batch_strided MKL_Zomatcopy_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zomatcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zomatcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Zomatcopy_batch_strided(char ordering, char trans,
                                 size_t rows, size_t cols,
                                 const MKL_Complex16 alpha,
                                 const MKL_Complex16 * A, size_t lda, size_t stridea,
                                 MKL_Complex16 *B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

#define mkl_simatcopy_batch MKL_Simatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(simatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(simatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Simatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const float * alpha_array, float ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

#define mkl_dimatcopy_batch MKL_Dimatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dimatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dimatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Dimatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const double * alpha_array, double ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

#define mkl_cimatcopy_batch MKL_Cimatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cimatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cimatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Cimatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex8 * alpha_array, MKL_Complex8 ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

#define mkl_zimatcopy_batch MKL_Zimatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zimatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:AB_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zimatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Zimatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex16 * alpha_array, MKL_Complex16 ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

#define mkl_somatcopy_batch MKL_Somatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(somatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_array,B_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(somatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Somatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const float * alpha_array, const float ** A_array,
    const size_t * lda_array, float ** B_array,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

#define mkl_domatcopy_batch MKL_Domatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(domatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_array,B_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(domatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Domatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const double * alpha_array, const double ** A_array,
    const size_t * lda_array, double ** B_array,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

#define mkl_comatcopy_batch MKL_Comatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(comatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_array,B_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(comatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Comatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex8 * alpha_array, const MKL_Complex8 ** A_array,
    const size_t * lda_array, MKL_Complex8 ** B_array,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

#define mkl_zomatcopy_batch MKL_Zomatcopy_batch
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zomatcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_array,B_array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zomatcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Zomatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex16 * alpha_array, const MKL_Complex16 ** A_array,
    const size_t * lda_array, MKL_Complex16 ** B_array,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

#define mkl_somatadd_batch_strided MKL_Somatadd_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(somatadd_batch_strided)) \
    match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(somatadd_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Somatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const float alpha, const float * A, size_t lda, size_t stridea,
    const float beta, const float * B, size_t ldb, size_t strideb,
    float * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

#define mkl_domatadd_batch_strided MKL_Domatadd_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(domatadd_batch_strided)) \
    match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(domatadd_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Domatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const double alpha, const double * A, size_t lda, size_t stridea,
    const double beta, const double * B, size_t ldb, size_t strideb,
    double * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

#define mkl_comatadd_batch_strided MKL_Comatadd_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(comatadd_batch_strided)) \
    match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(comatadd_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Comatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha, const MKL_Complex8 * A, size_t lda, size_t stridea,
    const MKL_Complex8 beta, const MKL_Complex8 * B, size_t ldb, size_t strideb,
    MKL_Complex8 * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

#define mkl_zomatadd_batch_strided MKL_Zomatadd_batch_strided
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zomatadd_batch_strided)) \
    match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zomatadd_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void MKL_Zomatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha, const MKL_Complex16 * A, size_t lda, size_t stridea,
    const MKL_Complex16 beta, const MKL_Complex16 * B, size_t ldb, size_t strideb,
    MKL_Complex16 * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

    
// BATCH APIs

// Level3

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const float *alpha_Array, const float **A_Array,
                       const MKL_INT *lda_Array, const float **B_Array, const MKL_INT *ldb_Array,
                       const float *beta_Array, float **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const double *alpha_Array, const double **A_Array,
                       const MKL_INT *lda_Array, const double **B_Array, const MKL_INT* ldb_Array,
                       const double *beta_Array, double **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                       const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                       const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemm_batch(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT *lda_array,
                 const float **b_array, const MKL_INT *ldb_array,
                 const float *beta_array, float **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemm_batch(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT *lda_array,
                 const double **b_array, const MKL_INT *ldb_array,
                 const double *beta_array, double **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemm_batch(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array,
                 const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemm_batch(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array,
                 const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const float alpha, const float *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const float *B, const MKL_INT ldb, const MKL_INT strideb,
                               const float beta, float *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const double alpha, const double *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const double *B, const MKL_INT ldb, const MKL_INT strideb,
                               const double beta, double *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *B, const MKL_INT ldb, const MKL_INT strideb,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *B, const MKL_INT ldb, const MKL_INT strideb,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const float *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const float *beta, float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const double *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const double *beta, double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                       const MKL_INT *N_Array, const MKL_INT *K_Array,
                       const float *alpha_Array, const float **A_Array, const MKL_INT *lda_Array,
                       const float *beta_Array, float **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                       const MKL_INT *N_Array, const MKL_INT *K_Array,
                       const double *alpha_Array, const double **A_Array, const MKL_INT *lda_Array,
                       const double *beta_Array, double **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                       const MKL_INT *N_Array, const MKL_INT *K_Array,
                       const void *alpha_Array, const void **A_Array, const MKL_INT *lda_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,C_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zsyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                       const MKL_INT *N_Array, const MKL_INT *K_Array,
                       const void *alpha_Array, const void **A_Array, const MKL_INT *lda_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                               const MKL_INT K, const float alpha, const float *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const float beta, float *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                               const MKL_INT K, const double alpha, const double *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const double beta, double *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;    

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zsyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;    
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyrk_batch(const char *uplo_array, const char *trans_array,
                 const MKL_INT *n_array, const MKL_INT *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT *lda_array,
                 const float *beta_array, float **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyrk_batch(const char *uplo_array, const char *trans_array,
                 const MKL_INT *n_array, const MKL_INT *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT *lda_array,
                 const double *beta_array, double **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void csyrk_batch(const char *uplo_array, const char *trans_array,
                 const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyrk_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,c_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyrk_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zsyrk_batch(const char *uplo_array, const char *trans_array,
                 const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyrk_batch_strided(const char *Uplo,
                         const char *Trans, const MKL_INT *N,
                         const MKL_INT *K, const float *alpha, const float *A,
                         const MKL_INT *lda, const MKL_INT *stridea,
                         const float *beta, float *C, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyrk_batch_strided(const char *Uplo,
                         const char *Trans, const MKL_INT *N,
                         const MKL_INT *K, const double *alpha, const double *A,
                         const MKL_INT *lda, const MKL_INT *stridea,
                         const double *beta, double *C, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;    

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void csyrk_batch_strided(const char *Uplo,
                         const char *Trans, const MKL_INT *N,
                         const MKL_INT *K, const MKL_Complex8 *alpha, const MKL_Complex8 *A,
                         const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *beta, MKL_Complex8 *C, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyrk_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyrk_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zsyrk_batch_strided(const char *Uplo,
                         const char *Trans, const MKL_INT *N,
                         const MKL_INT *K, const MKL_Complex16 *alpha, const MKL_Complex16 *A,
                         const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *beta, MKL_Complex16 *C, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;    
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_strsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                           const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *TransA_Array,
                           const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                           const MKL_INT *N_Array, const float *alpha_Array,
                           const float **A_Array, const MKL_INT *lda_Array,
                           float **B_Array, const MKL_INT *ldb_Array,
                           const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtrsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                           const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                           const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                           const MKL_INT *N_Array, const double *alpha_Array,
                           const double **A_Array, const MKL_INT *lda_Array,
                           double **B_Array, const MKL_INT *ldb_Array,
                           const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctrsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                           const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                           const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                           const MKL_INT *N_Array, const void *alpha_Array,
                           const void **A_Array, const MKL_INT *lda_Array,
                           void **B_Array, const MKL_INT *ldb_Array,
                           const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A_Array,B_Array)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztrsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                           const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                           const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                           const MKL_INT *N_Array, const void *alpha_Array,
                           const void **A_Array, const MKL_INT *lda_Array,
                           void **B_Array, const MKL_INT *ldb_Array,
                           const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_strsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                   const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_DIAG Diag, const MKL_INT M,
                                   const MKL_INT N, const float alpha,
                                   const float *A, const MKL_INT lda, const MKL_INT stridea,
                                   float *B, const MKL_INT ldb, const MKL_INT strideb,
                                   const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtrsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                   const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_DIAG Diag, const MKL_INT M,
                                   const MKL_INT N, const double alpha,
                                   const double *A, const MKL_INT lda, const MKL_INT stridea,
                                   double *B, const MKL_INT ldb, const MKL_INT strideb,
                                   const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctrsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                   const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_DIAG Diag, const MKL_INT M,
                                   const MKL_INT N, const void *alpha,
                                   const void *A, const MKL_INT lda, const MKL_INT stridea,
                                   void *B, const MKL_INT ldb, const MKL_INT strideb,
                                   const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztrsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                   const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_DIAG Diag, const MKL_INT M,
                                   const MKL_INT N, const void *alpha,
                                   const void *A, const MKL_INT lda, const MKL_INT stridea,
                                   void *B, const MKL_INT ldb, const MKL_INT strideb,
                                   const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void strsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                     const MKL_INT *m_array, const MKL_INT *n_array, const float *alpha_array, const float **a_array,
                     const MKL_INT *lda_array, float **b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                     const MKL_INT *m_array, const MKL_INT *n_array, const double *alpha_array, const double **a_array,
                     const MKL_INT *lda_array, double **b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                     const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array,
                     const MKL_INT *lda_array, MKL_Complex8 **b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a_array,b_array)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                     const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array,
                     const MKL_INT *lda_array, MKL_Complex16 **b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void strsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                             const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a,
                             const MKL_INT *lda, const MKL_INT *stridea, float *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                             const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a,
                             const MKL_INT *lda, const MKL_INT *stridea, double *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                             const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a,
                             const MKL_INT *lda, const MKL_INT *stridea, MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                             const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a,
                             const MKL_INT *lda, const MKL_INT *stridea, MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size) NOTHROW;
    
// Level2

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                 const float **a, const MKL_INT *lda, const float **x, const MKL_INT *incx,
                 const float *beta, float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex, 
                         const float *beta, float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                 const double **a, const MKL_INT *lda, const double **x, const MKL_INT *incx,
                 const double *beta, double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex, 
                         const double *beta, double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **a, const MKL_INT *lda, const MKL_Complex8 **x, const MKL_INT *incx,
                 const MKL_Complex8 *beta, MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex, 
                         const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **a, const MKL_INT *lda, const MKL_Complex16 **x, const MKL_INT *incx,
                 const MKL_Complex16 *beta, MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex, 
                         const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const float *alpha, const float **A, const MKL_INT *lda,
                       const float **X, const MKL_INT *incX, const float *beta,
                       float **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const float alpha, const float *A, const MKL_INT lda, const MKL_INT stridea,
                               const float *X, const MKL_INT incX, const MKL_INT stridex, const float beta,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const double *alpha, const double **A, const MKL_INT *lda,
                       const double **X, const MKL_INT *incX, const double *beta,
                       double **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const double alpha, const double *A, const MKL_INT lda, const MKL_INT stridea,
                               const double *X, const MKL_INT incX, const MKL_INT stridex, const double beta,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const void *alpha, const void **A, const MKL_INT *lda,
                       const void **X, const MKL_INT *incX, const void *beta,
                       void **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                               const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemv_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemv_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const void *alpha, const void **A, const MKL_INT *lda,
                       const void **X, const MKL_INT *incX, const void *beta,
                       void **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemv_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemv_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                               const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void sdgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const float **a, const MKL_INT *lda,
                 const float **x, const MKL_INT *incx,
                 float **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sdgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ddgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ddgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void ddgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const double **a, const MKL_INT *lda,
                 const double **x, const MKL_INT *incx,
                 double **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ddgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ddgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void ddgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cdgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex8 **a, const MKL_INT *lda,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cdgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;
    
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zdgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex16 **a, const MKL_INT *lda,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zdgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sdgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const float **a, const MKL_INT *lda,
                       const float **x, const MKL_INT *incx,
                       float **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sdgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const float *a, const MKL_INT lda, const MKL_INT stridea,
                               const float *x, const MKL_INT incx, const MKL_INT stridex,
                               float *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ddgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ddgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ddgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const double **a, const MKL_INT *lda,
                       const double **x, const MKL_INT *incx,
                       double **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ddgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ddgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ddgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const double *a, const MKL_INT lda, const MKL_INT stridea,
                               const double *x, const MKL_INT incx, const MKL_INT stridex,
                               double *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cdgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const void **a, const MKL_INT *lda,
                       const void **x, const MKL_INT *incx,
                       void **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cdgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const void *a, const MKL_INT lda, const MKL_INT stridea,
                               const void *x, const MKL_INT incx, const MKL_INT stridex,
                               void *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdgmm_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdgmm_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zdgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const void **a, const MKL_INT *lda,
                       const void **x, const MKL_INT *incx,
                       void **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdgmm_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,c)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdgmm_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zdgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const void *a, const MKL_INT lda, const MKL_INT stridea,
                               const void *x, const MKL_INT incx, const MKL_INT stridex,
                               void *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

// Level 1

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_saxpy_batch(const MKL_INT *n, const float *alpha,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_daxpy_batch(const MKL_INT *n, const double *alpha,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)    
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})    
void cblas_caxpy_batch(const MKL_INT *n, const void *alpha,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zaxpy_batch(const MKL_INT *n, const void *alpha,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void saxpy_batch(const MKL_INT *n, const float *alpha,
                 const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void daxpy_batch(const MKL_INT *n, const double *alpha,
                 const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void caxpy_batch(const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zaxpy_batch(const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_saxpy_batch_strided(const MKL_INT N, const float alpha,
                               const float *X, const MKL_INT incX, const MKL_INT stridex,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_daxpy_batch_strided(const MKL_INT N, const double alpha,
                               const double *X, const MKL_INT incX, const MKL_INT stridex,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_caxpy_batch_strided(const MKL_INT N, const void *alpha,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zaxpy_batch_strided(const MKL_INT N, const void *alpha,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void saxpy_batch_strided(const MKL_INT *n, const float *alpha,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void daxpy_batch_strided(const MKL_INT *n, const double *alpha,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void caxpy_batch_strided(const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zaxpy_batch_strided(const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_scopy_batch(const MKL_INT *n,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dcopy_batch(const MKL_INT *n,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ccopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ccopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ccopy_batch(const MKL_INT *n,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zcopy_batch(const MKL_INT *n,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void scopy_batch(const MKL_INT *n,
                 const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dcopy_batch(const MKL_INT *n,
                 const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ccopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ccopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void ccopy_batch(const MKL_INT *n,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zcopy_batch)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zcopy_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zcopy_batch(const MKL_INT *n,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_scopy_batch_strided(const MKL_INT N,
                               const float *X, const MKL_INT incX, const MKL_INT stridex,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dcopy_batch_strided(const MKL_INT N,
                               const double *X, const MKL_INT incX, const MKL_INT stridex,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ccopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ccopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ccopy_batch_strided(const MKL_INT N,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zcopy_batch_strided(const MKL_INT N,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void scopy_batch_strided(const MKL_INT *N,
                         const float *X, const MKL_INT *incX, const MKL_INT *stridex,
                         float *Y, const MKL_INT *incY, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dcopy_batch_strided(const MKL_INT *N,
                         const double *X, const MKL_INT *incX, const MKL_INT *stridex,
                         double *Y, const MKL_INT *incY, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ccopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ccopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void ccopy_batch_strided(const MKL_INT *N,
                         const MKL_Complex8 *X, const MKL_INT *incX, const MKL_INT *stridex,
                         MKL_Complex8 *Y, const MKL_INT *incY, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zcopy_batch_strided)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zcopy_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zcopy_batch_strided(const MKL_INT *N,
                         const MKL_Complex16 *X, const MKL_INT *incX, const MKL_INT *stridex,
                         MKL_Complex16 *Y, const MKL_INT *incY, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

// CBLAS API

// Level3 

// Routines with S, D, C, Z prefixes (Standard)
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const float alpha, const float *A,
                 const MKL_INT lda, const float *B, const MKL_INT ldb,
                 const float beta, float *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const float alpha, const float *A, const MKL_INT lda,
                  const float *B, const MKL_INT ldb, const float beta,
                  float *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssymm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 const float *B, const MKL_INT ldb, const float beta,
                 float *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const float alpha, const float *A, const MKL_INT lda,
                  const float *B, const MKL_INT ldb, const float beta,
                  float *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const float alpha, const float *A, const MKL_INT lda,
                 const float beta, float *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strmm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_strmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 float *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_strsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 float *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const double alpha, const double *A,
                 const MKL_INT lda, const double *B, const MKL_INT ldb,
                 const double beta, double *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const double alpha, const double *A, const MKL_INT lda,
                  const double *B, const MKL_INT ldb, const double beta,
                  double *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsymm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 const double *B, const MKL_INT ldb, const double beta,
                 double *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const double alpha, const double *A, const MKL_INT lda,
                  const double *B, const MKL_INT ldb, const double beta,
                  double *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const double alpha, const double *A, const MKL_INT lda,
                 const double beta, double *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrmm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 double *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtrsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 double *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *B, const MKL_INT ldb,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csymm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrmm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctrsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *B, const MKL_INT ldb,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsymm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zsymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zsyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zsyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zsyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrmm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztrsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;

// Routines with C, Z prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_chemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cher2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cher2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cher2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const float beta,
                  void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cherk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cherk)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cherk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const float alpha, const void *A, const MKL_INT lda,
                 const float beta, void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zhemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zher2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,B,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zher2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zher2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const double beta,
                  void *C, const MKL_INT ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zherk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,C)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zherk)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zherk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const double alpha, const void *A, const MKL_INT lda,
                 const double beta, void *C, const MKL_INT ldc) NOTHROW;


// Level2

// Routines with S, D, C, Z prefixes (Standard)
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 const float *X, const MKL_INT incX, const float beta,
                 float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const float alpha,
                 const float *A, const MKL_INT lda, const float *X,
                 const MKL_INT incX, const float beta, float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_strmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *A, const MKL_INT lda,
                 float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_stbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const float *A, const MKL_INT lda,
                 float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_stpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *Ap, float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(strsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_strsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *A, const MKL_INT lda, float *X,
                 const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_stbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const float *A, const MKL_INT lda,
                 float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(stpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_stpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *Ap, float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 const double *X, const MKL_INT incX, const double beta,
                 double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const double alpha,
                 const double *A, const MKL_INT lda, const double *X,
                 const MKL_INT incX, const double beta, double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtrmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *A, const MKL_INT lda,
                 double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const double *A, const MKL_INT lda,
                 double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *Ap, double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtrsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtrsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *A, const MKL_INT lda, double *X,
                 const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const double *A, const MKL_INT lda,
                 double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dtpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dtpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *Ap, double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *X, const MKL_INT incX, const void *beta,
                 void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const void *alpha,
                 const void *A, const MKL_INT lda, const void *X,
                 const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctrmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctrsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctrsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda, void *X,
                 const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ctpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ctpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *X, const MKL_INT incX, const void *beta,
                 void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const void *alpha,
                 const void *A, const MKL_INT lda, const void *X,
                 const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztrmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztrsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztrsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda, void *X,
                 const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ztpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ztpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;

// Routines with S, D prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssymv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssymv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssymv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *A,
                 const MKL_INT lda, const float *X, const MKL_INT incX,
                 const float beta, float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                 const MKL_INT lda, const float *X, const MKL_INT incX,
                 const float beta, float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sspmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sspmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sspmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *Ap,
                 const float *X, const MKL_INT incX,
                 const float beta, float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sger)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sger)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sger(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                const float alpha, const float *X, const MKL_INT incX,
                const float *Y, const MKL_INT incY, float *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyr)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssyr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const float *X,
                const MKL_INT incX, float *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sspr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sspr)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sspr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const float *X,
                const MKL_INT incX, float *Ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ssyr2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ssyr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *X,
                 const MKL_INT incX, const float *Y, const MKL_INT incY, float *A,
                 const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sspr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sspr2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sspr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *X,
                 const MKL_INT incX, const float *Y, const MKL_INT incY, float *A) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsymv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsymv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsymv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *A,
                 const MKL_INT lda, const double *X, const MKL_INT incX,
                 const double beta, double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const double alpha, const double *A,
                 const MKL_INT lda, const double *X, const MKL_INT incX,
                 const double beta, double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dspmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dspmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dspmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *Ap,
                 const double *X, const MKL_INT incX,
                 const double beta, double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dger)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dger)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dger(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                const double alpha, const double *X, const MKL_INT incX,
                const double *Y, const MKL_INT incY, double *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyr)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsyr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const double *X,
                const MKL_INT incX, double *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dspr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dspr)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dspr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const double *X,
                const MKL_INT incX, double *Ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsyr2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dsyr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *X,
                 const MKL_INT incX, const double *Y, const MKL_INT incY, double *A,
                 const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dspr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dspr2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dspr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *X,
                 const MKL_INT incX, const double *Y, const MKL_INT incY, double *A) NOTHROW;

// Routines with C, Z prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_chemv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_chbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_chpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *Ap,
                 const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgeru)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgeru)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgeru(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgerc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cgerc)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cgerc(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cher)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cher)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cher(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const void *X, const MKL_INT incX,
                void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chpr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chpr)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_chpr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const void *X,
                const MKL_INT incX, void *A) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cher2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cher2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cher2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chpr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(chpr2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_chpr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *Ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zhemv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zhbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zhpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *Ap,
                 const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgeru)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgeru)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgeru(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgerc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zgerc)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zgerc(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zher)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zher)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zher(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const void *X, const MKL_INT incX,
                void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhpr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhpr)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zhpr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const void *X,
                const MKL_INT incX, void *A) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zher2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:A,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zher2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zher2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhpr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:Ap,X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zhpr2)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zhpr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *Ap) NOTHROW;


// Level1

// Routines with S, D, DS, SDS prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdot)) match(construct={target variant dispatch}, device={arch(gen)})
float cblas_sdot(const MKL_INT N, const float  *X, const MKL_INT incX,
                 const float  *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ddot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ddot)) match(construct={target variant dispatch}, device={arch(gen)})
double cblas_ddot(const MKL_INT N, const double *X, const MKL_INT incX,
                  const double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsdot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dsdot)) match(construct={target variant dispatch}, device={arch(gen)})
double cblas_dsdot(const MKL_INT N, const float  *X, const MKL_INT incX,
                   const float  *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdsdot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sdsdot)) match(construct={target variant dispatch}, device={arch(gen)})
float cblas_sdsdot(const MKL_INT N, const float sb, const float  *X,
                   const MKL_INT incX, const float  *Y, const MKL_INT incY) NOTHROW;

// Routines with C, Z prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdotu)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y,dotu)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdotu)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cdotu_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotu) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdotc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y,dotc)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cdotc)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cdotc_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdotu)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y,dotu)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdotu)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zdotu_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotu) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdotc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y,dotc)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdotc)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zdotc_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotc) NOTHROW;

// Routines with S, D, SC, DZ prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(snrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(snrm2)) match(construct={target variant dispatch}, device={arch(gen)})
float cblas_snrm2(const MKL_INT N, const float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sasum)) match(construct={target variant dispatch}, device={arch(gen)})
float cblas_sasum(const MKL_INT N, const float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dnrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dnrm2)) match(construct={target variant dispatch}, device={arch(gen)})
double cblas_dnrm2(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dasum)) match(construct={target variant dispatch}, device={arch(gen)})
double cblas_dasum(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scnrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scnrm2)) match(construct={target variant dispatch}, device={arch(gen)})
float cblas_scnrm2(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scasum)) match(construct={target variant dispatch}, device={arch(gen)})
float cblas_scasum(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dznrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dznrm2)) match(construct={target variant dispatch}, device={arch(gen)})
double cblas_dznrm2(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dzasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dzasum)) match(construct={target variant dispatch}, device={arch(gen)})
double cblas_dzasum(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;


// Routines with S, D, C, Z prefixes (Standard)

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(isamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(isamax)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_isamax(const MKL_INT N, const float  *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(idamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(idamax)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_idamax(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(icamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(icamax)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_icamax(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(izamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(izamax)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_izamax(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(isamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(isamin)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_isamin(const MKL_INT N, const float  *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(idamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(idamin)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_idamin(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(icamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(icamin)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_icamin(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(izamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(izamin)) match(construct={target variant dispatch}, device={arch(gen)})
CBLAS_INDEX cblas_izamin(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sswap)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sswap(const MKL_INT N, float *X, const MKL_INT incX,
                 float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(scopy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_scopy(const MKL_INT N, const float *X, const MKL_INT incX,
                 float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_saxpy(const MKL_INT N, const float alpha, const float *X,
                 const MKL_INT incX, float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srotg)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_srotg(float *a, float *b, float *c, float *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dswap)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dswap(const MKL_INT N, double *X, const MKL_INT incX,
                 double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dcopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dcopy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dcopy(const MKL_INT N, const double *X, const MKL_INT incX,
                 double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_daxpy(const MKL_INT N, const double alpha, const double *X,
                 const MKL_INT incX, double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drotg)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_drotg(double *a, double *b, double *c, double *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cswap)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cswap(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ccopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(ccopy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_ccopy(const MKL_INT N, const void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_caxpy(const MKL_INT N, const void *alpha, const void *X,
                 const MKL_INT incX, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(crotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(crotg)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_crotg(void *a, const void *b, float *c, void *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zswap)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zswap(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zcopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zcopy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zcopy(const MKL_INT N, const void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zaxpy(const MKL_INT N, const void *alpha, const void *X,
                 const MKL_INT incX, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zrotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zrotg)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zrotg(void *a, const void *b, double *c, void *s) NOTHROW;

// Routines with S, D prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srotmg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:d1,d2,b1,P)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srotmg)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srot)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_srot(const MKL_INT N, float *X, const MKL_INT incX,
                float *Y, const MKL_INT incY, const float c, const float s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srotm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y,P)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(srotm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_srotm(const MKL_INT N, float *X, const MKL_INT incX,
                 float *Y, const MKL_INT incY, const float *P) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drotmg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:d1,d2,b1,P)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drotmg)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drot)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_drot(const MKL_INT N, double *X, const MKL_INT incX,
                double *Y, const MKL_INT incY, const double c, const double  s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drotm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y,P)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(drotm)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_drotm(const MKL_INT N, double *X, const MKL_INT incX,
                 double *Y, const MKL_INT incY, const double *P) NOTHROW;

// Routines with CS, ZD prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csrot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csrot)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csrot(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY, const float c, const float s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdrot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdrot)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zdrot(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY, const double c, const double s) NOTHROW;

// Routines with S D C Z CS and ZD prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(sscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_sscal(const MKL_INT N, const float alpha, float *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(dscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_dscal(const MKL_INT N, const double alpha, double *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(cscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_cscal(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zscal(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(csscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_csscal(const MKL_INT N, const float alpha, void *X, const MKL_INT incX) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zdscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zdscal(const MKL_INT N, const double alpha, void *X, const MKL_INT incX) NOTHROW;




// BLAS API

// Level3

// Routines with S, D, C, Z prefixes (Standard)
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemm(const char *transa, const char *transb, 
           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, 
           const float *b, const MKL_INT *ldb,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemmt(const char *uplo, const char *transa, const char *transb, 
            const MKL_INT *n, const MKL_INT *k,
            const float *alpha, const float *a, const MKL_INT *lda, 
            const float *b, const MKL_INT *ldb,
            const float *beta, float *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssymm)) match(construct={target variant dispatch}, device={arch(gen)})
void ssymm(const char *side, const char *uplo, 
           const MKL_INT *m, const MKL_INT *n,
           const float *alpha, const float *a, const MKL_INT *lda, 
           const float *b, const MKL_INT *ldb,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyr2k(const char *uplo, const char *trans, 
            const MKL_INT *n, const MKL_INT *k,
            const float *alpha, const float *a, const MKL_INT *lda, 
            const float *b, const MKL_INT *ldb,
            const float *beta, float *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyrk(const char *uplo, const char *trans, 
           const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, 
           const float *beta,
           float *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strmm)) match(construct={target variant dispatch}, device={arch(gen)})
void strmm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const float *alpha, const float *a, const MKL_INT *lda,
           float *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsm)) match(construct={target variant dispatch}, device={arch(gen)})
void strsm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const float *alpha, const float *a, const MKL_INT *lda,
           float *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemm(const char *transa, const char *transb, 
           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, 
           const double *b, const MKL_INT *ldb,
           const double *beta, double *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemmt(const char *uplo, const char *transa, const char *transb, 
            const MKL_INT *n, const MKL_INT *k,
            const double *alpha, const double *a, const MKL_INT *lda, 
            const double *b, const MKL_INT *ldb,
            const double *beta, double *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsymm)) match(construct={target variant dispatch}, device={arch(gen)})
void dsymm(const char *side, const char *uplo, 
           const MKL_INT *m, const MKL_INT *n,
           const double *alpha, const double *a, const MKL_INT *lda, 
           const double *b, const MKL_INT *ldb,
           const double *beta, double *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyr2k(const char *uplo, const char *trans, 
            const MKL_INT *n, const MKL_INT *k,
            const double *alpha, const double *a, const MKL_INT *lda, 
            const double *b, const MKL_INT *ldb,
            const double *beta, double *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyrk(const char *uplo, const char *trans, 
           const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, 
           const MKL_INT *lda, const double *beta,
           double *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrmm)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrmm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const double *alpha, const double *a, const MKL_INT *lda,
           double *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsm)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrsm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const double *alpha, const double *a, const MKL_INT *lda,
           double *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemm(const char *transa, const char *transb, 
           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemmt(const char *uplo, const char *transa, const char *transb, 
            const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csymm)) match(construct={target variant dispatch}, device={arch(gen)})
void csymm(const char *side, const char *uplo, 
           const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void csyr2k(const char *uplo, const char *trans, 
            const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void csyrk(const char *uplo, const char *trans, 
           const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrmm)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrmm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, 
           MKL_Complex8 *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsm)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrsm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, 
           MKL_Complex8 *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemm)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemm(const char *transa, const char *transb, 
           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemmt)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemmt)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemmt(const char *uplo, const char *transa, const char *transb, 
            const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsymm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsymm)) match(construct={target variant dispatch}, device={arch(gen)})
void zsymm(const char *side, const char *uplo, 
           const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyr2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyr2k)) match(construct={target variant dispatch}, device={arch(gen)})
void zsyr2k(const char *uplo, const char *trans, 
            const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyrk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zsyrk)) match(construct={target variant dispatch}, device={arch(gen)})
void zsyrk(const char *uplo, const char *trans, 
           const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrmm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrmm)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrmm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, 
           MKL_Complex16 *b, const MKL_INT *ldb) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsm)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrsm(const char *side, const char *uplo, const char *transa, 
           const char *diag, const MKL_INT *m, const MKL_INT *n, 
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, 
           MKL_Complex16 *b, const MKL_INT *ldb) NOTHROW;

// Routines with C, Z prefixes

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chemm)) match(construct={target variant dispatch}, device={arch(gen)})
void chemm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cher2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cher2k)) match(construct={target variant dispatch}, device={arch(gen)})
void cher2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const float *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cherk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cherk)) match(construct={target variant dispatch}, device={arch(gen)})
void cherk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const float *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhemm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhemm)) match(construct={target variant dispatch}, device={arch(gen)})
void zhemm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zher2k)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zher2k)) match(construct={target variant dispatch}, device={arch(gen)})
void zher2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const double *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zherk)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,c)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zherk)) match(construct={target variant dispatch}, device={arch(gen)})
void zherk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const double *beta, MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;


// Level2

// Routines with S, D, C, Z prefixes (Standard)
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const float *alpha, const float *a, const MKL_INT *lda, 
           const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void sgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const MKL_INT *kl, const MKL_INT *ku, const float *alpha, 
           const float *a, const MKL_INT *lda, const float *x, 
           const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strmv)) match(construct={target variant dispatch}, device={arch(gen)})
void strmv(const char *uplo, const char *transa, const char *diag, 
           const MKL_INT *n, const float *a,
           const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void stbmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k, const float *a, 
           const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void stpmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const float *ap,
           float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(strsv)) match(construct={target variant dispatch}, device={arch(gen)})
void strsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const float *a, const MKL_INT *lda, 
           float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void stbsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k, const float *a, 
           const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(stpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void stpsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const float *ap,
           float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const double *alpha, const double *a, const MKL_INT *lda, 
           const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void dgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const MKL_INT *kl, const MKL_INT *ku, const double *alpha, 
           const double *a, const MKL_INT *lda, const double *x, 
           const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrmv)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrmv(const char *uplo, const char *transa, const char *diag, 
           const MKL_INT *n, const double *a, 
           const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void dtbmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k, const double *a, 
           const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void dtpmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const double *ap, 
           double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtrsv)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const double *a, const MKL_INT *lda, 
           double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void dtbsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k, const double *a, 
           const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dtpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void dtpsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const double *ap, 
           double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void cgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, 
           const MKL_INT *lda, const MKL_Complex8 *x, 
           const MKL_INT *incx, const MKL_Complex8 *beta, 
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void cgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const MKL_INT *kl, const MKL_INT *ku,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a,
           const MKL_INT *lda,
           const MKL_Complex8 *x, const MKL_INT *incx, const
           MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrmv(const char *uplo, const char *transa, const char *diag, 
           const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *lda,
           MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ctbmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *a, const MKL_INT *lda,
           MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ctpmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_Complex8 *ap, MKL_Complex8 *x, 
           const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctrsv)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *lda,
           MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void ctbsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *a, const MKL_INT *lda,
           MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ctpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void ctpsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_Complex8 *ap, MKL_Complex8 *x, 
           const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgemv)) match(construct={target variant dispatch}, device={arch(gen)})
void zgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, 
           const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void zgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, 
           const MKL_INT *kl, const MKL_INT *ku,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a,
           const MKL_INT *lda,
           const MKL_Complex16 *x, const MKL_INT *incx, 
           const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrmv(const char *uplo, const char *transa, const char *diag, 
           const MKL_INT *n,
           const MKL_Complex16 *a, const MKL_INT *lda,
           MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ztbmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *a, const MKL_INT *lda,
           MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ztpmv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_Complex16 *ap, MKL_Complex16 *x, 
           const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztrsv)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n,
           const MKL_Complex16 *a, const MKL_INT *lda,
           MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztbsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztbsv)) match(construct={target variant dispatch}, device={arch(gen)})
void ztbsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *a, const MKL_INT *lda,
           MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztpsv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ztpsv)) match(construct={target variant dispatch}, device={arch(gen)})
void ztpsv(const char *uplo, const char *trans, const char *diag, 
           const MKL_INT *n, const MKL_Complex16 *ap, MKL_Complex16 *x, 
           const MKL_INT *incx) NOTHROW;

// Routines with S, D prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssymv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssymv)) match(construct={target variant dispatch}, device={arch(gen)})
void ssymv(const char *uplo, const MKL_INT *n, const float *alpha, 
           const float *a, const MKL_INT *lda,
           const float *x, const MKL_INT *incx, const float *beta, 
           float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void ssbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k, 
           const float *alpha, const float *a, const MKL_INT *lda, 
           const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sspmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sspmv)) match(construct={target variant dispatch}, device={arch(gen)})
void sspmv(const char *uplo, const MKL_INT *n, const float *alpha, 
           const float *ap, const float *x, const MKL_INT *incx, 
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sger)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sger)) match(construct={target variant dispatch}, device={arch(gen)})
void sger(const MKL_INT *m, const MKL_INT *n, const float *alpha, 
          const float *x, const MKL_INT *incx,
          const float *y, const MKL_INT *incy, float *a, 
          const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyr)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyr(const char *uplo, const MKL_INT *n, const float *alpha, 
          const float *x, const MKL_INT *incx,
          float *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sspr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sspr)) match(construct={target variant dispatch}, device={arch(gen)})
void sspr(const char *uplo, const MKL_INT *n, const float *alpha, 
          const float *x, const MKL_INT *incx, float *ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ssyr2)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyr2(const char *uplo, const MKL_INT *n, const float *alpha, 
           const float *x, const MKL_INT *incx,
           const float *y, const MKL_INT *incy, float *a, 
           const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sspr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sspr2)) match(construct={target variant dispatch}, device={arch(gen)})
void sspr2(const char *uplo, const MKL_INT *n, const float *alpha, 
           const float *x, const MKL_INT *incx,
           const float *y, const MKL_INT *incy, float *ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsymv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsymv)) match(construct={target variant dispatch}, device={arch(gen)})
void dsymv(const char *uplo, const MKL_INT *n, const double *alpha, 
           const double *a, const MKL_INT *lda,
           const double *x, const MKL_INT *incx, 
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void dsbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, 
           const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dspmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dspmv)) match(construct={target variant dispatch}, device={arch(gen)})
void dspmv(const char *uplo, const MKL_INT *n, const double *alpha, 
           const double *ap, const double *x, const MKL_INT *incx, 
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dger)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dger)) match(construct={target variant dispatch}, device={arch(gen)})
void dger(const MKL_INT *m, const MKL_INT *n, const double *alpha, 
          const double *x, const MKL_INT *incx,
          const double *y, const MKL_INT *incy, double *a, 
          const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyr)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyr(const char *uplo, const MKL_INT *n, const double *alpha, 
          const double *x, const MKL_INT *incx,
          double *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dspr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dspr)) match(construct={target variant dispatch}, device={arch(gen)})
void dspr(const char *uplo, const MKL_INT *n, const double *alpha, 
          const double *x, const MKL_INT *incx, double *ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsyr2)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyr2(const char *uplo, const MKL_INT *n, const double *alpha, 
           const double *x, const MKL_INT *incx,
           const double *y, const MKL_INT *incy, double *a,
           const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dspr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dspr2)) match(construct={target variant dispatch}, device={arch(gen)})
void dspr2(const char *uplo, const MKL_INT *n, const double *alpha, 
           const double *x, const MKL_INT *incx,
           const double *y, const MKL_INT *incy, double *ap) NOTHROW;

// Routines with C, Z prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chemv)) match(construct={target variant dispatch}, device={arch(gen)})
void chemv(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, 
           const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void chbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, 
           const MKL_INT *lda, const MKL_Complex8 *x, 
           const MKL_INT *incx, const MKL_Complex8 *beta, 
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void chpmv(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, 
           const MKL_Complex8 *ap, const MKL_Complex8 *x, 
           const MKL_INT *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgeru)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgeru)) match(construct={target variant dispatch}, device={arch(gen)})
void cgeru(const MKL_INT *m, const MKL_INT *n, const  MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, 
           const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgerc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cgerc)) match(construct={target variant dispatch}, device={arch(gen)})
void cgerc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, 
           const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cher)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cher)) match(construct={target variant dispatch}, device={arch(gen)})
void cher(const char *uplo, const MKL_INT *n, const float *alpha, 
          const MKL_Complex8 *x, const MKL_INT *incx,
          MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chpr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chpr)) match(construct={target variant dispatch}, device={arch(gen)})
void chpr(const char *uplo, const MKL_INT *n, const float *alpha, 
          const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cher2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cher2)) match(construct={target variant dispatch}, device={arch(gen)})
void cher2(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, 
           const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chpr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(chpr2)) match(construct={target variant dispatch}, device={arch(gen)})
void chpr2(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, 
           const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhemv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhemv)) match(construct={target variant dispatch}, device={arch(gen)})
void zhemv(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, 
           const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhbmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhbmv)) match(construct={target variant dispatch}, device={arch(gen)})
void zhbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k, 
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, 
           const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhpmv)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhpmv)) match(construct={target variant dispatch}, device={arch(gen)})
void zhpmv(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, 
           const MKL_Complex16 *ap, const MKL_Complex16 *x, 
           const MKL_INT *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgeru)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgeru)) match(construct={target variant dispatch}, device={arch(gen)})
void zgeru(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, 
           const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgerc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zgerc)) match(construct={target variant dispatch}, device={arch(gen)})
void zgerc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, 
           const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zher)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zher)) match(construct={target variant dispatch}, device={arch(gen)})
void zher(const char *uplo, const MKL_INT *n, const double *alpha, 
          const MKL_Complex16 *x, const MKL_INT *incx,
          MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhpr)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhpr)) match(construct={target variant dispatch}, device={arch(gen)})
void zhpr(const char *uplo, const MKL_INT *n, const double *alpha, 
          const MKL_Complex16 *x, const MKL_INT *incx,
          MKL_Complex16 *ap) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zher2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zher2)) match(construct={target variant dispatch}, device={arch(gen)})
void zher2(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT *incx, 
           const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhpr2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:ap,x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zhpr2)) match(construct={target variant dispatch}, device={arch(gen)})
void zhpr2(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, 
           const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *ap) NOTHROW;


// Level1

// Routines with S, D, DS, SDS prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdot)) match(construct={target variant dispatch}, device={arch(gen)})
float sdot(const MKL_INT *n, const float *x, const MKL_INT *incx, 
           const float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ddot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ddot)) match(construct={target variant dispatch}, device={arch(gen)})
double ddot(const MKL_INT *n, const double *x, const MKL_INT *incx, 
            const double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsdot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dsdot)) match(construct={target variant dispatch}, device={arch(gen)})
double dsdot(const MKL_INT *n, const float *x, const MKL_INT *incx, 
             const float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdsdot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sdsdot)) match(construct={target variant dispatch}, device={arch(gen)})
float sdsdot(const MKL_INT *n, const float *sb, const float *x, 
             const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;

// Routines with C, Z prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdotc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdotc)) match(construct={target variant dispatch}, device={arch(gen)})
void cdotc(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, 
           const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdotu)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cdotu)) match(construct={target variant dispatch}, device={arch(gen)})
void cdotu(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, 
           const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdotc)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdotc)) match(construct={target variant dispatch}, device={arch(gen)})
void zdotc(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, 
           const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdotu)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdotu)) match(construct={target variant dispatch}, device={arch(gen)})
void zdotu(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, 
           const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

// Routines with S, D, SC, DZ prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(snrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(snrm2)) match(construct={target variant dispatch}, device={arch(gen)})
float snrm2(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sasum)) match(construct={target variant dispatch}, device={arch(gen)})
float sasum(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dnrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dnrm2)) match(construct={target variant dispatch}, device={arch(gen)})
double dnrm2(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dasum)) match(construct={target variant dispatch}, device={arch(gen)})
double dasum(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scnrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scnrm2)) match(construct={target variant dispatch}, device={arch(gen)})
float scnrm2(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scasum)) match(construct={target variant dispatch}, device={arch(gen)})
float scasum(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dznrm2)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dznrm2)) match(construct={target variant dispatch}, device={arch(gen)})
double dznrm2(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dzasum)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dzasum)) match(construct={target variant dispatch}, device={arch(gen)})
double dzasum(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

// Routines with S, D, C, Z prefixes (Standard)
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(isamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(isamax)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT isamax(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(idamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(idamax)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT idamax(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(icamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(icamax)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT icamax(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(izamax)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(izamax)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT izamax(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(isamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(isamin)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT isamin(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(icamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(icamin)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT icamin(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(idamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(idamin)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT idamin(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(izamin)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(izamin)) match(construct={target variant dispatch}, device={arch(gen)})
MKL_INT izamin(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sswap)) match(construct={target variant dispatch}, device={arch(gen)})
void sswap(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(scopy)) match(construct={target variant dispatch}, device={arch(gen)})
void scopy(const MKL_INT *n, const float *x, const MKL_INT *incx, float *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void saxpy(const MKL_INT *n, const float *alpha, const float *x, 
           const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srotg)) match(construct={target variant dispatch}, device={arch(gen)})
void srotg(float *a,float *b,float *c,float *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dswap)) match(construct={target variant dispatch}, device={arch(gen)})
void dswap(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dcopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dcopy)) match(construct={target variant dispatch}, device={arch(gen)})
void dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx, double *y, 
           const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void daxpy(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, 
           double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drotg)) match(construct={target variant dispatch}, device={arch(gen)})
void drotg(double *a, double *b, double *c, double *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cswap)) match(construct={target variant dispatch}, device={arch(gen)})
void cswap(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, 
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ccopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(ccopy)) match(construct={target variant dispatch}, device={arch(gen)})
void ccopy(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, 
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void caxpy(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, 
           const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(crotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(crotg)) match(construct={target variant dispatch}, device={arch(gen)})
void crotg(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zswap)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zswap)) match(construct={target variant dispatch}, device={arch(gen)})
void zswap(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, 
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zcopy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zcopy)) match(construct={target variant dispatch}, device={arch(gen)})
void zcopy(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, 
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpy)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpy)) match(construct={target variant dispatch}, device={arch(gen)})
void zaxpy(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, 
           const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zrotg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:a,b,c,s)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zrotg)) match(construct={target variant dispatch}, device={arch(gen)})
void zrotg(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s) NOTHROW;

// Routines with S, D prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srotmg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:d1,d2,x1,param)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srotmg)) match(construct={target variant dispatch}, device={arch(gen)})
void srotmg(float *d1, float *d2, float *x1, const float *y1, float *param) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srot)) match(construct={target variant dispatch}, device={arch(gen)})
void srot(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, 
          const MKL_INT *incy, const float *c, const float *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srotm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y,param)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(srotm)) match(construct={target variant dispatch}, device={arch(gen)})
void srotm(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, 
           const MKL_INT *incy, const float *param) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drotmg)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:d1,d2,x1,param)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drotmg)) match(construct={target variant dispatch}, device={arch(gen)})
void drotmg(double *d1, double *d2, double *x1, const double *y1, double *param) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drot)) match(construct={target variant dispatch}, device={arch(gen)})
void drot(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, 
          const MKL_INT *incy, const double *c, const double *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drotm)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y,param)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(drotm)) match(construct={target variant dispatch}, device={arch(gen)})
void drotm(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, 
           const MKL_INT *incy, const double *param) NOTHROW;

// Routines with CS, ZD prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csrot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csrot)) match(construct={target variant dispatch}, device={arch(gen)})
void csrot(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, 
           const MKL_INT *incy, const float *c, const float *s) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdrot)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdrot)) match(construct={target variant dispatch}, device={arch(gen)})
void zdrot(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, 
           const MKL_INT *incy, const double *c, const double *s) NOTHROW;

// Routines with S D C Z CS and ZD prefixes
#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(sscal)) match(construct={target variant dispatch}, device={arch(gen)})
void sscal(const MKL_INT *n, const float *a, float *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(dscal)) match(construct={target variant dispatch}, device={arch(gen)})
void dscal(const MKL_INT *n, const double *a, double *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(cscal)) match(construct={target variant dispatch}, device={arch(gen)})
void cscal(const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zscal)) match(construct={target variant dispatch}, device={arch(gen)})
void zscal(const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(csscal)) match(construct={target variant dispatch}, device={arch(gen)})
void csscal(const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdscal)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zdscal)) match(construct={target variant dispatch}, device={arch(gen)})
void zdscal(const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(saxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void saxpby(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(caxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void caxpby(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(daxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void daxpby(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:x,y)
#endif
#pragma omp declare variant (MKL_BLAS_VARIANT_NAME(zaxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void zaxpby(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(saxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_saxpby(const MKL_INT N, const float alpha, const float *X,
                  const MKL_INT incX, const float beta, float *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(daxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_daxpby(const MKL_INT N, const double alpha, const double *X,
                  const MKL_INT incX, const double beta, double *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(caxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_caxpby(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW;

#if (_OPENMP >= 202011)
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpby)) match(construct={dispatch}, device={arch(gen)}) \
    append_args(interop(targetsync)) \
    adjust_args(need_device_ptr:X,Y)
#endif
#pragma omp declare variant (MKL_CBLAS_VARIANT_NAME(zaxpby)) match(construct={target variant dispatch}, device={arch(gen)})
void cblas_zaxpby(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW ;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

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

#ifndef _MKL_BLAS_OMP_VARIANT_H_
#define _MKL_BLAS_OMP_VARIANT_H_

#include "mkl_types.h"

#include "mkl_omp_variant.h"

#define MKL_BLAS_VARIANT_NAME(func) MKL_VARIANT_NAME(blas, func)
#define MKL_CBLAS_VARIANT_NAME(func) MKL_VARIANT_NAME(cblas, func)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

// Matrix transposition and copy API

void MKL_CBLAS_VARIANT_NAME(simatcopy_batch_strided)(const char ordering, const char trans,
                                                     size_t rows, size_t cols,
                                                     const float alpha,
                                                     float * AB, size_t lda, size_t ldb,
                                                     size_t stride, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dimatcopy_batch_strided)(const char ordering, const char trans,
                                                     size_t rows, size_t cols,
                                                     const double alpha,
                                                     double * AB, size_t lda, size_t ldb,
                                                     size_t stride, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cimatcopy_batch_strided)(const char ordering, const char trans,
                                                     size_t rows, size_t cols,
                                                     const MKL_Complex8 alpha,
                                                     MKL_Complex8 * AB, size_t lda, size_t ldb,
                                                     size_t stride, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zimatcopy_batch_strided)(const char ordering, const char trans,
                                                     size_t rows, size_t cols,
                                                     const MKL_Complex16 alpha,
                                                     MKL_Complex16 * AB, size_t lda, size_t ldb,
                                                     size_t stride, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(somatcopy_batch_strided)(char ordering, char trans,
                                                     size_t rows, size_t cols,
                                                     const float alpha,
                                                     const float * A, size_t lda, size_t stridea,
                                                     float *B, size_t ldb, size_t strideb,
                                                     size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(domatcopy_batch_strided)(char ordering, char trans,
                                                     size_t rows, size_t cols,
                                                     const double alpha,
                                                     const double * A, size_t lda, size_t stridea,
                                                     double *B, size_t ldb, size_t strideb,
                                                     size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(comatcopy_batch_strided)(char ordering, char trans,
                                                     size_t rows, size_t cols,
                                                     const MKL_Complex8 alpha,
                                                     const MKL_Complex8 * A, size_t lda, size_t stridea,
                                                     MKL_Complex8 *B, size_t ldb, size_t strideb,
                                                     size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zomatcopy_batch_strided)(char ordering, char trans,
                                                     size_t rows, size_t cols,
                                                     const MKL_Complex16 alpha,
                                                     const MKL_Complex16 * A, size_t lda, size_t stridea,
                                                     MKL_Complex16 *B, size_t ldb, size_t strideb,
                                                     size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(simatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const float * alpha_array, float ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dimatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const double * alpha_array, double ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cimatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex8 * alpha_array, MKL_Complex8 ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zimatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex16 * alpha_array, MKL_Complex16 ** AB_array,
    const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(somatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const float * alpha_array, const float ** A_array,
    const size_t * lda_array, float ** B,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(domatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const double * alpha_array, const double ** A_array,
    const size_t * lda_array, double ** B,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(comatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex8 * alpha_array, const MKL_Complex8 ** A_array,
    const size_t * lda_array, MKL_Complex8 ** B,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zomatcopy_batch)(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex16 * alpha_array, const MKL_Complex16 ** A_array,
    const size_t * lda_array, MKL_Complex16 ** B,
    const size_t * ldb_array, size_t group_count,
    const size_t * group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(somatadd_batch_strided)(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const float alpha, const float * A, size_t lda, size_t stridea,
    const float beta, const float * B, size_t ldb, size_t strideb,
    float * C, size_t ldc, size_t stridec, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(domatadd_batch_strided)(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const double alpha, const double * A, size_t lda, size_t stridea,
    const double beta, const double * B, size_t ldb, size_t strideb,
    double * C, size_t ldc, size_t stridec, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(comatadd_batch_strided)(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha, const MKL_Complex8 * A, size_t lda, size_t stridea,
    const MKL_Complex8 beta, const MKL_Complex8 * B, size_t ldb, size_t strideb,
    MKL_Complex8 * C, size_t ldc, size_t stridec, size_t batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zomatadd_batch_strided)(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha, const MKL_Complex16 * A, size_t lda, size_t stridea,
    const MKL_Complex16 beta, const MKL_Complex16 * B, size_t ldb, size_t strideb,
    MKL_Complex16 * C, size_t ldc, size_t stridec, size_t batch_size, void *interop_obj) NOTHROW;


// BATCH APIs

// Level3

void MKL_CBLAS_VARIANT_NAME(sgemm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const float *alpha_Array, const float **A_Array,
                       const MKL_INT *lda_Array, const float **B_Array, const MKL_INT *ldb_Array,
                       const float *beta_Array, float **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const double *alpha_Array, const double **A_Array,
                       const MKL_INT *lda_Array, const double **B_Array, const MKL_INT* ldb_Array,
                       const double *beta_Array, double **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                       const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                       const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sgemm_batch)(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT *lda_array,
                 const float **b_array, const MKL_INT *ldb_array,
                 const float *beta_array, float **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemm_batch)(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT *lda_array,
                 const double **b_array, const MKL_INT *ldb_array,
                 const double *beta_array, double **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemm_batch)(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array,
                 const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemm_batch)(const char *transa_array, const char *transb_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array,
                 const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sgemm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const float alpha, const float *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const float *B, const MKL_INT ldb, const MKL_INT strideb,
                               const float beta, float *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const double alpha, const double *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const double *B, const MKL_INT ldb, const MKL_INT strideb,
                               const double beta, double *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *B, const MKL_INT ldb, const MKL_INT strideb,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *B, const MKL_INT ldb, const MKL_INT strideb,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sgemm_batch_strided)(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const float *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const float *beta, float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemm_batch_strided)(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const double *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const double *beta, double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemm_batch_strided)(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemm_batch_strided)(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssyrk_batch)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                                         const MKL_INT *N_Array, const MKL_INT *K_Array,
                                         const float *alpha_Array, const float **A_Array, const MKL_INT *lda_Array,
                                         const float *beta_Array, float **C_Array, const MKL_INT *ldc_Array,
                                         const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsyrk_batch)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                                         const MKL_INT *N_Array, const MKL_INT *K_Array,
                                         const double *alpha_Array, const double **A_Array, const MKL_INT *lda_Array,
                                         const double *beta_Array, double **C_Array, const MKL_INT *ldc_Array,
                                         const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(csyrk_batch)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                                         const MKL_INT *N_Array, const MKL_INT *K_Array,
                                         const void *alpha_Array, const void **A_Array, const MKL_INT *lda_Array,
                                         const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                                         const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zsyrk_batch)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Trans_Array,
                                         const MKL_INT *N_Array, const MKL_INT *K_Array,
                                         const void *alpha_Array, const void **A_Array, const MKL_INT *lda_Array,
                                         const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                                         const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssyrk_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                                                 const MKL_INT K, const float alpha, const float *A,
                                                 const MKL_INT lda, const MKL_INT stridea,
                                                 const float beta, float *C, const MKL_INT ldc, const MKL_INT stridec,
                                                 const MKL_INT batch_strided_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsyrk_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                                                 const MKL_INT K, const double alpha, const double *A,
                                                 const MKL_INT lda, const MKL_INT stridea,
                                                 const double beta, double *C, const MKL_INT ldc, const MKL_INT stridec,
                                                 const MKL_INT batch_strided_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(csyrk_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                                                 const MKL_INT K, const void *alpha, const void *A,
                                                 const MKL_INT lda, const MKL_INT stridea,
                                                 const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                                                 const MKL_INT batch_strided_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zsyrk_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N,
                                                 const MKL_INT K, const void *alpha, const void *A,
                                                 const MKL_INT lda, const MKL_INT stridea,
                                                 const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                                                 const MKL_INT batch_strided_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssyrk_batch)(const char *uplo_array, const char *trans_array,
                                        const MKL_INT *n_array, const MKL_INT *k_array,
                                        const float *alpha_array, const float **a_array, const MKL_INT *lda_array,
                                        const float *beta_array, float **c_array, const MKL_INT *ldc_array,
                                        const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsyrk_batch)(const char *uplo_array, const char *trans_array,
                                        const MKL_INT *n_array, const MKL_INT *k_array,
                                        const double *alpha_array, const double **a_array, const MKL_INT *lda_array,
                                        const double *beta_array, double **c_array, const MKL_INT *ldc_array,
                                        const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(csyrk_batch)(const char *uplo_array, const char *trans_array,
                                        const MKL_INT *n_array, const MKL_INT *k_array,
                                        const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array,
                                        const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array,
                                        const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zsyrk_batch)(const char *uplo_array, const char *trans_array,
                                        const MKL_INT *n_array, const MKL_INT *k_array,
                                        const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array,
                                        const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array,
                                        const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssyrk_batch_strided)(const char *Uplo,
                                                const char *Trans, const MKL_INT *N,
                                                const MKL_INT *K, const float *alpha, const float *A,
                                                const MKL_INT *lda, const MKL_INT *stridea,
                                                const float *beta, float *C, const MKL_INT *ldc, const MKL_INT *stridec,
                                                const MKL_INT *batch_strided_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsyrk_batch_strided)(const char *Uplo,
                                                const char *Trans, const MKL_INT *N,
                                                const MKL_INT *K, const double *alpha, const double *A,
                                                const MKL_INT *lda, const MKL_INT *stridea,
                                                const double *beta, double *C, const MKL_INT *ldc, const MKL_INT *stridec,
                                                const MKL_INT *batch_strided_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(csyrk_batch_strided)(const char *Uplo,
                                                const char *Trans, const MKL_INT *N,
                                                const MKL_INT *K, const MKL_Complex8 *alpha, const MKL_Complex8 *A,
                                                const MKL_INT *lda, const MKL_INT *stridea,
                                                const MKL_Complex8 *beta, MKL_Complex8 *C, const MKL_INT *ldc, const MKL_INT *stridec,
                                                const MKL_INT *batch_strided_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zsyrk_batch_strided)(const char *Uplo,
                                                const char *Trans, const MKL_INT *N,
                                                const MKL_INT *K, const MKL_Complex16 *alpha, const MKL_Complex16 *A,
                                                const MKL_INT *lda, const MKL_INT *stridea,
                                                const MKL_Complex16 *beta, MKL_Complex16 *C, const MKL_INT *ldc, const MKL_INT *stridec,
                                                const MKL_INT *batch_strided_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(strsm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                                         const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *TransA_Array,
                                         const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                                         const MKL_INT *N_Array, const float *alpha_Array,
                                         const float **A_Array, const MKL_INT *lda_Array,
                                         float **B_Array, const MKL_INT *ldb_Array,
                                         const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;


void MKL_CBLAS_VARIANT_NAME(dtrsm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                                          const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                                          const MKL_INT *N_Array, const double *alpha_Array,
                                          const double **A_Array, const MKL_INT *lda_Array,
                                          double **B_Array, const MKL_INT *ldb_Array,
                                          const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctrsm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                                          const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                                          const MKL_INT *N_Array, const void *alpha_Array,
                                          const void **A_Array, const MKL_INT *lda_Array,
                                          void **B_Array, const MKL_INT *ldb_Array,
                                          const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztrsm_batch)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                                          const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                                          const MKL_INT *N_Array, const void *alpha_Array,
                                          const void **A_Array, const MKL_INT *lda_Array,
                                          void **B_Array, const MKL_INT *ldb_Array,
                                          const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(strsm_batch)(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                                         const MKL_INT *m_array, const MKL_INT *n_array, const float *alpha_array, const float **a_array,
                                         const MKL_INT *lda_array, float **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtrsm_batch)(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                                         const MKL_INT *m_array, const MKL_INT *n_array, const double *alpha_array, const double **a_array,
                                         const MKL_INT *lda_array, double **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctrsm_batch)(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                                         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array,
                                         const MKL_INT *lda_array, MKL_Complex8 **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztrsm_batch)(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                                         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array,
                                         const MKL_INT *lda_array, MKL_Complex16 **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(strsm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                                  const CBLAS_DIAG Diag, const MKL_INT M,
                                                  const MKL_INT N, const float alpha,
                                                  const float *A, const MKL_INT lda, const MKL_INT stridea,
                                                  float *B, const MKL_INT ldb, const MKL_INT strideb,
                                                  const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtrsm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                                  const CBLAS_DIAG Diag, const MKL_INT M,
                                                  const MKL_INT N, const double alpha,
                                                  const double *A, const MKL_INT lda, const MKL_INT stridea,
                                                  double *B, const MKL_INT ldb, const MKL_INT strideb,
                                                  const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctrsm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                                  const CBLAS_DIAG Diag, const MKL_INT M,
                                                  const MKL_INT N, const void *alpha,
                                                  const void *A, const MKL_INT lda, const MKL_INT stridea,
                                                  void *B, const MKL_INT ldb, const MKL_INT strideb,
                                                  const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztrsm_batch_strided)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                                  const CBLAS_DIAG Diag, const MKL_INT M,
                                                  const MKL_INT N, const void *alpha,
                                                  const void *A, const MKL_INT lda, const MKL_INT stridea,
                                                  void *B, const MKL_INT ldb, const MKL_INT strideb,
                                                  const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(strsm_batch_strided)(const char *side, const char *uplo, const char *transa, const char *diag,
                                                 const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a,
                                                 const MKL_INT *lda, const MKL_INT *stridea, float *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtrsm_batch_strided)(const char *side, const char *uplo, const char *transa, const char *diag,
                                                 const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a,
                                                 const MKL_INT *lda, const MKL_INT *stridea, double *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctrsm_batch_strided)(const char *side, const char *uplo, const char *transa, const char *diag,
                                                 const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a,
                                                 const MKL_INT *lda, const MKL_INT *stridea, MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztrsm_batch_strided)(const char *side, const char *uplo, const char *transa, const char *diag,
                                                 const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a,
                                                 const MKL_INT *lda, const MKL_INT *stridea, MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb, const MKL_INT *batch_size, void *interop_obj) NOTHROW;


// Level2

void MKL_BLAS_VARIANT_NAME(sgemv_batch)(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                 const float **a, const MKL_INT *lda, const float **x, const MKL_INT *incx,
                 const float *beta, float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sgemv_batch_strided)(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const float *beta, float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemv_batch)(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                 const double **a, const MKL_INT *lda, const double **x, const MKL_INT *incx,
                 const double *beta, double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemv_batch_strided)(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const double *beta, double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemv_batch)(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **a, const MKL_INT *lda, const MKL_Complex8 **x, const MKL_INT *incx,
                 const MKL_Complex8 *beta, MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemv_batch_strided)(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemv_batch)(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **a, const MKL_INT *lda, const MKL_Complex16 **x, const MKL_INT *incx,
                 const MKL_Complex16 *beta, MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemv_batch_strided)(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sgemv_batch)(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const float *alpha, const float **A, const MKL_INT *lda,
                       const float **X, const MKL_INT *incX, const float *beta,
                       float **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sgemv_batch_strided)(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const float alpha, const float *A, const MKL_INT lda, const MKL_INT stridea,
                               const float *X, const MKL_INT incX, const MKL_INT stridex, const float beta,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemv_batch)(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const double *alpha, const double **A, const MKL_INT *lda,
                       const double **X, const MKL_INT *incX, const double *beta,
                       double **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemv_batch_strided)(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const double alpha, const double *A, const MKL_INT lda, const MKL_INT stridea,
                               const double *X, const MKL_INT incX, const MKL_INT stridex, const double beta,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemv_batch)(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const void *alpha, const void **A, const MKL_INT *lda,
                       const void **X, const MKL_INT *incX, const void *beta,
                       void **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemv_batch_strided)(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                               const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemv_batch)(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const void *alpha, const void **A, const MKL_INT *lda,
                       const void **X, const MKL_INT *incX, const void *beta,
                       void **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemv_batch_strided)(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                               const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sdgmm_batch)(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const float **a, const MKL_INT *lda,
                 const float **x, const MKL_INT *incx,
                 float **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sdgmm_batch_strided)(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ddgmm_batch)(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const double **a, const MKL_INT *lda,
                 const double **x, const MKL_INT *incx,
                 double **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ddgmm_batch_strided)(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cdgmm_batch)(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex8 **a, const MKL_INT *lda,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cdgmm_batch_strided)(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zdgmm_batch)(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex16 **a, const MKL_INT *lda,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zdgmm_batch_strided)(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;


void MKL_CBLAS_VARIANT_NAME(sdgmm_batch)(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const float **a, const MKL_INT *lda,
                       const float **x, const MKL_INT *incx,
                       float **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sdgmm_batch_strided)(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const float *a, const MKL_INT lda, const MKL_INT stridea,
                               const float *x, const MKL_INT incx, const MKL_INT stridex,
                               float *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ddgmm_batch)(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const double **a, const MKL_INT *lda,
                       const double **x, const MKL_INT *incx,
                       double **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ddgmm_batch_strided)(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const double *a, const MKL_INT lda, const MKL_INT stridea,
                               const double *x, const MKL_INT incx, const MKL_INT stridex,
                               double *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cdgmm_batch)(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const void **a, const MKL_INT *lda,
                       const void **x, const MKL_INT *incx,
                       void **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cdgmm_batch_strided)(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const void *a, const MKL_INT lda, const MKL_INT stridea,
                               const void *x, const MKL_INT incx, const MKL_INT stridex,
                               void *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zdgmm_batch)(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const void **a, const MKL_INT *lda,
                       const void **x, const MKL_INT *incx,
                       void **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zdgmm_batch_strided)(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const void *a, const MKL_INT lda, const MKL_INT stridea,
                               const void *x, const MKL_INT incx, const MKL_INT stridex,
                               void *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

// Level1

void MKL_CBLAS_VARIANT_NAME(saxpy_batch)(const MKL_INT *n, const float *alpha,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(daxpy_batch)(const MKL_INT *n, const double *alpha,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(caxpy_batch)(const MKL_INT *n, const void *alpha,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zaxpy_batch)(const MKL_INT *n, const void *alpha,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(saxpy_batch)(const MKL_INT *n, const float *alpha,
                 const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(daxpy_batch)(const MKL_INT *n, const double *alpha,
                 const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(caxpy_batch)(const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zaxpy_batch)(const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(saxpy_batch_strided)(const MKL_INT N, const float alpha,
                               const float *X, const MKL_INT incX, const MKL_INT stridex,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(daxpy_batch_strided)(const MKL_INT N, const double alpha,
                               const double *X, const MKL_INT incX, const MKL_INT stridex,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(caxpy_batch_strided)(const MKL_INT N, const void *alpha,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zaxpy_batch_strided)(const MKL_INT N, const void *alpha,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(saxpy_batch_strided)(const MKL_INT *n, const float *alpha,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(daxpy_batch_strided)(const MKL_INT *n, const double *alpha,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(caxpy_batch_strided)(const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zaxpy_batch_strided)(const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(scopy_batch)(const MKL_INT *n,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dcopy_batch)(const MKL_INT *n,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ccopy_batch)(const MKL_INT *n,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zcopy_batch)(const MKL_INT *n,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(scopy_batch)(const MKL_INT *n,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dcopy_batch)(const MKL_INT *n,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ccopy_batch)(const MKL_INT *n,
                       const MKL_Complex8 **x, const MKL_INT *incx,
                       MKL_Complex8 **y, const MKL_INT *incy,
                       const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zcopy_batch)(const MKL_INT *n,
                       const MKL_Complex16 **x, const MKL_INT *incx,
                       MKL_Complex16 **y, const MKL_INT *incy,
                       const MKL_INT *group_count, const MKL_INT *group_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(scopy_batch_strided)(const MKL_INT N,
                               const float *X, const MKL_INT incX, const MKL_INT stridex,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dcopy_batch_strided)(const MKL_INT N,
                               const double *X, const MKL_INT incX, const MKL_INT stridex,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ccopy_batch_strided)(const MKL_INT N,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zcopy_batch_strided)(const MKL_INT N,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(scopy_batch_strided)(const MKL_INT *N,
                               const float *X, const MKL_INT *incX, const MKL_INT *stridex,
                               float *Y, const MKL_INT *incY, const MKL_INT *stridey,
                               const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dcopy_batch_strided)(const MKL_INT *N,
                               const double *X, const MKL_INT *incX, const MKL_INT *stridex,
                               double *Y, const MKL_INT *incY, const MKL_INT *stridey,
                               const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ccopy_batch_strided)(const MKL_INT *N,
                               const MKL_Complex8 *X, const MKL_INT *incX, const MKL_INT *stridex,
                               MKL_Complex8 *Y, const MKL_INT *incY, const MKL_INT *stridey,
                               const MKL_INT *batch_size, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zcopy_batch_strided)(const MKL_INT *N,
                               const MKL_Complex16 *X, const MKL_INT *incX, const MKL_INT *stridex,
                               MKL_Complex16 *Y, const MKL_INT *incY, const MKL_INT *stridey,
                               const MKL_INT *batch_size, void *interop_obj) NOTHROW;

// CBLAS API

// Level3

// Routines with S, D, C, Z prefixes (Standard)
void MKL_CBLAS_VARIANT_NAME(sgemm)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                                const MKL_INT K, const float alpha, const float *A,
                                const MKL_INT lda, const float *B, const MKL_INT ldb,
                                const float beta, float *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sgemmt)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                                 const MKL_INT N, const MKL_INT K,
                                 const float alpha, const float *A, const MKL_INT lda,
                                 const float *B, const MKL_INT ldb, const float beta,
                                 float *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssymm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                                const float alpha, const float *A, const MKL_INT lda,
                                const float *B, const MKL_INT ldb, const float beta,
                                float *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssyr2k)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                 const float alpha, const float *A, const MKL_INT lda,
                                 const float *B, const MKL_INT ldb, const float beta,
                                 float *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssyrk)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                const float alpha, const float *A, const MKL_INT lda,
                                const float beta, float *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(strmm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const float alpha, const float *A, const MKL_INT lda,
                                float *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(strsm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const float alpha, const float *A, const MKL_INT lda,
                                float *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemm)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                                const MKL_INT K, const double alpha, const double *A,
                                const MKL_INT lda, const double *B, const MKL_INT ldb,
                                const double beta, double *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemmt)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                                 const MKL_INT N, const MKL_INT K,
                                 const double alpha, const double *A, const MKL_INT lda,
                                 const double *B, const MKL_INT ldb, const double beta,
                                 double *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsymm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                                const double alpha, const double *A, const MKL_INT lda,
                                const double *B, const MKL_INT ldb, const double beta,
                                double *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsyr2k)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                 const double alpha, const double *A, const MKL_INT lda,
                                 const double *B, const MKL_INT ldb, const double beta,
                                 double *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsyrk)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                const double alpha, const double *A, const MKL_INT lda,
                                const double beta, double *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtrmm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const double alpha, const double *A, const MKL_INT lda,
                                double *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtrsm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const double alpha, const double *A, const MKL_INT lda,
                                double *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemm)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                                const MKL_INT K, const void *alpha, const void *A,
                                const MKL_INT lda, const void *B, const MKL_INT ldb,
                                const void *beta, void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemmt)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                                 const MKL_INT N, const MKL_INT K,
                                 const void *alpha, const void *A, const MKL_INT lda,
                                 const void *B, const MKL_INT ldb, const void *beta,
                                 void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(csymm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *B, const MKL_INT ldb, const void *beta,
                                void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(csyr2k)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                 const void *alpha, const void *A, const MKL_INT lda,
                                 const void *B, const MKL_INT ldb, const void *beta,
                                 void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(csyrk)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *beta, void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctrmm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                void *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctrsm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                void *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemm)(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                                const MKL_INT K, const void *alpha, const void *A,
                                const MKL_INT lda, const void *B, const MKL_INT ldb,
                                const void *beta, void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemmt)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                                 const MKL_INT N, const MKL_INT K,
                                 const void *alpha, const void *A, const MKL_INT lda,
                                 const void *B, const MKL_INT ldb, const void *beta,
                                 void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zsymm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *B, const MKL_INT ldb, const void *beta,
                                void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zsyr2k)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                 const void *alpha, const void *A, const MKL_INT lda,
                                 const void *B, const MKL_INT ldb, const void *beta,
                                 void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zsyrk)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *beta, void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztrmm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                void *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztrsm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                void *B, const MKL_INT ldb, void *interop_obj) NOTHROW;

// Routines with C, Z prefixes
void MKL_CBLAS_VARIANT_NAME(chemm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *B, const MKL_INT ldb, const void *beta,
                                void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cher2k)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                 const void *alpha, const void *A, const MKL_INT lda,
                                 const void *B, const MKL_INT ldb, const float beta,
                                 void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cherk)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                const float alpha, const void *A, const MKL_INT lda,
                                const float beta, void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zhemm)(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *B, const MKL_INT ldb, const void *beta,
                                void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zher2k)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                 const void *alpha, const void *A, const MKL_INT lda,
                                 const void *B, const MKL_INT ldb, const double beta,
                                 void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zherk)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                                const double alpha, const void *A, const MKL_INT lda,
                                const double beta, void *C, const MKL_INT ldc, void *interop_obj) NOTHROW;

// Level2

// Routines with S, D, C, Z prefixes (Standard)
void MKL_CBLAS_VARIANT_NAME(sgemv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const float alpha, const float *A, const MKL_INT lda,
                                const float *X, const MKL_INT incX, const float beta,
                                float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sgbmv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const MKL_INT KL, const MKL_INT KU, const float alpha,
                                const float *A, const MKL_INT lda, const float *X,
                                const MKL_INT incX, const float beta, float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(strmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const float *A, const MKL_INT lda,
                                float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(stbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const float *A, const MKL_INT lda,
                                float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(stpmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const float *Ap, float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(strsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const float *A, const MKL_INT lda, float *X,
                                const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(stbsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const float *A, const MKL_INT lda,
                                float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(stpsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const float *Ap, float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgemv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const double alpha, const double *A, const MKL_INT lda,
                                const double *X, const MKL_INT incX, const double beta,
                                double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dgbmv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const MKL_INT KL, const MKL_INT KU, const double alpha,
                                const double *A, const MKL_INT lda, const double *X,
                                const MKL_INT incX, const double beta, double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtrmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const double *A, const MKL_INT lda,
                                double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const double *A, const MKL_INT lda,
                                double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtpmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const double *Ap, double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtrsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const double *A, const MKL_INT lda, double *X,
                                const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtbsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const double *A, const MKL_INT lda,
                                double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dtpsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const double *Ap, double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgemv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *X, const MKL_INT incX, const void *beta,
                                void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgbmv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const MKL_INT KL, const MKL_INT KU, const void *alpha,
                                const void *A, const MKL_INT lda, const void *X,
                                const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctrmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *A, const MKL_INT lda,
                                void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                                void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctpmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *Ap, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctrsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *A, const MKL_INT lda, void *X,
                                const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctbsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                                void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ctpsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *Ap, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgemv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *A, const MKL_INT lda,
                                const void *X, const MKL_INT incX, const void *beta,
                                void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgbmv)(const CBLAS_LAYOUT Layout,
                                const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                                const MKL_INT KL, const MKL_INT KU, const void *alpha,
                                const void *A, const MKL_INT lda, const void *X,
                                const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztrmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *A, const MKL_INT lda,
                                void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                                void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztpmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *Ap, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztrsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *A, const MKL_INT lda, void *X,
                                const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztbsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                                void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ztpsv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                                const MKL_INT N, const void *Ap, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

// Routines with S, D prefixes
void MKL_CBLAS_VARIANT_NAME(ssymv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const float alpha, const float *A,
                                const MKL_INT lda, const float *X, const MKL_INT incX,
                                const float beta, float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                                const MKL_INT lda, const float *X, const MKL_INT incX,
                                const float beta, float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sspmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const float alpha, const float *Ap,
                                const float *X, const MKL_INT incX,
                                const float beta, float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sger)(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                               const float alpha, const float *X, const MKL_INT incX,
                               const float *Y, const MKL_INT incY, float *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssyr)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const float alpha, const float *X,
                               const MKL_INT incX, float *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sspr)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const float alpha, const float *X,
                               const MKL_INT incX, float *Ap, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ssyr2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const float alpha, const float *X,
                                const MKL_INT incX, const float *Y, const MKL_INT incY, float *A,
                                const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sspr2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const float alpha, const float *X,
                                const MKL_INT incX, const float *Y, const MKL_INT incY, float *A, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsymv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const double alpha, const double *A,
                                const MKL_INT lda, const double *X, const MKL_INT incX,
                                const double beta, double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const MKL_INT K, const double alpha, const double *A,
                                const MKL_INT lda, const double *X, const MKL_INT incX,
                                const double beta, double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dspmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const double alpha, const double *Ap,
                                const double *X, const MKL_INT incX,
                                const double beta, double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dger)(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                               const double alpha, const double *X, const MKL_INT incX,
                               const double *Y, const MKL_INT incY, double *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsyr)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const double alpha, const double *X,
                               const MKL_INT incX, double *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dspr)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const double alpha, const double *X,
                               const MKL_INT incX, double *Ap, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dsyr2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const double alpha, const double *X,
                                const MKL_INT incX, const double *Y, const MKL_INT incY, double *A,
                                const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dspr2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const double alpha, const double *X,
                                const MKL_INT incX, const double *Y, const MKL_INT incY, double *A, void *interop_obj) NOTHROW;

// Routines with C, Z prefixes
void MKL_CBLAS_VARIANT_NAME(chemv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const void *alpha, const void *A,
                                const MKL_INT lda, const void *X, const MKL_INT incX,
                                const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(chbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const MKL_INT K, const void *alpha, const void *A,
                                const MKL_INT lda, const void *X, const MKL_INT incX,
                                const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(chpmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const void *alpha, const void *Ap,
                                const void *X, const MKL_INT incX,
                                const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgeru)(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cgerc)(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cher)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const float alpha, const void *X, const MKL_INT incX,
                               void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(chpr)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const float alpha, const void *X,
                               const MKL_INT incX, void *A, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cher2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(chpr2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *Ap, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zhemv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const void *alpha, const void *A,
                                const MKL_INT lda, const void *X, const MKL_INT incX,
                                const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zhbmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const MKL_INT K, const void *alpha, const void *A,
                                const MKL_INT lda, const void *X, const MKL_INT incX,
                                const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zhpmv)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                const MKL_INT N, const void *alpha, const void *Ap,
                                const void *X, const MKL_INT incX,
                                const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgeru)(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zgerc)(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zher)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const double alpha, const void *X, const MKL_INT incX,
                               void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zhpr)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                               const MKL_INT N, const double alpha, const void *X,
                               const MKL_INT incX, void *A, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zher2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *A, const MKL_INT lda, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zhpr2)(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                                const void *alpha, const void *X, const MKL_INT incX,
                                const void *Y, const MKL_INT incY, void *Ap, void *interop_obj) NOTHROW;


// Level1

// Routines with S, D, DS, SDS prefixes
float MKL_CBLAS_VARIANT_NAME(sdot)(const MKL_INT N, const float  *X, const MKL_INT incX,
                                const float  *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

double MKL_CBLAS_VARIANT_NAME(ddot)(const MKL_INT N, const double *X, const MKL_INT incX,
                                 const double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

double MKL_CBLAS_VARIANT_NAME(dsdot)(const MKL_INT N, const float  *X, const MKL_INT incX,
                                  const float  *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

float MKL_CBLAS_VARIANT_NAME(sdsdot)(const MKL_INT N, const float sb, const float  *X,
		                  const MKL_INT incX, const float  *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

// Routines with C, Z prefixes
void MKL_CBLAS_VARIANT_NAME(cdotu)(const MKL_INT N, const void *X, const MKL_INT incX,
                                    const void *Y, const MKL_INT incY, void *dotu, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cdotc)(const MKL_INT N, const void *X, const MKL_INT incX,
                                    const void *Y, const MKL_INT incY, void *dotc, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zdotu)(const MKL_INT N, const void *X, const MKL_INT incX,
                                    const void *Y, const MKL_INT incY, void *dotu, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zdotc)(const MKL_INT N, const void *X, const MKL_INT incX,
                                    const void *Y, const MKL_INT incY, void *dotc, void *interop_obj) NOTHROW;

// Routines with S, D, SC, DZ prefixes
float MKL_CBLAS_VARIANT_NAME(snrm2)(const MKL_INT N, const float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

float MKL_CBLAS_VARIANT_NAME(sasum)(const MKL_INT N, const float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

double MKL_CBLAS_VARIANT_NAME(dnrm2)(const MKL_INT N, const double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

double MKL_CBLAS_VARIANT_NAME(dasum)(const MKL_INT N, const double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

float MKL_CBLAS_VARIANT_NAME(scnrm2)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

float MKL_CBLAS_VARIANT_NAME(scasum)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

double MKL_CBLAS_VARIANT_NAME(dznrm2)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

double MKL_CBLAS_VARIANT_NAME(dzasum)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;


// Routines with S, D, C, Z prefixes (Standard)
CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(isamax)(const MKL_INT N, const float  *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(idamax)(const MKL_INT N, const double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(icamax)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(izamax)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(isamin)(const MKL_INT N, const float  *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(idamin)(const MKL_INT N, const double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(icamin)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

CBLAS_INDEX MKL_CBLAS_VARIANT_NAME(izamin)(const MKL_INT N, const void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(sswap)(const MKL_INT N, float *X, const MKL_INT incX,
                                float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(scopy)(const MKL_INT N, const float *X, const MKL_INT incX,
                                float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(saxpy)(const MKL_INT N, const float alpha, const float *X,
                                const MKL_INT incX, float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(srotg)(float *a, float *b, float *c, float *s, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dswap)(const MKL_INT N, double *X, const MKL_INT incX,
                                double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dcopy)(const MKL_INT N, const double *X, const MKL_INT incX,
                                double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(daxpy)(const MKL_INT N, const double alpha, const double *X,
                                const MKL_INT incX, double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(drotg)(double *a, double *b, double *c, double *s, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cswap)(const MKL_INT N, void *X, const MKL_INT incX,
                                void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(ccopy)(const MKL_INT N, const void *X, const MKL_INT incX,
                                void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(caxpy)(const MKL_INT N, const void *alpha, const void *X,
                                const MKL_INT incX, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(crotg)(void *a, const void *b, float *c, void *s, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zswap)(const MKL_INT N, void *X, const MKL_INT incX,
                                void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zcopy)(const MKL_INT N, const void *X, const MKL_INT incX,
                                void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zaxpy)(const MKL_INT N, const void *alpha, const void *X,
                                const MKL_INT incX, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zrotg)(void *a, const void *b, double *c, void *s, void *interop_obj) NOTHROW;

// Routines with S, D prefixes
void MKL_CBLAS_VARIANT_NAME(srotmg)(float *d1, float *d2, float *b1, const float b2, float *P, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(srot)(const MKL_INT N, float *X, const MKL_INT incX,
                               float *Y, const MKL_INT incY, const float c, const float s, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(srotm)(const MKL_INT N, float *X, const MKL_INT incX,
                                float *Y, const MKL_INT incY, const float *P, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(drotmg)(double *d1, double *d2, double *b1, const double b2, double *P, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(drot)(const MKL_INT N, double *X, const MKL_INT incX,
                               double *Y, const MKL_INT incY, const double c, const double  s, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(drotm)(const MKL_INT N, double *X, const MKL_INT incX,
                                double *Y, const MKL_INT incY, const double *P, void *interop_obj) NOTHROW;

// Routines with CS, ZD prefixes
void MKL_CBLAS_VARIANT_NAME(csrot)(const MKL_INT N, void *X, const MKL_INT incX,
                                void *Y, const MKL_INT incY, const float c, const float s, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zdrot)(const MKL_INT N, void *X, const MKL_INT incX,
                                void *Y, const MKL_INT incY, const double c, const double  s, void *interop_obj) NOTHROW;

// Routines with S D C Z CS and ZD prefixes
void MKL_CBLAS_VARIANT_NAME(sscal)(const MKL_INT N, const float alpha, float *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(dscal)(const MKL_INT N, const double alpha, double *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(cscal)(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zscal)(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(csscal)(const MKL_INT N, const float alpha, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zdscal)(const MKL_INT N, const double alpha, void *X, const MKL_INT incX, void *interop_obj) NOTHROW;



// BLAS API

// Level3

// Routines with S, D, C, Z prefixes (Standard)
void MKL_BLAS_VARIANT_NAME(sgemm)(const char *transa, const char *transb,
                               const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               const float *b, const MKL_INT *ldb,
                               const float *beta, float *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sgemmt)(const char *uplo, const char *transa, const char *transb,
                                const MKL_INT *n, const MKL_INT *k,
                                const float *alpha, const float *a, const MKL_INT *lda,
                                const float *b, const MKL_INT *ldb,
                                const float *beta, float *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssymm)(const char *side, const char *uplo,
                               const MKL_INT *m, const MKL_INT *n,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               const float *b, const MKL_INT *ldb,
                               const float *beta, float *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssyr2k)(const char *uplo, const char *trans,
                                const MKL_INT *n, const MKL_INT *k,
                                const float *alpha, const float *a, const MKL_INT *lda,
                                const float *b, const MKL_INT *ldb,
                                const float *beta, float *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssyrk)(const char *uplo, const char *trans,
                               const MKL_INT *n, const MKL_INT *k,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               const float *beta,
                               float *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(strmm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               float *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(strsm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               float *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemm)(const char *transa, const char *transb,
                               const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                               const double *alpha, const double *a, const MKL_INT *lda,
                               const double *b, const MKL_INT *ldb,
                               const double *beta, double *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemmt)(const char *uplo, const char *transa, const char *transb,
                                const MKL_INT *n, const MKL_INT *k,
                                const double *alpha, const double *a, const MKL_INT *lda,
                                const double *b, const MKL_INT *ldb,
                                const double *beta, double *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsymm)(const char *side, const char *uplo,
                               const MKL_INT *m, const MKL_INT *n,
                               const double *alpha, const double *a, const MKL_INT *lda,
                               const double *b, const MKL_INT *ldb,
                               const double *beta, double *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsyr2k)(const char *uplo, const char *trans,
                                const MKL_INT *n, const MKL_INT *k,
                                const double *alpha, const double *a, const MKL_INT *lda,
                                const double *b, const MKL_INT *ldb,
                                const double *beta, double *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsyrk)(const char *uplo, const char *trans,
                               const MKL_INT *n, const MKL_INT *k,
                               const double *alpha, const double *a,
                               const MKL_INT *lda, const double *beta,
                               double *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtrmm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const double *alpha, const double *a, const MKL_INT *lda,
                               double *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtrsm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const double *alpha, const double *a, const MKL_INT *lda,
                               double *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemm)(const char *transa, const char *transb,
                               const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                               const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                               MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemmt)(const char *uplo, const char *transa, const char *transb,
                                const MKL_INT *n, const MKL_INT *k,
                                const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                                const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                                MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(csymm)(const char *side, const char *uplo,
                               const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                               const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                               MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(csyr2k)(const char *uplo, const char *trans,
                                const MKL_INT *n, const MKL_INT *k,
                                const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                                const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                                MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(csyrk)(const char *uplo, const char *trans,
                               const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                               const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctrmm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                               MKL_Complex8 *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctrsm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                               MKL_Complex8 *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemm)(const char *transa, const char *transb,
                               const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                               MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemmt)(const char *uplo, const char *transa, const char *transb,
                                const MKL_INT *n, const MKL_INT *k,
                                const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                                const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                                MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zsymm)(const char *side, const char *uplo,
                               const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                               MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zsyr2k)(const char *uplo, const char *trans,
                                const MKL_INT *n, const MKL_INT *k,
                                const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                                const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                                MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zsyrk)(const char *uplo, const char *trans,
                               const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztrmm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               MKL_Complex16 *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztrsm)(const char *side, const char *uplo, const char *transa,
                               const char *diag, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               MKL_Complex16 *b, const MKL_INT *ldb, void *interop_obj) NOTHROW;

// Routines with C, Z prefixes

void MKL_BLAS_VARIANT_NAME(chemm)(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                               const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                               MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cher2k)(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
                                const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                                const MKL_Complex8 *b, const MKL_INT *ldb, const float *beta,
                                MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cherk)(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
                               const float *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const float *beta,
                               MKL_Complex8 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zhemm)(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                               MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zher2k)(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
                                const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                                const MKL_Complex16 *b, const MKL_INT *ldb, const double *beta,
                                MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zherk)(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
                               const double *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                               const double *beta, MKL_Complex16 *c, const MKL_INT *ldc, void *interop_obj) NOTHROW;


// Level2

// Routines with S, D, C, Z prefixes (Standard)
void MKL_BLAS_VARIANT_NAME(sgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               const float *x, const MKL_INT *incx,
                               const float *beta, float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sgbmv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const MKL_INT *kl, const MKL_INT *ku, const float *alpha,
                               const float *a, const MKL_INT *lda, const float *x,
                               const MKL_INT *incx,
                               const float *beta, float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(strmv)(const char *uplo, const char *transa, const char *diag,
                               const MKL_INT *n, const float *a,
                               const MKL_INT *lda, float *b, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(stbmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k, const float *a,
                               const MKL_INT *lda, float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(stpmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const float *ap,
                               float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(strsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const float *a, const MKL_INT *lda,
                               float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(stbsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k, const float *a,
                               const MKL_INT *lda, float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(stpsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const float *ap,
                               float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const double *alpha, const double *a, const MKL_INT *lda,
                               const double *x, const MKL_INT *incx,
                               const double *beta, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dgbmv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const MKL_INT *kl, const MKL_INT *ku, const double *alpha,
                               const double *a, const MKL_INT *lda, const double *x,
                               const MKL_INT *incx,
                               const double *beta, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtrmv)(const char *uplo, const char *transa, const char *diag,
                               const MKL_INT *n, const double *a,
                               const MKL_INT *lda, double *b, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtbmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k, const double *a,
                               const MKL_INT *lda, double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtpmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const double *ap,
                               double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtrsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const double *a, const MKL_INT *lda,
                               double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtbsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k, const double *a,
                               const MKL_INT *lda, double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dtpsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const double *ap,
                               double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a,
                               const MKL_INT *lda, const MKL_Complex8 *x,
                               const MKL_INT *incx, const MKL_Complex8 *beta,
                               MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgbmv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const MKL_INT *kl, const MKL_INT *ku,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a,
                               const MKL_INT *lda,
                               const MKL_Complex8 *x, const MKL_INT *incx, const
                               MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctrmv)(const char *uplo, const char *transa, const char *diag,
                               const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *lda,
                               MKL_Complex8 *b, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctbmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex8 *a, const MKL_INT *lda,
                               MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctpmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_Complex8 *ap, MKL_Complex8 *x,
                               const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctrsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *lda,
                               MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctbsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex8 *a, const MKL_INT *lda,
                               MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ctpsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_Complex8 *ap, MKL_Complex8 *x,
                               const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const MKL_Complex16 *alpha,
                               const MKL_Complex16 *a, const MKL_INT *lda,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *beta, MKL_Complex16 *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgbmv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                               const MKL_INT *kl, const MKL_INT *ku,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a,
                               const MKL_INT *lda,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *beta,
                               MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztrmv)(const char *uplo, const char *transa, const char *diag,
                               const MKL_INT *n,
                               const MKL_Complex16 *a, const MKL_INT *lda,
                               MKL_Complex16 *b, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztbmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex16 *a, const MKL_INT *lda,
                               MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztpmv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_Complex16 *ap, MKL_Complex16 *x,
                               const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztrsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n,
                               const MKL_Complex16 *a, const MKL_INT *lda,
                               MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztbsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex16 *a, const MKL_INT *lda,
                               MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ztpsv)(const char *uplo, const char *trans, const char *diag,
                               const MKL_INT *n, const MKL_Complex16 *ap, MKL_Complex16 *x,
                               const MKL_INT *incx, void *interop_obj) NOTHROW;

// Routines with S, D prefixes
void MKL_BLAS_VARIANT_NAME(ssymv)(const char *uplo, const MKL_INT *n, const float *alpha,
                               const float *a, const MKL_INT *lda,
                               const float *x, const MKL_INT *incx, const float *beta,
                               float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssbmv)(const char *uplo, const MKL_INT *n, const MKL_INT *k,
                               const float *alpha, const float *a, const MKL_INT *lda,
                               const float *x, const MKL_INT *incx,
                               const float *beta, float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sspmv)(const char *uplo, const MKL_INT *n, const float *alpha,
                               const float *ap, const float *x, const MKL_INT *incx,
                               const float *beta, float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sger)(const MKL_INT *m, const MKL_INT *n, const float *alpha,
                              const float *x, const MKL_INT *incx,
                              const float *y, const MKL_INT *incy, float *a,
                              const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssyr)(const char *uplo, const MKL_INT *n, const float *alpha,
                              const float *x, const MKL_INT *incx,
                              float *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sspr)(const char *uplo, const MKL_INT *n, const float *alpha,
                              const float *x, const MKL_INT *incx, float *ap, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ssyr2)(const char *uplo, const MKL_INT *n, const float *alpha,
                               const float *x, const MKL_INT *incx,
                               const float *y, const MKL_INT *incy, float *a,
                               const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sspr2)(const char *uplo, const MKL_INT *n, const float *alpha,
                               const float *x, const MKL_INT *incx,
                               const float *y, const MKL_INT *incy, float *ap, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsymv)(const char *uplo, const MKL_INT *n, const double *alpha,
                               const double *a, const MKL_INT *lda,
                               const double *x, const MKL_INT *incx,
                               const double *beta, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsbmv)(const char *uplo, const MKL_INT *n, const MKL_INT *k,
                               const double *alpha, const double *a, const MKL_INT *lda,
                               const double *x, const MKL_INT *incx,
                               const double *beta, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dspmv)(const char *uplo, const MKL_INT *n, const double *alpha,
                               const double *ap, const double *x, const MKL_INT *incx,
                               const double *beta, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dger)(const MKL_INT *m, const MKL_INT *n, const double *alpha,
                              const double *x, const MKL_INT *incx,
                              const double *y, const MKL_INT *incy, double *a,
                              const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsyr)(const char *uplo, const MKL_INT *n, const double *alpha,
                              const double *x, const MKL_INT *incx,
                              double *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dspr)(const char *uplo, const MKL_INT *n, const double *alpha,
                              const double *x, const MKL_INT *incx, double *ap, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dsyr2)(const char *uplo, const MKL_INT *n, const double *alpha,
                               const double *x, const MKL_INT *incx,
                               const double *y, const MKL_INT *incy, double *a,
                               const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dspr2)(const char *uplo, const MKL_INT *n, const double *alpha,
                               const double *x, const MKL_INT *incx,
                               const double *y, const MKL_INT *incy, double *ap, void *interop_obj) NOTHROW;

// Routines with C, Z prefixes
void MKL_BLAS_VARIANT_NAME(chemv)(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
                               const MKL_Complex8 *a, const MKL_INT *lda,
                               const MKL_Complex8 *x, const MKL_INT *incx,
                               const MKL_Complex8 *beta, MKL_Complex8 *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(chbmv)(const char *uplo, const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex8 *alpha, const MKL_Complex8 *a,
                               const MKL_INT *lda, const MKL_Complex8 *x,
                               const MKL_INT *incx, const MKL_Complex8 *beta,
                               MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(chpmv)(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
                               const MKL_Complex8 *ap, const MKL_Complex8 *x,
                               const MKL_INT *incx, const MKL_Complex8 *beta,
                               MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgeru)(const MKL_INT *m, const MKL_INT *n, const  MKL_Complex8 *alpha,
                               const MKL_Complex8 *x, const MKL_INT *incx,
                               const MKL_Complex8 *y, const MKL_INT *incy,
                               MKL_Complex8 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cgerc)(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                               const MKL_Complex8 *x, const MKL_INT *incx,
                               const MKL_Complex8 *y, const MKL_INT *incy,
                               MKL_Complex8 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cher)(const char *uplo, const MKL_INT *n, const float *alpha,
                              const MKL_Complex8 *x, const MKL_INT *incx,
                              MKL_Complex8 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(chpr)(const char *uplo, const MKL_INT *n, const float *alpha,
                              const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *ap, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cher2)(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
                               const MKL_Complex8 *x, const MKL_INT *incx,
                               const MKL_Complex8 *y, const MKL_INT *incy,
                               MKL_Complex8 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(chpr2)(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
                               const MKL_Complex8 *x, const MKL_INT *incx,
                               const MKL_Complex8 *y, const MKL_INT *incy,
                               MKL_Complex8 *ap, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zhemv)(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
                               const MKL_Complex16 *a, const MKL_INT *lda,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *beta, MKL_Complex16 *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zhbmv)(const char *uplo, const MKL_INT *n, const MKL_INT *k,
                               const MKL_Complex16 *alpha, const MKL_Complex16 *a,
                               const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *beta, MKL_Complex16 *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zhpmv)(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
                               const MKL_Complex16 *ap, const MKL_Complex16 *x,
                               const MKL_INT *incx, const MKL_Complex16 *beta,
                               MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgeru)(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *y, const MKL_INT *incy,
                               MKL_Complex16 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zgerc)(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *y, const MKL_INT *incy,
                               MKL_Complex16 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zher)(const char *uplo, const MKL_INT *n, const double *alpha,
                              const MKL_Complex16 *x, const MKL_INT *incx,
                              MKL_Complex16 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zhpr)(const char *uplo, const MKL_INT *n, const double *alpha,
                              const MKL_Complex16 *x, const MKL_INT *incx,
                              MKL_Complex16 *ap, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zher2)(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *y, const MKL_INT *incy,
                               MKL_Complex16 *a, const MKL_INT *lda, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zhpr2)(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
                               const MKL_Complex16 *x, const MKL_INT *incx,
                               const MKL_Complex16 *y, const MKL_INT *incy,
                               MKL_Complex16 *ap, void *interop_obj) NOTHROW;


// Level1

// Routines with S, D, DS, SDS prefixes
float MKL_BLAS_VARIANT_NAME(sdot)(const MKL_INT *n, const float *x, const MKL_INT *incx,
                               const float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

double MKL_BLAS_VARIANT_NAME(ddot)(const MKL_INT *n, const double *x, const MKL_INT *incx,
                                const double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

double MKL_BLAS_VARIANT_NAME(dsdot)(const MKL_INT *n, const float *x, const MKL_INT *incx,
                                 const float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

float MKL_BLAS_VARIANT_NAME(sdsdot)(const MKL_INT *n, const float *sb, const float *x,
                                 const MKL_INT *incx, const float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

// Routines with C, Z prefixes
void MKL_BLAS_VARIANT_NAME(cdotc)(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x,
                               const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cdotu)(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x,
                               const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zdotc)(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x,
                               const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zdotu)(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x,
                               const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

// Routines with S, D, SC, DZ prefixes
float MKL_BLAS_VARIANT_NAME(snrm2)(const MKL_INT *n, const float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

float MKL_BLAS_VARIANT_NAME(sasum)(const MKL_INT *n, const float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

double MKL_BLAS_VARIANT_NAME(dnrm2)(const MKL_INT *n, const double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

double MKL_BLAS_VARIANT_NAME(dasum)(const MKL_INT *n, const double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

float MKL_BLAS_VARIANT_NAME(scnrm2)(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

float MKL_BLAS_VARIANT_NAME(scasum)(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

double MKL_BLAS_VARIANT_NAME(dznrm2)(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

double MKL_BLAS_VARIANT_NAME(dzasum)(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

// Routines with S, D, C, Z prefixes (Standard)
MKL_INT MKL_BLAS_VARIANT_NAME(isamax)(const MKL_INT *n, const float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(idamax)(const MKL_INT *n, const double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(icamax)(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(izamax)(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(isamin)(const MKL_INT *n, const float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(icamin)(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(idamin)(const MKL_INT *n, const double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

MKL_INT MKL_BLAS_VARIANT_NAME(izamin)(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(sswap)(const MKL_INT *n, float *x, const MKL_INT *incx, float *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(scopy)(const MKL_INT *n, const float *x, const MKL_INT *incx, float *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(saxpy)(const MKL_INT *n, const float *alpha, const float *x,
                               const MKL_INT *incx, float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(srotg)(float *a,float *b,float *c,float *s, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dswap)(const MKL_INT *n, double *x, const MKL_INT *incx, double *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dcopy)(const MKL_INT *n, const double *x, const MKL_INT *incx, double *y,
                               const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(daxpy)(const MKL_INT *n, const double *alpha, const double *x,
                               const MKL_INT *incx, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(drotg)(double *a, double *b, double *c, double *s, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cswap)(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx,
                               MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(ccopy)(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx,
                               MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(caxpy)(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x,
                               const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(crotg)(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zswap)(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx,
                               MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zcopy)(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx,
                               MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zaxpy)(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x,
                               const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zrotg)(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s, void *interop_obj) NOTHROW;

// Routines with S, D prefixes
void MKL_BLAS_VARIANT_NAME(srotmg)(float *d1, float *d2, float *x1, const float *y1, float *param, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(srot)(const MKL_INT *n, float *x, const MKL_INT *incx, float *y,
                              const MKL_INT *incy, const float *c, const float *s, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(srotm)(const MKL_INT *n, float *x, const MKL_INT *incx, float *y,
                               const MKL_INT *incy, const float *param, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(drotmg)(double *d1, double *d2, double *x1, const double *y1, double *param, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(drot)(const MKL_INT *n, double *x, const MKL_INT *incx, double *y,
                              const MKL_INT *incy, const double *c, const double *s, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(drotm)(const MKL_INT *n, double *x, const MKL_INT *incx, double *y,
                               const MKL_INT *incy, const double *param, void *interop_obj) NOTHROW;

// Routines with CS, ZD prefixes
void MKL_BLAS_VARIANT_NAME(csrot)(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y,
                               const MKL_INT *incy, const float *c, const float *s, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zdrot)(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y,
                               const MKL_INT *incy, const double *c, const double *s, void *interop_obj) NOTHROW;

// Routines with S D C Z CS and ZD prefixes
void MKL_BLAS_VARIANT_NAME(sscal)(const MKL_INT *n, const float *a, float *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(dscal)(const MKL_INT *n, const double *a, double *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(cscal)(const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zscal)(const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(csscal)(const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zdscal)(const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *incx, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(saxpby)(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                                   const float *beta, float *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(caxpby)(const MKL_INT *n, const MKL_Complex8 *alpha,
                                   const MKL_Complex8 *x, const MKL_INT *incx,
                                   const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(daxpby)(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                                   const double *beta, double *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_BLAS_VARIANT_NAME(zaxpby)(const MKL_INT *n, const MKL_Complex16 *alpha,
                                   const MKL_Complex16 *x, const MKL_INT *incx,
                                   const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(saxpby)(const MKL_INT N, const float alpha, const float *X,
                  const MKL_INT incX, const float beta, float *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(daxpby)(const MKL_INT N, const double alpha, const double *X,
                  const MKL_INT incX, const double beta, double *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(caxpby)(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW;

void MKL_CBLAS_VARIANT_NAME(zaxpby)(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY, void *interop_obj) NOTHROW ;


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

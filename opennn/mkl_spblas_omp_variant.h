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
!    Intel(R) oneAPI Math Kernel Library (oneMKL) C OpenMP offload
!    interface for Sparse BLAS
!******************************************************************************/

#ifndef _MKL_SPBLAS_OMP_VARIANT_H_
#define _MKL_SPBLAS_OMP_VARIANT_H_

#include "mkl_types.h"
#include "mkl_spblas.h"
#include "mkl_omp_variant.h"

#define MKL_SPBLAS_VARIANT_NAME(func) MKL_VARIANT_NAME(sparse, func)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(s_create_csr)( sparse_matrix_t           *A,
                                                         const sparse_index_base_t indexing,
                                                         const MKL_INT             rows,
                                                         const MKL_INT             cols,
                                                               MKL_INT             *rows_start,
                                                               MKL_INT             *rows_end,
                                                               MKL_INT             *col_indx,
                                                               float               *values );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(d_create_csr)( sparse_matrix_t           *A,
                                                         const sparse_index_base_t indexing,
                                                         const MKL_INT             rows,
                                                         const MKL_INT             cols,
                                                               MKL_INT             *rows_start,
                                                               MKL_INT             *rows_end,
                                                               MKL_INT             *col_indx,
                                                               double              *values );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(s_export_csr)( const sparse_matrix_t  source,
                                                           sparse_index_base_t    *indexing,
                                                           MKL_INT                *rows,
                                                           MKL_INT                *cols,
                                                           MKL_INT                **rows_start,
                                                           MKL_INT                **rows_end,
                                                           MKL_INT                **col_indx,
                                                           float                  **values );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(d_export_csr)( const sparse_matrix_t  source,
                                                           sparse_index_base_t    *indexing,
                                                           MKL_INT                *rows,
                                                           MKL_INT                *cols,
                                                           MKL_INT                **rows_start,
                                                           MKL_INT                **rows_end,
                                                           MKL_INT                **col_indx,
                                                           double                 **values );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(destroy)( sparse_matrix_t  A );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(set_sv_hint)( const sparse_matrix_t     A,
                                                          const sparse_operation_t  operation,
                                                          const struct matrix_descr descr,
                                                          const MKL_INT             expected_calls );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(optimize)( sparse_matrix_t  A );

    /* Level 2 */

    /*   Computes y = alpha * A * x + beta * y   */
    sparse_status_t MKL_SPBLAS_VARIANT_NAME(s_mv)( const sparse_operation_t  operation,
                                                 const float               alpha,
                                                 const sparse_matrix_t     A,
                                                 const struct matrix_descr descr,
                                                 const float               *x,
                                                 const float               beta,
                                                 float                     *y );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(d_mv)( const sparse_operation_t  operation,
                                                 const double              alpha,
                                                 const sparse_matrix_t     A,
                                                 const struct matrix_descr descr,
                                                 const double              *x,
                                                 const double              beta,
                                                 double                    *y );

    /*   Solves triangular system y = alpha * A^{-1} * x   */
    sparse_status_t MKL_SPBLAS_VARIANT_NAME(s_trsv) ( const sparse_operation_t  operation,
                                                      const float               alpha,
                                                      const sparse_matrix_t     A,
                                                      const struct matrix_descr descr,
                                                      const float               *x,
                                                      float                     *y );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(d_trsv) ( const sparse_operation_t  operation,
                                                      const double              alpha,
                                                      const sparse_matrix_t     A,
                                                      const struct matrix_descr descr,
                                                      const double              *x,
                                                      double                    *y );

    /* Level 3 */

    /*   Computes y = alpha * A * x + beta * y   */
    sparse_status_t MKL_SPBLAS_VARIANT_NAME(s_mm)( const sparse_operation_t  operation,
                                                   const float               alpha,
                                                   const sparse_matrix_t     A,
                                                   const struct matrix_descr descr,
                                                   const sparse_layout_t     layout,
                                                   const float               *x,
                                                   const MKL_INT             columns,
                                                   const MKL_INT             ldx,
                                                   const float               beta,
                                                   float                     *y,
                                                   const MKL_INT             ldy );

    sparse_status_t MKL_SPBLAS_VARIANT_NAME(d_mm)( const sparse_operation_t  operation,
                                                   const double              alpha,
                                                   const sparse_matrix_t     A,
                                                   const struct matrix_descr descr,
                                                   const sparse_layout_t     layout,
                                                   const double              *x,
                                                   const MKL_INT             columns,
                                                   const MKL_INT             ldx,
                                                   const double              beta,
                                                   double                    *y,
                                                   const MKL_INT             ldy );

    /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse   */
    sparse_status_t MKL_SPBLAS_VARIANT_NAME(sp2m) ( const sparse_operation_t  transA, 
                                                    const struct matrix_descr descrA, 
                                                    const sparse_matrix_t     A,
                                                    const sparse_operation_t  transB, 
                                                    const struct matrix_descr descrB, 
                                                    const sparse_matrix_t     B,
                                                    const sparse_request_t    request, 
                                                    sparse_matrix_t           *C );

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif /*_MKL_SPBLAS_OMP_VARIANT_H_ */

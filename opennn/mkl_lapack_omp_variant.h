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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) C interface for
!      OpenMP offload for LAPACK
!******************************************************************************/

#ifndef _MKL_LAPACK_OMP_VARIANT_H_
#define _MKL_LAPACK_OMP_VARIANT_H_

#include "mkl_types.h"
#include "mkl_omp_variant.h"

#define MKL_LAPACK_OPENMP_OFFLOAD(name) MKL_VARIANT_NAME(lapack, name)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void MKL_LAPACK_OPENMP_OFFLOAD(cgebrd)(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                                       float* d, float* e, MKL_Complex8* tauq, MKL_Complex8* taup, MKL_Complex8* work,
                                       const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgebrd)(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* d,
                                       double* e, double* tauq, double* taup, double* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgebrd)(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* d,
                                       float* e, float* tauq, float* taup, float* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgebrd)(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                                       double* d, double* e, MKL_Complex16* tauq, MKL_Complex16* taup,
                                       MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgeqrf)(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                                       MKL_Complex8* tau, MKL_Complex8* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgeqrf)(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* tau,
                                       double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgeqrf)(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* tau,
                                       float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgeqrf)(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                                       MKL_Complex16* tau, MKL_Complex16* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgesvd)(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n,
                                       MKL_Complex8* a, const MKL_INT* lda, float* s, MKL_Complex8* u,
                                       const MKL_INT* ldu, MKL_Complex8* vt, const MKL_INT* ldvt, MKL_Complex8* work,
                                       const MKL_INT* lwork, float* rwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgesvd)(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n,
                                       double* a, const MKL_INT* lda, double* s, double* u, const MKL_INT* ldu,
                                       double* vt, const MKL_INT* ldvt, double* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgesvd)(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n,
                                       float* a, const MKL_INT* lda, float* s, float* u, const MKL_INT* ldu, float* vt,
                                       const MKL_INT* ldvt, float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgesvd)(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n,
                                       MKL_Complex16* a, const MKL_INT* lda, double* s, MKL_Complex16* u,
                                       const MKL_INT* ldu, MKL_Complex16* vt, const MKL_INT* ldvt, MKL_Complex16* work,
                                       const MKL_INT* lwork, double* rwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrf)(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                                       MKL_INT* ipiv, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrf)(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* ipiv,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrf)(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* ipiv,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrf)(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                                       MKL_INT* ipiv, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetri)(const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, const MKL_INT* ipiv,
                                       MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetri)(const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* ipiv,
                                       double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetri)(const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* ipiv, float* work,
                                       const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetri)(const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, const MKL_INT* ipiv,
                                       MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrs)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a,
                                       const MKL_INT* lda, const MKL_INT* ipiv, MKL_Complex8* b, const MKL_INT* ldb,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrs)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
                                       const MKL_INT* lda, const MKL_INT* ipiv, double* b, const MKL_INT* ldb,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrs)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
                                       const MKL_INT* lda, const MKL_INT* ipiv, float* b, const MKL_INT* ldb,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrs)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a,
                                       const MKL_INT* lda, const MKL_INT* ipiv, MKL_Complex16* b, const MKL_INT* ldb,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cheev)(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a,
                                      const MKL_INT* lda, float* w, MKL_Complex8* work, const MKL_INT* lwork,
                                      float* rwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zheev)(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a,
                                      const MKL_INT* lda, double* w, MKL_Complex16* work, const MKL_INT* lwork,
                                      double* rwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cheevd)(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a,
                                       const MKL_INT* lda, float* w, MKL_Complex8* work, const MKL_INT* lwork,
                                       float* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zheevd)(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a,
                                       const MKL_INT* lda, double* w, MKL_Complex16* work, const MKL_INT* lwork,
                                       double* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cheevx)(const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
                                       MKL_Complex8* a, const MKL_INT* lda, const float* vl, const float* vu,
                                       const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w,
                                       MKL_Complex8* z, const MKL_INT* ldz, MKL_Complex8* work, const MKL_INT* lwork,
                                       float* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zheevx)(const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
                                       MKL_Complex16* a, const MKL_INT* lda, const double* vl, const double* vu,
                                       const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m,
                                       double* w, MKL_Complex16* z, const MKL_INT* ldz, MKL_Complex16* work,
                                       const MKL_INT* lwork, double* rwork, MKL_INT* iwork, MKL_INT* ifail,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(chegvd)(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n,
                                       MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb,
                                       float* w, MKL_Complex8* work, const MKL_INT* lwork, float* rwork,
                                       const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zhegvd)(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n,
                                       MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb,
                                       double* w, MKL_Complex16* work, const MKL_INT* lwork, double* rwork,
                                       const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(chegvx)(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo,
                                       const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b,
                                       const MKL_INT* ldb, const float* vl, const float* vu, const MKL_INT* il,
                                       const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, MKL_Complex8* z,
                                       const MKL_INT* ldz, MKL_Complex8* work, const MKL_INT* lwork, float* rwork,
                                       MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zhegvx)(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo,
                                       const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b,
                                       const MKL_INT* ldb, const double* vl, const double* vu, const MKL_INT* il,
                                       const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, MKL_Complex16* z,
                                       const MKL_INT* ldz, MKL_Complex16* work, const MKL_INT* lwork, double* rwork,
                                       MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(chetrd)(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                                       float* d, float* e, MKL_Complex8* tau, MKL_Complex8* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zhetrd)(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                                       double* d, double* e, MKL_Complex16* tau, MKL_Complex16* work,
                                       const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dorgqr)(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, double* a,
                                       const MKL_INT* lda, const double* tau, double* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sorgqr)(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, float* a,
                                       const MKL_INT* lda, const float* tau, float* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dormqr)(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n,
                                       const MKL_INT* k, const double* a, const MKL_INT* lda, const double* tau,
                                       double* c, const MKL_INT* ldc, double* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sormqr)(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n,
                                       const MKL_INT* k, const float* a, const MKL_INT* lda, const float* tau, float* c,
                                       const MKL_INT* ldc, float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(csteqr)(const char* compz, const MKL_INT* n, float* d, float* e, MKL_Complex8* z,
                                       const MKL_INT* ldz, float* work, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dsteqr)(const char* compz, const MKL_INT* n, double* d, double* e, double* z,
                                       const MKL_INT* ldz, double* work, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssteqr)(const char* compz, const MKL_INT* n, float* d, float* e, float* z,
                                       const MKL_INT* ldz, float* work, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zsteqr)(const char* compz, const MKL_INT* n, double* d, double* e, MKL_Complex16* z,
                                       const MKL_INT* ldz, double* work, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dsyev)(const char* jobz, const char* uplo, const MKL_INT* n, double* a,
                                      const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
                                      MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssyev)(const char* jobz, const char* uplo, const MKL_INT* n, float* a,
                                      const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
                                      MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dsyevd)(const char* jobz, const char* uplo, const MKL_INT* n, double* a,
                                       const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
                                       MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssyevd)(const char* jobz, const char* uplo, const MKL_INT* n, float* a,
                                       const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork, MKL_INT* iwork,
                                       const MKL_INT* liwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dsyevx)(const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
                                       double* a, const MKL_INT* lda, const double* vl, const double* vu,
                                       const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m,
                                       double* w, double* z, const MKL_INT* ldz, double* work, const MKL_INT* lwork,
                                       MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssyevx)(const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
                                       float* a, const MKL_INT* lda, const float* vl, const float* vu,
                                       const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w,
                                       float* z, const MKL_INT* ldz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
                                       MKL_INT* ifail, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dsygvd)(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n,
                                       double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, double* w,
                                       double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssygvd)(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n,
                                       float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb, float* w,
                                       float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dsygvx)(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo,
                                       const MKL_INT* n, double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb,
                                       const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
                                       const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
                                       double* work, const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssygvx)(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo,
                                       const MKL_INT* n, float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb,
                                       const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
                                       const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
                                       float* work, const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail,
                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(dsytrd)(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* d,
                                       double* e, double* tau, double* work, const MKL_INT* lwork,
                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ssytrd)(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* d,
                                       float* e, float* tau, float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(ctrtrs)(const char* uplo, const char* trans, const char* diag, const MKL_INT* n,
                                       const MKL_INT* nrhs, const MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b,
                                       const MKL_INT* ldb, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dtrtrs)(const char* uplo, const char* trans, const char* diag, const MKL_INT* n,
                                       const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
                                       const MKL_INT* ldb, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(strtrs)(const char* uplo, const char* trans, const char* diag, const MKL_INT* n,
                                       const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
                                       const MKL_INT* ldb, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(ztrtrs)(const char* uplo, const char* trans, const char* diag, const MKL_INT* n,
                                       const MKL_INT* nrhs, const MKL_Complex16* a, const MKL_INT* lda,
                                       MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cungqr)(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, MKL_Complex8* a,
                                       const MKL_INT* lda, const MKL_Complex8* tau, MKL_Complex8* work,
                                       const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zungqr)(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, MKL_Complex16* a,
                                       const MKL_INT* lda, const MKL_Complex16* tau, MKL_Complex16* work,
                                       const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cunmqr)(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n,
                                       const MKL_INT* k, const MKL_Complex8* a, const MKL_INT* lda,
                                       const MKL_Complex8* tau, MKL_Complex8* c, const MKL_INT* ldc, MKL_Complex8* work,
                                       const MKL_INT* lwork, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zunmqr)(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n,
                                       const MKL_INT* k, const MKL_Complex16* a, const MKL_INT* lda,
                                       const MKL_Complex16* tau, MKL_Complex16* c, const MKL_INT* ldc,
                                       MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrf_batch_strided)(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a,
                                                     const MKL_INT* lda, const MKL_INT* stride_a, MKL_INT* ipiv,
                                                     const MKL_INT* stride_ipiv, const MKL_INT* batch_size,
                                                     MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrf_batch_strided)(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda,
                                                     const MKL_INT* stride_a, MKL_INT* ipiv, const MKL_INT* stride_ipiv,
                                                     const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrf_batch_strided)(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda,
                                                     const MKL_INT* stride_a, MKL_INT* ipiv, const MKL_INT* stride_ipiv,
                                                     const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrf_batch_strided)(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a,
                                                     const MKL_INT* lda, const MKL_INT* stride_a, MKL_INT* ipiv,
                                                     const MKL_INT* stride_ipiv, const MKL_INT* batch_size,
                                                     MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetri_oop_batch_strided)(const MKL_INT* n, const MKL_Complex8* a, const MKL_INT* lda,
                                                         const MKL_INT* stride_a, const MKL_INT* ipiv,
                                                         const MKL_INT* stride_ipiv, MKL_Complex8* ainv,
                                                         const MKL_INT* ldainv, const MKL_INT* stride_ainv,
                                                         const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetri_oop_batch_strided)(const MKL_INT* n, const double* a, const MKL_INT* lda,
                                                         const MKL_INT* stride_a, const MKL_INT* ipiv,
                                                         const MKL_INT* stride_ipiv, double* ainv,
                                                         const MKL_INT* ldainv, const MKL_INT* stride_ainv,
                                                         const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetri_oop_batch_strided)(const MKL_INT* n, const float* a, const MKL_INT* lda,
                                                         const MKL_INT* stride_a, const MKL_INT* ipiv,
                                                         const MKL_INT* stride_ipiv, float* ainv, const MKL_INT* ldainv,
                                                         const MKL_INT* stride_ainv, const MKL_INT* batch_size,
                                                         MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetri_oop_batch_strided)(const MKL_INT* n, const MKL_Complex16* a, const MKL_INT* lda,
                                                         const MKL_INT* stride_a, const MKL_INT* ipiv,
                                                         const MKL_INT* stride_ipiv, MKL_Complex16* ainv,
                                                         const MKL_INT* ldainv, const MKL_INT* stride_ainv,
                                                         const MKL_INT* batch_size, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetri_oop_batch)(const MKL_INT* n, const MKL_Complex8** a, const MKL_INT* lda, const MKL_INT** ipiv,
                                                 MKL_Complex8** ainv, const MKL_INT* ldainv, const MKL_INT* group_count,
                                                 const MKL_INT* group_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetri_oop_batch)(const MKL_INT* n, const double** a, const MKL_INT* lda, const MKL_INT** ipiv,
                                                 double** ainv, const MKL_INT* ldainv, const MKL_INT* group_count,
                                                 const MKL_INT* group_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetri_oop_batch)(const MKL_INT* n, const float** a, const MKL_INT* lda, const MKL_INT** ipiv,
                                                 float** ainv, const MKL_INT* ldainv, const MKL_INT* group_count,
                                                 const MKL_INT* group_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetri_oop_batch)(const MKL_INT* n, const MKL_Complex16** a, const MKL_INT* lda, const MKL_INT** ipiv,
                                                 MKL_Complex16** ainv, const MKL_INT* ldainv, const MKL_INT* group_count,
                                                 const MKL_INT* group_size, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrs_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                     const MKL_Complex8* a, const MKL_INT* lda, const MKL_INT* stride_a,
                                                     const MKL_INT* ipiv, const MKL_INT* stride_ipiv, MKL_Complex8* b,
                                                     const MKL_INT* ldb, const MKL_INT* stride_b,
                                                     const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrs_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                     const double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                                                     const MKL_INT* ipiv, const MKL_INT* stride_ipiv, double* b,
                                                     const MKL_INT* ldb, const MKL_INT* stride_b,
                                                     const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrs_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                     const float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                                                     const MKL_INT* ipiv, const MKL_INT* stride_ipiv, float* b,
                                                     const MKL_INT* ldb, const MKL_INT* stride_b,
                                                     const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrs_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                     const MKL_Complex16* a, const MKL_INT* lda,
                                                     const MKL_INT* stride_a, const MKL_INT* ipiv,
                                                     const MKL_INT* stride_ipiv, MKL_Complex16* b, const MKL_INT* ldb,
                                                     const MKL_INT* stride_b, const MKL_INT* batch_size,
                                                     MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrfnp_batch_strided)(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a,
                                                       const MKL_INT* lda, const MKL_INT* stride_a,
                                                       const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrfnp_batch_strided)(const MKL_INT* m, const MKL_INT* n, double* a,
                                                       const MKL_INT* lda, const MKL_INT* stride_a,
                                                       const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrfnp_batch_strided)(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda,
                                                       const MKL_INT* stride_a, const MKL_INT* batch_size,
                                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrfnp_batch_strided)(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a,
                                                       const MKL_INT* lda, const MKL_INT* stride_a,
                                                       const MKL_INT* batch_size, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrsnp_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                       const MKL_Complex8* a, const MKL_INT* lda,
                                                       const MKL_INT* stride_a, MKL_Complex8* b, const MKL_INT* ldb,
                                                       const MKL_INT* stride_b, const MKL_INT* batch_size,
                                                       MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrsnp_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                       const double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                                                       double* b, const MKL_INT* ldb, const MKL_INT* stride_b,
                                                       const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrsnp_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                       const float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                                                       float* b, const MKL_INT* ldb, const MKL_INT* stride_b,
                                                       const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrsnp_batch_strided)(const char* trans, const MKL_INT* n, const MKL_INT* nrhs,
                                                       const MKL_Complex16* a, const MKL_INT* lda,
                                                       const MKL_INT* stride_a, MKL_Complex16* b, const MKL_INT* ldb,
                                                       const MKL_INT* stride_b, const MKL_INT* batch_size,
                                                       MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cpotrf)(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dpotrf)(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(spotrf)(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zpotrf)(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cpotri)(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dpotri)(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(spotri)(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zpotri)(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cpotrs)(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dpotrs)(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(spotrs)(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zpotrs)(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

void MKL_LAPACK_OPENMP_OFFLOAD(cgetrf_batch)(const MKL_INT* m, const MKL_INT* n, MKL_Complex8** a, const MKL_INT* lda,
                                             MKL_INT** ipiv, const MKL_INT* group_count, const MKL_INT* group_size,
                                             MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(dgetrf_batch)(const MKL_INT* m, const MKL_INT* n, double** a, const MKL_INT* lda,
                                             MKL_INT** ipiv, const MKL_INT* group_count, const MKL_INT* group_size,
                                             MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(sgetrf_batch)(const MKL_INT* m, const MKL_INT* n, float** a, const MKL_INT* lda,
                                             MKL_INT** ipiv, const MKL_INT* group_count, const MKL_INT* group_size,
                                             MKL_INT* info) NOTHROW;
void MKL_LAPACK_OPENMP_OFFLOAD(zgetrf_batch)(const MKL_INT* m, const MKL_INT* n, MKL_Complex16** a, const MKL_INT* lda,
                                             MKL_INT** ipiv, const MKL_INT* group_count, const MKL_INT* group_size,
                                             MKL_INT* info) NOTHROW;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _MKL_LAPACK_OMP_VARIANT_H_ */

    

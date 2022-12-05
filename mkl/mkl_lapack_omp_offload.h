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

#ifndef _MKL_LAPACK_OMP_OFFLOAD_H_
#define _MKL_LAPACK_OMP_OFFLOAD_H_

#include "mkl_types.h"
#include "mkl_lapack_omp_variant.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgebrd)) match(construct={target variant dispatch}, device={arch(gen)})
void cgebrd(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* d, float* e,
            MKL_Complex8* tauq, MKL_Complex8* taup, MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgebrd)) match(construct={target variant dispatch}, device={arch(gen)})
void dgebrd(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* d, double* e, double* tauq,
            double* taup, double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgebrd)) match(construct={target variant dispatch}, device={arch(gen)})
void sgebrd(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* d, float* e, float* tauq,
            float* taup, float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgebrd)) match(construct={target variant dispatch}, device={arch(gen)})
void zgebrd(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* d, double* e,
            MKL_Complex16* tauq, MKL_Complex16* taup, MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgeqrf)) match(construct={target variant dispatch}, device={arch(gen)})
void cgeqrf(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* tau,
            MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgeqrf)) match(construct={target variant dispatch}, device={arch(gen)})
void dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* tau, double* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgeqrf)) match(construct={target variant dispatch}, device={arch(gen)})
void sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* tau, float* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgeqrf)) match(construct={target variant dispatch}, device={arch(gen)})
void zgeqrf(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* tau,
            MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgesvd)) match(construct={target variant dispatch}, device={arch(gen)})
void cgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a,
            const MKL_INT* lda, float* s, MKL_Complex8* u, const MKL_INT* ldu, MKL_Complex8* vt, const MKL_INT* ldvt,
            MKL_Complex8* work, const MKL_INT* lwork, float* rwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgesvd)) match(construct={target variant dispatch}, device={arch(gen)})
void dgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda,
            double* s, double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgesvd)) match(construct={target variant dispatch}, device={arch(gen)})
void sgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda,
            float* s, float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgesvd)) match(construct={target variant dispatch}, device={arch(gen)})
void zgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a,
            const MKL_INT* lda, double* s, MKL_Complex16* u, const MKL_INT* ldu, MKL_Complex16* vt, const MKL_INT* ldvt,
            MKL_Complex16* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrf)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrf(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_INT* ipiv,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrf)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrf(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrf)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrf(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrf)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrf(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_INT* ipiv,
            MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetri)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetri(const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, const MKL_INT* ipiv, MKL_Complex8* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetri)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetri(const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* ipiv, double* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetri)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetri(const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* ipiv, float* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetri)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetri(const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, const MKL_INT* ipiv, MKL_Complex16* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrs)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a, const MKL_INT* lda,
            const MKL_INT* ipiv, MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrs)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
            const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrs)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda,
            const MKL_INT* ipiv, float* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrs)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a, const MKL_INT* lda,
            const MKL_INT* ipiv, MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cheev)) match(construct={target variant dispatch}, device={arch(gen)})
void cheev(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* w,
           MKL_Complex8* work, const MKL_INT* lwork, float* rwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zheev)) match(construct={target variant dispatch}, device={arch(gen)})
void zheev(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* w,
           MKL_Complex16* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cheevd)) match(construct={target variant dispatch}, device={arch(gen)})
void cheevd(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* w,
            MKL_Complex8* work, const MKL_INT* lwork, float* rwork, const MKL_INT* lrwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zheevd)) match(construct={target variant dispatch}, device={arch(gen)})
void zheevd(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* w,
            MKL_Complex16* work, const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cheevx)) match(construct={target variant dispatch}, device={arch(gen)})
void cheevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, MKL_Complex8* a,
            const MKL_INT* lda, const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
            const float* abstol, MKL_INT* m, float* w, MKL_Complex8* z, const MKL_INT* ldz, MKL_Complex8* work,
            const MKL_INT* lwork, float* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zheevx)) match(construct={target variant dispatch}, device={arch(gen)})
void zheevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, MKL_Complex16* a,
            const MKL_INT* lda, const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
            const double* abstol, MKL_INT* m, double* w, MKL_Complex16* z, const MKL_INT* ldz, MKL_Complex16* work,
            const MKL_INT* lwork, double* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(chegvd)) match(construct={target variant dispatch}, device={arch(gen)})
void chegvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a,
            const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, float* w, MKL_Complex8* work, const MKL_INT* lwork,
            float* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zhegvd)) match(construct={target variant dispatch}, device={arch(gen)})
void zhegvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a,
            const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, double* w, MKL_Complex16* work,
            const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork,
            MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(chegvx)) match(construct={target variant dispatch}, device={arch(gen)})
void chegvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
            MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, const float* vl, const float* vu,
            const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, MKL_Complex8* z,
            const MKL_INT* ldz, MKL_Complex8* work, const MKL_INT* lwork, float* rwork, MKL_INT* iwork, MKL_INT* ifail,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zhegvx)) match(construct={target variant dispatch}, device={arch(gen)})
void zhegvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
            MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, const double* vl,
            const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w,
            MKL_Complex16* z, const MKL_INT* ldz, MKL_Complex16* work, const MKL_INT* lwork, double* rwork,
            MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(chetrd)) match(construct={target variant dispatch}, device={arch(gen)})
void chetrd(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* d, float* e,
            MKL_Complex8* tau, MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zhetrd)) match(construct={target variant dispatch}, device={arch(gen)})
void zhetrd(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* d, double* e,
            MKL_Complex16* tau, MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dorgqr)) match(construct={target variant dispatch}, device={arch(gen)})
void dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
            double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sorgqr)) match(construct={target variant dispatch}, device={arch(gen)})
void sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
            float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dormqr)) match(construct={target variant dispatch}, device={arch(gen)})
void dormqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
            const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc, double* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sormqr)) match(construct={target variant dispatch}, device={arch(gen)})
void sormqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
            const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc, float* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(csteqr)) match(construct={target variant dispatch}, device={arch(gen)})
void csteqr(const char* compz, const MKL_INT* n, float* d, float* e, MKL_Complex8* z, const MKL_INT* ldz, float* work,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsteqr)) match(construct={target variant dispatch}, device={arch(gen)})
void dsteqr(const char* compz, const MKL_INT* n, double* d, double* e, double* z, const MKL_INT* ldz, double* work,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssteqr)) match(construct={target variant dispatch}, device={arch(gen)})
void ssteqr(const char* compz, const MKL_INT* n, float* d, float* e, float* z, const MKL_INT* ldz, float* work,
            MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zsteqr)) match(construct={target variant dispatch}, device={arch(gen)})
void zsteqr(const char* compz, const MKL_INT* n, double* d, double* e, MKL_Complex16* z, const MKL_INT* ldz,
            double* work, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsyev)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyev(const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
           const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssyev)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyev(const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
           const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsyevd)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyevd(const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w,
            double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssyevd)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyevd(const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
            const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsyevx)) match(construct={target variant dispatch}, device={arch(gen)})
void dsyevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
            const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m,
            double* w, double* z, const MKL_INT* ldz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
            MKL_INT* ifail, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssyevx)) match(construct={target variant dispatch}, device={arch(gen)})
void ssyevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
            const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m,
            float* w, float* z, const MKL_INT* ldz, float* work, const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail,
            MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsygvd)) match(construct={target variant dispatch}, device={arch(gen)})
void dsygvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
            double* b, const MKL_INT* ldb, double* w, double* work, const MKL_INT* lwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssygvd)) match(construct={target variant dispatch}, device={arch(gen)})
void ssygvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
            float* b, const MKL_INT* ldb, float* w, float* work, const MKL_INT* lwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsygvx)) match(construct={target variant dispatch}, device={arch(gen)})
void dsygvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, double* a,
            const MKL_INT* lda, double* b, const MKL_INT* ldb, const double* vl, const double* vu, const MKL_INT* il,
            const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz, double* work,
            const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssygvx)) match(construct={target variant dispatch}, device={arch(gen)})
void ssygvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, float* a,
            const MKL_INT* lda, float* b, const MKL_INT* ldb, const float* vl, const float* vu, const MKL_INT* il,
            const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz, float* work,
            const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dsytrd)) match(construct={target variant dispatch}, device={arch(gen)})
void dsytrd(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* d, double* e, double* tau,
            double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ssytrd)) match(construct={target variant dispatch}, device={arch(gen)})
void ssytrd(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* d, float* e, float* tau,
            float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ctrtrs)) match(construct={target variant dispatch}, device={arch(gen)})
void ctrtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dtrtrs)) match(construct={target variant dispatch}, device={arch(gen)})
void dtrtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(strtrs)) match(construct={target variant dispatch}, device={arch(gen)})
void strtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(ztrtrs)) match(construct={target variant dispatch}, device={arch(gen)})
void ztrtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cungqr)) match(construct={target variant dispatch}, device={arch(gen)})
void cungqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, MKL_Complex8* a, const MKL_INT* lda,
            const MKL_Complex8* tau, MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zungqr)) match(construct={target variant dispatch}, device={arch(gen)})
void zungqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, MKL_Complex16* a, const MKL_INT* lda,
            const MKL_Complex16* tau, MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cunmqr)) match(construct={target variant dispatch}, device={arch(gen)})
void cunmqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
            const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* tau, MKL_Complex8* c, const MKL_INT* ldc,
            MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zunmqr)) match(construct={target variant dispatch}, device={arch(gen)})
void zunmqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
            const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* tau, MKL_Complex16* c, const MKL_INT* ldc,
            MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrf_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                          const MKL_INT* stride_a, MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrf_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                          MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrf_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                          MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrf_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                          const MKL_INT* stride_a, MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetri_oop_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetri_oop_batch_strided(const MKL_INT* n, const MKL_Complex8* a, const MKL_INT* lda, const MKL_INT* stride_a,
                              const MKL_INT* ipiv, const MKL_INT* stride_ipiv, MKL_Complex8* ainv,
                              const MKL_INT* ldainv, const MKL_INT* stride_ainv, const MKL_INT* batch_size,
                              MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetri_oop_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetri_oop_batch_strided(const MKL_INT* n, const double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                              const MKL_INT* ipiv, const MKL_INT* stride_ipiv, double* ainv, const MKL_INT* ldainv,
                              const MKL_INT* stride_ainv, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetri_oop_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetri_oop_batch_strided(const MKL_INT* n, const float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                              const MKL_INT* ipiv, const MKL_INT* stride_ipiv, float* ainv, const MKL_INT* ldainv,
                              const MKL_INT* stride_ainv, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetri_oop_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetri_oop_batch_strided(const MKL_INT* n, const MKL_Complex16* a, const MKL_INT* lda, const MKL_INT* stride_a,
                              const MKL_INT* ipiv, const MKL_INT* stride_ipiv, MKL_Complex16* ainv,
                              const MKL_INT* ldainv, const MKL_INT* stride_ainv, const MKL_INT* batch_size,
                              MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetri_oop_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetri_oop_batch(const MKL_INT* n, const MKL_Complex8** a, const MKL_INT* lda, const MKL_INT** ipiv, MKL_Complex8** ainv,
                      const MKL_INT* ldainv, const MKL_INT* group_count, const MKL_INT* group_size,
                      MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetri_oop_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetri_oop_batch(const MKL_INT* n, const double** a, const MKL_INT* lda, const MKL_INT** ipiv, double** ainv,
                      const MKL_INT* ldainv, const MKL_INT* group_count, const MKL_INT* group_size,
                      MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetri_oop_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetri_oop_batch(const MKL_INT* n, const float** a, const MKL_INT* lda, const MKL_INT** ipiv, float** ainv, const MKL_INT* ldainv,
                      const MKL_INT* group_count, const MKL_INT* group_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetri_oop_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetri_oop_batch(const MKL_INT* n, const MKL_Complex16** a, const MKL_INT* lda, const MKL_INT** ipiv, MKL_Complex16** ainv,
                      const MKL_INT* ldainv, const MKL_INT* group_count, const MKL_INT* group_size,
                      MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrs_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrs_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a,
                          const MKL_INT* lda, const MKL_INT* stride_a, const MKL_INT* ipiv, const MKL_INT* stride_ipiv,
                          MKL_Complex8* b, const MKL_INT* ldb, const MKL_INT* stride_b, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrs_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrs_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
                          const MKL_INT* stride_a, const MKL_INT* ipiv, const MKL_INT* stride_ipiv, double* b,
                          const MKL_INT* ldb, const MKL_INT* stride_b, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrs_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrs_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda,
                          const MKL_INT* stride_a, const MKL_INT* ipiv, const MKL_INT* stride_ipiv, float* b,
                          const MKL_INT* ldb, const MKL_INT* stride_b, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrs_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrs_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a,
                          const MKL_INT* lda, const MKL_INT* stride_a, const MKL_INT* ipiv, const MKL_INT* stride_ipiv,
                          MKL_Complex16* b, const MKL_INT* ldb, const MKL_INT* stride_b, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrfnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                            const MKL_INT* stride_a, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrfnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                            const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrfnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                            const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrfnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                            const MKL_INT* stride_a, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrsnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, MKL_Complex8* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrsnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, double* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrsnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, float* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrsnp_batch_strided)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, MKL_Complex16* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cpotrf)) match(construct={target variant dispatch}, device={arch(gen)})
void cpotrf(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dpotrf)) match(construct={target variant dispatch}, device={arch(gen)})
void dpotrf(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(spotrf)) match(construct={target variant dispatch}, device={arch(gen)})
void spotrf(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zpotrf)) match(construct={target variant dispatch}, device={arch(gen)})
void zpotrf(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cpotri)) match(construct={target variant dispatch}, device={arch(gen)})
void cpotri(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dpotri)) match(construct={target variant dispatch}, device={arch(gen)})
void dpotri(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(spotri)) match(construct={target variant dispatch}, device={arch(gen)})
void spotri(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zpotri)) match(construct={target variant dispatch}, device={arch(gen)})
void zpotri(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cpotrs)) match(construct={target variant dispatch}, device={arch(gen)})
void cpotrs(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dpotrs)) match(construct={target variant dispatch}, device={arch(gen)})
void dpotrs(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(spotrs)) match(construct={target variant dispatch}, device={arch(gen)})
void spotrs(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zpotrs)) match(construct={target variant dispatch}, device={arch(gen)})
void zpotrs(const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(cgetrf_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void cgetrf_batch(const MKL_INT* m, const MKL_INT* n, MKL_Complex8** a, const MKL_INT* lda, MKL_INT** ipiv,
                  const MKL_INT* group_count, const MKL_INT* group_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrf_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void dgetrf_batch(const MKL_INT* m, const MKL_INT* n, double** a, const MKL_INT* lda, MKL_INT** ipiv,
                  const MKL_INT* group_count, const MKL_INT* group_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(sgetrf_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void sgetrf_batch(const MKL_INT* m, const MKL_INT* n, float** a, const MKL_INT* lda, MKL_INT** ipiv,
                  const MKL_INT* group_count, const MKL_INT* group_size, MKL_INT* info) NOTHROW;
#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(zgetrf_batch)) match(construct={target variant dispatch}, device={arch(gen)})
void zgetrf_batch(const MKL_INT* m, const MKL_INT* n, MKL_Complex16** a, const MKL_INT* lda, MKL_INT** ipiv,
                  const MKL_INT* group_count, const MKL_INT* group_size, MKL_INT* info) NOTHROW;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _MKL_LAPACK_OMP_VARIANT_H_ */

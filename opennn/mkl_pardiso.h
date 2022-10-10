/*******************************************************************************
* Copyright 2004-2022 Intel Corporation.
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
*   Content : oneMKL PARDISO C header file
*
*           Contains interface to PARDISO.
*
********************************************************************************
*/
#if !defined( __MKL_PARDISO_H )

#define __MKL_PARDISO_H

#include "mkl_dss.h"

#ifdef __GNUC__
#define MKL_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define MKL_DEPRECATED __declspec(deprecated)
#else
#define MKL_DEPRECATED
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#if !defined(MKL_CALL_CONV)
#   if defined(__MIC__) || defined(__TARGET_ARCH_MIC)
#       define MKL_CALL_CONV
#   else
#       define MKL_CALL_CONV __cdecl
#   endif
#endif

#if  !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)   extern rtype MKL_CALL_CONV   name    arg;
#endif

#if  !defined(_Mkl_Deprecated_Api)
#define _Mkl_Deprecated_Api(rtype,name,arg)   extern MKL_DEPRECATED rtype MKL_CALL_CONV   name    arg;
#endif

_Mkl_Api(void,pardiso,(
    _MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *, const MKL_INT *,
    const MKL_INT *,   const MKL_INT *, const void *,    const MKL_INT *,
    const MKL_INT *,   MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *,   void *,          void *,          MKL_INT *))

_Mkl_Api(void,PARDISO,(
    _MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *, const MKL_INT *,
    const MKL_INT *,   const MKL_INT *, const void *,    const MKL_INT *,
    const MKL_INT *,   MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *,   void *,          void *,          MKL_INT *))

_Mkl_Api(void,pardisoinit,(
    _MKL_DSS_HANDLE_t,   const MKL_INT *,  MKL_INT *))

_Mkl_Api(void,PARDISOINIT,(
    _MKL_DSS_HANDLE_t,   const MKL_INT *,   MKL_INT *))

_Mkl_Api(void,pardiso_64,(
    _MKL_DSS_HANDLE_t,     const long long int *, const long long int *, const long long int *,
    const long long int *, const long long int *, const void *,          const long long int *,
    const long long int *, long long int *, const long long int *, long long int *,
    const long long int *, void *,                void *,                long long int *))
_Mkl_Api(void,PARDISO_64,(
    _MKL_DSS_HANDLE_t,     const long long int *, const long long int *, const long long int *,
    const long long int *, const long long int *, const void *,          const long long int *,
    const long long int *, long long int *, const long long int *, long long int *,
    const long long int *, void *,                void *,                long long int *))

_Mkl_Api(void,pardiso_handle_store_64,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));
_Mkl_Api(void,PARDISO_HANDLE_STORE_64,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));

_Mkl_Api(void,pardiso_handle_restore_64,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));
_Mkl_Api(void,PARDISO_HANDLE_RESTORE_64,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));

_Mkl_Api(void,pardiso_handle_delete_64,( const char*, MKL_INT *));
_Mkl_Api(void,PARDISO_HANDLE_DELETE_64,( const char*, MKL_INT *));

/* Error classes */
#define PARDISO_NO_ERROR                 0
#define PARDISO_UNIMPLEMENTED         -101
#define PARDISO_NULL_HANDLE           -102
#define PARDISO_MEMORY_ERROR          -103

_Mkl_Api(void,pardiso_handle_store,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));
_Mkl_Api(void,PARDISO_HANDLE_STORE,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));

_Mkl_Api(void,pardiso_handle_restore,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));
_Mkl_Api(void,PARDISO_HANDLE_RESTORE,( _MKL_DSS_HANDLE_t, const char*, MKL_INT *));

_Mkl_Api(void,pardiso_handle_delete,( const char*, MKL_INT *));
_Mkl_Api(void,PARDISO_HANDLE_DELETE,( const char*, MKL_INT *));

/* oneMKL Progress routine */
#ifndef _MKL_PARDISO_PIVOT_H_
#define _MKL_PARDISO_PIVOT_H_
_Mkl_Api(int,MKL_PARDISO_PIVOT, ( const double* aii, double* bii, const double* eps ))
_Mkl_Api(int,MKL_PARDISO_PIVOT_,( const double* aii, double* bii, const double* eps ))
_Mkl_Api(int,mkl_pardiso_pivot, ( const double* aii, double* bii, const double* eps ))
_Mkl_Api(int,mkl_pardiso_pivot_,( const double* aii, double* bii, const double* eps ))
#endif /* _MKL_PARDISO_PIVOT_H_ */
_Mkl_Api(void,pardiso_getdiag,( const _MKL_DSS_HANDLE_t, void *,       void *, const MKL_INT *, MKL_INT *  ))
_Mkl_Api(void,pardiso_export,( void *, void *, MKL_INT *, MKL_INT *, const MKL_INT *, const MKL_INT *, MKL_INT * ))

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

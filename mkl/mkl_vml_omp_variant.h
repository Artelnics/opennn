/* -== file: mkl_vml_omp_variant.h ==- */
/*******************************************************************************
* Copyright 2006-2022 Intel Corporation.
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

#ifndef _MKL_VML_OMP_VARIANT_H_
#define _MKL_VML_OMP_VARIANT_H_ 1

#include "mkl_types.h"
#include "mkl_omp_variant.h"

#ifdef __cplusplus
extern "C" {
#endif


unsigned int MKL_VARIANT_NAME(vm, vmlsetmode)(const MKL_UINT mode) NOTHROW;
unsigned int MKL_VARIANT_NAME(vm, vmlgetmode)(void) NOTHROW;


int MKL_VARIANT_NAME(vm, vmlseterrstatus)(const MKL_INT new_status) NOTHROW;
int MKL_VARIANT_NAME(vm, vmlgeterrstatus)(void) NOTHROW;
int MKL_VARIANT_NAME(vm, vmlclearerrstatus)(void) NOTHROW;





/* function: Abs, indexing: simple */
void MKL_VARIANT_NAME(vm, vsabs)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsabs)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdabs)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdabs)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcabs)(const MKL_INT n, const MKL_Complex8 * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcabs)(const MKL_INT n, const MKL_Complex8 * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzabs)(const MKL_INT n, const MKL_Complex16 * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzabs)(const MKL_INT n, const MKL_Complex16 * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Abs, indexing: strided */
void MKL_VARIANT_NAME(vm, vsabsi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsabsi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdabsi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdabsi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcabsi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcabsi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzabsi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzabsi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Acos, indexing: simple */
void MKL_VARIANT_NAME(vm, vsacos)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsacos)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdacos)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdacos)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcacos)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcacos)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzacos)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzacos)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Acos, indexing: strided */
void MKL_VARIANT_NAME(vm, vsacosi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsacosi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdacosi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdacosi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcacosi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcacosi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzacosi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzacosi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Acosh, indexing: simple */
void MKL_VARIANT_NAME(vm, vsacosh)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsacosh)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdacosh)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdacosh)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcacosh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcacosh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzacosh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzacosh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Acosh, indexing: strided */
void MKL_VARIANT_NAME(vm, vsacoshi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsacoshi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdacoshi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdacoshi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcacoshi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcacoshi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzacoshi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzacoshi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Acospi, indexing: simple */
void MKL_VARIANT_NAME(vm, vsacospi)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsacospi)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdacospi)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdacospi)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Acospi, indexing: strided */
void MKL_VARIANT_NAME(vm, vsacospii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsacospii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdacospii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdacospii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Add, indexing: simple */
void MKL_VARIANT_NAME(vm, vsadd)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsadd)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdadd)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdadd)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcadd)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcadd)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzadd)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzadd)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Add, indexing: strided */
void MKL_VARIANT_NAME(vm, vsaddi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsaddi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdaddi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdaddi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcaddi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcaddi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzaddi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzaddi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Arg, indexing: simple */
void MKL_VARIANT_NAME(vm, vcarg)(const MKL_INT n, const MKL_Complex8 * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcarg)(const MKL_INT n, const MKL_Complex8 * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzarg)(const MKL_INT n, const MKL_Complex16 * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzarg)(const MKL_INT n, const MKL_Complex16 * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Arg, indexing: strided */
void MKL_VARIANT_NAME(vm, vcargi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcargi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzargi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzargi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Asin, indexing: simple */
void MKL_VARIANT_NAME(vm, vsasin)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsasin)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdasin)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdasin)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcasin)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcasin)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzasin)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzasin)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Asin, indexing: strided */
void MKL_VARIANT_NAME(vm, vsasini)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsasini)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdasini)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdasini)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcasini)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcasini)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzasini)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzasini)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Asinh, indexing: simple */
void MKL_VARIANT_NAME(vm, vsasinh)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsasinh)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdasinh)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdasinh)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcasinh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcasinh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzasinh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzasinh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Asinh, indexing: strided */
void MKL_VARIANT_NAME(vm, vsasinhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsasinhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdasinhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdasinhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcasinhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcasinhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzasinhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzasinhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Asinpi, indexing: simple */
void MKL_VARIANT_NAME(vm, vsasinpi)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsasinpi)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdasinpi)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdasinpi)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Asinpi, indexing: strided */
void MKL_VARIANT_NAME(vm, vsasinpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsasinpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdasinpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdasinpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Atan, indexing: simple */
void MKL_VARIANT_NAME(vm, vsatan)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatan)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatan)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatan)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcatan)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcatan)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzatan)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzatan)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Atan, indexing: strided */
void MKL_VARIANT_NAME(vm, vsatani)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatani)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatani)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatani)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcatani)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcatani)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzatani)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzatani)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Atan2, indexing: simple */
void MKL_VARIANT_NAME(vm, vsatan2)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatan2)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatan2)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatan2)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Atan2, indexing: strided */
void MKL_VARIANT_NAME(vm, vsatan2i)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatan2i)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatan2i)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatan2i)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Atan2pi, indexing: simple */
void MKL_VARIANT_NAME(vm, vsatan2pi)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatan2pi)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatan2pi)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatan2pi)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Atan2pi, indexing: strided */
void MKL_VARIANT_NAME(vm, vsatan2pii)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatan2pii)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatan2pii)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatan2pii)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Atanh, indexing: simple */
void MKL_VARIANT_NAME(vm, vsatanh)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatanh)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatanh)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatanh)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcatanh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcatanh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzatanh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzatanh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Atanh, indexing: strided */
void MKL_VARIANT_NAME(vm, vsatanhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatanhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatanhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatanhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcatanhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcatanhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzatanhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzatanhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Atanpi, indexing: simple */
void MKL_VARIANT_NAME(vm, vsatanpi)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatanpi)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatanpi)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatanpi)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Atanpi, indexing: strided */
void MKL_VARIANT_NAME(vm, vsatanpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsatanpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdatanpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdatanpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Cbrt, indexing: simple */
void MKL_VARIANT_NAME(vm, vscbrt)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscbrt)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcbrt)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcbrt)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Cbrt, indexing: strided */
void MKL_VARIANT_NAME(vm, vscbrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscbrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcbrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcbrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: CdfNorm, indexing: simple */
void MKL_VARIANT_NAME(vm, vscdfnorm)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscdfnorm)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcdfnorm)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcdfnorm)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: CdfNorm, indexing: strided */
void MKL_VARIANT_NAME(vm, vscdfnormi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscdfnormi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcdfnormi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcdfnormi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: CdfNormInv, indexing: simple */
void MKL_VARIANT_NAME(vm, vscdfnorminv)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscdfnorminv)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcdfnorminv)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcdfnorminv)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: CdfNormInv, indexing: strided */
void MKL_VARIANT_NAME(vm, vscdfnorminvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscdfnorminvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcdfnorminvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcdfnorminvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Ceil, indexing: simple */
void MKL_VARIANT_NAME(vm, vsceil)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsceil)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdceil)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdceil)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Ceil, indexing: strided */
void MKL_VARIANT_NAME(vm, vsceili)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsceili)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdceili)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdceili)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: CIS, indexing: simple */
void MKL_VARIANT_NAME(vm, vccis)(const MKL_INT n, const float * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmccis)(const MKL_INT n, const float * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzcis)(const MKL_INT n, const double * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzcis)(const MKL_INT n, const double * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: CIS, indexing: strided */
void MKL_VARIANT_NAME(vm, vccisi)(const MKL_INT n, const float * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmccisi)(const MKL_INT n, const float * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzcisi)(const MKL_INT n, const double * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzcisi)(const MKL_INT n, const double * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Conj, indexing: simple */
void MKL_VARIANT_NAME(vm, vcconj)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcconj)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzconj)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzconj)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Conj, indexing: strided */
void MKL_VARIANT_NAME(vm, vcconji)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcconji)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzconji)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzconji)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: CopySign, indexing: simple */
void MKL_VARIANT_NAME(vm, vscopysign)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscopysign)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcopysign)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcopysign)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: CopySign, indexing: strided */
void MKL_VARIANT_NAME(vm, vscopysigni)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscopysigni)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcopysigni)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcopysigni)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Cos, indexing: simple */
void MKL_VARIANT_NAME(vm, vscos)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscos)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcos)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcos)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vccos)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmccos)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzcos)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzcos)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Cos, indexing: strided */
void MKL_VARIANT_NAME(vm, vscosi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscosi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcosi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcosi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vccosi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmccosi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzcosi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzcosi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Cosd, indexing: simple */
void MKL_VARIANT_NAME(vm, vscosd)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscosd)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcosd)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcosd)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Cosd, indexing: strided */
void MKL_VARIANT_NAME(vm, vscosdi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscosdi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcosdi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcosdi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Cosh, indexing: simple */
void MKL_VARIANT_NAME(vm, vscosh)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscosh)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcosh)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcosh)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vccosh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmccosh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzcosh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzcosh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Cosh, indexing: strided */
void MKL_VARIANT_NAME(vm, vscoshi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscoshi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcoshi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcoshi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vccoshi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmccoshi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzcoshi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzcoshi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Cospi, indexing: simple */
void MKL_VARIANT_NAME(vm, vscospi)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscospi)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcospi)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcospi)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Cospi, indexing: strided */
void MKL_VARIANT_NAME(vm, vscospii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmscospii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdcospii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdcospii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Div, indexing: simple */
void MKL_VARIANT_NAME(vm, vsdiv)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsdiv)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vddiv)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmddiv)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcdiv)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcdiv)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzdiv)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzdiv)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Div, indexing: strided */
void MKL_VARIANT_NAME(vm, vsdivi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsdivi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vddivi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmddivi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcdivi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcdivi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzdivi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzdivi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Erf, indexing: simple */
void MKL_VARIANT_NAME(vm, vserf)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserf)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderf)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderf)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Erf, indexing: strided */
void MKL_VARIANT_NAME(vm, vserfi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Erfc, indexing: simple */
void MKL_VARIANT_NAME(vm, vserfc)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfc)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfc)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfc)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Erfc, indexing: strided */
void MKL_VARIANT_NAME(vm, vserfci)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfci)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfci)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfci)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: ErfcInv, indexing: simple */
void MKL_VARIANT_NAME(vm, vserfcinv)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfcinv)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfcinv)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfcinv)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: ErfcInv, indexing: strided */
void MKL_VARIANT_NAME(vm, vserfcinvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfcinvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfcinvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfcinvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: ErfInv, indexing: simple */
void MKL_VARIANT_NAME(vm, vserfinv)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfinv)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfinv)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfinv)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: ErfInv, indexing: strided */
void MKL_VARIANT_NAME(vm, vserfinvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmserfinvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vderfinvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmderfinvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Exp, indexing: simple */
void MKL_VARIANT_NAME(vm, vsexp)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexp)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexp)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexp)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcexp)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcexp)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzexp)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzexp)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Exp, indexing: strided */
void MKL_VARIANT_NAME(vm, vsexpi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexpi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexpi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexpi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcexpi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcexpi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzexpi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzexpi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Exp10, indexing: simple */
void MKL_VARIANT_NAME(vm, vsexp10)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexp10)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexp10)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexp10)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Exp10, indexing: strided */
void MKL_VARIANT_NAME(vm, vsexp10i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexp10i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexp10i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexp10i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Exp2, indexing: simple */
void MKL_VARIANT_NAME(vm, vsexp2)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexp2)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexp2)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexp2)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Exp2, indexing: strided */
void MKL_VARIANT_NAME(vm, vsexp2i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexp2i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexp2i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexp2i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: ExpInt1, indexing: simple */
void MKL_VARIANT_NAME(vm, vsexpint1)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexpint1)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexpint1)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexpint1)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: ExpInt1, indexing: strided */
void MKL_VARIANT_NAME(vm, vsexpint1i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexpint1i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexpint1i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexpint1i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Expm1, indexing: simple */
void MKL_VARIANT_NAME(vm, vsexpm1)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexpm1)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexpm1)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexpm1)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Expm1, indexing: strided */
void MKL_VARIANT_NAME(vm, vsexpm1i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsexpm1i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdexpm1i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdexpm1i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Fdim, indexing: simple */
void MKL_VARIANT_NAME(vm, vsfdim)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfdim)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfdim)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfdim)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Fdim, indexing: strided */
void MKL_VARIANT_NAME(vm, vsfdimi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfdimi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfdimi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfdimi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Floor, indexing: simple */
void MKL_VARIANT_NAME(vm, vsfloor)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfloor)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfloor)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfloor)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Floor, indexing: strided */
void MKL_VARIANT_NAME(vm, vsfloori)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfloori)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfloori)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfloori)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Fmax, indexing: simple */
void MKL_VARIANT_NAME(vm, vsfmax)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfmax)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfmax)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfmax)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Fmax, indexing: strided */
void MKL_VARIANT_NAME(vm, vsfmaxi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfmaxi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfmaxi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfmaxi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Fmin, indexing: simple */
void MKL_VARIANT_NAME(vm, vsfmin)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfmin)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfmin)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfmin)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Fmin, indexing: strided */
void MKL_VARIANT_NAME(vm, vsfmini)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfmini)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfmini)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfmini)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Fmod, indexing: simple */
void MKL_VARIANT_NAME(vm, vsfmod)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfmod)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfmod)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfmod)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Fmod, indexing: strided */
void MKL_VARIANT_NAME(vm, vsfmodi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfmodi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfmodi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfmodi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Frac, indexing: simple */
void MKL_VARIANT_NAME(vm, vsfrac)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfrac)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfrac)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfrac)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Frac, indexing: strided */
void MKL_VARIANT_NAME(vm, vsfraci)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsfraci)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdfraci)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdfraci)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Hypot, indexing: simple */
void MKL_VARIANT_NAME(vm, vshypot)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmshypot)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdhypot)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdhypot)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Hypot, indexing: strided */
void MKL_VARIANT_NAME(vm, vshypoti)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmshypoti)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdhypoti)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdhypoti)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Inv, indexing: simple */
void MKL_VARIANT_NAME(vm, vsinv)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsinv)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdinv)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdinv)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Inv, indexing: strided */
void MKL_VARIANT_NAME(vm, vsinvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsinvi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdinvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdinvi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: InvCbrt, indexing: simple */
void MKL_VARIANT_NAME(vm, vsinvcbrt)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsinvcbrt)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdinvcbrt)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdinvcbrt)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: InvCbrt, indexing: strided */
void MKL_VARIANT_NAME(vm, vsinvcbrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsinvcbrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdinvcbrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdinvcbrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: InvSqrt, indexing: simple */
void MKL_VARIANT_NAME(vm, vsinvsqrt)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsinvsqrt)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdinvsqrt)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdinvsqrt)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: InvSqrt, indexing: strided */
void MKL_VARIANT_NAME(vm, vsinvsqrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsinvsqrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdinvsqrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdinvsqrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: LGamma, indexing: simple */
void MKL_VARIANT_NAME(vm, vslgamma)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslgamma)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlgamma)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlgamma)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: LGamma, indexing: strided */
void MKL_VARIANT_NAME(vm, vslgammai)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslgammai)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlgammai)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlgammai)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: LinearFrac, indexing: simple */
void MKL_VARIANT_NAME(vm, vslinearfrac)(const MKL_INT n, const float * a, const float * b, const float c, const float d, const float e, const float f, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslinearfrac)(const MKL_INT n, const float * a, const float * b, const float c, const float d, const float e, const float f, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlinearfrac)(const MKL_INT n, const double * a, const double * b, const double c, const double d, const double e, const double f, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlinearfrac)(const MKL_INT n, const double * a, const double * b, const double c, const double d, const double e, const double f, double * y, MKL_INT64 mode) NOTHROW;

/* function: LinearFrac, indexing: strided */
void MKL_VARIANT_NAME(vm, vslinearfraci)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, const float c, const float d, const float e, const float f, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslinearfraci)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, const float c, const float d, const float e, const float f, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlinearfraci)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, const double c, const double d, const double e, const double f, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlinearfraci)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, const double c, const double d, const double e, const double f, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Ln, indexing: simple */
void MKL_VARIANT_NAME(vm, vsln)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsln)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdln)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdln)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcln)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcln)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzln)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzln)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Ln, indexing: strided */
void MKL_VARIANT_NAME(vm, vslni)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslni)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlni)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlni)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vclni)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmclni)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzlni)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzlni)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Log10, indexing: simple */
void MKL_VARIANT_NAME(vm, vslog10)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslog10)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlog10)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlog10)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vclog10)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmclog10)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzlog10)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzlog10)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Log10, indexing: strided */
void MKL_VARIANT_NAME(vm, vslog10i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslog10i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlog10i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlog10i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vclog10i)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmclog10i)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzlog10i)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzlog10i)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Log1p, indexing: simple */
void MKL_VARIANT_NAME(vm, vslog1p)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslog1p)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlog1p)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlog1p)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Log1p, indexing: strided */
void MKL_VARIANT_NAME(vm, vslog1pi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslog1pi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlog1pi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlog1pi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Log2, indexing: simple */
void MKL_VARIANT_NAME(vm, vslog2)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslog2)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlog2)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlog2)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Log2, indexing: strided */
void MKL_VARIANT_NAME(vm, vslog2i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslog2i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlog2i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlog2i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Logb, indexing: simple */
void MKL_VARIANT_NAME(vm, vslogb)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslogb)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlogb)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlogb)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Logb, indexing: strided */
void MKL_VARIANT_NAME(vm, vslogbi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmslogbi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdlogbi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdlogbi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: MaxMag, indexing: simple */
void MKL_VARIANT_NAME(vm, vsmaxmag)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsmaxmag)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdmaxmag)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdmaxmag)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: MaxMag, indexing: strided */
void MKL_VARIANT_NAME(vm, vsmaxmagi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsmaxmagi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdmaxmagi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdmaxmagi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: MinMag, indexing: simple */
void MKL_VARIANT_NAME(vm, vsminmag)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsminmag)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdminmag)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdminmag)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: MinMag, indexing: strided */
void MKL_VARIANT_NAME(vm, vsminmagi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsminmagi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdminmagi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdminmagi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Modf, indexing: simple */
void MKL_VARIANT_NAME(vm, vsmodf)(const MKL_INT n, const float * a, float * y, float * z) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsmodf)(const MKL_INT n, const float * a, float * y, float * z, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdmodf)(const MKL_INT n, const double * a, double * y, double * z) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdmodf)(const MKL_INT n, const double * a, double * y, double * z, MKL_INT64 mode) NOTHROW;

/* function: Modf, indexing: strided */
void MKL_VARIANT_NAME(vm, vsmodfi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, float * z, const MKL_INT incz) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsmodfi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, float * z, const MKL_INT incz, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdmodfi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, double * z, const MKL_INT incz) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdmodfi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, double * z, const MKL_INT incz, MKL_INT64 mode) NOTHROW;



/* function: Mul, indexing: simple */
void MKL_VARIANT_NAME(vm, vsmul)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsmul)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdmul)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdmul)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcmul)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcmul)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzmul)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzmul)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Mul, indexing: strided */
void MKL_VARIANT_NAME(vm, vsmuli)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsmuli)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdmuli)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdmuli)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcmuli)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcmuli)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzmuli)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzmuli)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: MulByConj, indexing: simple */
void MKL_VARIANT_NAME(vm, vcmulbyconj)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcmulbyconj)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzmulbyconj)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzmulbyconj)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: MulByConj, indexing: strided */
void MKL_VARIANT_NAME(vm, vcmulbyconji)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcmulbyconji)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzmulbyconji)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzmulbyconji)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: NearbyInt, indexing: simple */
void MKL_VARIANT_NAME(vm, vsnearbyint)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsnearbyint)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdnearbyint)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdnearbyint)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: NearbyInt, indexing: strided */
void MKL_VARIANT_NAME(vm, vsnearbyinti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsnearbyinti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdnearbyinti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdnearbyinti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: NextAfter, indexing: simple */
void MKL_VARIANT_NAME(vm, vsnextafter)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsnextafter)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdnextafter)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdnextafter)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: NextAfter, indexing: strided */
void MKL_VARIANT_NAME(vm, vsnextafteri)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsnextafteri)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdnextafteri)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdnextafteri)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Pow, indexing: simple */
void MKL_VARIANT_NAME(vm, vspow)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspow)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpow)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpow)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcpow)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcpow)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzpow)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzpow)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Pow, indexing: strided */
void MKL_VARIANT_NAME(vm, vspowi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspowi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpowi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpowi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcpowi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcpowi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzpowi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzpowi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Pow2o3, indexing: simple */
void MKL_VARIANT_NAME(vm, vspow2o3)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspow2o3)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpow2o3)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpow2o3)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Pow2o3, indexing: strided */
void MKL_VARIANT_NAME(vm, vspow2o3i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspow2o3i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpow2o3i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpow2o3i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Pow3o2, indexing: simple */
void MKL_VARIANT_NAME(vm, vspow3o2)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspow3o2)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpow3o2)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpow3o2)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Pow3o2, indexing: strided */
void MKL_VARIANT_NAME(vm, vspow3o2i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspow3o2i)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpow3o2i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpow3o2i)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Powr, indexing: simple */
void MKL_VARIANT_NAME(vm, vspowr)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspowr)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpowr)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpowr)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Powr, indexing: strided */
void MKL_VARIANT_NAME(vm, vspowri)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspowri)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpowri)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpowri)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Powx, indexing: simple */
void MKL_VARIANT_NAME(vm, vspowx)(const MKL_INT n, const float * a, const float b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspowx)(const MKL_INT n, const float * a, const float b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpowx)(const MKL_INT n, const double * a, const double b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpowx)(const MKL_INT n, const double * a, const double b, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcpowx)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcpowx)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzpowx)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzpowx)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Powx, indexing: strided */
void MKL_VARIANT_NAME(vm, vspowxi)(const MKL_INT n, const float * a, const MKL_INT inca, const float b, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmspowxi)(const MKL_INT n, const float * a, const MKL_INT inca, const float b, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdpowxi)(const MKL_INT n, const double * a, const MKL_INT inca, const double b, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdpowxi)(const MKL_INT n, const double * a, const MKL_INT inca, const double b, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcpowxi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 b, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcpowxi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 b, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzpowxi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 b, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzpowxi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 b, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Remainder, indexing: simple */
void MKL_VARIANT_NAME(vm, vsremainder)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsremainder)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdremainder)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdremainder)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;

/* function: Remainder, indexing: strided */
void MKL_VARIANT_NAME(vm, vsremainderi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsremainderi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdremainderi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdremainderi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Rint, indexing: simple */
void MKL_VARIANT_NAME(vm, vsrint)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsrint)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdrint)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdrint)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Rint, indexing: strided */
void MKL_VARIANT_NAME(vm, vsrinti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsrinti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdrinti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdrinti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Round, indexing: simple */
void MKL_VARIANT_NAME(vm, vsround)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsround)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdround)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdround)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Round, indexing: strided */
void MKL_VARIANT_NAME(vm, vsroundi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmsroundi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdroundi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdroundi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Sin, indexing: simple */
void MKL_VARIANT_NAME(vm, vssin)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssin)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsin)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsin)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsin)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsin)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsin)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsin)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Sin, indexing: strided */
void MKL_VARIANT_NAME(vm, vssini)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssini)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsini)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsini)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsini)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsini)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsini)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsini)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: SinCos, indexing: simple */
void MKL_VARIANT_NAME(vm, vssincos)(const MKL_INT n, const float * a, float * y, float * z) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssincos)(const MKL_INT n, const float * a, float * y, float * z, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsincos)(const MKL_INT n, const double * a, double * y, double * z) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsincos)(const MKL_INT n, const double * a, double * y, double * z, MKL_INT64 mode) NOTHROW;

/* function: SinCos, indexing: strided */
void MKL_VARIANT_NAME(vm, vssincosi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, float * z, const MKL_INT incz) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssincosi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, float * z, const MKL_INT incz, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsincosi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, double * z, const MKL_INT incz) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsincosi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, double * z, const MKL_INT incz, MKL_INT64 mode) NOTHROW;



/* function: Sind, indexing: simple */
void MKL_VARIANT_NAME(vm, vssind)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssind)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsind)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsind)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Sind, indexing: strided */
void MKL_VARIANT_NAME(vm, vssindi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssindi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsindi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsindi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Sinh, indexing: simple */
void MKL_VARIANT_NAME(vm, vssinh)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssinh)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsinh)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsinh)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsinh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsinh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsinh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsinh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Sinh, indexing: strided */
void MKL_VARIANT_NAME(vm, vssinhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssinhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsinhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsinhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsinhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsinhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsinhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsinhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Sinpi, indexing: simple */
void MKL_VARIANT_NAME(vm, vssinpi)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssinpi)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsinpi)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsinpi)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Sinpi, indexing: strided */
void MKL_VARIANT_NAME(vm, vssinpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssinpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsinpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsinpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Sqr, indexing: simple */
void MKL_VARIANT_NAME(vm, vssqr)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssqr)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsqr)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsqr)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Sqr, indexing: strided */
void MKL_VARIANT_NAME(vm, vssqri)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssqri)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsqri)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsqri)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Sqrt, indexing: simple */
void MKL_VARIANT_NAME(vm, vssqrt)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssqrt)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsqrt)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsqrt)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsqrt)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsqrt)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsqrt)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsqrt)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Sqrt, indexing: strided */
void MKL_VARIANT_NAME(vm, vssqrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssqrti)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsqrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsqrti)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsqrti)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsqrti)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsqrti)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsqrti)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Sub, indexing: simple */
void MKL_VARIANT_NAME(vm, vssub)(const MKL_INT n, const float * a, const float * b, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssub)(const MKL_INT n, const float * a, const float * b, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsub)(const MKL_INT n, const double * a, const double * b, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsub)(const MKL_INT n, const double * a, const double * b, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsub)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsub)(const MKL_INT n, const MKL_Complex8 * a, const MKL_Complex8 * b, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsub)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsub)(const MKL_INT n, const MKL_Complex16 * a, const MKL_Complex16 * b, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Sub, indexing: strided */
void MKL_VARIANT_NAME(vm, vssubi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmssubi)(const MKL_INT n, const float * a, const MKL_INT inca, const float * b, const MKL_INT incb, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdsubi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdsubi)(const MKL_INT n, const double * a, const MKL_INT inca, const double * b, const MKL_INT incb, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vcsubi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmcsubi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, const MKL_Complex8 * b, const MKL_INT incb, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vzsubi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmzsubi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, const MKL_Complex16 * b, const MKL_INT incb, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Tan, indexing: simple */
void MKL_VARIANT_NAME(vm, vstan)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstan)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtan)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtan)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vctan)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmctan)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vztan)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmztan)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Tan, indexing: strided */
void MKL_VARIANT_NAME(vm, vstani)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstani)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtani)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtani)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vctani)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmctani)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vztani)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmztani)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Tand, indexing: simple */
void MKL_VARIANT_NAME(vm, vstand)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstand)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtand)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtand)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Tand, indexing: strided */
void MKL_VARIANT_NAME(vm, vstandi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstandi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtandi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtandi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Tanh, indexing: simple */
void MKL_VARIANT_NAME(vm, vstanh)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstanh)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtanh)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtanh)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vctanh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmctanh)(const MKL_INT n, const MKL_Complex8 * a, MKL_Complex8 * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vztanh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmztanh)(const MKL_INT n, const MKL_Complex16 * a, MKL_Complex16 * y, MKL_INT64 mode) NOTHROW;

/* function: Tanh, indexing: strided */
void MKL_VARIANT_NAME(vm, vstanhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstanhi)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtanhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtanhi)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vctanhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmctanhi)(const MKL_INT n, const MKL_Complex8 * a, const MKL_INT inca, MKL_Complex8 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vztanhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmztanhi)(const MKL_INT n, const MKL_Complex16 * a, const MKL_INT inca, MKL_Complex16 * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Tanpi, indexing: simple */
void MKL_VARIANT_NAME(vm, vstanpi)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstanpi)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtanpi)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtanpi)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Tanpi, indexing: strided */
void MKL_VARIANT_NAME(vm, vstanpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstanpii)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtanpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtanpii)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: TGamma, indexing: simple */
void MKL_VARIANT_NAME(vm, vstgamma)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstgamma)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtgamma)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtgamma)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: TGamma, indexing: strided */
void MKL_VARIANT_NAME(vm, vstgammai)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstgammai)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtgammai)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtgammai)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;



/* function: Trunc, indexing: simple */
void MKL_VARIANT_NAME(vm, vstrunc)(const MKL_INT n, const float * a, float * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstrunc)(const MKL_INT n, const float * a, float * y, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtrunc)(const MKL_INT n, const double * a, double * y) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtrunc)(const MKL_INT n, const double * a, double * y, MKL_INT64 mode) NOTHROW;

/* function: Trunc, indexing: strided */
void MKL_VARIANT_NAME(vm, vstrunci)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmstrunci)(const MKL_INT n, const float * a, const MKL_INT inca, float * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;
void MKL_VARIANT_NAME(vm, vdtrunci)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy) NOTHROW;
void MKL_VARIANT_NAME(vm, vmdtrunci)(const MKL_INT n, const double * a, const MKL_INT inca, double * y, const MKL_INT incy, MKL_INT64 mode) NOTHROW;


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ifndef _MKL_VML_OMP_VARIANT_H_ */


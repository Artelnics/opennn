/*
 *
 *
 * Copyright (C) 2005 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials, and your use of them
 * is governed by the express license under which they were provided to you ("License"). Unless
 * the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose
 * or transmit this software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
*/

// matrix multiply routines
#include "multiply.h"

#ifdef USE_MKL
#include "mkl.h"
#endif //USE_MKL

#ifdef USE_THR
void multiply0(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
    int i,j,k;

// Basic serial implementation
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
            for(k=0; k<msize; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}

void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k;

// This naive implementation of matrix multiply contains an inefficient memory access pattern.
// Each iteration of the inner loop strides across the full width of a row from matrix 'b'
// because the iterator 'k' is used in the first dimension of b[k][j].
// This leads to bad cache reuse and significant memory stalls.
// Use Microarchitecture and Memory access analysis to estimate an impact of this performance bottleneck.
// See the 'multiply2' function implementation to overcome the issue.
    for(i=tidx; i<msize; i=i+numt) {
        for(j=0; j<msize; j++) {
            for(k=0; k<msize; k++) {
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}

void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k;

// This implementation interchanges the 'j' and 'k' loop iterations.
// The loop interchange technique removes the bottleneck caused by the inefficient 
// memory access pattern in the 'multiply1' function.
    for(i=tidx; i<msize; i=i+numt) {
        for(k=0; k<msize; k++) {
            for(j=0; j<msize; j++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}

void multiply3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k;

// Add compile option for vectorization report Windows: /Qvec-report3 Linux -vec-report3
    for(i=tidx; i<msize; i=i+numt) {
        for(k=0; k<msize; k++) {
#pragma ivdep
            for(j=0; j<msize; j++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}

void multiply4(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k,i0,j0,k0,ibeg,ibound,istep,mblock;

// Cache blocking
// Add current platform optimization for Windows: /QxHost Linux: -xHost
// Define the ALIGNED in the preprocessor definitions and compile option Windows: /Oa Linux: -fno-alias
    istep = msize / numt;
    ibeg = tidx * istep;
    ibound = ibeg + istep;
    mblock = MATRIX_BLOCK_SIZE;

    for (i0 = ibeg; i0 < ibound; i0 +=mblock) {
        for (k0 = 0; k0 < msize; k0 += mblock) {
            for (j0 =0; j0 < msize; j0 += mblock) {
                for (i = i0; i < i0 + mblock; i++) {
                    for (k = k0; k < k0 + mblock; k++) {
#pragma ivdep
#ifdef ALIGNED 
    #pragma vector aligned
#endif //ALIGNED
                        for (j = j0; j < j0 + mblock; j++) {
                            c[i][j]  = c[i][j] + a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }
}

void multiply5(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k,istep,ibeg,ibound;
//transpose b
    for(i=0;i<msize;i++) {
        for(k=0;k<msize;k++) {
        t[i][k] = b[k][i];
        }
    }

    istep = msize / numt;
    ibeg = tidx * istep;
    ibound = ibeg + istep;
/*  for(i=0;i<msize;i+=4) { // use instead for single threaded impl.*/
    for(i=ibeg;i<ibound;i+=4) {
        for(j=0;j<msize;j+=4) {
#pragma loop count (NUM)
#pragma ivdep
            for(k=0;k<msize;k++) {
                c[i][j] = c[i][j] + a[i][k] * t[j][k];
                c[i+1][j] = c[i+1][j] + a[i+1][k] * t[j][k];
                c[i+2][j] = c[i+2][j] + a[i+2][k] * t[j][k];
                c[i+3][j] = c[i+3][j] + a[i+3][k] * t[j][k];

                c[i][j+1] = c[i][j+1] + a[i][k] * t[j+1][k];
                c[i+1][j+1] = c[i+1][j+1] + a[i+1][k] * t[j+1][k];
                c[i+2][j+1] = c[i+2][j+1] + a[i+2][k] * t[j+1][k];
                c[i+3][j+1] = c[i+3][j+1] + a[i+3][k] * t[j+1][k];

                c[i][j+2] = c[i][j+2] + a[i][k] * t[j+2][k];
                c[i+1][j+2] = c[i+1][j+2] + a[i+1][k] * t[j+2][k];
                c[i+2][j+2] = c[i+2][j+2] + a[i+2][k] * t[j+2][k];
                c[i+3][j+2] = c[i+3][j+2] + a[i+3][k] * t[j+2][k];

                c[i][j+3] = c[i][j+3] + a[i][k] * t[j+3][k];
                c[i+1][j+3] = c[i+1][j+3] + a[i+1][k] * t[j+3][k];
                c[i+2][j+3] = c[i+2][j+3] + a[i+2][k] * t[j+3][k];
                c[i+3][j+3] = c[i+3][j+3] + a[i+3][k] * t[j+3][k];
              }
        }
    }

/*
    // it's the same to the loop above?
    for(i=ibeg;i<ibound;i++) {
        for(j=0;j<msize;j++) {

#pragma ivdep
#pragma vector aligned

            for(k=0;k<msize;k++) {
                c[i][j] = c[i][j] + a[i][k] * t[j][k];}}}
*/
}
#endif // USE_THR

#ifdef USE_OMP

void multiply0(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
    int i,j,k;

// Basic serial implementation
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
            for(k=0; k<msize; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}

void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
    int i,j,k;

    // Basic parallel implementation
    #pragma omp parallel for
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
            for(k=0; k<msize; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}

void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k;

    // Parallel with merged outer loops
    #pragma omp parallel for collapse (2)
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
            for(k=0; k<msize; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    } 
}
void multiply3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k;

    #pragma omp parallel for collapse (2)
    for(i=0; i<msize; i++) {
        for(k=0; k<msize; k++) {
#pragma ivdep
            for(j=0; j<msize; j++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}
void multiply4(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
    int i,j,k;

    #pragma omp parallel for collapse (2)
    for(i=0; i<msize; i++) {
        for(k=0; k<msize; k++) {
#pragma unroll(8)
#pragma ivdep
            for(j=0; j<msize; j++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}

#endif // USE_OMP

#ifdef USE_MKL
// DGEMM way of matrix multiply using Intel MKL
// Link with Intel MKL library: With MSFT VS and Intel Composer integration: Select build components in the Project context menu.
// For command line - check out the Intel® Math Kernel Library Link Line Advisor 
void multiply5(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{

    double alpha = 1.0, beta = 0.;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NUM,NUM,NUM,alpha,(const double *)b,NUM,(const double *)a,NUM,beta,(double *)c,NUM);
}
#endif //USE_MKL
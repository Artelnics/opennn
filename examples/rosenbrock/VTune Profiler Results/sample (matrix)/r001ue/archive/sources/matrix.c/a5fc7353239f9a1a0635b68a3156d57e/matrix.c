/*
    Copyright 2005-2013 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#ifdef WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <stdlib.h>
	#ifdef ICC // The ICC is defined (by default) for enabling Intel Compiler specific headers and calls
	#include <immintrin.h>
	#endif
#endif
#include <stdio.h>
#include <time.h>
#include <malloc.h>

#include "multiply.h"

typedef unsigned long long  UINT64;

double TRIP_COUNT = (double)NUM * (double)NUM * (double)NUM;
int FLOP_PER_ITERATION = 2; // basic matrix multiplication

extern int getCPUCount();
extern double getCPUFreq();

// routine to initialize an array with data
void init_arr(TYPE row, TYPE col, TYPE off, TYPE a[][NUM])
{
  int i,j;

  for (i=0; i< NUM;i++) {
    for (j=0; j<NUM;j++) {
      a[i][j] = row*i+col*j+off;
    }
  }
}

// routine to print out contents of small arrays
void print_arr(char *name, TYPE array[][NUM])
{
  int i,j;
  
  printf("\n%s\n", name);
  for (i=0;i<NUM;i++){
    for (j=0;j<NUM;j++) {
      printf("%g\t",array[i][j]);
    }
    printf("\n"); fflush(stdout);
  }
}

int main()
{
#ifdef WIN32
	clock_t start=0.0, stop=0.0;
#else // Pthreads
	double start=0.0, stop=0.0;
	struct timeval  before, after;
#endif
	double secs;
   	double flops;
	double mflops;

	char *buf1, *buf2, *buf3, *buf4;
	char *addr1, *addr2, *addr3, *addr4;
	array *a, *b, *c, *t;
	int Offset_Addr1=128, Offset_Addr2=192, Offset_Addr3=0,Offset_Addr4=64;

// malloc arrays space
// Define ALIGNED in the preprocessor
// Also add '/Oa' for Windows and '-fno-alias' for Linux
#ifdef ALIGNED

#ifdef WIN32 
#ifdef ICC
	buf1 = _mm_malloc((sizeof (double))*NUM*NUM, 64);  
	buf2 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
	buf3 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
	buf4 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
#else
	buf1 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
	buf2 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
	buf3 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
	buf4 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
#endif //ICC
#else // WIN32
	buf1 = _mm_malloc((sizeof (double))*NUM*NUM, 64);  
	buf2 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
	buf3 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
	buf4 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
#endif //WIN32
	addr1 = buf1;
	addr2 = buf2;
	addr3 = buf3;
	addr4 = buf4;

#else //!ALIGNED
	buf1 = (char *) malloc(NUM*NUM*(sizeof (double))+1024);
	printf("Addr of buf1 = %p\n",buf1); fflush(stdout);
	addr1 = buf1 + 256 - ((UINT64)buf1%256) + (UINT64)Offset_Addr1;
	printf("Offs of buf1 = %p\n",addr1); fflush(stdout);
	
	buf2 = (char *) malloc(NUM*NUM*(sizeof (double))+1024);
	printf("Addr of buf2 = %p\n",buf2); fflush(stdout);
	addr2 = buf2 + 256 - ((UINT64)buf2%256) + (UINT64)Offset_Addr2;
	printf("Offs of buf2 = %p\n",addr2); fflush(stdout);
	
	buf3 = (char *) malloc(NUM*NUM*(sizeof (double))+1024);
	printf("Addr of buf3 = %p\n",buf3); fflush(stdout);
	addr3 = buf3 + 256 - ((UINT64)buf3%256) + (UINT64)Offset_Addr3;
	printf("Offs of buf3 = %p\n",addr3); fflush(stdout);

	buf4 = (char *) malloc(NUM*NUM*(sizeof (double))+1024);
	printf("Addr of buf4 = %p\n",buf4); fflush(stdout);
	addr4 = buf4 + 256 - ((UINT64)buf4%256) + (UINT64)Offset_Addr4;
	printf("Offs of buf4 = %p\n",addr4); fflush(stdout);

#endif //ALIGNED

	a = (array *) addr1;
	b = (array *) addr2;
	c = (array *) addr3;
	t = (array *) addr4;

// initialize the arrays with data
	init_arr(3,-2,1,a);
	init_arr(-2,1,3,b);
	
	// Printing model parameters
	GetModelParams(0, 0, 1);
	
// start timing the matrix multiply code
#ifdef WIN32		
	start = clock();
#else
	#ifdef ICC
	start = (double)_rdtsc();
	#else
	gettimeofday(&before, NULL);
	#endif
#endif

	ParallelMultiply(NUM, a, b, c, t);

#ifdef WIN32
	stop = clock();
	secs = ((double)(stop - start)) / CLOCKS_PER_SEC;
#else
	#ifdef ICC
	stop = (double)_rdtsc();
	secs = ((double)(stop - start)) / (double) getCPUFreq();
	#else
	gettimeofday(&after, NULL);
	secs = (after.tv_sec - before.tv_sec) + (after.tv_usec - before.tv_usec)/1000000.0;
	#endif
#endif

	flops = TRIP_COUNT * FLOP_PER_ITERATION;
	mflops = flops / 1000000.0f / secs;
	printf("Execution time = %2.3lf seconds\n",secs);	fflush(stdout);
	//printf("MFLOPS: %2.3f mflops\n", mflops);

// print simple test case of data to be sure multiplication is correct
  if (NUM < 5) {
    print_arr("a", a); fflush(stdout);
    print_arr("b", b); fflush(stdout);
    print_arr("c", c); fflush(stdout);
  }

	//free memory
#ifdef ALIGNED
#ifdef WIN32 
#ifdef ICC
	_mm_free(buf1);
	_mm_free(buf2);
	_mm_free(buf3);
	_mm_free(buf4);
#else
	_aligned_free(buf1);
	_aligned_free(buf2);
	_aligned_free(buf3);
	_aligned_free(buf4);
#endif //ICC
#else // ICC or GCC Linux
	_mm_free(buf1);
	_mm_free(buf2);
	_mm_free(buf3);
	_mm_free(buf4);
#endif //WIN32
#else //ALIGNED
	free (buf1);
	free (buf2);
	free (buf3);
	free (buf4);
#endif //ALIGNED
}
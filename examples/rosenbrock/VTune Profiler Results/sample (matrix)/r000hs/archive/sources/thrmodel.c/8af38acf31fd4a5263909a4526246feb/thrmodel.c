#ifdef WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <stdlib.h>
#endif
#ifdef USE_OMP
#include <omp.h>
#endif // USE_OMP
#ifdef USE_MKL
#include <omp.h>
#endif // USE_MKL

#include <assert.h>
#include <stdio.h>
#include "multiply.h"

#define xstr(s) x_str(s)
#define x_str(s) #s

#ifdef USE_THR
//=========================================================================================
// Native threading model
//=========================================================================================

typedef struct tparam
{
	array *a, *b, *c, *t;
	int msize;
	int tidx;
	int numt;
} _tparam;

#ifdef WIN32
DWORD WINAPI ThreadFunction(LPVOID ptr)
#else
void *ThreadFunction(void *ptr)
#endif
{
	_tparam* par = (_tparam*)ptr;
	assert(par->numt > 0);
	assert(par->a != NULL);
	assert(par->b != NULL);
	assert(par->c != NULL);
	assert(par->t != NULL);
	assert( (par->msize % par->numt) == 0);

	MULTIPLY(	par->msize,
				par->tidx, 
				par->numt, 
				par->a, 
				par->b, 
				par->c,
				par->t
				);
#ifdef WIN32
	return 0;
#else
	//printf("exit thread function\n");
	pthread_exit( (void *)0 );
#endif
}


void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int NTHREADS = MAXTHREADS;
	int MSIZE = NUM;

#ifdef WIN32
	HANDLE ht[MAXTHREADS];
	DWORD tid[MAXTHREADS];
#else 
	pthread_t ht[MAXTHREADS];
	int tret[MAXTHREADS]; 
	int rc; 
	void* status;
#endif
	_tparam par[MAXTHREADS];
	int tidx;

	GetModelParams(&NTHREADS, &MSIZE, 0);

	for (tidx=0; tidx<NTHREADS; tidx++)
	{
		par[tidx].msize = MSIZE;
		par[tidx].numt = NTHREADS;
		par[tidx].tidx = tidx;
		par[tidx].a = a;
		par[tidx].b = b;
		par[tidx].c = c;
		par[tidx].t = t;
#ifdef WIN32		
		ht[tidx] = (HANDLE)CreateThread(NULL, 0, ThreadFunction, &par[tidx], 0, &tid[tidx]);
#else
		tret[tidx] = pthread_create( &ht[tidx], NULL, (void*)ThreadFunction, (void*) &par[tidx]);
#endif
	}
#ifdef WIN32
	WaitForMultipleObjects(NTHREADS, ht, TRUE, INFINITE);
#else // Pthreads
	for (tidx=0; tidx<NTHREADS; tidx++)
	{
	//	printf("Enter join\n"); fflush(stdout);
		rc = pthread_join(ht[tidx], (void **)&status);
	//	printf("Exit join\n"); fflush(stdout);
	}
#endif

}

extern int getCPUCount();

void GetModelParams(int* p_nthreads, int* p_msize, int print)
{
	int nthr = MAXTHREADS;
	int msize = NUM;
	int ncpu = getCPUCount();
	if (ncpu < MAXTHREADS) {
		nthr = ncpu;
	}
	// Making sure the matrix size and the nthreads are aligned
	// If you want more robust threading implementation, take care
	// of the matrix tails
	while ((msize % nthr) != 0 )
		nthr--;
	// If kernel multiply0, set single threaded execution
	if (MULTIPLY == multiply0)
		nthr = 1;

	if(p_nthreads != 0)
		*p_nthreads = nthr;
	if(p_msize != 0)
		*p_msize = msize;

	if(print)
	{
		printf("Threads #: %d %s\n",nthr,
#ifdef WIN32
	 "Win threads"
#else
	  "Pthreads"
#endif
			);	fflush(stdout);
		printf("Matrix size: %d\n",msize); fflush(stdout);
		printf("Using multiply kernel: %s\n", xstr(MULTIPLY)); fflush(stdout);
	}
}
#endif // USE_THR

#ifdef USE_OMP
//=========================================================================================
// OpenMP threading model
//=========================================================================================


void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int NTHREADS = MAXTHREADS;
	int MSIZE = NUM;

	GetModelParams(&NTHREADS, &MSIZE, 0);

	MULTIPLY(MSIZE, NTHREADS, 0, a, b, c, t);
}

void GetModelParams(int* p_nthreads, int* p_msize, int print)
{
	int msize = NUM;
	int nthr = MAXTHREADS;
	int ncpu = omp_get_max_threads();
	if (ncpu < nthr)
		nthr = ncpu;
	omp_set_num_threads(nthr);

	if(p_nthreads != 0)
		*p_nthreads = nthr;
	if(p_msize != 0)
		*p_msize = msize;

	if(print)
	{
		printf("Threads #: %d %s\n",nthr,"OpenMP threads");	fflush(stdout);
		printf("Matrix size: %d\n",msize); fflush(stdout);
		printf("Using multiply kernel: %s\n", xstr(MULTIPLY)); fflush(stdout);
	}
}
#endif // USE_OMP

#ifdef USE_MKL
//=========================================================================================
// MKL library
//=========================================================================================


void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int NTHREADS = MAXTHREADS;
	int MSIZE = NUM;

	GetModelParams(&NTHREADS, &MSIZE, 0);
	if(strncmp(xstr(MULTIPLY), "multiply5", 16) != 0)
	{
		printf("===== Error: Change matrix kernel to 'multiply5' for compilation with MKL =====\n"); fflush(stdout);
		return;
	}
	MULTIPLY(MSIZE, NTHREADS, 0, a, b, c, t);
}

void GetModelParams(int* p_nthreads, int* p_msize, int print)
{
	int msize = NUM;
	int nthr = MAXTHREADS;
	int ncpu = omp_get_max_threads();
	if (ncpu < nthr)
		nthr = ncpu;
	omp_set_num_threads(nthr);

	if(p_nthreads != 0)
		*p_nthreads = nthr;
	if(p_msize != 0)
		*p_msize = msize;

	if(print)
	{
		printf("Threads #: %d %s\n",nthr,"requested OpenMP threads");	fflush(stdout);
		printf("Matrix size: %d\n",msize); fflush(stdout);
		printf("Using multiply kernel: %s\n", xstr(MULTIPLY)); fflush(stdout);
	}
}
#endif // USE_MKL
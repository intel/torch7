#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif
#include "AVX2.h"

#ifdef _OPENMP
#include <omp.h>
#define TH_OMP_OVERHEAD_THRESHOLD_VEC 1000
#endif


void THDoubleVector_cadd_AVX2(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i) 
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256d YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_pd(y+i);
    YMM1 = _mm256_loadu_pd(y+i+4);
    YMM2 = _mm256_loadu_pd(x+i);
    YMM3 = _mm256_loadu_pd(x+i+4);
    YMM2 = _mm256_fmadd_pd(YMM0, YMM15, YMM2);
    YMM3 = _mm256_fmadd_pd(YMM1, YMM15, YMM3);
    _mm256_storeu_pd(z+i, YMM2);
    _mm256_storeu_pd(z+i+4, YMM3);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

void THFloatVector_cadd_AVX2(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i) 
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m256 YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_ps(y+i);
    YMM1 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_loadu_ps(x+i);
    YMM3 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_fmadd_ps(YMM0, YMM15, YMM2);
    YMM3 = _mm256_fmadd_ps(YMM1, YMM15, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

#endif // defined(__AVX2__)

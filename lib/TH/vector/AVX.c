#if defined(__AVX__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#include "AVX.h"

#ifdef _OPENMP
#include <omp.h>
#define TH_OMP_OVERHEAD_THRESHOLD_VEC 1000
#define TH_OMP_OVERHEAD_THRESHOLD_VEC 4352
#endif

void THDoubleVector_copy_AVX(double *y, const double *x, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) ) private (i)
#endif
  for (i=0; i<=((n)-4); i+=4) {
    _mm256_storeu_pd(y+i, _mm256_loadu_pd(x+i));
  }
  off = (n) - ((n)%4);
  for (i=0; i<((n)%4); i++) {
    y[off+i] = x[off+i];
  }
}

void THDoubleVector_fill_AVX(double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM0 = _mm256_set_pd(c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    _mm256_storeu_pd((x)+i  , YMM0);
    _mm256_storeu_pd((x)+i+4, YMM0);
    _mm256_storeu_pd((x)+i+8, YMM0);
    _mm256_storeu_pd((x)+i+12, YMM0);
  }
  off = (n) - ((n)%16);
  for (i=0; i<((n)%16); i++) {
    x[off+i] = c;
  }
}

void THDoubleVector_cdiv_AVX(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256d YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM2 = _mm256_loadu_pd(y+i);
    YMM3 = _mm256_loadu_pd(y+i+4);
    YMM2 = _mm256_div_pd(YMM0, YMM2);
    YMM3 = _mm256_div_pd(YMM1, YMM3);
    _mm256_storeu_pd(z+i, YMM2);
    _mm256_storeu_pd(z+i+4, YMM3);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

void THDoubleVector_divs_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256d YMM0, YMM1;
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM0 = _mm256_div_pd(YMM0, YMM15);
    YMM1 = _mm256_div_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM0);
    _mm256_storeu_pd(y+i+4, YMM1);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

void THDoubleVector_cmul_AVX(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256d YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM2 = _mm256_loadu_pd(y+i);
    YMM3 = _mm256_loadu_pd(y+i+4);
    YMM2 = _mm256_mul_pd(YMM0, YMM2);
    YMM3 = _mm256_mul_pd(YMM1, YMM3);
    _mm256_storeu_pd(z+i, YMM2);
    _mm256_storeu_pd(z+i+4, YMM3);
  }
  off = (n) - ((n)%8);
  for (i=off; i<n; i++) {
    z[i] = x[i] * y[i];
  }
}

void THDoubleVector_muls_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256d YMM0, YMM1;
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM0 = _mm256_mul_pd(YMM0, YMM15);
    YMM1 = _mm256_mul_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM0);
    _mm256_storeu_pd(y+i+4, YMM1);
  }
  off = (n) - ((n)%8);
  for (i=off; i<n; i++) {
    y[i] = x[i] * c;
  }
}

void THDoubleVector_cadd_AVX(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);

#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-4); i+=4) {
    __m256d YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_pd(y+i);
    YMM1 = _mm256_loadu_pd(x+i);
    YMM2 = _mm256_mul_pd(YMM0, YMM15);
    YMM3 = _mm256_add_pd(YMM1, YMM2);
    _mm256_storeu_pd(z+i, YMM3);
  }
  off = (n) - ((n)%4);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

void THDoubleVector_adds_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);

#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256d YMM0, YMM1;
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM0 = _mm256_add_pd(YMM0, YMM15);
    YMM1 = _mm256_add_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM0);
    _mm256_storeu_pd(y+i+4, YMM1);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

void THFloatVector_copy_AVX(float *y, const float *x, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    _mm256_storeu_ps(y+i, _mm256_loadu_ps(x+i));
    _mm256_storeu_ps(y+i+8, _mm256_loadu_ps(x+i+8));
  }
  off = (n) - ((n)%16);
  for (i=0; i<((n)%16); i++) {
    y[off+i] = x[off+i];
  }
}

void THFloatVector_fill_AVX(float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM0 = _mm256_set_ps(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-32); i+=32) {
    _mm256_storeu_ps((x)+i  , YMM0);
    _mm256_storeu_ps((x)+i+8, YMM0);
    _mm256_storeu_ps((x)+i+16, YMM0);
    _mm256_storeu_ps((x)+i+24, YMM0);
  }
  off = (n) - ((n)%32);
  for (i=0; i<((n)%32); i++) {
    x[off+i] = c;
  }
}

void THFloatVector_cdiv_AVX(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m256 YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_loadu_ps(y+i);
    YMM3 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_div_ps(YMM0, YMM2);
    YMM3 = _mm256_div_ps(YMM1, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

void THFloatVector_divs_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m256 YMM0, YMM1;
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM0 = _mm256_div_ps(YMM0, YMM15);
    YMM1 = _mm256_div_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM0);
    _mm256_storeu_ps(y+i+8, YMM1);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

void THFloatVector_cmul_AVX(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m256 YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_loadu_ps(y+i);
    YMM3 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_mul_ps(YMM0, YMM2);
    YMM3 = _mm256_mul_ps(YMM1, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  off = (n) - ((n)%16);
  for (i=off; i<n; i++) {
    z[i] = x[i] * y[i];
  }
}

void THFloatVector_muls_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m256 YMM0, YMM1;
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM0 = _mm256_mul_ps(YMM0, YMM15);
    YMM1 = _mm256_mul_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM0);
    _mm256_storeu_ps(y+i+8, YMM1);
  }
  off = (n) - ((n)%16);
  for (i=off; i<n; i++) {
    y[i] = x[i] * c;
  }
}

void THFloatVector_cadd_AVX(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m256 YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm256_loadu_ps(y+i);
    YMM1 = _mm256_loadu_ps(x+i);
    YMM2 = _mm256_mul_ps(YMM0, YMM15);
    YMM3 = _mm256_add_ps(YMM1, YMM2);
    _mm256_storeu_ps(z+i, YMM3);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

void THFloatVector_adds_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m256 YMM0, YMM1;
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM0 = _mm256_add_ps(YMM0, YMM15);
    YMM1 = _mm256_add_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM0);
    _mm256_storeu_ps(y+i+8, YMM1);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

#endif // defined(__AVX__)

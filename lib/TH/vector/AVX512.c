#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <zmmintrin.h>
#endif

#include "AVX512.h"

#ifdef _OPENMP
#include <omp.h>
#define TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512 4352
#endif

void THDoubleVector_copy_AVX512(double *y, const double *x, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) ) private (i)
#endif
  for (i=0; i<=((n)-8); i+=8) {
    _mm512_storeu_pd(y+i, _mm512_loadu_pd(x+i));
  }
  off = (n) - ((n)%8);
  for (i=off; i< n; i++) {
    y[i] = x[i];
  }
}

void THDoubleVector_fill_AVX512(double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512d YMM0 = _mm512_set_pd(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    _mm512_storeu_pd((x)+i, YMM0);
  }
  off = (n) - ((n)%8);
  for (i=off; i<n; i++) {
    x[i] = c;
  }
}

void THDoubleVector_cdiv_AVX512(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0, YMM1;
    YMM0 = _mm512_loadu_pd(x+i);
    YMM1 = _mm512_loadu_pd(y+i);
    YMM1 = _mm512_div_pd(YMM0, YMM1);
    _mm512_storeu_pd(z+i, YMM1);
   
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

void THDoubleVector_divs_AVX512(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512d YMM15 = _mm512_set_pd(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0;
    YMM0 = _mm512_loadu_pd(x+i);
    YMM0 = _mm512_div_pd(YMM0, YMM15);
    _mm512_storeu_pd(y+i, YMM0);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

void THDoubleVector_cmul_AVX512(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0, YMM1;
    YMM0 = _mm512_loadu_pd(x+i);
    YMM1 = _mm512_loadu_pd(y+i);
    YMM1 = _mm512_mul_pd(YMM0, YMM1);
    _mm512_storeu_pd(z+i, YMM1);
  }
  off = (n) - ((n)%8);
  for (i=off; i<n; i++) {
    z[i] = x[i] * y[i];
  }
}

void THDoubleVector_muls_AVX512(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512d YMM15 = _mm512_set_pd(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0;
    YMM0 = _mm512_loadu_pd(x+i);
    YMM0 = _mm512_mul_pd(YMM0, YMM15);
    _mm512_storeu_pd(y+i, YMM0);
  }
  off = (n) - ((n)%8);
  for (i=off; i<n; i++) {
    y[i] = x[i] * c;
  }
}
/*
void THDoubleVector_cadd_AVX512(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512d YMM15 = _mm512_set_pd(c, c, c, c, c, c, c, c);

#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm512_loadu_pd(y+i);
    YMM1 = _mm512_loadu_pd(x+i);
    YMM2 = _mm512_mul_pd(YMM0, YMM15);
    YMM3 = _mm512_add_pd(YMM1, YMM2);
    _mm512_storeu_pd(z+i, YMM3);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}
*/
void THDoubleVector_adds_AVX512(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512d YMM15 = _mm512_set_pd(c, c, c, c, c, c, c, c);

#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0;
    YMM0 = _mm512_loadu_pd(x+i);
    YMM0 = _mm512_add_pd(YMM0, YMM15);
    _mm512_storeu_pd(y+i, YMM0);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

void THFloatVector_copy_AVX512(float *y, const float *x, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    _mm512_storeu_ps(y+i, _mm512_loadu_ps(x+i));
  }
  off = (n) - ((n)%16);
  for (i=off; i<n; i++) {
    y[i] = x[i];
  }
}

void THFloatVector_fill_AVX512(float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512 YMM0 = _mm512_set_ps(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    _mm512_storeu_ps((x)+i  , YMM0);
  }
  off = (n) - ((n)%16);
  for (i=off; i<n; i++) {
    x[i] = c;
  }
}

void THFloatVector_cdiv_AVX512(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0, YMM1;
    YMM0 = _mm512_loadu_ps(x+i);
    YMM1 = _mm512_loadu_ps(y+i);    
    YMM1 = _mm512_div_ps(YMM0, YMM1);    
    _mm512_storeu_ps(z+i, YMM1);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

void THFloatVector_divs_AVX512(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512 YMM15 = _mm512_set_ps(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0;
    YMM0 = _mm512_loadu_ps(x+i);
    YMM0 = _mm512_div_ps(YMM0, YMM15);
    _mm512_storeu_ps(y+i, YMM0);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

void THFloatVector_cmul_AVX512(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm512_loadu_ps(x+i);
    YMM1 = _mm512_loadu_ps(y+i);
    YMM1 = _mm512_mul_ps(YMM0, YMM1);
    _mm512_storeu_ps(z+i, YMM1);
  }
  off = (n) - ((n)%16);
  for (i=off; i<n; i++) {
    z[i] = x[i] * y[i];
  }
}

void THFloatVector_muls_AVX512(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512 YMM15 = _mm512_set_ps(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0;
    YMM0 = _mm512_loadu_ps(x+i);
    YMM0 = _mm512_mul_ps(YMM0, YMM15);
    _mm512_storeu_ps(y+i, YMM0);
  }
  off = (n) - ((n)%16);
  for (i=off; i<n; i++) {
    y[i] = x[i] * c;
  }
}
/*
void THFloatVector_cadd_AVX512(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512 YMM15 = _mm512_set_ps(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0, YMM1, YMM2, YMM3;
    YMM0 = _mm512_loadu_ps(y+i);
    YMM1 = _mm512_loadu_ps(x+i);
    YMM2 = _mm512_mul_ps(YMM0, YMM15);
    YMM3 = _mm512_add_ps(YMM1, YMM2);
    _mm512_storeu_ps(z+i, YMM3);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}
*/
void THFloatVector_adds_AVX512(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512 YMM15 = _mm512_set_ps(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i)  
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0;
    YMM0 = _mm512_loadu_ps(x+i);
    YMM0 = _mm512_add_ps(YMM0, YMM15);
    _mm512_storeu_ps(y+i, YMM0);
  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

void THDoubleVector_cadd_AVX512(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512d YMM15 = _mm512_set_pd(c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i) 
#endif
  for (i=0; i<=((n)-8); i+=8) {
    __m512d YMM0, YMM1;
    YMM0 = _mm512_loadu_pd(y+i);   
    YMM1 = _mm512_loadu_pd(x+i);
    YMM1 = _mm512_fmadd_pd(YMM0, YMM15, YMM1);
    _mm512_storeu_pd(z+i, YMM1);
  }
  off = (n) - ((n)%8);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

void THFloatVector_cadd_AVX512(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m512 YMM15 = _mm512_set_ps(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
#ifdef _OPENMP
  int omp_flag = omp_in_parallel();
  #pragma omp parallel for if ( (n > TH_OMP_OVERHEAD_THRESHOLD_VEC_AVX512) && ( 0 == omp_flag) )private (i) 
#endif
  for (i=0; i<=((n)-16); i+=16) {
    __m512 YMM0, YMM1;
    YMM0 = _mm512_loadu_ps(y+i);
    YMM1 = _mm512_loadu_ps(x+i);
    YMM1 = _mm512_fmadd_ps(YMM0, YMM15, YMM1);
    _mm512_storeu_ps(z+i, YMM1);

  }
  off = (n) - ((n)%16);
  for (i=off; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

#endif // defined(__AVX512__)

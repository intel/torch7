#ifndef TH_AVX512_H
#define TH_AVX512_H

#include <stddef.h>

void THDoubleVector_copy_AVX512(double *y, const double *x, const ptrdiff_t n);
void THDoubleVector_fill_AVX512(double *x, const double c, const ptrdiff_t n);
void THDoubleVector_cdiv_AVX512(double *z, const double *x, const double *y, const ptrdiff_t n);
void THDoubleVector_divs_AVX512(double *y, const double *x, const double c, const ptrdiff_t n);
void THDoubleVector_cmul_AVX512(double *z, const double *x, const double *y, const ptrdiff_t n);
void THDoubleVector_muls_AVX512(double *y, const double *x, const double c, const ptrdiff_t n);
void THDoubleVector_cadd_AVX512(double *z, const double *x, const double *y, const double c, const ptrdiff_t n);
void THDoubleVector_adds_AVX512(double *y, const double *x, const double c, const ptrdiff_t n);
void THFloatVector_copy_AVX512(float *y, const float *x, const ptrdiff_t n);
void THFloatVector_fill_AVX512(float *x, const float c, const ptrdiff_t n);
void THFloatVector_cdiv_AVX512(float *z, const float *x, const float *y, const ptrdiff_t n);
void THFloatVector_divs_AVX512(float *y, const float *x, const float c, const ptrdiff_t n);
void THFloatVector_cmul_AVX512(float *z, const float *x, const float *y, const ptrdiff_t n);
void THFloatVector_muls_AVX512(float *y, const float *x, const float c, const ptrdiff_t n);
void THFloatVector_cadd_AVX512(float *z, const float *x, const float *y, const float c, const ptrdiff_t n);
void THFloatVector_adds_AVX512(float *y, const float *x, const float c, const ptrdiff_t n);

#endif

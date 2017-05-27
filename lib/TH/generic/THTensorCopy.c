#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

#ifdef _OPENMP
#include <omp.h>
#define TH_OMP_OVERHEAD_THRESHOLD_COPY 100
#endif

#define THTENSOR_MAX_DIM 100

#ifdef _OPENMP


#ifndef _WIN32
#define _STR(s)   #s
#define STR(s)    _STR(s)
//#define PRAGMA(P) _Pragma( #P )
#define PRAGMA2(P) _Pragma( STR(P) )
#else
#define PRAGMA(P) __pragma(P)
#endif


#define TH_TENSOR_APPLY2_ADVANCED_INDEX(TYPE1, TENSOR1, TYPE2, TENSOR2, ADV_CODE, ORI_CODE) \
{                                                                                           \
  int TENSOR2##Dim = TENSOR2->nDimension;                                     \
  int TENSOR1##Dim = TENSOR1->nDimension;                                     \
  ptrdiff_t TENSOR2##Size = THTensor_(nElement)(TENSOR2);                     \
  ptrdiff_t TENSOR1##Size = THTensor_(nElement)(TENSOR1);                     \
  int TENSOR2##Contg = THTensor_(isContiguous)(TENSOR2)? 1:0;                 \
  int TENSOR1##Contg = THTensor_(isContiguous)(TENSOR1)? 1:0;                 \
  /* size not equal */                                                          \
  int omp_flag = omp_in_parallel();                                                           \
  if( (TENSOR2##Size == TENSOR1##Size) && (0 == omp_flag) ){                                         \
    int TENSOR2##StrideContg = 1;                                             \
    int TENSOR1##StrideContg = 1;                                             \
    /* all strides below are for advanced searching index*/                     \
    ptrdiff_t TENSOR2##Stride[THTENSOR_MAX_DIM] = {0};                        \
    ptrdiff_t TENSOR1##Stride[THTENSOR_MAX_DIM] = {0};                        \
                                                                              \
    ptrdiff_t strideSomeDim = 1;                                              \
    int dim;                                                                  \
    for (dim = TENSOR2##Dim; dim > 0; dim--){                                 \
      strideSomeDim *= TENSOR2->size[dim-1];                                  \
      TENSOR2##Stride[dim-1] = strideSomeDim;                                 \
      if(0 == TENSOR2->stride[dim])                                           \
      TENSOR2##StrideContg = 0;                                               \
    }                                                                         \
                                                                              \
    strideSomeDim = 1;                                                        \
    for (dim = TENSOR1##Dim; dim > 0; dim--){                                 \
      strideSomeDim *= TENSOR1->size[dim-1];                                  \
      TENSOR1##Stride[dim-1] = strideSomeDim;                                 \
      if(0 == TENSOR1->stride[dim])                                           \
      TENSOR1##StrideContg = 0;                                               \
    }                                                                         \
                                                                              \
    if((TENSOR2##StrideContg != 0) && (TENSOR1##StrideContg != 0) ){          \
      /* for adveanced searching index*/                                        \
      TYPE2 *tp = THTensor_(data)(TENSOR2);                                    \
      TYPE1 *rp = THTensor_(data)(TENSOR1);                                    \
      if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && TENSOR1##Contg){              \
        ptrdiff_t TENSOR2##BasicIndex = 0;                                                     \
        ptrdiff_t index = 0;                                                                   \
        TYPE1 *TENSOR1##Local = NULL;                                                           \
        TYPE2 *TENSOR2##Local = NULL;                                                           \
        ptrdiff_t iter = 0;                                                                    \
        ptrdiff_t dim = 0;                                                                     \
        ptrdiff_t i = 0;                                                                       \
        ptrdiff_t j = 0;                                                                       \
                                                                                               \
        PRAGMA2( omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR2##BasicIndex, index, TENSOR1##Local, TENSOR2##Local, iter, dim, i, j) )  \
        /* there is no parallelism below this level*/                                             \
        for(iter=0; iter < TENSOR1##Size; iter+=TENSOR1->stride[TENSOR1##Dim-2]) {             \
          /*not -1 to make use of vectorization*/                                                \
          index = iter/TENSOR1->stride[0];                                                     \
          TENSOR2##BasicIndex = index*TENSOR2->stride[0];                                      \
          for(dim = 1; dim < TENSOR1##Dim-1; dim++) {                                          \
             index = (iter%TENSOR1->stride[dim-1])/TENSOR1->stride[dim];\
             TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
\
          TENSOR1##Local = rp+iter;\
          TENSOR2##Local = tp+TENSOR2##BasicIndex;\
          j=0;\
          PRAGMA2( ivdep ) \
          for(i=0; i < TENSOR1->stride[TENSOR2##Dim-2]; i++) {   \
          /* not contiguous requirement*/          \
          /*  TENSOR1##Local[i] = TENSOR2##Local[j];*/ \
            ADV_CODE                                \
            j+= TENSOR2->stride[TENSOR2##Dim-1];\
          }\
        }      \
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && TENSOR2##Contg){\
        ptrdiff_t TENSOR1##BasicIndex = 0; \
        ptrdiff_t iter = 0; \
        ptrdiff_t dim = 0; \
        ptrdiff_t i = 0; \
        ptrdiff_t j = 0; \
        ptrdiff_t index = 0; \
        TYPE1 *TENSOR1##Local = NULL;\
        TYPE2 *TENSOR2##Local = NULL;\
                                    \
        PRAGMA2(  omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##BasicIndex, index, TENSOR1##Local, TENSOR2##Local, iter, dim, i, j)  )   \
        /*there is no parallelism below this level*/ \
        for(iter=0; iter < TENSOR2##Size; iter+=TENSOR2->stride[TENSOR2##Dim-2]){  \
          /*not -1 to make use of vectorization*/    \
          index = iter/TENSOR2->stride[0];\
          TENSOR1##BasicIndex = index*TENSOR1->stride[0];\
          for(dim = 1; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR2->stride[dim-1])/TENSOR2->stride[dim];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
\
          TENSOR2##Local = tp+iter;\
          TENSOR1##Local = rp+TENSOR1##BasicIndex;\
          i = 0;\
          PRAGMA2(ivdep) \
          for(j=0; j < TENSOR2->stride[TENSOR2##Dim-2]; j++) { \
          /*  TENSOR1##Local[i] = TENSOR2##Local[j];*/ \
            ADV_CODE                                \
            i+= TENSOR1->stride[TENSOR2##Dim-1];\
          }\
        }\
      } else {\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##Local = NULL;\
        TYPE2 *TENSOR2##Local = NULL;\
        ptrdiff_t i = 0;\
        ptrdiff_t j = 0;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2(  omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY)  private(TENSOR2##BasicIndex, TENSOR1##BasicIndex, index, TENSOR1##Local, TENSOR2##Local, iter, dim, i, j)  )  \
        /*there is no parallelism below this level*/ \
        for (iter = 0; iter < TENSOR1##Size; iter++) {\
          TENSOR2##BasicIndex = 0;\
          TENSOR1##BasicIndex = 0;\
\
          for(dim = 0; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR2##Stride[dim])/TENSOR2##Stride[dim+1];\
            TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
          index = iter%TENSOR2##Stride[dim];\
          TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
\
          for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
          index = iter%TENSOR1##Stride[dim];\
          TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          TENSOR2##Local = tp+TENSOR2##BasicIndex;\
          TENSOR1##Local = rp+TENSOR1##BasicIndex;\
          i = 0;\
          j = 0;\
          \
          /*  TENSOR1##Local[i] = TENSOR2##Local[j];*/ \
            ADV_CODE                                \
        }\
      }\
    } else {\
      TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE1, TENSOR2, ORI_CODE)\
    }\
  } else {\
    TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, ORI_CODE)\
  }\
\
}
#endif

/*
void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  int srcDim = src->nDimension;
  int tensorDim = tensor->nDimension;
  
  ptrdiff_t srcSize = THTensor_(nElement)(src);
  ptrdiff_t tensorSize = THTensor_(nElement)(tensor);
  int srcContg = THTensor_(isContiguous)(src)? 1:0;
  int tensorContg = THTensor_(isContiguous)(tensor)? 1:0;

  if ( srcContg && tensorContg && (srcSize == tensorSize)) {
    real *sp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    
#ifndef TH_REAL_IS_HALF
    THVector_(copy)(rp, sp, srcSize); 
#else
#ifdef _OPENMP
    ptrdiff_t i;
    
    int omp_flag = omp_in_parallel();
    #pragma omp parallel for if ( (srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) && ( 0 == omp_flag) )private (i) 
    #pragma ivdep
    for(i=0; i<srcSize; i++){
      rp[i] = sp[i];
    }   
#else
    memcpy(rp, sp, srcSize * sizeof(real));
#endif
#endif
  } else {

#ifdef _OPENMP      
    int omp_flag = omp_in_parallel();
    if( (srcSize == tensorSize) && (0 == omp_flag) ){        
      int srcStrideContg = 1;
      int tensorStrideContg = 1;
      // all strides below are for advanced searching index 
      ptrdiff_t srcStride[THTENSOR_MAX_DIM] = {0};
      ptrdiff_t tensorStride[THTENSOR_MAX_DIM] = {0};
      
      ptrdiff_t strideSomeDim = 1;     
      int dim;
      for (dim = srcDim; dim > 0; dim--){
        strideSomeDim *= src->size[dim-1];
        srcStride[dim-1] = strideSomeDim;
        if(0 == src->stride[dim])
          srcStrideContg = 0;
      }
      
      strideSomeDim = 1;     
      for (dim = tensorDim; dim > 0; dim--){
        strideSomeDim *= tensor->size[dim-1];
        tensorStride[dim-1] = strideSomeDim;
        if(0 == tensor->stride[dim])
          tensorStrideContg = 0;
      }
      
      if((srcStrideContg != 0) && (tensorStrideContg != 0) ){ // for adveanced searching index
        real *tp = THTensor_(data)(src);
        real *rp = THTensor_(data)(tensor);
        if((srcDim == tensorDim) && (srcDim > 2) && tensorContg){

          ptrdiff_t srcBasicIndex = 0;
          ptrdiff_t index = 0;
          real *rpLocal = NULL;
          real *tpLocal = NULL;
          ptrdiff_t iter = 0;
          ptrdiff_t dim = 0;
          ptrdiff_t i = 0;
          ptrdiff_t j = 0;
          
          #pragma omp parallel for if (srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(srcBasicIndex, index, rpLocal, tpLocal, iter, dim, i, j)   //there is no parallelism below this level
          for(iter=0; iter < tensorSize; iter+=tensor->stride[tensorDim-2]) {  //not -1 to make use of vectorization          

            index = iter/tensor->stride[0];
            srcBasicIndex = index*src->stride[0];
            for(dim = 1; dim < tensorDim-1; dim++)
            {
              index = (iter%tensor->stride[dim-1])/tensor->stride[dim];
              srcBasicIndex += index*src->stride[dim];
            }

            rpLocal = rp+iter;
            tpLocal = tp+srcBasicIndex;
            j=0;
            #pragma ivdep
            for(i=0; i < tensor->stride[srcDim-2]; i++)  // not contiguous requirement
            {
              rpLocal[i] = tpLocal[j];
              j+= src->stride[srcDim-1];
            }
          }      
        } else if((srcDim == tensorDim) && (srcDim > 2) && srcContg){
          ptrdiff_t tensorBasicIndex = 0;
          ptrdiff_t iter = 0;
          ptrdiff_t dim = 0;
          ptrdiff_t i = 0;
          ptrdiff_t j = 0;
          ptrdiff_t index = 0;

          real *rpLocal = NULL;
          real *tpLocal = NULL;
          #pragma omp parallel for if (srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(tensorBasicIndex, index, rpLocal, tpLocal, iter, dim, i, j)   //there is no parallelism below this level
          for(iter=0; iter < srcSize; iter+=src->stride[srcDim-2]){  //not -1 to make use of vectorization         
            index = iter/src->stride[0];
            tensorBasicIndex = index*tensor->stride[0];
            for(dim = 1; dim < srcDim-1; dim++) {
              index = (iter%src->stride[dim-1])/src->stride[dim];
              tensorBasicIndex += index*tensor->stride[dim];
            }

            tpLocal = tp+iter;
            rpLocal = rp+tensorBasicIndex;
            j=0;
            #pragma ivdep
            for(i=0; i < src->stride[srcDim-2]; i++)
            {
              rpLocal[j] = tpLocal[i];
              j+= tensor->stride[srcDim-1];
            }
          }
        } else {
          ptrdiff_t srcBasicIndex = 0;
          ptrdiff_t tensorBasicIndex = 0;

          ptrdiff_t i = 0;
          ptrdiff_t j = 0;
          ptrdiff_t index = 0;
          ptrdiff_t iter = 0;
          ptrdiff_t dim = 0;
          #pragma omp parallel for if (srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(srcBasicIndex, tensorBasicIndex, index,  iter, dim, i, j)   //there is no parallelism below this level
          for (iter = 0; iter < tensorSize; iter++) {
            srcBasicIndex = 0;
            tensorBasicIndex = 0;

            for(dim = 0; dim < srcDim-1; dim++) {
              index = (iter%srcStride[dim])/srcStride[dim+1];
              srcBasicIndex += index*src->stride[dim];
            }
            index = iter%srcStride[dim];
            srcBasicIndex += index*src->stride[dim];

            for(dim = 0; dim < tensorDim-1; dim++) {
              index = (iter%tensorStride[dim])/tensorStride[dim+1];
              tensorBasicIndex += index*tensor->stride[dim];
            }
            index = iter%tensorStride[dim];
            tensorBasicIndex += index*tensor->stride[dim];

            *(rp+tensorBasicIndex) = *(tp+srcBasicIndex);
          }
        }
      } else {
        TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
      }
    } else {
      TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
    }
    
#else
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
#endif        
  }
}

*/
void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  if (THTensor_(isContiguous)(tensor) && THTensor_(isContiguous)(src) && THTensor_(nElement)(tensor) == THTensor_(nElement)(src)) {
    real *sp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    ptrdiff_t sz = THTensor_(nElement)(tensor);
#ifndef TH_REAL_IS_HALF
    THVector_(copy)(rp, sp, sz); 
#else
    memcpy(rp, sp, sz * sizeof(real));
#endif
  } else {
#ifdef _OPENMP
    TH_TENSOR_APPLY2_ADVANCED_INDEX(real, tensor, real, src, tensorLocal[i] = srcLocal[j];, *tensor_data = *src_data;)
#else
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
#endif
  }
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_TO_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = TH_float2half((float)*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)TH_half2float(*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_TO_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = *src_data;) \
}

#ifndef TH_REAL_IS_HALF
IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)
IMPLEMENT_THTensor_COPY_FROM_HALF(Half, THHalf)
#else
/* only allow pass-through for Half */
IMPLEMENT_THTensor_COPY_TO_FROM_HALF(Half, THHalf)
IMPLEMENT_THTensor_COPY_TO_HALF(Byte, unsigned char)
IMPLEMENT_THTensor_COPY_TO_HALF(Char, char)
IMPLEMENT_THTensor_COPY_TO_HALF(Short, short)
IMPLEMENT_THTensor_COPY_TO_HALF(Int, int)
IMPLEMENT_THTensor_COPY_TO_HALF(Long, long)
IMPLEMENT_THTensor_COPY_TO_HALF(Float, float)
IMPLEMENT_THTensor_COPY_TO_HALF(Double, double)

#endif /* REAL_IS_HALF */

#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

#ifdef _OPENMP
#include <omp.h>
#define TH_OMP_OVERHEAD_THRESHOLD_COPY 10000
#endif

#define THTENSOR_MAX_DIM 100

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
    
    #pragma omp parallel for if (srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) private (i)
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
    if(srcSize == tensorSize){        
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

void THTensor_(copy2)(THTensor *tensor, THTensor *src)
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
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
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

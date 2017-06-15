#ifndef TH_TENSOR_APPLY_INC
#define TH_TENSOR_APPLY_INC

/*
 * The basic strategy for apply is as follows:
 *
 * 1. Starting with the outermost index, loop until we reach a dimension where the
 * data is no longer contiguous, i.e. the stride at that dimension is not equal to
 * the size of the tensor defined by the outer dimensions. Let's call this outer
 * (contiguous) tensor A. Note that if the Tensor is contiguous, then A is equal
 * to the entire Tensor. Let's call the inner tensor B.
 *
 * 2. We loop through the indices in B, starting at its outermost dimension. For
 * example, if B is a 2x2 matrix, then we do:
 *
 * B[0][0]
 * B[0][1]
 * B[1][0]
 * B[1][1]
 *
 * We set the offset into the underlying storage as (storageOffset + stride_B * index_B),
 * i.e. basically we compute the offset into the storage as we would normally for a
 * Tensor. But because we are guaranteed the subsequent data is contiguous in memory, we
 * can simply loop for sizeof(A) iterations and perform the operation, without having to
 * follow the order described by the strides of A.
 *
 * 3. As an optimization, we merge dimensions of A that are contiguous in memory. For
 * example, if A is a 3x3x3x3 tensor narrowed from a 3x3x4x3 tensor, then the first two
 * dimensions can be merged for the purposes of APPLY, reducing the number of nested
 * loops.
 */

#define __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, DIM, ALLOW_CONTIGUOUS) \
  TYPE *TENSOR##_data = NULL; \
  long *TENSOR##_counter = NULL, *TENSOR##_sizes = NULL, *TENSOR##_strides = NULL, *TENSOR##_dimOffset = NULL; \
  long TENSOR##_stride = 0, TENSOR##_size = 0, TENSOR##_dim = 0, TENSOR##_i, TENSOR##_n; \
  int TENSOR##_contiguous = ALLOW_CONTIGUOUS && DIM < 0; \
  TENSOR##_n = (TENSOR->nDimension ? 1 : 0); \
  for(TENSOR##_i = 0; TENSOR##_i < TENSOR->nDimension; TENSOR##_i++) \
    TENSOR##_n *= TENSOR->size[TENSOR##_i]; \
\
  if(TENSOR->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR##_data = TENSOR->storage->data+TENSOR->storageOffset; \
    TENSOR##_size = 1; \
    TENSOR##_stride = 1; \
    for(TENSOR##_i = TENSOR->nDimension-1; TENSOR##_i >= 0; TENSOR##_i--) { \
      if(TENSOR->size[TENSOR##_i] != 1) { \
        if(TENSOR->stride[TENSOR##_i] == TENSOR##_size && TENSOR##_i != DIM) \
          TENSOR##_size *= TENSOR->size[TENSOR##_i]; \
        else{ \
          TENSOR##_contiguous = 0; \
          break; \
        } \
      } \
    } \
    if (!TENSOR##_contiguous) { \
      /* Find the dimension of contiguous sections */ \
      TENSOR##_dim = 1; \
      for(TENSOR##_i = TENSOR->nDimension-2; TENSOR##_i >= 0; TENSOR##_i--) \
      { \
        if(TENSOR->stride[TENSOR##_i] != TENSOR->stride[TENSOR##_i+1] * TENSOR->size[TENSOR##_i+1] || TENSOR##_i == DIM || TENSOR##_i+1 == DIM) \
          TENSOR##_dim++; \
      } \
      /* Allocate an array of 3*dim elements, where dim is the number of contiguous sections */ \
      TENSOR##_counter = (long*)THAlloc(sizeof(long)*(3*TENSOR##_dim)); \
      TENSOR##_sizes = TENSOR##_counter + TENSOR##_dim; \
      TENSOR##_strides = TENSOR##_counter + 2*TENSOR##_dim; \
      TH_TENSOR_dim_index = TENSOR##_dim-1; \
      TENSOR##_dimOffset = (DIM == TENSOR->nDimension-1) ? &TENSOR##_i : &TENSOR##_counter[DIM]; \
      TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size[TENSOR->nDimension-1]; \
      TENSOR##_strides[TH_TENSOR_dim_index] = TENSOR->stride[TENSOR->nDimension-1]; \
      /* TENSOR##_counter tracks where we are in the storage. The offset into the */ \
      /* storage is given by storage_offset + (i * j), where i is the stride */ \
      /* vector and j is tensor_counter vector. This sets the starting position for the loop. */ \
      for(TENSOR##_i = TENSOR##_dim-1; TENSOR##_i >= 0; --TENSOR##_i) { \
        TENSOR##_counter[TENSOR##_i] = 0; \
      } \
      for(TENSOR##_i = TENSOR->nDimension-2; TENSOR##_i >= 0; --TENSOR##_i) { \
        if (TENSOR->stride[TENSOR##_i] == TENSOR->stride[TENSOR##_i+1] * TENSOR->size[TENSOR##_i+1] && TENSOR##_i != DIM && TENSOR##_i+1 != DIM) { \
          TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size[TENSOR##_i] * TENSOR##_sizes[TH_TENSOR_dim_index]; \
          if (DIM != TENSOR->nDimension-1 && TENSOR##_i < DIM) \
            TENSOR##_dimOffset--; \
        } else { \
          --TH_TENSOR_dim_index; \
          TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size[TENSOR##_i]; \
          TENSOR##_strides[TH_TENSOR_dim_index] = TENSOR->stride[TENSOR##_i]; \
        } \
      } \
      /* Size of the inner most section */ \
      TENSOR##_size = TENSOR##_sizes[TENSOR##_dim-1]; \
      /* Stride of the inner most section */ \
      TENSOR##_stride = TENSOR##_strides[TENSOR##_dim-1]; \
    } \
  } \
  TENSOR##_i = 0;

#define  __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR, ALWAYS_UPDATE) \
  if(TENSOR##_i == TENSOR##_size || ALWAYS_UPDATE) \
  { \
    if(TENSOR##_contiguous) \
      break; \
\
    if(TENSOR##_dim == 1) \
       break; \
\
    /* Reset pointer to beginning of loop */ \
    TENSOR##_data -= TENSOR##_size*TENSOR##_stride; \
    for(TENSOR##_i = TENSOR##_dim-2; TENSOR##_i >= 0; TENSOR##_i--) \
    { \
      TENSOR##_counter[TENSOR##_i]++; \
      /* Jump ahread by the stride of this dimension */ \
      TENSOR##_data += TENSOR##_strides[TENSOR##_i]; \
\
      if(TENSOR##_counter[TENSOR##_i]  == TENSOR##_sizes[TENSOR##_i]) \
      { \
        if(TENSOR##_i == 0) \
        { \
          TH_TENSOR_APPLY_hasFinished = 1; \
          break; \
        } \
          else \
        { \
          /* Reset the pointer to the beginning of the chunk defined by this dimension */ \
          TENSOR##_data -= TENSOR##_counter[TENSOR##_i]*TENSOR##_strides[TENSOR##_i]; \
          TENSOR##_counter[TENSOR##_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
    TENSOR##_i = 0; \
  } \

#define TH_TENSOR_APPLY3_D(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  long TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, DIM, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, DIM, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE3, TENSOR3, DIM, 1) \
\
  if(TENSOR1##_n != TENSOR2##_n || TENSOR1##_n != TENSOR3##_n) /* should we do the check in the function instead? i think so */ \
    THError("inconsistent tensor size"); \
\
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the inner most region of the Tensor */ \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size && TENSOR3##_i < TENSOR3##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR3##_i++, TENSOR1##_data += TENSOR1##_stride, TENSOR2##_data += TENSOR2##_stride, TENSOR3##_data += TENSOR3##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR1, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR2, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR3, 0) \
  } \
  if(TENSOR1##_counter != NULL) \
    THFree(TENSOR1##_counter); \
  if(TENSOR2##_counter != NULL) \
    THFree(TENSOR2##_counter); \
  if(TENSOR3##_counter != NULL) \
    THFree(TENSOR3##_counter); \
}

#define TH_TENSOR_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
  TH_TENSOR_APPLY3_D(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, -1, CODE)

#ifdef _OPENMP

#ifndef _WIN32
#define _STR(s)   #s
#define STR(s)    _STR(s)
//#define PRAGMA(P) _Pragma( #P )
#define PRAGMA2(P) _Pragma( STR(P) )
#else
#define PRAGMA(P) __pragma(P)
#endif

#define THTENSOR_MAX_DIM 100
#define TH_OMP_OVERHEAD_THRESHOLD_COPY 1000 

#define TH_TENSOR_APPLY_REDUCTION_ADVANCED_INDEX(TYPE1, TENSOR1, OPERATION, CODE) \
{                                                                               \
    int TENSOR1##Dim = TENSOR1->nDimension;                                     \
    ptrdiff_t TENSOR1##Size = THTensor_(nElement)(TENSOR1);                     \
    int TENSOR1##Contg = THTensor_(isContiguous)(TENSOR1)? 1:0;                 \
    int omp_flag = omp_in_parallel();                                            \
    if(0 == omp_flag) {                                                         \
        int TENSOR1##StrideContg = 1;                                             \
        ptrdiff_t TENSOR1##Stride[THTENSOR_MAX_DIM] = {0};                        \
        ptrdiff_t strideSomeDim = 1;                                              \
        int dim;                                                                  \
        strideSomeDim = 1;                                                        \
        for (dim = TENSOR1##Dim; dim > 0; dim--){                                 \
          if(0 == TENSOR1->stride[dim])  {                                         \
              TENSOR1##StrideContg = 0;                                              \
              break;                                                                \
          }                                                                       \
          strideSomeDim *= TENSOR1->size[dim-1];                                  \
          TENSOR1##Stride[dim-1] = strideSomeDim;                                 \
        }                                                                         \
        if(TENSOR1##StrideContg != 0) {                                          \
          TYPE1 *rp = THTensor_(data)(TENSOR1);                                    \
          if(TENSOR1##Contg){                                    \
            TYPE1 *TENSOR1##_data = NULL;         \
            ptrdiff_t index = 0;\
            ptrdiff_t iter = 0;\
            ptrdiff_t dim = 0;\
            PRAGMA2( omp parallel for if (TENSOR1##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##_data,  iter) reduction(OPERATION) ) \
            for (iter = 0; iter < TENSOR1##Size; iter++) {\
              TENSOR1##_data = rp+iter;\
              CODE                                \
            }\
          } else { \
            ptrdiff_t TENSOR1##BasicIndex = 0;\
            TYPE1 *TENSOR1##_data = NULL;         \
            ptrdiff_t index = 0;\
            ptrdiff_t iter = 0;\
            ptrdiff_t dim = 0;\
            PRAGMA2( omp parallel for if (TENSOR1##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##BasicIndex, TENSOR1##_data, index, iter, dim) reduction(OPERATION) ) \
            for (iter = 0; iter < TENSOR1##Size; iter++) {\
              TENSOR1##BasicIndex = 0;\
              for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
                index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
                TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
              }\
              index = iter%TENSOR1##Stride[dim];\
              TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
              TENSOR1##_data = rp+TENSOR1##BasicIndex;\
              CODE                                \
            }\
          }\
        } else {\
            TH_TENSOR_APPLY(TYPE1, TENSOR1, CODE);\
        }\
    } else {\
        TH_TENSOR_APPLY(TYPE1, TENSOR1, CODE);\
    }\
}

#define TH_TENSOR_APPLY2_ADVANCED_INDEX(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{                                                                             \
  int TENSOR1##Dim = TENSOR1->nDimension;                                     \
  int TENSOR2##Dim = TENSOR2->nDimension;                                     \
  ptrdiff_t TENSOR1##Size = THTensor_(nElement)(TENSOR1);                     \
  ptrdiff_t TENSOR2##Size = THTensor_(nElement)(TENSOR2);                     \
  int TENSOR1##Contg = THTensor_(isContiguous)(TENSOR1)? 1:0;                 \
  int TENSOR2##Contg = THTensor_(isContiguous)(TENSOR2)? 1:0;                 \
  /* size not equal */                                                        \
  int omp_flag = omp_in_parallel();                                                           \
  if( (TENSOR2##Size == TENSOR1##Size) && (0 == omp_flag) ){                                         \
    int TENSOR2##StrideContg = 1;                                             \
    int TENSOR1##StrideContg = 1;                                             \
    /* all strides below are for advanced searching index*/                   \
    ptrdiff_t TENSOR2##Stride[THTENSOR_MAX_DIM] = {0};                        \
    ptrdiff_t TENSOR1##Stride[THTENSOR_MAX_DIM] = {0};                        \
                                                                              \
    ptrdiff_t strideSomeDim = 1;                                              \
    int dim;                                                                  \
    for (dim = TENSOR2##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR2->stride[dim]) {                                         \
        TENSOR2##StrideContg = 0;                                             \
        break;                                               \
      }                                                                        \
      strideSomeDim *= TENSOR2->size[dim-1];                                  \
      TENSOR2##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
                                                                              \
    strideSomeDim = 1;                                                        \
    for (dim = TENSOR1##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR1->stride[dim])  {                                         \
        TENSOR1##StrideContg = 0;                                              \
        break;                                                                \
      }                                                                       \
      strideSomeDim *= TENSOR1->size[dim-1];                                  \
      TENSOR1##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
                                                                              \
    if((TENSOR2##StrideContg != 0) && (TENSOR1##StrideContg != 0) ){          \
      /* for adveanced searching index*/                                       \
      TYPE2 *tp = THTensor_(data)(TENSOR2);                                    \
      TYPE1 *rp = THTensor_(data)(TENSOR1);                                    \
      if(TENSOR1##Contg && TENSOR2##Contg){                                    \
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t iter = 0;\
        PRAGMA2( omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private( TENSOR1##_data, TENSOR2##_data, iter) ) \
        for (iter = 0; iter < TENSOR1##Size; iter++) {\
          TENSOR1##_data = rp+iter;\
          TENSOR2##_data = tp+iter;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && TENSOR1##Contg){              \
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR2##BasicIndex, TENSOR1##_data, TENSOR2##_data, index, iter, dim) )  \
        for (iter = 0; iter < TENSOR1##Size; iter++) {\
          TENSOR2##BasicIndex = 0;\
          for(dim = 0; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR2##Stride[dim])/TENSOR2##Stride[dim+1];\
            TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
          index = iter%TENSOR2##Stride[dim];\
          TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
                                                             \
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+iter;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && TENSOR2##Contg){\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##BasicIndex, TENSOR1##_data, TENSOR2##_data, index, iter, dim) )  \
        for (iter = 0; iter < TENSOR1##Size; iter++) {\
          TENSOR1##BasicIndex = 0;\
\
          for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
          index = iter%TENSOR1##Stride[dim];\
          TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          TENSOR2##_data = tp+iter;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      } else {\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (TENSOR2##Size > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR2##BasicIndex, TENSOR1##BasicIndex, TENSOR1##_data, TENSOR2##_data, index, iter, dim) )  \
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
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      }\
    } else {\
      TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE)\
    }\
  } else {\
    TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE)\
  }\
\
}

#define TH_TENSOR_APPLY2_ADVANCED_INDEX2(SIZE, CONTIG1, CONTIG2, TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{                                                                             \
  int TENSOR1##Dim = TENSOR1->nDimension;                                     \
  int TENSOR2##Dim = TENSOR2->nDimension;                                     \
  /* size not equal */                                                        \
  int omp_flag = omp_in_parallel();                                           \
  if(0 == omp_flag){                                         \
    int TENSOR2##StrideContg = 1;                                             \
    int TENSOR1##StrideContg = 1;                                             \
    /* all strides below are for advanced searching index*/                   \
    ptrdiff_t TENSOR2##Stride[THTENSOR_MAX_DIM] = {0};                        \
    ptrdiff_t TENSOR1##Stride[THTENSOR_MAX_DIM] = {0};                        \
                                                                              \
    ptrdiff_t strideSomeDim = 1;                                              \
    int dim;                                                                  \
    for (dim = TENSOR2##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR2->stride[dim]) {                                         \
        TENSOR2##StrideContg = 0;                                             \
        break;                                               \
      }                                                                        \
      strideSomeDim *= TENSOR2->size[dim-1];                                  \
      TENSOR2##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
                                                                              \
    strideSomeDim = 1;                                                        \
    for (dim = TENSOR1##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR1->stride[dim])  {                                         \
        TENSOR1##StrideContg = 0;                                              \
        break;                                                                \
      }                                                                       \
      strideSomeDim *= TENSOR1->size[dim-1];                                  \
      TENSOR1##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
                                                                              \
    if((TENSOR2##StrideContg != 0) && (TENSOR1##StrideContg != 0) ){          \
      /* for adveanced searching index*/                                       \
      TYPE2 *tp = THTensor_(data)(TENSOR2);                                    \
      TYPE1 *rp = THTensor_(data)(TENSOR1);                                    \
      if(CONTIG1 && CONTIG2){                                    \
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t iter = 0;\
        if(tp != rp) { \
          PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) )\
          PRAGMA2(ivdep) \
          for (iter = 0; iter < SIZE; iter++) {\
            TENSOR2##_data = tp+iter;\
            TENSOR1##_data = rp+iter;\
            CODE                                \
          }\
        } else {\
          PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) )\
          for (iter = 0; iter < SIZE; iter++) {\
            TENSOR2##_data = tp+iter;\
            TENSOR1##_data = rp+iter;\
            CODE                                \
          }\
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && CONTIG1){              \
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##_data, TENSOR2##_data) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR2##BasicIndex = 0;\
          for(dim = 0; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR2##Stride[dim])/TENSOR2##Stride[dim+1];\
            TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
          index = iter%TENSOR2##Stride[dim];\
          TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
                                                             \
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+iter;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && CONTIG2){\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##_data, TENSOR2##_data) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR1##BasicIndex = 0;\
\
          for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
          index = iter%TENSOR1##Stride[dim];\
          TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          TENSOR2##_data = tp+iter;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      } else {\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##_data, TENSOR2##_data )  )\
        /*there is no parallelism below this level*/ \
        for (iter = 0; iter < SIZE; iter++) {\
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
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      }\
    } else {\
      TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE)\
    }\
  } else {\
    TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE)\
  }\
\
}

#define TH_TENSOR_APPLY3_ADVANCED_INDEX2(SIZE, CONTIG1, CONTIG2, CONTIG3, TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{                                                                             \
  int TENSOR1##Dim = TENSOR1->nDimension;                                     \
  int TENSOR2##Dim = TENSOR2->nDimension;                                     \
  int TENSOR3##Dim = TENSOR3->nDimension;                                     \
  /* size not equal */                                                        \
  int omp_flag = omp_in_parallel();                                           \
  if(0 == omp_flag){                                                          \
    int TENSOR1##StrideContg = 1;                                             \
    int TENSOR2##StrideContg = 1;                                             \
    int TENSOR3##StrideContg = 1;                                             \
    /* all strides below are for advanced searching index*/                   \
    ptrdiff_t TENSOR1##Stride[THTENSOR_MAX_DIM] = {0};                        \
    ptrdiff_t TENSOR2##Stride[THTENSOR_MAX_DIM] = {0};                        \
    ptrdiff_t TENSOR3##Stride[THTENSOR_MAX_DIM] = {0};                        \
                                                                              \
    ptrdiff_t strideSomeDim = 1;                                              \
    int dim;                                                                  \
    for (dim = TENSOR1##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR1->stride[dim])  {                                        \
        TENSOR1##StrideContg = 0;                                             \
        break;                                                                \
      }                                                                       \
      strideSomeDim *= TENSOR1->size[dim-1];                                  \
      TENSOR1##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
    strideSomeDim = 1;                                                        \
    for (dim = TENSOR2##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR2->stride[dim]) {                                         \
        TENSOR2##StrideContg = 0;                                             \
        break;                                                                \
      }                                                                       \
      strideSomeDim *= TENSOR2->size[dim-1];                                  \
      TENSOR2##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
    strideSomeDim = 1;                                                        \
    for (dim = TENSOR3##Dim; dim > 0; dim--){                                 \
      if(0 == TENSOR3->stride[dim]) {                                         \
        TENSOR3##StrideContg = 0;                                             \
        break;                                                                \
      }                                                                       \
      strideSomeDim *= TENSOR3->size[dim-1];                                  \
      TENSOR3##Stride[dim-1] = strideSomeDim;                                 \
    }                                                                         \
                                                                              \
    if((TENSOR2##StrideContg != 0) && (TENSOR1##StrideContg != 0) && (TENSOR3##StrideContg != 0)){          \
      /* for adveanced searching index*/                                                                    \
      TYPE1 *rp = THTensor_(data)(TENSOR1);                                                                 \
      TYPE2 *tp = THTensor_(data)(TENSOR2);                                                                 \
      TYPE3 *srcp = THTensor_(data)(TENSOR3);                                                               \
      if(CONTIG1 && CONTIG2 && CONTIG3){                                                                    \
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t iter = 0;\
        if (rp != tp) { \
          PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) )\
          PRAGMA2(ivdep) \
          for (iter = 0; iter < SIZE; iter++) {\
            TENSOR1##_data = rp+iter;\
            TENSOR2##_data = tp+iter; \
            TENSOR3##_data = srcp+iter;\
            CODE                                \
          } \
        } else {\
          PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) )\
          for (iter = 0; iter < SIZE; iter++) {\
            TENSOR1##_data = rp+iter;\
            TENSOR2##_data = tp+iter; \
            TENSOR3##_data = srcp+iter;\
            CODE                                \
          } \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR3##Dim == TENSOR1##Dim) && (TENSOR1##Dim > 2) && CONTIG1 && CONTIG2){              \
        /*TENSOR3 is not contig*/ \
        ptrdiff_t TENSOR3##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR3##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR3##BasicIndex = 0;\
          for(dim = 0; dim < TENSOR3##Dim-1; dim++) {\
            index = (iter%TENSOR3##Stride[dim])/TENSOR3##Stride[dim+1];\
            TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
          }\
          index = iter%TENSOR3##Stride[dim];\
          TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
                                                             \
          TENSOR3##_data = srcp+TENSOR3##BasicIndex;\
          TENSOR2##_data = tp+iter;\
          TENSOR1##_data = rp+iter;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR3##Dim == TENSOR1##Dim) && (TENSOR1##Dim > 2) && CONTIG1 && CONTIG3){              \
        /*TENSOR2 is not contig*/ \
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR2##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR2##BasicIndex = 0;\
          for(dim = 0; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR2##Stride[dim])/TENSOR2##Stride[dim+1];\
            TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
          index = iter%TENSOR2##Stride[dim];\
          TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
                                                             \
          TENSOR3##_data = srcp+iter;\
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+iter;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR3##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && CONTIG2 && CONTIG3){              \
        /*TENSOR1 is not contig*/ \
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR1##BasicIndex = 0;\
          for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
          index = iter%TENSOR1##Stride[dim];\
          TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
                                                             \
          TENSOR3##_data = srcp+iter;\
          TENSOR2##_data = tp+iter;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR3##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && CONTIG3){\
        /* only tensor3 is contig*/ \
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##BasicIndex, TENSOR2##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR1##BasicIndex = 0;\
          TENSOR2##BasicIndex = 0;\
\
          for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
          index = iter%TENSOR1##Stride[dim];\
          TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          \
          for(dim = 0; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR2##Stride[dim])/TENSOR2##Stride[dim+1];\
            TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
          index = iter%TENSOR2##Stride[dim];\
          TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
\
          TENSOR3##_data = srcp+iter;\
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR3##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && CONTIG2){\
        /* only tensor2 is contig*/ \
        ptrdiff_t TENSOR3##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR1##BasicIndex, TENSOR3##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR1##BasicIndex = 0;\
          TENSOR3##BasicIndex = 0;\
\
          for(dim = 0; dim < TENSOR1##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR1##Stride[dim+1];\
            TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          }\
          index = iter%TENSOR1##Stride[dim];\
          TENSOR1##BasicIndex += index*TENSOR1->stride[dim];\
          \
          for(dim = 0; dim < TENSOR3##Dim-1; dim++) {\
            index = (iter%TENSOR3##Stride[dim])/TENSOR3##Stride[dim+1];\
            TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
          }\
          index = iter%TENSOR3##Stride[dim];\
          TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
\
          TENSOR3##_data = srcp+TENSOR3##BasicIndex;\
          TENSOR2##_data = tp+iter;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                \
        }\
      } else if((TENSOR2##Dim == TENSOR1##Dim) && (TENSOR3##Dim == TENSOR1##Dim) && (TENSOR2##Dim > 2) && CONTIG1){\
        /* only tensor1 is contig*/ \
        ptrdiff_t TENSOR3##BasicIndex = 0;\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR2##BasicIndex, TENSOR3##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR3##BasicIndex = 0;\
          TENSOR2##BasicIndex = 0;\
\
          for(dim = 0; dim < TENSOR2##Dim-1; dim++) {\
            index = (iter%TENSOR1##Stride[dim])/TENSOR2##Stride[dim+1];\
            TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          }\
          index = iter%TENSOR2##Stride[dim];\
          TENSOR2##BasicIndex += index*TENSOR2->stride[dim];\
          \
          for(dim = 0; dim < TENSOR3##Dim-1; dim++) {\
            index = (iter%TENSOR3##Stride[dim])/TENSOR3##Stride[dim+1];\
            TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
          }\
          index = iter%TENSOR3##Stride[dim];\
          TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
\
          TENSOR3##_data = srcp+TENSOR3##BasicIndex;\
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+iter;\
          CODE                                \
        }\
      } else {\
        ptrdiff_t TENSOR3##BasicIndex = 0;\
        ptrdiff_t TENSOR2##BasicIndex = 0;\
        ptrdiff_t TENSOR1##BasicIndex = 0;\
        TYPE1 *TENSOR1##_data = NULL;\
        TYPE2 *TENSOR2##_data = NULL;\
        TYPE3 *TENSOR3##_data = NULL;\
        ptrdiff_t index = 0;\
        ptrdiff_t iter = 0;\
        ptrdiff_t dim = 0;\
                          \
        PRAGMA2( omp parallel for if (SIZE > TH_OMP_OVERHEAD_THRESHOLD_COPY) private(TENSOR3##BasicIndex, TENSOR2##BasicIndex, TENSOR1##BasicIndex, TENSOR1##_data, TENSOR2##_data, TENSOR3##_data, index, iter, dim) )  \
        /*there is no parallelism below this level*/ \
        for (iter = 0; iter < SIZE; iter++) {\
          TENSOR3##BasicIndex = 0;\
          TENSOR2##BasicIndex = 0;\
          TENSOR1##BasicIndex = 0;\
          \
          for(dim = 0; dim < TENSOR3##Dim-1; dim++) {\
            index = (iter%TENSOR3##Stride[dim])/TENSOR3##Stride[dim+1];\
            TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
          }\
          index = iter%TENSOR3##Stride[dim];\
          TENSOR3##BasicIndex += index*TENSOR3->stride[dim];\
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
          \
          TENSOR3##_data = srcp+TENSOR3##BasicIndex;\
          TENSOR2##_data = tp+TENSOR2##BasicIndex;\
          TENSOR1##_data = rp+TENSOR1##BasicIndex;\
          CODE                                    \
        }\
      }\
    } else {\
      TH_TENSOR_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE)\
    }\
  } else {\
    TH_TENSOR_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE)\
  }\
\
}
#endif

#define TH_TENSOR_APPLY2_D(TYPE1, TENSOR1, TYPE2, TENSOR2, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  long TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE1, TENSOR1, DIM, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE2, TENSOR2, DIM, 1) \
\
  if(TENSOR1##_n != TENSOR2##_n) /* should we do the check in the function instead? i think so */ \
    THError("inconsistent tensor size"); \
\
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the inner most region of the Tensor */ \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR1##_data += TENSOR1##_stride, TENSOR2##_data += TENSOR2##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR1, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR2, 0) \
  } \
  if(TENSOR1##_counter != NULL) \
    THFree(TENSOR1##_counter); \
  if(TENSOR2##_counter != NULL) \
    THFree(TENSOR2##_counter); \
}

#define TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
  TH_TENSOR_APPLY2_D(TYPE1, TENSOR1, TYPE2, TENSOR2, -1, CODE)

#define TH_TENSOR_APPLY_D(TYPE, TENSOR, DIM, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  long TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, DIM, 0) \
\
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    /* Loop through the inner most region of the Tensor */ \
    for(; TENSOR##_i < TENSOR##_size; TENSOR##_i++, TENSOR##_data += TENSOR##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR, 1) \
  } \
  THFree(TENSOR##_counter); \
}

#define TH_TENSOR_APPLY(TYPE, TENSOR, CODE) \
  TH_TENSOR_APPLY_D(TYPE, TENSOR, -1, CODE)

#endif

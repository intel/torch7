# - Find INTEL MKLML library
#
# This module finds the Intel Mkl ml libraries.
#
# This module sets the following variables:
#  MKLML_FOUND - set to true if a library implementing the CBLAS interface is found
#  MKLML_VERSION - best guess
#  MKLML_INCLUDE_DIR - path to include dir.
#  MKLML_LIBRARIES - list of libraries for base mkl


# Do nothing if MKLML_FOUND was set before!
IF (NOT MKLML_FOUND)

SET(MKLML_VERSION)
SET(MKLML_INCLUDE_DIR)
SET(MKLML_LIBRARIES)

# Includes
INCLUDE(CheckTypeSize)
INCLUDE(CheckFunctionExists)

# Intel Compiler Suite
SET(INTEL_COMPILER_DIR CACHE STRING
  "Root directory of the Intel Compiler Suite (contains ipp, mkl, etc.)")
SET(INTEL_MKLML_DIR CACHE STRING
  "Root directory of the Intel MKLML (standalone)")
SET(INTEL_MKLML_SEQUENTIAL OFF CACHE BOOL
  "Force using the sequential (non threaded) libraries")

IF(CMAKE_COMPILER_IS_GNUCC)
  SET(mklml_lib_list
        mklml_gnu)
  SET(mklml "mklml_gnu")
  SET(mklomp "iomp5")
ELSE(CMAKE_COMPILER_IS_GNUCC)
  SET(mklml "mklml_intel")
  SET(mklomp "iomp5")
ENDIF (CMAKE_COMPILER_IS_GNUCC)

#SET(mklml_lib_list mklml_gnu mklml_intel iomp)
SET(mklml_header_list mkl_blas.h i_malloc.h mkl_cblas.h mkl_dnn_types.h mkl_service.h mkl_trans.h mkl_types.h mkl_version.h mkl_vml_defines.h mkl_vml_functions.h mkl_vml.h mkl_vml_types.h mkl_vsl_defines.h mkl_vsl_functions.h mkl_vsl.h mkl_vsl_types.h)


# Paths
SET(saved_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})
SET(saved_CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH})
IF (INTEL_MKLML_DIR)
  # TODO: diagnostic if dir does not exist
  SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${INTEL_MKLML_DIR}/include")
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_MKLML_DIR}/lib/${mklvers}")
ENDIF (INTEL_MKLML_DIR)

macro (mklml_find_lib VAR NAME DIRS)
    find_path(${VAR} ${NAME} ${DIRS} NO_DEFAULT_PATH)
    set(${VAR} ${${VAR}}/${NAME})
    unset(${VAR} CACHE)
endmacro()


macro(GET_MKLML_VERSION VERSION_FILE)
    # read MKL version info from file
    file(STRINGS ${VERSION_FILE} STR1 REGEX "__INTEL_MKL__")
    file(STRINGS ${VERSION_FILE} STR2 REGEX "__INTEL_MKL_MINOR__")
    file(STRINGS ${VERSION_FILE} STR3 REGEX "__INTEL_MKL_UPDATE__")
    #file(STRINGS ${VERSION_FILE} STR4 REGEX "INTEL_MKL_VERSION")

    # extract info and assign to variables
    string(REGEX MATCHALL "[0-9]+" MKLML_VERSION_MAJOR ${STR1})
    string(REGEX MATCHALL "[0-9]+" MKLML_VERSION_MINOR ${STR2})
    string(REGEX MATCHALL "[0-9]+" MKLML_VERSION_UPDATE ${STR3})
    set(MKLML_VERSION_STR "${MKLML_VERSION_MAJOR}.${MKLML_VERSION_MINOR}.${MKLML_VERSION_UPDATE}" CACHE STRING "MKLML version" FORCE)
endmacro()


#check current MKL_ROOT_DIR
if(NOT MKLML_ROOT_DIR OR NOT EXISTS ${MKLML_ROOT_DIR}/include/mkldnn.h OR NOT EXISTS ${MKLML_ROOT_DIR}/include/mklcblas.h)
    set(mklml_root_paths "")
    if (NOT MKLML_ROOT_DIR)
        LIST(APPEND mklml_root_paths ${MKLML_ROOT_DIR})
    endif()
    set(ENV_MKLML "$ENV{MKLML_PATH}")
    if(ENV_MKLML)

        list(APPEND mklml_root_paths ${ENV_MKLML})
    endif()
    if(UNIX)
        list(APPEND mklml_root_paths "/opt/intel/mklml")
    endif()
    message("+++++++++++" $ENV{MKLML_PATH}   "       " ${mklml_root_paths})
    find_path(MKLML_ROOT_DIR include/mkl_dnn.h include/mkl_cblas.h include/mkl_blas.h PATHS ${mklml_root_paths})
endif()

if(MKLML_ROOT_DIR)
    set(MKLML_INCLUDE_DIRS ${MKLML_ROOT_DIR}/include)
    GET_MKLML_VERSION(${MKLML_INCLUDE_DIRS}/mkl_version.h)    
    if(${MKLML_VERSION_STR} VERSION_GREATER "17.0.0" OR ${MKLML_VERSION_STR} VERSION_EQUAL "17.0.0")
        set(mklml_lib_find_paths
            ${MKLML_ROOT_DIR}/lib )
        
        set(mklml_header_find_paths
            ${MKLML_ROOT_DIR}/include ) 
    
    else()
        message(STATUS "MKL version ${MKLML_VERSION_STR} is obsoleting")
    endif()
    
    
    set(MKLML_LIBRARIES "")
    foreach(lib ${mklml_lib_list})
        FIND_LIBRARY(${lib} ${lib} ${mklml_lib_find_paths})
        MARK_AS_ADVANCED(${lib})
        list(APPEND MKLML_LIBRARIES ${${lib}})
    endforeach()

    # Include files
    set(MKLML_INCLUDE_DIR "")
    FOREACH(header ${mklml_header_list})
       FIND_PATH(${header} ${header} ${mklml_header_find_paths})
       MARK_AS_ADVANCED(${header})
       LIST(APPEND MKLML_INCLUDE_DIR ${${header}})
    ENDFOREACH()
message("-----------------------------------" ${MKLML_VERSION_STR} "   " ${MKLML_LIBRARIES} "  " ${mklml_lib_find_paths}  "   "  ${MKLML_INCLUDE_DIR})
endif()


# Final
SET(CMAKE_LIBRARY_PATH ${saved_CMAKE_LIBRARY_PATH})
SET(CMAKE_INCLUDE_PATH ${saved_CMAKE_INCLUDE_PATH})
IF (MKLML_LIBRARIES AND MKLML_INCLUDE_DIR)
  SET(MKLML_FOUND TRUE)
ELSE (MKLML_LIBRARIES AND MKLML_INCLUDE_DIR)
  SET(MKLML_FOUND FALSE)
  SET(MKLML_VERSION)
ENDIF (MKLML_LIBRARIES AND MKLML_INCLUDE_DIR)

# Standard termination
IF(NOT MKLML_FOUND AND MKLML_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "MKLML library not found. Please specify library  location")
ENDIF(NOT MKLML_FOUND AND MKLML_FIND_REQUIRED)
IF(NOT MKLML_FIND_QUIETLY)
  IF(MKLML_FOUND)
    MESSAGE(STATUS "MKLML library found")
  ELSE(MKLML_FOUND)
    MESSAGE(STATUS "MKLML library not found")
  ENDIF(MKLML_FOUND)
ENDIF(NOT MKLML_FIND_QUIETLY)

ENDIF(NOT MKLML_FOUND)

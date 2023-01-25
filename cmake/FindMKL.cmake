#===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

include_guard()
set(MKL_SYCL mkl_sycl)
set(MKL_IFACE mkl_intel_ilp64)
set(MKL_SEQ mkl_sequential)
set(MKL_TBB mkl_tbb_thread)
set(MKL_CORE mkl_core)

set(MKL_C ${MKL_IFACE})

if(ENABLE_MKLCPU_THREAD_TBB)
  find_package(TBB REQUIRED)
  list(APPEND MKL_C ${MKL_TBB})
else()
  list(APPEND MKL_C ${MKL_SEQ})
endif()

list(APPEND MKL_C ${MKL_CORE})

if(ENABLE_MKLGPU_BACKEND)
  set(USE_DPCPP_API ON)
endif()

if (ENABLE_MKLCPU_BACKEND OR ENABLE_MKLGPU_BACKEND)
  if(USE_DPCPP_API)
    list(APPEND MKL_LIBRARIES ${MKL_SYCL})
  endif()
  list(APPEND MKL_LIBRARIES ${MKL_C})
endif()

include(FindPackageHandleStandardArgs)
foreach(lib ${MKL_LIBRARIES})
  find_library(${lib}_file NAMES ${lib}
          HINTS $ENV{MKLROOT} ${MKL_ROOT}
          PATH_SUFFIXES lib/intel64)
  find_package_handle_standard_args(MKL
          REQUIRED_VARS ${lib}_file
          VERSION_VAR MKL_VERSION)
endforeach()

get_filename_component(MKL_LIB_DIR ${mkl_core_file} DIRECTORY)

find_path(MKL_INCLUDE mkl.h
          HINTS $ENV{MKLROOT} ${MKL_ROOT}
          PATH_SUFFIXES include)

file(READ "${MKL_INCLUDE}/mkl_version.h" mkl_version_h)
string(REGEX MATCH "INTEL_MKL_VERSION      ([0-9]*)" _ ${mkl_version_h})
set(MKL_VERSION ${CMAKE_MATCH_1})

if(${CMAKE_SIZEOF_VOID_P} EQUAL 8 OR USE_DPCPP_API)
  set(MKL_COPT "-DMKL_ILP64")
else()
  set(MKL_COPT "")
endif()
list(APPEND MKL_COPT "-DINTEL_MKL_VERSION=${MKL_VERSION}")

if(UNIX)
  list(APPEND MKL_LINK_PREFIX "-Wl,-rpath,${MKL_LIB_DIR}")
  list(APPEND MKL_LINK_PREFIX "-L${MKL_LIB_DIR}")
  set(LIB_PREFIX "-l")
  set(LIB_SUFFIX "")
else()
  set(LIB_PREFIX "${MKL_LIB_DIR}/")
  set(LIB_SUFFIX ".lib")
endif()

if (ENABLE_MKLCPU_BACKEND OR ENABLE_MKLGPU_BACKEND)
  set(MKL_LINK_C ${MKL_LINK_PREFIX})
  foreach(lib ${MKL_C})
    list(APPEND MKL_LINK_C ${LIB_PREFIX}${lib}${LIB_SUFFIX})
  endforeach()
  if(ENABLE_MKLCPU_THREAD_TBB)
    list(APPEND MKL_LINK_C ${TBB_LINK})
  endif()
  if(USE_DPCPP_API)
    find_package(OpenCL QUIET)
    # Try to find OpenCL library in the environment
    if(${OpenCL_LIBRARY} STREQUAL "OpenCL_LIBRARY-NOTFOUND")
      find_library(OPENCL_LIBNAME NAMES libOpenCL.so OpenCL.lib OpenCL HINTS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH ENV LIB ENV PATH)
    else()
      set(OPENCL_LIBNAME ${OpenCL_LIBRARY})
    endif()
    find_package_handle_standard_args(MKL REQUIRED_VARS OPENCL_LIBNAME)
    set(MKL_LINK_SYCL ${MKL_LINK_PREFIX} ${LIB_PREFIX}${MKL_SYCL}${LIB_SUFFIX} ${MKL_LINK_C} ${OPENCL_LIBNAME} )
  endif()
endif()

if (USE_DPCPP_API)
  find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE MKL_COPT MKL_LINK_SYCL
    VERSION_VAR MKL_VERSION)
else(ENABLE_MKLCPU_BACKEND)
  find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE MKL_COPT MKL_LINK_C
    VERSION_VAR MKL_VERSION)
endif()

#===============================================================================
# Copyright 2020 Intel Corporation
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


if (ENABLE_MKLCPU_BACKEND OR ENABLE_MKLGPU_BACKEND)
  if(ENABLE_MKLGPU_BACKEND)
    list(APPEND MKL_LIBRARIES ${MKL_SYCL})
  endif()
  list(APPEND MKL_LIBRARIES ${MKL_C})
endif()

foreach(lib ${MKL_LIBRARIES})
  find_library(${lib}_file NAMES ${lib}
          HINTS $ENV{MKLROOT} ${MKL_ROOT}
          PATH_SUFFIXES lib/intel64)
  find_package_handle_standard_args(MKL REQUIRED_VARS ${lib}_file)
endforeach()

if(UNIX)
  set(MKL_CORE_LIBNAME libmkl_core.so)
else()
  set(MKL_CORE_LIBNAME mkl_core.lib)
endif()

find_path(MKL_LIB_DIR ${MKL_CORE_LIBNAME}
          HINTS $ENV{MKLROOT} ${MKL_ROOT}
          PATH_SUFFIXES lib/intel64)

find_path(MKL_INCLUDE mkl.h
          HINTS $ENV{MKLROOT} ${MKL_ROOT}
          PATH_SUFFIXES include)

if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
  set(MKL_COPT "-DMKL_ILP64")
else()
  set(MKL_COPT "")
endif()

#Workaround for soname problem
if(UNIX)
  list(APPEND MKL_LINK_PREFIX "-Wl,-rpath,${MKL_LIB_DIR}")
  list(APPEND MKL_LINK_PREFIX "-L${MKL_LIB_DIR}")
  if (ENABLE_MKLCPU_BACKEND OR ENABLE_MKLGPU_BACKEND)
    set(MKL_LINK_C ${MKL_LINK_PREFIX})
    foreach(lib ${MKL_C})
      list(APPEND MKL_LINK_C -l${lib})
    endforeach()
    if(ENABLE_MKLCPU_THREAD_TBB)
      list(APPEND MKL_LINK_C ${TBB_LINK})
    endif()
    if(ENABLE_MKLGPU_BACKEND)
      set(MKL_LINK_SYCL ${MKL_LINK_PREFIX} -l${MKL_SYCL} ${MKL_LINK_C} -lOpenCL)
    endif()
  endif()
endif()


include(FindPackageHandleStandardArgs)
if (ENABLE_MKLCPU_BACKEND)
  find_package_handle_standard_args(MKL REQUIRED_VARS MKL_INCLUDE MKL_COPT MKL_LINK_C)
else(ENABLE_MKLGPU_BACKEND)
  find_package_handle_standard_args(MKL REQUIRED_VARS MKL_INCLUDE MKL_COPT MKL_LINK_SYCL)
endif()

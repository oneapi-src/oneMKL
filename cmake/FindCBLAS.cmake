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

list(APPEND BLAS_LIBS cblas)
list(APPEND BLAS_LIBS blas)

foreach(lib ${BLAS_LIBS})
  find_library(${lib}_file NAMES ${lib} HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES lib lib64)
  find_package_handle_standard_args(CBLAS REQUIRED_VARS ${lib}_file)
endforeach()

if(UNIX)
  set(CBLAS_LIBNAME libcblas.so)
else()
  set(CBLAS_LIBNAME cblas.lib)
endif()

find_path(CBLAS_LIB_DIR ${CBLAS_LIBNAME} HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES lib lib64)

find_path(CBLAS_INCLUDE cblas.h HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES include)


if(UNIX)
  list(APPEND CBLAS_LINK "-Wl,-rpath,${CBLAS_LIB_DIR}")
  list(APPEND CBLAS_LINK "-L${CBLAS_LIB_DIR}")
  foreach(lib ${BLAS_LIBS})
    list(APPEND CBLAS_LINK -l${lib})
  endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS REQUIRED_VARS CBLAS_INCLUDE CBLAS_LINK)

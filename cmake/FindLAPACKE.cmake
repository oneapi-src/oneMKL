#===============================================================================
# Copyright 2021 Intel Corporation
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

find_library(LAPACKE64_file NAMES lapacke64.dll.lib lapacke64.lib lapacke HINTS ${REF_LAPACK_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(LAPACKE REQUIRED_VARS LAPACKE64_file)
find_library(LAPACK64_file NAMES lapack64.dll.lib lapack64.lib lapack HINTS ${REF_LAPACK_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(LAPACKE REQUIRED_VARS LAPACK64_file)
find_library(CBLAS64_file NAMES cblas64.dll.lib cblas64.lib cblas HINTS ${REF_LAPACK_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(LAPACKE REQUIRED_VARS CBLAS64_file)
find_library(BLAS64_file NAMES blas64.dll.lib blas64.lib blas HINTS ${REF_LAPACK_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(LAPACKE REQUIRED_VARS BLAS64_file)

get_filename_component(LAPACKE64_LIB_DIR ${LAPACKE64_file} DIRECTORY)
find_path(LAPACKE_INCLUDE lapacke.h HINTS ${REF_LAPACK_ROOT} PATH_SUFFIXES include)

if(UNIX)
    list(APPEND LAPACKE_LINK "-Wl,-rpath,${LAPACKE64_LIB_DIR}")
endif()
list(APPEND LAPACKE_LINK ${LAPACKE64_file})
list(APPEND LAPACKE_LINK ${LAPACK64_file})
list(APPEND LAPACKE_LINK ${CBLAS64_file})
list(APPEND LAPACKE_LINK ${BLAS64_file})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE REQUIRED_VARS LAPACKE_INCLUDE LAPACKE_LINK)

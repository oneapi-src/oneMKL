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

find_library(NETLIB_CBLAS_LIBRARY NAMES cblas.dll.lib cblas.lib cblas HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(NETLIB REQUIRED_VARS NETLIB_CBLAS_LIBRARY)
find_library(NETLIB_BLAS_LIBRARY NAMES blas.dll.lib blas.lib blas HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES lib lib64)
find_package_handle_standard_args(NETLIB REQUIRED_VARS NETLIB_BLAS_LIBRARY)

get_filename_component(NETLIB_LIB_DIR ${NETLIB_CBLAS_LIBRARY} DIRECTORY)
find_path(NETLIB_INCLUDE cblas.h HINTS ${REF_BLAS_ROOT} PATH_SUFFIXES include)

if(UNIX)
  list(APPEND NETLIB_LINK "-Wl,-rpath,${NETLIB_LIB_DIR}")
endif()
list(APPEND NETLIB_LINK ${NETLIB_CBLAS_LIBRARY})
list(APPEND NETLIB_LINK ${NETLIB_BLAS_LIBRARY})


set(NETLIB_COPT "")

if(UNIX)
  set(NETLIB_COPT "")
else()
  if(${BUILD_SHARED_LIBS})
    set(NETLIB_COPT ${NETLIB_COPT} "-Donemkl_EXPORTS")
  endif()
endif()

include(FindPackageHandleStandardArgs)
if(UNIX)
  find_package_handle_standard_args(NETLIB REQUIRED_VARS NETLIB_INCLUDE NETLIB_LINK)
else()
  find_package_handle_standard_args(NETLIB REQUIRED_VARS NETLIB_INCLUDE NETLIB_COPT NETLIB_LINK)
endif()

add_library(ONEMKL::NETLIB::NETLIB UNKNOWN IMPORTED)
set_target_properties(ONEMKL::NETLIB::NETLIB PROPERTIES IMPORTED_LOCATION ${NETLIB_CBLAS_LIBRARY})


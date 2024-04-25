#==========================================================================
#  Copyright 2020-2022 Intel Corporation
#  Copyright (C) Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake"
)

find_package(HIP QUIET)
find_package(rocblas REQUIRED)
set(ROCBLAS_VERSION ${rocblas_VERSION})

# this is work around to avoid duplication half creation in both HIP and SYCL
add_compile_definitions(HIP_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocBLAS
    REQUIRED_VARS
      HIP_INCLUDE_DIRS
      HIP_LIBRARIES
      ROCBLAS_INCLUDE_DIR
      ROCBLAS_LIBRARIES
    VERSION_VAR
      ROCBLAS_VERSION
)

if (DEFINED rocblas_INCLUDE_DIR)
  set(ROCBLAS_LIB_PATH "${rocblas_INCLUDE_DIR}/../lib/librocblas.so")
else()
  set(ROCBLAS_LIB_PATH "${HIP_PATH}/../rocblas/lib/librocblas.so")
endif()

# OPENCL_INCLUDE_DIR
if(NOT TARGET ONEMKL::rocBLAS::rocBLAS)
  add_library(ONEMKL::rocBLAS::rocBLAS SHARED IMPORTED)
  set_target_properties(ONEMKL::rocBLAS::rocBLAS PROPERTIES
      IMPORTED_LOCATION "${ROCBLAS_LIB_PATH}"
      INTERFACE_INCLUDE_DIRECTORIES "${ROCBLAS_INCLUDE_DIR};${HIP_INCLUDE_DIRS};"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;${ROCBLAS_LIBRARIES};"
  )

endif()

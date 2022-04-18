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

#Build object from all sources
set(BLAS_CT_SOURCES "")
if(ENABLE_MKLCPU_BACKEND)
  list(APPEND BLAS_CT_SOURCES "gemm_usm_mklcpu")
endif()
if(ENABLE_MKLGPU_BACKEND)
  list(APPEND BLAS_CT_SOURCES "gemm_usm_mklgpu")
endif()
if(ENABLE_CUBLAS_BACKEND)
  list(APPEND BLAS_CT_SOURCES "gemm_usm_cublas")
endif()
if(ENABLE_ROCBLAS_BACKEND)
  list(APPEND BLAS_CT_SOURCES "gemm_usm_rocblas")
endif()

foreach(blas_ct_source ${BLAS_CT_SOURCES})
  add_executable(example_${domain}_${blas_ct_source} ${blas_ct_source}.cpp)
  target_compile_options(example_${domain}_${blas_ct_source} PRIVATE  -DNOMINMAX)
  target_include_directories(example_${domain}_${blas_ct_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  if(ENABLE_MKLCPU_BACKEND)
    add_dependencies(example_${domain}_${blas_ct_source} onemkl_${domain}_mklcpu)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklcpu)
  endif()

  if(ENABLE_MKLGPU_BACKEND)
    add_dependencies(example_${domain}_${blas_ct_source} onemkl_${domain}_mklgpu)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklgpu)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_CUBLAS_BACKEND)
    add_dependencies(example_${domain}_${blas_ct_source} onemkl_${domain}_cublas)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_cublas)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_NETLIB_BACKEND)
    add_dependencies(example_${domain}_${blas_ct_source} onemkl_${domain}_netlib)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_netlib)
  endif()

  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    # add_sycl_to_target(TARGET example_${domain}_${blas_ct_source} SOURCES ${BLAS_CT_SOURCES})
  else()
    target_link_libraries(example_${domain}_${blas_ct_source} PUBLIC
        ${ONEMKL_LIBRARIES_${domain}}
        ONEMKL::SYCL::SYCL
    )
  endif()

  # Register example as ctest
  add_test(NAME example_${domain}_${blas_ct_source} COMMAND example_${domain}_${blas_ct_source})

endforeach(blas_ct_source)

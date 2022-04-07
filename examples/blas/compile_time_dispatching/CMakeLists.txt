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
if(ENABLE_MKLCPU_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_mklcpu.cpp")
endif()
if(ENABLE_MKLGPU_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_mklgpu.cpp")
endif()
if(ENABLE_CUBLAS_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_cublas.cpp")
endif()
if(ENABLE_ROCBLAS_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_rocblas.cpp")
endif()

add_executable(example_blas_source_ct ${BLAS_CT_SOURCES})
target_compile_options(example_blas_source_ct PRIVATE  -DNOMINMAX)
target_include_directories(example_blas_source_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
)

if(ENABLE_MKLCPU_BACKEND)
  add_dependencies(example_blas_source_ct onemkl_${domain}_mklcpu)
  list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklcpu)
endif()

if(ENABLE_MKLGPU_BACKEND)
  add_dependencies(example_blas_source_ct onemkl_${domain}_mklgpu)
  list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklgpu)
endif()

if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET example_blas_source_ct SOURCES ${BLAS_CT_SOURCES})
else()
  target_link_libraries(example_blas_source_ct PUBLIC
      ${CMAKE_BINARY_DIR}/lib/libonemkl_blas_mklgpu.a
      ${ONEMKL_LIBRARIES_${domain}}
      ONEMKL::SYCL::SYCL
  )
endif()

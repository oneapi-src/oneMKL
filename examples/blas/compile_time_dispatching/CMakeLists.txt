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
elseif(ENABLE_MKLGPU_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_mklgpu.cpp")
elseif(ENABLE_CUBLAS_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_cublas.cpp")
elseif(ENABLE_ROCBLAS_BACKEND)
  set(BLAS_CT_SOURCES "gemm_usm_rocblas.cpp")
endif()

add_library(blas_source_ct OBJECT ${BLAS_CT_SOURCES}))
target_compile_options(blas_source_ct PRIVATE  -DNOMINMAX)
target_include_directories(blas_source_ct
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_BINARY_DIR}/bin
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET blas_source_ct SOURCES ${BLAS_CT_SOURCES})
else()
  target_link_libraries(blas_source_ct PUBLIC ONEMKL::SYCL::SYCL)
endif()

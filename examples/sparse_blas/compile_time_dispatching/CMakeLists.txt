#===============================================================================
# Copyright 2023 Intel Corporation
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
set(SPARSE_BLAS_BACKENDS "")

if(ENABLE_MKLCPU_BACKEND)
  list(APPEND SPARSE_BLAS_BACKENDS "mklcpu")
endif()

include(WarningsUtils)

foreach(backend ${SPARSE_BLAS_BACKENDS})
  set(EXAMPLE_NAME example_sparse_blas_gemv_usm_${backend})
  add_executable(${EXAMPLE_NAME} sparse_blas_gemv_usm_${backend}.cpp)
  target_include_directories(${EXAMPLE_NAME}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  add_dependencies(${EXAMPLE_NAME} onemkl_sparse_blas_${backend})
  target_link_libraries(${EXAMPLE_NAME} PRIVATE ONEMKL::SYCL::SYCL onemkl_sparse_blas_${backend})

  # Register example as ctest
  add_test(NAME sparse_blas/EXAMPLE/CT/sparse_blas_gemv_usm_${backend} COMMAND ${EXAMPLE_NAME})
endforeach(backend)


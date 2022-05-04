#===============================================================================
# Copyright 2022 Intel Corporation
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
set(LAPACK_CT_SOURCES "")
if(ENABLE_MKLCPU_BACKEND AND ENABLE_CUSOLVER_BACKEND)
  list(APPEND LAPACK_CT_SOURCES "getrs_usm_mklcpu_cusolver")
endif()

if(domain STREQUAL "lapack" AND ENABLE_MKLCPU_BACKEND)
  find_library(OPENCL_LIBRARY NAMES OpenCL)
  message(STATUS "Found OpenCL: ${OPENCL_LIBRARY}")
endif()

foreach(lapack_ct_source ${LAPACK_CT_SOURCES})
  add_executable(example_${domain}_${lapack_ct_source} ${lapack_ct_source}.cpp)
  target_compile_options(example_${domain}_${lapack_ct_source} PRIVATE  -DNOMINMAX)
  target_include_directories(example_${domain}_${lapack_ct_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )
  if(domain STREQUAL "lapack" AND ENABLE_MKLCPU_BACKEND AND ENABLE_CUSOLVER_BACKEND)
    add_dependencies(example_${domain}_${lapack_ct_source} onemkl_${domain}_mklcpu onemkl_${domain}_cusolver)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklcpu onemkl_${domain}_cusolver)
    target_link_libraries(example_${domain}_${lapack_ct_source} PUBLIC ${OPENCL_LIBRARY})
  endif()
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    # Uncomment this after testing with hipSYCL
    # add_sycl_to_target(TARGET example_${domain}_${lapack_ct_source} SOURCES ${LAPACK_CT_SOURCES})
  else()
    target_link_libraries(example_${domain}_${lapack_ct_source} PUBLIC
        ${ONEMKL_LIBRARIES_${domain}}
        ONEMKL::SYCL::SYCL
    )
  endif()
  # Register example as ctest
 add_test(NAME ${domain}/EXAMPLE/CT/${lapack_ct_source} COMMAND example_${domain}_${lapack_ct_source})
endforeach(lapack_ct_source)

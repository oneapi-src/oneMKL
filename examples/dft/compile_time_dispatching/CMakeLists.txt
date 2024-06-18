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
set(DFTI_CT_SOURCES "")
if (ENABLE_MKLCPU_BACKEND AND ENABLE_CUFFT_BACKEND)
  list(APPEND DFTI_CT_SOURCES "complex_fwd_usm_mklcpu_cufft")
endif()

foreach(dfti_ct_source ${DFTI_CT_SOURCES})
  set(EXAMPLE_NAME example_${domain}_${dfti_ct_source})
  add_executable(${EXAMPLE_NAME} ${dfti_ct_source}.cpp)
  target_include_directories(${EXAMPLE_NAME}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

if(domain STREQUAL "dft" AND ENABLE_MKLCPU_BACKEND AND ENABLE_CUFFT_BACKEND)
  add_dependencies(${EXAMPLE_NAME} onemkl_${domain}_mklcpu onemkl_${domain}_cufft)
  list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklcpu onemkl_${domain}_cufft)
endif()

target_link_libraries(${EXAMPLE_NAME} PUBLIC
  ${ONEMKL_LIBRARIES_${domain}}
  ONEMKL::SYCL::SYCL
  )

  # Register example as ctest
  add_test(NAME dft/EXAMPLE/CT/${dfti_ct_source} COMMAND ${EXAMPLE_NAME})

endforeach(dfti_ct_source)


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
set(DFTI_CT_SOURCES "")
if(ENABLE_MKLCPU_BACKEND)
  list(APPEND DFTI_CT_SOURCES "complex_fwd_usm_mklcpu")
endif()

if(domain STREQUAL "dft" AND ENABLE_MKLCPU_BACKEND)
  find_library(OPENCL_LIBRARY NAMES OpenCL)
  message(STATUS "Found OpenCL: ${OPENCL_LIBRARY}")
endif()

foreach(dfti_ct_sources ${DFTI_CT_SOURCES})
  add_executable(example_${domain}_${dfti_ct_sources} ${dfti_ct_sources}.cpp)
  target_include_directories(example_${domain}_${dfti_ct_sources}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )
  if(domain STREQUAL "dft" AND ENABLE_MKLCPU_BACKEND)
    add_dependencies(example_${domain}_${dfti_ct_sources} onemkl_${domain}_mklcpu)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklcpu)
    target_link_libraries(example_${domain}_${dfti_ct_sources} PUBLIC ${OPENCL_LIBRARY})
  endif()
  target_link_libraries(example_${domain}_${dfti_ct_sources} PUBLIC
      ${ONEMKL_LIBRARIES_${domain}}
      ONEMKL::SYCL::SYCL
  )
  # Register example as ctest
 add_test(NAME ${domain}/EXAMPLE/CT/${dfti_ct_sources} COMMAND example_${domain}_${dfti_ct_sources})
endforeach(dfti_ct_sources)

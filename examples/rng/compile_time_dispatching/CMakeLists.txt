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
set(RNG_CT_SOURCES "")
if(ENABLE_MKLCPU_BACKEND AND ENABLE_CURAND_BACKEND)
  list(APPEND RNG_CT_SOURCES "uniform_usm_mklcpu_curand")
endif()

foreach(rng_ct_source ${RNG_CT_SOURCES})
  add_executable(example_${domain}_${rng_ct_source} ${rng_ct_source}.cpp)
  target_compile_options(example_${domain}_${rng_ct_source} PRIVATE  -DNOMINMAX)
  target_include_directories(example_${domain}_${rng_ct_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  if(domain STREQUAL "rng" AND ENABLE_MKLCPU_BACKEND AND ENABLE_CURAND_BACKEND)
    add_dependencies(example_${domain}_${rng_ct_source}
        onemkl_${domain}_mklcpu
        onemkl_${domain}_curand)
    list(APPEND ONEMKL_LIBRARIES_${domain}
        onemkl_${domain}_mklcpu
        onemkl_${domain}_curand)
  endif()

  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    # Uncomment this after testing with hipSYCL
    # add_sycl_to_target(TARGET example_${domain}_${rng_ct_source} SOURCES ${RNG_CT_SOURCES})
  else()
    target_link_libraries(example_${domain}_${rng_ct_source} PUBLIC
        ${ONEMKL_LIBRARIES_${domain}}
        ONEMKL::SYCL::SYCL
    )
  endif()

  # Register example as ctest
  add_test(NAME ${domain}/EXAMPLE/CT/${rng_ct_source} COMMAND example_${domain}_${rng_ct_source})
endforeach(rng_ct_source)

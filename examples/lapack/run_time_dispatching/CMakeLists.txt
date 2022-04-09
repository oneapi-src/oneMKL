
#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

# NOTE: user need to set env var SYCL_DEVICE_FILTER to use runtime example without specifying backend in CMake

# Build object from all example sources
set(LAPACK_RT_SOURCES "getrf_usm")

foreach(lapack_rt_source ${LAPACK_RT_SOURCES})
  add_executable(example_${domain}_${lapack_rt_source} ${lapack_rt_source}.cpp)
  target_compile_options(example_${domain}_${lapack_rt_source} PRIVATE -DNOMINMAX)
  target_include_directories(example_${domain}_${lapack_rt_source}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  add_dependencies(example_${domain}_${lapack_rt_source} onemkl)

  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    # add_sycl_to_target(TARGET example_${domain}_${lapack_rt_source} SOURCES ${LAPACK_RT_SOURCES})
  else()
    target_link_libraries(example_${domain}_${lapack_rt_source} PUBLIC
        onemkl
        ONEMKL::SYCL::SYCL
    )
  endif()
endforeach(lapack_rt_source)

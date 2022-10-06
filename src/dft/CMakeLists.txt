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

# Build backends
add_subdirectory(backends)

# Recipe for DFT loader object
if(BUILD_SHARED_LIBS)
add_library(onemkl_dft OBJECT)
target_sources(onemkl_dft PRIVATE dft_loader.cpp)
target_include_directories(onemkl_dft
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
          ${PROJECT_SOURCE_DIR}/src/include
          ${CMAKE_BINARY_DIR}/bin
          $<TARGET_FILE_DIR:onemkl>
)

target_compile_options(onemkl_dft PRIVATE ${ONEMKL_BUILD_COPT})

set_target_properties(onemkl_dft PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET onemkl_dft SOURCES dft_loader.cpp)
else()
  target_link_libraries(onemkl_dft PUBLIC ONEMKL::SYCL::SYCL)
endif()

endif()

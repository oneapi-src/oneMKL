#===============================================================================
# Copyright 2020 Intel Corporation
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

# build blas_loader and backends
# add_subdirectory(blas)

# build rng_loader and backends
add_subdirectory(rng)

# generate header with enabled backends for testing
configure_file(config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/config.hpp.configured")
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/detail/config.hpp"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/config.hpp.configured"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/oneapi/mkl/detail/config.hpp"
  DESTINATION include/oneapi/mkl/detail
  COMPONENT Devel
)

# Add recipe for onemkl loader library
if(BUILD_SHARED_LIBS)
add_library(onemkl SHARED)

target_include_directories(onemkl
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<INSTALL_INTERFACE:include>
)
set_target_properties(onemkl PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Build dispatcher library
# target_link_libraries(onemkl PUBLIC onemkl_blas onemkl_rng)
target_link_libraries(onemkl PUBLIC onemkl_rng)

# Add the library to install package
# install(TARGETS onemkl_blas EXPORT oneMKLTargets)
install(TARGETS onemkl_rng EXPORT oneMKLTargets)
install(TARGETS onemkl EXPORT oneMKLTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
endif()

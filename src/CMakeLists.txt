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

# Define common build flags for oneMKL libraries
set(ONEMKL_BUILD_COPT "")
if(WIN32 AND BUILD_SHARED_LIBS)
  list(APPEND ONEMKL_BUILD_COPT "-Donemkl_EXPORTS")
endif()

# SYCL-BLAS backend variables must be accessible here to correctly
# generate the config file.
set(ENABLE_SYCLBLAS_BACKEND_INTEL_CPU OFF CACHE INTERNAL "")
set(ENABLE_SYCLBLAS_BACKEND_INTEL_GPU OFF CACHE INTERNAL "")
set(ENABLE_SYCLBLAS_BACKEND_AMD_GPU OFF CACHE INTERNAL "")
set(ENABLE_SYCLBLAS_BACKEND_NVIDIA_GPU OFF CACHE INTERNAL "")

# Build loader and backends for each domain
foreach(domain ${TARGET_DOMAINS})
  add_subdirectory(${domain})
endforeach()

# Generate header with enabled backends for testing
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
  set (ONEMKL_LIBS ${TARGET_DOMAINS})
  list(TRANSFORM ONEMKL_LIBS PREPEND onemkl_)
  target_link_libraries(onemkl PUBLIC ${ONEMKL_LIBS})

  # Add the library to install package
  foreach(domain_lib ${ONEMKL_LIBS})
    install(TARGETS ${domain_lib} EXPORT oneMKLTargets)
  endforeach()
  install(TARGETS onemkl EXPORT oneMKLTargets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )
endif()

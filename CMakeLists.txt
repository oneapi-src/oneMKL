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

cmake_minimum_required (VERSION 3.13)

project(oneMKL VERSION 0.1.0 LANGUAGES CXX)

# Default options. Override by defining in command
## Build type
option(BUILD_DEBUG "" OFF)

if(BUILD_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)
## Backends
option(ENABLE_MKLCPU_BACKEND "" ON)
option(ENABLE_MKLGPU_BACKEND "" ON)
if(ENABLE_MKLCPU_BACKEND)
    option(ENABLE_MKLCPU_THREAD_TBB "" ON)
endif()
## Testing
option(BUILD_FUNCTIONAL_TESTS "" ON)
## Documentation
option(BUILD_DOC "" OFF)

# Set compilation flags
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set output directories for the project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add CMake Finders
add_subdirectory(cmake)

# Include general cmake config files
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Find necessary packages
find_package(Compiler REQUIRED)

# Add source directory and output to bin/
add_subdirectory(src bin)

# Functional Tests
if(BUILD_FUNCTIONAL_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_DOC)
  add_subdirectory(docs)
endif()

install(DIRECTORY include/
  DESTINATION include
  COMPONENT Devel
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/oneMKLConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT oneMKLTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/oneMKLTargets.cmake"
  NAMESPACE ONEMKL::
)
configure_file("${PROJECT_SOURCE_DIR}/cmake/oneMKLConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/oneMKLConfig.cmake"
  COPYONLY
)

set(config_package_location "lib/cmake/${PROJECT_NAME}")
install(EXPORT oneMKLTargets
  FILE oneMKLTargets.cmake
  NAMESPACE MKL::
  DESTINATION ${config_package_location}
)
install(
  FILES
  "${PROJECT_SOURCE_DIR}/cmake/oneMKLConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/oneMKLConfigVersion.cmake"
  DESTINATION ${config_package_location}
  COMPONENT Devel
)

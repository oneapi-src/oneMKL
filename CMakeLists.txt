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

cmake_minimum_required (VERSION 3.13)

# Set flags from Conan if Conan is used
if(EXISTS "${CMAKE_BINARY_DIR}/conanbuildinfo.cmake")
  include("${CMAKE_BINARY_DIR}/conanbuildinfo.cmake" REQUIRED)
  conan_basic_setup()
endif()

# Define build type
set(DEFAULT_BUILD_TYPE "Release")

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE: None, set to ${DEFAULT_BUILD_TYPE} by default")
    set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE} CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)
else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# Build options
option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

## Backends
option(ENABLE_MKLCPU_BACKEND "" ON)
option(ENABLE_MKLGPU_BACKEND "" ON)
if(ENABLE_MKLCPU_BACKEND)
  option(ENABLE_MKLCPU_THREAD_TBB "" ON)
endif()
option(ENABLE_CUBLAS_BACKEND "" OFF)
option(ENABLE_CURAND_BACKEND "" OFF)
option(ENABLE_NETLIB_BACKEND "" OFF)
set(ONEMKL_SYCL_IMPLEMENTATION "dpc++" CACHE STRING "Name of the SYCL compiler")

## Testing
option(BUILD_FUNCTIONAL_TESTS "" ON)

## Documentation
option(BUILD_DOC "" OFF)

## Supported domains
set(DOMAINS_LIST "")
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND
        OR ENABLE_CUBLAS_BACKEND
        OR ENABLE_NETLIB_BACKEND)
  list(APPEND DOMAINS_LIST "blas")
endif()
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND)
  list(APPEND DOMAINS_LIST "lapack")
endif()
if(ENABLE_MKLCPU_BACKEND
        OR ENABLE_MKLGPU_BACKEND
        OR ENABLE_CURAND_BACKEND)
  list(APPEND DOMAINS_LIST "rng")
endif()

# Define required CXX compilers before project
if(CMAKE_CXX_COMPILER OR NOT ONEMKL_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  if(WIN32)
    string(REPLACE "\\" "/" CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
else()
  if(ENABLE_CUBLAS_BACKEND OR ENABLE_CURAND_BACKEND)
    set(CMAKE_CXX_COMPILER "clang++")
  elseif(ENABLE_MKLGPU_BACKEND)
    set(CMAKE_CXX_COMPILER "dpcpp")
  else()
    find_program(DPCPP_PATH dpcpp)
    if(DPCPP_PATH)
      set(CMAKE_CXX_COMPILER "dpcpp")
    else()
      if(WIN32)
        message(STATUS "dpcpp was not found in PATH, using clang-cl instead")
        set(CMAKE_CXX_COMPILER "clang-cl")
      else()
        message(STATUS "dpcpp was not found in PATH, using clang++ instead")
        set(CMAKE_CXX_COMPILER "clang++")
      endif()
    endif()
  endif()
endif()

# Define required C compilers for Windows before project
if(WIN32 AND ONEMKL_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  set(CMAKE_C_COMPILER "clang-cl")
endif()

project(oneMKL VERSION 0.2.0 LANGUAGES CXX)

# Override default CXX compile/link lines for Windows after project
if(WIN32 AND ONEMKL_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -w")
  foreach (flag_var
           CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
           CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REPLACE "/MD" "" ${flag_var} "${${flag_var}}")
  endforeach()
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fsycl /nologo <DEFINES> <INCLUDES> /EHsc <FLAGS> /Fo<OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib /nologo <OBJECTS> /out:<TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl /nologo <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

# Temporary disable sycl 2020 deprecations warnings
if(ONEMKL_SYCL_IMPLEMENTATION STREQUAL "dpc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL2020_DISABLE_DEPRECATION_WARNINGS")
endif()

# Target domains
if(NOT TARGET_DOMAINS OR TARGET_DOMAINS STREQUAL "None")
  # Set to all by default
  set(TARGET_DOMAINS ${DOMAINS_LIST})
else()
  # Make sure the input was converted to list
  string(REPLACE " " ";" TARGET_DOMAINS ${TARGET_DOMAINS})
  set(NOT_FOUND 0)
  foreach(domain ${TARGET_DOMAINS})
    if(NOT ${domain} IN_LIST DOMAINS_LIST)
      set(NOT_FOUND 1)
      break()
    endif()
  endforeach()
  if(NOT_FOUND)
    message(STATUS "TARGET_DOMAINS contains unsupported options, reset to all")
    set(TARGET_DOMAINS ${DOMAINS_LIST})
  endif()
endif()
message(STATUS "TARGET_DOMAINS: ${TARGET_DOMAINS}")

# Set output directories for the project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add CMake Finders
add_subdirectory(cmake)

# Include general cmake config files
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Add DPC++ options for Linux
if(WIN32)
  add_library(ONEMKL::SYCL::SYCL INTERFACE IMPORTED)
else()
  # Find necessary packages
  if(ONEMKL_SYCL_IMPLEMENTATION)
    string( TOLOWER "${ONEMKL_SYCL_IMPLEMENTATION}" ONEMKL_SYCL_IMPLEMENTATION)
    if (ONEMKL_SYCL_IMPLEMENTATION STREQUAL "hipsycl")
      message(STATUS "Looking for hipSYCL")
      find_package(hipSYCL CONFIG REQUIRED)
      set(USE_ADD_SYCL_TO_TARGET_INTEGRATION true)
      add_library(ONEMKL::SYCL::SYCL INTERFACE IMPORTED)
    elseif(ONEMKL_SYCL_IMPLEMENTATION STREQUAL "dpc++")
      message(STATUS "Looking for dpc++")
      set(USE_ADD_SYCL_TO_TARGET_INTEGRATION false)
      find_package(Compiler REQUIRED)
    else()
      message(FATAL_ERROR "SYCL implementation ${ONEMKL_SYCL_IMPLEMENTATION} is not known")
    endif()
  else()
    message(STATUS "Looking for dpc++")
    set(USE_ADD_SYCL_TO_TARGET_INTEGRATION false)
    find_package(Compiler REQUIRED)
  endif()
endif()

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

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

if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed") 
  else() 
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed") 
  endif() 
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH}) 
list(APPEND CMAKE_PREFIX_PATH 
	"${HIP_PATH}/lib/cmake"
	"${HIP_PATH}/../lib/cmake"
	"${HIP_PATH}/../lib/cmake/rocrand/rocrand")

find_package(rocrand REQUIRED)
find_package(hip QUIET)

# this is work around to avoid duplication half creation in both hip and SYCL
add_compile_definitions(HIP_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocRAND
  REQUIRED_VARS
      HIP_INCLUDE_DIRS
      rocrand_INCLUDE_DIR
      rocrand_LIBRARIES)

if(NOT TARGET ONEMKL::rocRAND::rocRAND)
  add_library(ONEMKL::rocRAND::rocRAND SHARED IMPORTED)
  set_target_properties(ONEMKL::rocRAND::rocRAND PROPERTIES
      IMPORTED_LOCATION "${HIP_PATH}/../rocrand/lib/librocrand.so"
      INTERFACE_INCLUDE_DIRECTORIES "${rocrand_INCLUDE_DIR};${HIP_INCLUDE_DIRS};"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;hip::host;${rocrand_LIBRARIES};")
endif()

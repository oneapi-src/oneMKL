#===============================================================================
# Copyright 2020-2022 Intel Corporation
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

if("blas" IN_LIST TARGET_DOMAINS)
  find_package(CBLAS REQUIRED)
endif()

if("lapack" IN_LIST TARGET_DOMAINS)
    find_package(LAPACKE REQUIRED)
endif()

foreach(domain ${TARGET_DOMAINS})
  # Build tests first
  add_subdirectory(${domain})
endforeach()

include(GoogleTest)

get_target_property(GTEST_INCLUDE_DIR gtest INTERFACE_INCLUDE_DIRECTORIES)

# Build final test binaries: test_main_rt is for testing RunTime API (RT), test_main_ct is for testing CompileTime API (CT)

# BLAS config
set(blas_TEST_LIST
        blas_level1
        blas_level2
        blas_level3
        blas_batch
        blas_extensions)

set(blas_TEST_LINK ${CBLAS_LINK})

# LAPACK config
set(lapack_TEST_LIST
        lapack_source)

set(lapack_TEST_LINK ${LAPACKE_LINK})

# RNG config
set(rng_TEST_LIST
      rng_statistics
      rng_service)

set(rng_TEST_LINK "")

foreach(domain ${TARGET_DOMAINS})
  # Generate RT and CT test lists
  set(${domain}_TEST_LIST_RT ${${domain}_TEST_LIST})
  set(${domain}_TEST_LIST_CT ${${domain}_TEST_LIST})
  list(TRANSFORM ${domain}_TEST_LIST_RT APPEND _rt)
  list(TRANSFORM ${domain}_TEST_LIST_CT APPEND _ct)

  add_executable(test_main_${domain}_ct main_test.cpp)
  target_include_directories(test_main_${domain}_ct PUBLIC ${GTEST_INCLUDE_DIR})
  
  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET test_main_${domain}_ct SOURCES main_test.cpp)
  else()
    target_compile_options(test_main_${domain}_ct PRIVATE -fsycl)
  endif()

  if(BUILD_SHARED_LIBS)
    add_executable(test_main_${domain}_rt main_test.cpp)
    target_include_directories(test_main_${domain}_rt PUBLIC ${GTEST_INCLUDE_DIR})
    target_compile_options(test_main_${domain}_rt PRIVATE -fsycl)
    target_link_libraries(test_main_${domain}_rt PUBLIC
      gtest
      gtest_main
      ${CMAKE_DL_LIBS}
      ${${domain}_TEST_LINK}
      ONEMKL::SYCL::SYCL
      onemkl
      ${${domain}_TEST_LIST_RT}
    )
    if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
      add_sycl_to_target(TARGET test_main_${domain}_rt SOURCES main_test.cpp)
    endif()
  endif()

  if(ENABLE_MKLCPU_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_mklcpu)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklcpu)
  endif()

  if(ENABLE_MKLGPU_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_mklgpu)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_mklgpu)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_CUBLAS_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_cublas)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_cublas)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_ROCBLAS_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_rocblas)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_rocblas)
  endif()

  if(domain STREQUAL "blas" AND ENABLE_NETLIB_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_netlib)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_netlib)
  endif()

  if(domain STREQUAL "lapack" AND ENABLE_CUSOLVER_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_cusolver)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_cusolver)
  endif()

  if(domain STREQUAL "lapack" AND ENABLE_ROCSOLVER_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_rocsolver)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_rocsolver)
  endif()
  
  if(domain STREQUAL "rng" AND ENABLE_CURAND_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_curand)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_curand)
  endif()

  if(domain STREQUAL "rng" AND ENABLE_ROCRAND_BACKEND)
    add_dependencies(test_main_${domain}_ct onemkl_${domain}_rocrand)
    list(APPEND ONEMKL_LIBRARIES_${domain} onemkl_${domain}_rocrand)
  endif()

  target_link_libraries(test_main_${domain}_ct PUBLIC
      gtest
      gtest_main
      ${CMAKE_DL_LIBS}
      ${${domain}_TEST_LINK}
      ${ONEMKL_LIBRARIES_${domain}}
      ONEMKL::SYCL::SYCL
      ${${domain}_TEST_LIST_CT}
  )

  string(TOUPPER ${domain} DOMAIN_PREFIX)

  if(BUILD_SHARED_LIBS)
    set_target_properties(test_main_${domain}_rt
      PROPERTIES BUILD_RPATH $<TARGET_FILE_DIR:onemkl>)
  # Find individual tests within executable
    gtest_discover_tests(test_main_${domain}_rt
      PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
      PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
      PROPERTIES TEST_PREFIX ${DOMAIN_PREFIX}/RT/
      DISCOVERY_TIMEOUT 30
    )
  endif()

  gtest_discover_tests(test_main_${domain}_ct
    PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
    PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
    PROPERTIES TEST_PREFIX ${DOMAIN_PREFIX}/CT/
    DISCOVERY_TIMEOUT 30
  )

  if(BUILD_SHARED_LIBS)
    if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
      add_sycl_to_target(TARGET test_main_${domain}_rt)
    endif()
  endif()
endforeach()

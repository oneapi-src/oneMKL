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

find_package(CBLAS REQUIRED)

# Build BLAS tests first
add_subdirectory(blas)

include(GoogleTest)

# Build final test binaries: test_main_rt is for testing RunTime API, test_main_ct is for testing CompileTime API
get_target_property(GTEST_INCLUDE_DIR gtest INTERFACE_INCLUDE_DIRECTORIES)

add_executable(test_main_ct main_test.cpp)
target_include_directories(test_main_ct PUBLIC ${GTEST_INCLUDE_DIR})
target_compile_options(test_main_ct PRIVATE -fsycl)

if(BUILD_SHARED_LIBS)
  add_executable(test_main_rt main_test.cpp)
  target_include_directories(test_main_rt PUBLIC ${GTEST_INCLUDE_DIR})
  target_compile_options(test_main_rt PRIVATE -fsycl)
  target_link_libraries(test_main_rt PUBLIC
    gtest
    gtest_main
    ${CMAKE_DL_LIBS}
    ${CBLAS_LINK}
    ONEMKL::SYCL::SYCL
    onemkl
    blas_level1_rt
    blas_level2_rt
    blas_level3_rt
  )
endif()

if(ENABLE_MKLCPU_BACKEND)
  add_dependencies(test_main_ct onemkl_blas_mklcpu)
  if(BUILD_SHARED_LIBS)
    list(APPEND ONEMKL_LIBRARIES onemkl_blas_mklcpu)
  else()
    list(APPEND ONEMKL_LIBRARIES -foffload-static-lib=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonemkl_blas_mklcpu.a)
    find_package(MKL REQUIRED)
    list(APPEND ONEMKL_LIBRARIES ${MKL_LINK_C})
  endif()
endif()

if(ENABLE_MKLGPU_BACKEND)
  add_dependencies(test_main_ct onemkl_blas_mklgpu)
  if(BUILD_SHARED_LIBS)
    list(APPEND ONEMKL_LIBRARIES onemkl_blas_mklgpu)
  else()
    list(APPEND ONEMKL_LIBRARIES -foffload-static-lib=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonemkl_blas_mklgpu.a)
    find_package(MKL REQUIRED)
    list(APPEND ONEMKL_LIBRARIES ${MKL_LINK_SYCL})
  endif()
endif()

if(ENABLE_CUBLAS_BACKEND)
  add_dependencies(test_main_ct onemkl_blas_cublas)
  if(BUILD_SHARED_LIBS)
    list(APPEND ONEMKL_LIBRARIES onemkl_blas_cublas)
  else()
    list(APPEND ONEMKL_LIBRARIES -foffload-static-lib=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonemkl_blas_cublas.a)
    find_package(cuBLAS REQUIRED)
    list(APPEND ONEMKL_LIBRARIES ONEMKL::cuBLAS::cuBLAS)
  endif()
endif()

target_link_libraries(test_main_ct PUBLIC
    gtest
    gtest_main
    ${CMAKE_DL_LIBS}
    ${CBLAS_LINK}
    ${ONEMKL_LIBRARIES}
    ONEMKL::SYCL::SYCL
    blas_level1_ct
    blas_level2_ct
    blas_level3_ct
)

if(BUILD_SHARED_LIBS)
  set_target_properties(test_main_rt
    PROPERTIES BUILD_RPATH $<TARGET_FILE_DIR:onemkl>)
# Find individual tests within executable
  gtest_discover_tests(test_main_rt
    PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
    PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
  )
endif()

gtest_discover_tests(test_main_ct
  PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
  PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
)

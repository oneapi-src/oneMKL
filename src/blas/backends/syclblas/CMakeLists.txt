#==========================================================================
#  Copyright (C) Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

set(LIB_NAME onemkl_blas_syclblas)
set(LIB_OBJ ${LIB_NAME}_obj)

find_package(sycl_blas REQUIRED)

set(SOURCES
  syclblas_level1_double.cpp syclblas_level1_float.cpp
  syclblas_level2_double.cpp syclblas_level2_float.cpp
  syclblas_level3_double.cpp syclblas_level3_float.cpp 
  syclblas_level3_half.cpp syclblas_level3_bfloat16.cpp 
  syclblas_batch.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: syclblas_wrappers.cpp>)
add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})

# Tune SYCL-BLAS for INTEL_GPU only.
# Once we support more platforms this define can be removed to allow users to tune for any platform.
target_compile_definitions(${LIB_OBJ} PUBLIC INTEL_GPU)

if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMKL_BUILD_COPT})
target_link_libraries(${LIB_OBJ} PUBLIC ONEMKL::SYCL::SYCL SYCL_BLAS::sycl_blas)

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})

if(BUILD_SHARED_LIBS)
  set_target_properties(${LIB_NAME} PROPERTIES
    INTERFACE_LINK_LIBRARIES ONEMKL::SYCL::SYCL
  )
endif()

# Add major version to the library
set_target_properties(${LIB_NAME} PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${LIB_NAME}>)

# Add the library to install package
install(TARGETS ${LIB_OBJ} EXPORT oneMKLTargets)
install(TARGETS ${LIB_NAME} EXPORT oneMKLTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

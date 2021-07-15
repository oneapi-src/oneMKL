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

set(LIB_NAME onemkl_blas_cublas)
set(LIB_OBJ ${LIB_NAME}_obj)
find_package(cuBLAS REQUIRED)
set(SOURCES cublas_level1.cpp 
                cublas_level2.cpp 
                cublas_level3.cpp 
                cublas_batch.cpp 
                cublas_extensions.cpp 
                cublas_scope_handle.cpp 
                $<$<BOOL:${BUILD_SHARED_LIBS}>: mkl_blas_cublas_wrappers.cpp>)
add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
)
target_compile_options(${LIB_OBJ} PRIVATE ${ONEMKL_BUILD_COPT})
target_link_libraries(${LIB_OBJ} PUBLIC ONEMKL::SYCL::SYCL ONEMKL::cuBLAS::cuBLAS)
target_compile_features(${LIB_OBJ} PUBLIC cxx_std_11)
set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
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

#=================================================================================
# Intel Math Kernel Library (oneMKL) Copyright (c) 2021, The Regents of
# the University of California, through Lawrence Berkeley National
# Laboratory (subject to receipt of any required approvals from the U.S.
# Dept. of Energy). All rights reserved.
# 
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at
# IPO@lbl.gov.
# 
# NOTICE.  This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.  As
# such, the U.S. Government has been granted for itself and others acting on
# its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative 
# works, and perform publicly and display publicly, and to permit others to do so.
#=================================================================================

set(LIB_NAME onemkl_rng_curand)
set(LIB_OBJ ${LIB_NAME}_obj)
find_package(cuRAND REQUIRED)

add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT
  philox4x32x10.cpp
  mrg32k3a.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: mkl_rng_curand_wrappers.cpp>
)

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}/bin
          ${MKL_INCLUDE}
)

target_link_libraries(${LIB_OBJ} PUBLIC ONEMKL::SYCL::SYCL ONEMKL::cuRAND::cuRAND)
target_compile_features(${LIB_OBJ} PUBLIC cxx_std_11)
set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})

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

#==========================================================================
#  Copyright 2022 Intel Corporation
#=========================================================================

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
	"${HIP_PATH}/../lib/cmake/rocsolver")

find_package(HIP QUIET) 
find_package(rocsolver REQUIRED)

# this is work around to avoid duplication half creation in both hip and SYCL
add_compile_definitions(HIP_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocSOLVER
  REQUIRED_VARS
      HIP_INCLUDE_DIRS
      rocsolver_INCLUDE_DIR
      rocsolver_LIBRARIES)

if(NOT TARGET ONEMKL::rocSOLVER::rocSOLVER)
	add_library(ONEMKL::rocSOLVER::rocSOLVER SHARED IMPORTED)
	set_target_properties(ONEMKL::rocSOLVER::rocSOLVER PROPERTIES
      IMPORTED_LOCATION "${HIP_PATH}/../rocsolver/lib/librocsolver.so"
      INTERFACE_INCLUDE_DIRECTORIES "${rocsolver_INCLUDE_DIR};${HIP_INCLUDE_DIRS};"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;hip::host;${rocsolver_LIBRARIES};")
endif()


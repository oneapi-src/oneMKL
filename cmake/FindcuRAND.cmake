#--===============================================================================
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

find_package(CUDA 10.0 REQUIRED)
get_filename_component(SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
# the OpenCL include file from cuda is opencl 1.1 and it is not compatible with DPC++
# the OpenCL include headers 1.2 onward is required. This is used to bypass NVIDIA OpenCL headers
find_path(OPENCL_INCLUDE_DIR CL/cl.h OpenCL/cl.h 
HINTS 
${OPENCL_INCLUDE_DIR}
${SYCL_BINARY_DIR}/../include/sycl/
)
# this is work around to avoid duplication half creation in both cuda and SYCL
add_compile_definitions(CUDA_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuRAND
    REQUIRED_VARS
	CUDA_TOOLKIT_INCLUDE
	CUDA_curand_LIBRARY
        CUDA_LIBRARIES
        CUDA_CUDA_LIBRARY
        OPENCL_INCLUDE_DIR
)
if(NOT TARGET ONEMKL::cuRAND::cuRAND)
  add_library(ONEMKL::cuRAND::cuRAND SHARED IMPORTED)
  set_target_properties(ONEMKL::cuRAND::cuRAND PROPERTIES
      IMPORTED_LOCATION ${CUDA_curand_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR};${CUDA_TOOLKIT_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}"
  )

endif()

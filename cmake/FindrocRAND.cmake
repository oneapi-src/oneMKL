#--===============================================================================
# cuRAND back-end Copyright (c) 2021, The Regents of the University of
# California, through Lawrence Berkeley National Laboratory (subject to receipt
# of any required approvals from the U.S. Dept. of Energy). All rights
# reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# (1) Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# (2) Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# (3) Neither the name of the University of California, Lawrence Berkeley
# National Laboratory, U.S. Dept. of Energy nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches,
# or upgrades to the features, functionality or performance of the source
# code ("Enhancements") to anyone; however, if you choose to make your
# Enhancements available either publicly, or directly to Lawrence Berkeley
# National Laboratory, without imposing a separate written license agreement
# for such Enhancements, then you hereby grant the following license: a
# non-exclusive, royalty-free perpetual license to install, use, modify,
# prepare derivative works, incorporate into other computer software,
# distribute, and sublicense such enhancements or derivative works thereof,
# in binary and source code form.
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
# works, and perform publicly and display publicly, and to permit others to do
# so.
#=================================================================================

if (NOT DEFINED ROCM_PATH)
if (NOT DEFINED ENV{ROCM_PATH})
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed") 
else() 
set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed") 
endif() 
endif()

if (NOT DEFINED HIP_PATH)
if (NOT DEFINED ENV{HIP_PATH})
set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed") 
else() 
set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed") 
endif() 
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH}) 
list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake" "${HIP_PATH}/../lib/cmake" "${ROCM_PATH}/rocrand/lib/cmake/rocrand")

#find_package(HIP QUIET)
find_package(hip QUIET)
find_package(rocrand REQUIRED)

get_filename_component(SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
# the OpenCL include file from hip is opencl 1.1 and it is not compatible with DPC++
# the OpenCL include headers 1.2 onward is required. This is used to bypass NVIDIA OpenCL headers
find_path(OPENCL_INCLUDE_DIR CL/cl.h OpenCL/cl.h 
HINTS 
${OPENCL_INCLUDE_DIR}
${SYCL_BINARY_DIR}/../include/sycl/
)
# this is work around to avoid duplication half creation in both hip and SYCL
add_compile_definitions(HIP_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocRAND
    REQUIRED_VARS
      HIP_INCLUDE_DIRS
      HIP_LIBRARIES
      rocrand_INCLUDE_DIR
      rocrand_LIBRARIES
      OPENCL_INCLUDE_DIR
)
if(NOT TARGET ONEMKL::rocRAND::rocRAND)
  add_library(ONEMKL::rocRAND::rocRAND SHARED IMPORTED)
  set_target_properties(ONEMKL::rocRAND::rocRAND PROPERTIES
      IMPORTED_LOCATION "/opt/rocm/lib/librocrand.so"
      INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR};${rocrand_INCLUDE_DIR};${HIP_INCLUDE_DIRS};"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;${rocrand_LIBRARIES};hip::host;"
  )

endif()

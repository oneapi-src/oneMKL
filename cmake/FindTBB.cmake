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

include_guard()

#Workaround for soname problem
if(UNIX)
  set(TBB_LIBNAME libtbb.so)
else()
  set(TBB_LIBNAME tbb.lib)
endif()

find_path(TBB_LIB_DIR ${TBB_LIBNAME}
      HINTS $ENV{TBBROOT} $ENV{MKLROOT} ${MKL_ROOT} ${TBB_ROOT}
      PATH_SUFFIXES "lib" "lib/intel64/gcc4.4" "lib/intel64/gcc4.8"
               "../tbb/lib/intel64/gcc4.4" "../tbb/lib/intel64/gcc4.8"
               "../../tbb/latest/lib/intel64/gcc4.8"
               "../tbb/lib/intel64/vc14"
)

find_library(TBB_LIBRARIES NAMES tbb
        HINTS $ENV{TBBROOT} $ENV{MKLROOT} ${MKL_ROOT} ${TBB_ROOT}
        PATH_SUFFIXES "lib" "lib/intel64/gcc4.4" "lib/intel64/gcc4.8"
                 "../tbb/lib/intel64/gcc4.4" "../tbb/lib/intel64/gcc4.8"
                 "../../tbb/latest/lib/intel64/gcc4.8"
                 "../tbb/lib/intel64/vc14"
                 "../tbb/lib/intel64/vc_mt"
                 )

#Workaround for ref problem
if(UNIX)
  set(TBB_LINK "-Wl,-rpath,${TBB_LIB_DIR} -L${TBB_LIB_DIR} -ltbb")
else()
  set(TBB_LINK "-LIBPATH:\"${TBB_LIB_DIR}\" tbb.lib")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB REQUIRED_VARS TBB_LIBRARIES TBB_LINK)


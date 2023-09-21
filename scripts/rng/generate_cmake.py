#!/usr/bin/env python
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

from sys import argv, exit, stdin
from subprocess import call
from pprint import pprint
from collections import defaultdict
import re
import os

def usage(err = None):
    if err:
        print('error: %s' % err)
    print('''\
Script to generate CMakeLists.txt for all files in the specified directory
Usage:

    {script} <path/to/directory> <backend>

Example:

    {script}  src/rng/backends/mklgpu mklgpu
'''.format(script = argv[0]))

if len(argv) <= 2:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_dir = argv[1]
libname = argv[2]

if not os.path.exists(in_dir):
    print("Error: directory " + in_dir + " doesn't exist\n")
    exit(1)

cmake_file = in_dir + "/CMakeLists.txt"

print("Generate " + cmake_file)

file_list = os.listdir(in_dir)

out_file = open(cmake_file, "w+")

out_file.write("""#
# copyright
#

set(LIB_NAME onemkl_blas_{libname})
set(LIB_OBJ ${{LIB_NAME}}_obj)

# Add third-party library
# find_package(XXX REQUIRED)

set(SOURCES
""".format(libname=libname))

for f in file_list:
    if re.search('wrappers.c', f):
        out_file.write("""  $<$<BOOL:${{BUILD_SHARED_LIBS}}>: {filename}>
""".format(filename=f))
    else:
        out_file.write("""  {filename}
""".format(filename=f))

out_file.write(""")

add_library(${{LIB_NAME}})
add_library(${{LIB_OBJ}} OBJECT ${{SOURCES}})

target_include_directories(${{LIB_OBJ}}
  PRIVATE ${{PROJECT_SOURCE_DIR}}/include
          ${{PROJECT_SOURCE_DIR}}/src
          ${{CMAKE_BINARY_DIR}}/bin
          ${{MKL_INCLUDE}}
)

target_link_libraries(${{LIB_OBJ}}
    PUBLIC ONEMKL::SYCL::SYCL
    # Add third party library to link with here
)

target_link_libraries(${{LIB_NAME}} PUBLIC ${{LIB_OBJ}})

# Add major version to the library
set_target_properties(${{LIB_NAME}} PROPERTIES
  SOVERSION ${{PROJECT_VERSION_MAJOR}}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${{LIB_NAME}}>)

# Add the library to install package
install(TARGETS ${{LIB_OBJ}} EXPORT oneMKLTargets)
install(TARGETS ${{LIB_NAME}} EXPORT oneMKLTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
""".format())

out_file.close()

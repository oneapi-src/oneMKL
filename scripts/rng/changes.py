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
Script to changes some files for the new backend
Usage:

    {script} <path/to/directory> <backend>

'''.format(script = argv[0]))

if len(argv) <= 2:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_dir = argv[1]
backend = argv[2]
cmake_file = in_dir + "/src/rng/backends/CMakeLists.txt"
config_file = in_dir + "/src/config.hpp.in"
engines_file = in_dir + "/include/oneapi/mkl/rng/engines.hpp"
backends_file = in_dir + "/include/oneapi/mkl/detail/backends.hpp"

if not os.path.exists(in_dir):
    print("Error: directory " + in_dir + " doesn't exist\n")
    exit(1)

with open(engines_file, 'r+') as f:
    contents = f.readlines()
    flag = 0
    flag2 = 0
    index = 0
    for line in contents:
        index=index+1
        if flag == 1:
            out = out + """
""" + line.strip()
        if line.strip() == "#ifdef ENABLE_CURAND_BACKEND":
            out = line.strip()
            flag = 1
        if (line.strip() == "#endif") & (flag == 1):
            flag = 0
            flag2 = 1
        if flag2 == 1:
            contents.insert(index, """
""" + out.replace("curand", backend).replace("CURAND", backend.upper()) + """
""")
            flag2 = 0
    
    with open(engines_file, 'r+') as f2:
        contents = "".join(contents) 
        f2.write(contents)

print("Formatting with clang-format " + engines_file)
try:
    lc = ["clang-format", "-style=file", "-i", engines_file]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: clang-format is not found")
    else:
        raise   


with open(config_file, 'r+') as f:
    contents = f.readlines()
    index = 0
    for line in contents:
        index=index+1
        if line.strip() == "#cmakedefine ENABLE_CURAND_BACKEND":
            out = line.strip()
            contents.insert(index,out.replace("CURAND", backend.upper()) + """
""")

    with open(config_file, 'r+') as f2:
        contents = "".join(contents) 
        f2.write(contents)



with open(cmake_file, 'r+') as f:
    contents = f.readlines()
    contents.append("""
if(ENABLE_{BACKEND}_BACKEND)
  add_subdirectory({backend})
endif()
""".format(backend=backend, BACKEND=backend.upper()))

    with open(cmake_file, 'w+') as f2:
        contents = "".join(contents) 
        f2.write(contents)


with open(backends_file, "r+") as file:
    x = file.read()
	
with open(backends_file, "wt") as file:
    x = x.replace(" cublas,", " cublas, {backend},".format(backend=backend))
    x = x.replace("""{ backend::cublas, "cublas" },""", """{{ backend::cublas, "cublas" }}, {{ backend::{backend}, "{backend}" }},""".format(backend=backend))
    file.write(x)

print("Formatting with clang-format " + backends_file)
try:
    lc = ["clang-format", "-style=file", "-i", backends_file]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: clang-format is not found")
    else:
        raise 
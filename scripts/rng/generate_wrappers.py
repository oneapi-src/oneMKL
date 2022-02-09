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
import errno
import re
import os

def usage(err = None):
    if err:
        print('error: %s' % err)
    print('''\
Script to generate wrappers for backends
Note: requires clang-format 9.0.0 tool to be installed
Usage:

    {script} <out_file> <backend>

Example:
The command below will generate "mkl_rng_curand_wrappers.cpp"
python3 {script} "../../src/rng/backends/curand/mkl_rng_curand_wrappers.cpp" curand
'''.format(script = argv[0]))

if len(argv) <= 2:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

out_filename = argv[1]
backend = argv[2]

table_list = "engine_list.txt"

print("Generate " + out_filename)

try:
    os.makedirs(os.path.dirname(out_filename))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

out_file = open(out_filename, "w+")
out_file.write("""//Copyright

#include "oneapi/mkl/rng/detail/{header}/onemkl_rng_{header}.hpp"
#include "rng/function_table.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMKL_EXPORT rng_function_table_t mkl_rng_table = {{
    WRAPPER_VERSION,
    """.format(header=backend))

with open(table_list, "r") as f:
    table = f.readlines()

    for t in table[:len(table) - 1]:
        out_file.write("""oneapi::mkl::rng::{namespace}::create_{engine},
    oneapi::mkl::rng::{namespace}::create_{engine},
    """.format(namespace=backend, engine=t.strip()))

    out_file.write("""oneapi::mkl::rng::{namespace}::create_{engine},
    oneapi::mkl::rng::{namespace}::create_{engine}
}}
""".format(namespace=backend, engine=table[len(table) - 1].strip()))

out_file.close()

print("Formatting with clang-format " + out_filename)
try:
    lc = ["clang-format", "-style=file", "-i", out_filename]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: clang-format is not found")
    else:
        raise
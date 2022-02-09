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
Script to create new backend
Note: requires clang-format 9.0.0 tool to be installed
Usage:

    {script} <dir to ONEMKL> <name_of_backend>

'''.format(script = argv[0]))

if len(argv) <= 2:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

backend = argv[2]

include_dir = argv[1] + "/include/oneapi/mkl/rng/detail/" + "{name}/".format(name=backend)
src_dir = argv[1] + "/src/rng/backends/" + "{name}/".format(name=backend)


print("Generate " + include_dir)
print("Generate " + src_dir)

try:
    os.mkdir(include_dir)
    os.mkdir(src_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

try:
    lc = ["python3", "generate_onemkl_rng_backend.py", include_dir +  "onemkl_rng_{name}.hpp".format(name=backend), backend]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: script generate_onemkl_rng_backend.py is not found")
    else:
        raise

try:
    lc = ["python3", "generate_wrappers.py", src_dir +  "mkl_rng_{name}_wrappers.cpp".format(name=backend), backend]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: script generate_wrappers.py is not found")
    else:
        raise

table_list = "engine_list.txt"
seed_list = "seed_type.txt"

with open(table_list, "r") as f:
    with open(seed_list, "r") as f2:
        table = f.readlines()
        types = f2.readlines()

        for i in range(len(table)):
            out_file = src_dir + "{name}.cpp".format(name=table[i].strip())
            try:
                lc = ["python3", "generate_engine.py", out_file, backend, table[i].strip(), types[i].strip()]
                call(lc)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    print("Error: script generate_engine.py is not found")
                else:
                    raise
                    
try:
    lc = ["python3", "generate_cmake.py", src_dir, backend]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: script generate_cmake.py is not found")
    else:
        raise

try:
    lc = ["python3", "changes.py", argv[1], backend]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: script changes.py is not found")
    else:
        raise
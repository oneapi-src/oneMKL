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
import json

def usage(err = None):
    if err:
        print('error: %s' % err)
    print('''\
Script to generate onemkl_rng_<backend>.hpp
Note: requires clang-format 9.0.0 tool to be installed
Usage:

    {script} <out_file> <backend>

Example:
The command below will generate "onemkl_rng_curand.hpp"
python3 {script} "../../include/oneapi/mkl/rng/detail/curand/onemkl_rng_curand.hpp" curand
'''.format(script = argv[0]))

if len(argv) <= 2:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

out_filename = argv[1]
backend = argv[2]

table_list = "engine_list.json"

print("Generate " + out_filename)

try:
    os.makedirs(os.path.dirname(out_filename))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise


out_file = open(out_filename, "w+")
out_file.write("""//Copyright

#ifndef _ONEMKL_RNG_{HEADER}_HPP_
#define _ONEMKL_RNG_{HEADER}_HPP_

#include <CL/sycl.hpp>
#include <cstdint>

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/rng/detail/engine_impl.hpp"

namespace oneapi::mkl::rng::{header} {{
""".format(HEADER=backend.upper(), header=backend))

with open(table_list, "r") as f:
    table = json.load(f)

    for i, seeds in table.items():
        for j in seeds:
            out_file.write("""
ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_{engine}(cl::sycl::queue queue, {type} seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_{engine}(cl::sycl::queue queue, std::initializer_list<{type}> seed);
""".format(type=j, engine=i))


out_file.write("""
}} // namespace oneapi::mkl::rng::{header}

#endif //_ONEMKL_RNG_{HEADER}_HPP_
""".format(header=backend, HEADER=backend.upper()))

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
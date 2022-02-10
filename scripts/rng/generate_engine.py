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
Script to generate engines files
Note: requires clang-format 9.0.0 tool to be installed
Usage:

    {script} <out_file> <backend> <engine> <seed_type>

Example:
The command below will generate "mcg59.cpp"
python3 {script} "../../src/rng/backends/curand/mcg59.cpp" curand mcg59 std::uint64_t
'''.format(script = argv[0]))

if len(argv) <= 4:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

out_filename = argv[1]
backend = argv[2]
engine = argv[3]
seed_type=[argv[i] for i in range(4, len(argv))]

# Generate wrappers
print("Generate " + out_filename)

try:
    os.makedirs(os.path.dirname(out_filename))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

out_file = open(out_filename, "w+")
out_file.write("""//Copyright

#include <iostream>
#include <CL/sycl.hpp>

#include "oneapi/mkl/rng/detail/engine_impl.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/detail/curand/onemkl_rng_{backend}.hpp"

namespace oneapi::mkl::rng::{backend} {{
""".format(backend=backend)) 
for t in seed_type:
    out_file.write("""
oneapi::mkl::rng::detail::engine_impl* create_{e}(cl::sycl::queue queue, {type} seed) {{
    //throw oneapi::mkl::unimplemented("rng", "{e} engine");
    //return nullptr;
}}

oneapi::mkl::rng::detail::engine_impl* create_{e}(cl::sycl::queue queue,
                                                    std::initializer_list<{type}> seed) {{
    //throw oneapi::mkl::unimplemented("rng", "{e} engine");
    //return nullptr;
}}
""".format(type=t, e=engine))
out_file.write("""
}} // namespace oneapi::mkl::rng::{backend}
""".format(backend=backend))

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
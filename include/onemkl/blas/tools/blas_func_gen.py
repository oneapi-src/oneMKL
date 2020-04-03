#!/usr/bin/env python
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

from sys import argv, exit, stdin
from pprint import pprint
from collections import defaultdict
import re

from func_parser import create_db

def usage(err = None):
    if err:
        print 'error: %s' % err
    print '''\
usage: {script} <in_header.h> <out_header.h> <library> <backend> <include> <namespace>

usage example:
{script}  blas.hpp new_blas.hpp intelmkl intelcpu mkl_sycl_blas.hpp onemkl::intelmkl::blas
'''.format(script = argv[0])

if len(argv) <= 1:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_filename = argv[1]
out_filename = argv[2]
lib = argv[3]
backend = argv[4]
include = argv[5]
namespace = argv[6]

header_db = create_db(in_filename)

def print_funcs(func_list):
    code=""
    for data in func_list:
        code +="""
template <onemkl::library lib, onemkl::backend backend> static inline {ret_type} {name}{par_str};
template<>
{ret_type} {name}<library::{lib}, backend::{backend}>{par_str} {{
    {name}_precondition{call_str};
    {namespace}::{name}{call_str};
    {name}_postcondition{call_str};
}}
""".format(lib=lib, namespace=namespace, backend=backend, **data)
    return code

print "Generate " + out_filename + "..."
out_file = open(out_filename, "w+")
out_file.write("""//
// Generated based on onemkl/blas/blas.hpp
//

#ifndef _{gard}_HPP_
#define _{gard}_HPP_

#include <CL/sycl.hpp>
#include <cstdint>

#include "onemkl/types.hpp"
#include "onemkl/detail/backends.hpp"
#include "onemkl/detail/libraries.hpp"

#include "{include}"

""".format(gard=(out_filename.split('.', 1)[0].upper()).replace('/','_'), include=include))


out_file.write("""

namespace onemkl {
namespace blas {""")

for func_name, func_list in header_db.iteritems():
    out_file.write("""
{funcs}""".format(funcs=print_funcs(func_list)))

out_file.write("""
}} //namespace blas
}} //namespace onemkl

#endif //_{gard}_HPP_""".format(gard=(out_filename.split('.', 1)[0].upper()).replace('/','_')))
out_file.close()

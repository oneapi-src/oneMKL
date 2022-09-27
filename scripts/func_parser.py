#!/usr/bin/env python
#===============================================================================
# Copyright 2020-2021 Intel Corporation
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
from collections import defaultdict
import re

def parse_item(item):
    func_type_and_name, par_str = item.split('(', 1)
    ret_type, func_name = func_type_and_name.strip().rsplit(' ', 1)
    """remove macros callback"""
    ret_type = re.sub('^[A-Z_]+ ','',ret_type)
    """remove macros calling convention"""
    ret_type = re.sub(' [A-Z_]+$','',ret_type.strip())
    """remove templates"""
    ret_type = re.sub('template\s+<[a-zA-Z, _*:]*>','',ret_type.strip())
    if func_name[0] == '*':
            ret_type += ' *'
            func_name = func_name[1:]

    par_str = re.sub('^\s*(void)\s*$', '', par_str.strip(');'))

    """Extract callback calls from parameter list and replace them by temp 'cllbck' param"""
    clbck_type = None
    clbck_param = None
    if par_str.find('(') != -1:
        par_str = re.sub('[)]\s*[(]', ')(', par_str)
        clbck_re = re.compile(r'\w*\s*[(]\s*[A-Z_]*\s*[*]\s*\w*\s*[)]\s*[(][\w*,\[\]\s]*[)]')
        clbck_list = clbck_re.findall(par_str)
        clbck_type = [ re.sub('[*]\s*\w*', '*', x.split(')(')[0]) for x in clbck_list]
        clbck_param = [ x.split(')(')[1] for x in clbck_list]

        par_str = re.sub('[)]\s*[(][\w*,\[\]\s]*[)]', '', par_str)
        par_str = re.sub('\w*\s*[(]\s*[A-Z_]*\s*[\w]*\s*[*]\s*', 'cllbck ', par_str)
    par_str_ = re.sub('[,]+(?![^<]+>)', '@', par_str)
    par_list = [x.strip() for x in par_str_.split('@') \
                    if len(x.strip()) > 0 ]
    """Split list of parameters to types and names"""
    if len(par_list) > 0:
            """Add parameter names (param1, param2, etc) if the declaration includes only types"""
            if re.search('(,|^)+\s*(const)*\s*[\w:]+\s*[*]*\s*(,|$)+', re.sub('<[\s\w\d,]*>', '', par_str)) is not None:
                par_list = [(x + ' param' + str(idx)).replace(" * ", " *" \
                     ).replace("[] param" + str(idx), "param" + str(idx) + "[]") \
                     for idx, x in enumerate(par_list)]

            """Extract names to call_list"""
            call_list = [x.split('=', 1)[0].strip().rsplit(' ', 1)[1].strip(' *').strip('\[\]').strip('&') \
                            for x in par_list]

            """Extract types to sig_list"""
            par_list_wo_st_arrays = [(x.rsplit(' ', 1)[0] + \
                    (lambda x: '* ' if x.find('[]') != -1 else ' ')(x.rsplit(' ', 1)[1]) + \
                    (x.rsplit(' ', 1)[1]).strip('\[\]')) for x in par_list]
            sig_list = [(x.rsplit(' ', 1)[0] + \
                                    (x.rsplit(' ', 1)[1].startswith('*') \
                                    and (' ' + x.rsplit(' ', 1)[1].count('*') * '*') or '')) \
                                    for x in par_list_wo_st_arrays]
    else:
            call_list = list()
            sig_list = list()
    par_str = '(' + ', '.join(par_list) + ')'
    call_str = '(' + ', '.join(call_list) + ')'
    sig_str = '(' + ', '.join(sig_list) + ')'

    """Put real callback call types back to the param_list and sig_str """
    if clbck_param is not None:
        for idx, x in enumerate(clbck_param):
            par_str = re.sub(r'(cllbck\s*\w*)[,]', r'\1(' + x + ',', par_str, idx)
            sig_str = re.sub(r'(cllbck\s*\w*)[,]', r'\1(' + x + ',', sig_str, idx)

    if clbck_type is not None:
        for idx, x in enumerate(clbck_type):
            par_str = re.sub(r'cllbck(\s*\w*)', x + r'\1)', par_str, idx)
            sig_str = re.sub(r'cllbck(\s*\w*)', x + r'\1)', sig_str, idx)
    return func_name, ret_type, func_name, par_str, call_str, sig_str, call_list, sig_list


def to_dict(func_data):
    """ convert (ret_type, 'name', par_str, call_str, sig_str, call_list, sig_list) tuple to
        dict with corresponding keys """
    return dict(zip(('ret_type', 'name', 'par_str', 'call_str', 'sig_str', 'call_list', 'sig_list'), func_data))

is_comment = 0
is_wrapperbody = 0
def strip_line(l):
    """ remove global variables"""
    if re.search('^\s*\w+\s*\w+[;]', l) is not None:
        l = ''
    """ remove namespaces"""
    if re.search('^\s*namespace\s*\w+\s*[{]', l) is not None:
        l = ''
    """ remove declaration keywords """
    l = re.sub("^extern ", "", l)
    l = re.sub("^static ", "", l)
    l = re.sub("^inline ", "", l)
    """ remove extra whitespace and comments from input line """
    l = re.sub("[)][A-Za-z0-9\s_]*[;]", ");", l)

    """ remove simple wrapper function body"""
    global is_wrapperbody
    if is_wrapperbody == 1:
        if re.search('^\s*}', l) is not None:
            l = l.split('}', 1)[1].strip()
            is_wrapperbody = 0
        else:
            return ""

    m = re.search(r'[)]\s*\n*\s*[{]', l)
    if m is not None:
        l = l[:m.end()].strip('{').strip() + ";"
        is_wrapperbody = 1

    global is_comment
    if is_comment == 1:
        if re.search('\*/', l) is not None:
            l = l.split('*/', 1)[1].strip()
            is_comment = 0
        else:
            return ""
    """ Delete comments """
    l1 = l.split('#', 1)[0].strip()
    l2 = l1.split('//', 1)[0].strip()
    l3 = l2.split('/*', 1)[0].strip()
    if re.search('/\*', l2) is not None:
        is_comment = 1
        if re.search('\*/', l2) is not None:
            is_comment = 0
            l4 = l2.split('*/', 1)[1].strip()
            l3 += l4
    """ Delete comments if there are several of them in one line """
    l3 = re.sub("[/][*][\w\s]*[*][/]", "", l3);
    """Delete all tabs"""
    return re.sub(' +',' ', l3)

def create_func_db(filenames):
    data=[]
    for filename in filenames.split(":"):
        with open(filename, 'r') as f:
            data.extend(f.readlines())
    funcs_db = defaultdict(list)
    whole_line = ""
    idx = 0
    for l in data:
        stripped = strip_line(l)
        if not stripped:
            continue
        """ Check if function contains 1 line """
        whole_line += stripped + ' '
        """ Check if there is function """
        if re.search('[(][\w\s\*/\&,_\[\]():<>={}]*[)]\s*[;]', whole_line) is None:
            """ Check if there is some other staff before the function """
            if re.search('[;{}]\s*$', whole_line) is not None:
                whole_line = ""
            continue
        else:
            stripped = whole_line.strip()
            whole_line = ""
        print(stripped)
        parsed = parse_item(stripped)
        func_name, func_data = parsed[0], parsed[1:]
        funcs_db[func_name].append(to_dict(func_data))
        idx = idx + 1
    return funcs_db

def get_namespaces(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    namespace_list = list()
    for l in data:
        stripped = strip_line(l)
        if re.search('^\s*namespace\s*\w+\s*[{]', l) is not None:
           l = l.split("namespace", 1)[1]
           l = l.split("{", 1)[0]
           namespace_list.append(l.strip())
    return namespace_list


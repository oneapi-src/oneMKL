/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _ONEMKL_LIBRARIES_HPP_
#define _ONEMKL_LIBRARIES_HPP_

#include <CL/sycl.hpp>
#include <map>
#include <string>

namespace oneapi {
namespace mkl {

enum class library { intelmkl, cublas };

typedef std::map<library, std::string> librarymap;

static librarymap library_map = { { library::intelmkl, "intelmkl" },
                                  { library::cublas, "cublas" } };

} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_LIBRARIES_HPP_

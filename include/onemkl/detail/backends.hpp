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

#ifndef _ONEMKL_BACKENDS_HPP_
#define _ONEMKL_BACKENDS_HPP_

#include <map>
#include <string>

namespace onemkl {

enum class backend { intelcpu, intelgpu, unsupported };

typedef std::map<backend, std::string> backendmap;

static backendmap backend_map = { { backend::intelcpu, "intelcpu" },
                                  { backend::intelgpu, "intelgpu" },
                                  { backend::unsupported, "unsupported" } };

} //namespace onemkl

#endif //_ONEMKL_BACKENDS_HPP_

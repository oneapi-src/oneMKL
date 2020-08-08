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

#ifndef _ONEMKL_BACKENDS_TABLE_HPP_
#define _ONEMKL_BACKENDS_TABLE_HPP_

#include <string>
#include <vector>
#include <map>
#include <CL/sycl.hpp>

#include "oneapi/mkl/detail/config.hpp"

#ifdef __linux__
    #define LIB_NAME(a) "libonemkl_" a ".so"
#elif defined(_WIN64)
    #define LIB_NAME(a) "onemkl_" a ".dll"
#endif

namespace oneapi {
namespace mkl {

enum class device: uint16_t { x86cpu, intelgpu, nvidiagpu };
enum class domain: uint16_t { blas };

static std::map<domain, std::map<device, std::vector<const char*>>>
    libraries = { {domain::blas,
                      {{device::x86cpu, {
#ifdef ENABLE_MKLCPU_BACKEND
                           LIB_NAME("blas_mklcpu")
#endif
                       }},
                       {device::intelgpu, {
#ifdef ENABLE_MKLGPU_BACKEND
                           LIB_NAME("blas_mklgpu")
#endif
                       }},
                       {device::nvidiagpu, {
#ifdef ENABLE_CUBLAS_BACKEND
                           LIB_NAME("blas_cublas")
#endif
                       }}
                      }}
                };

} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BACKENDS_TABLE_HPP_

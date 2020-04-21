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

#ifndef _ONEMKL_BACKENDS_SELECTOR_HPP_
#define _ONEMKL_BACKENDS_SELECTOR_HPP_

#include <CL/sycl.hpp>
#include <map>
#include <string>
#include "onemkl/detail/backends.hpp"

#ifdef __linux__
    #define LIB_NAME(a) "lib" a ".so"
#endif

#define INTEL_ID  32902
#define NVIDIA_ID 4318

namespace onemkl {
inline char *select_backend(cl::sycl::queue &queue) {
    if (queue.is_host()) {
        return (char *)LIB_NAME("onemkl_blas_mklcpu");
    }
    else if (queue.get_device().is_cpu()) {
        return (char *)LIB_NAME("onemkl_blas_mklcpu");
    }
    else if (queue.get_device().is_gpu()) {
        unsigned int vendor_id = static_cast<unsigned int>(
            queue.get_device().get_info<cl::sycl::info::device::vendor_id>());

        if (vendor_id == INTEL_ID)
            return (char *)LIB_NAME("onemkl_blas_mklgpu");
        else if (vendor_id == NVIDIA_ID)
            return (char *)LIB_NAME("onemkl_blas_cublas");
        return (char *)"unsupported";
    }
    else {
        return (char *)"unsupported";
    }
}

} //namespace onemkl

#endif //_ONEMKL_BACKENDS_SELECTOR_HPP_

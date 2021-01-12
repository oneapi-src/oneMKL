/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _ONEMKL_GET_DEVICE_ID_HPP_
#define _ONEMKL_GET_DEVICE_ID_HPP_

#include <CL/sycl.hpp>

#include "oneapi/mkl/detail/backends_table.hpp"
#include "oneapi/mkl/exceptions.hpp"

#define INTEL_ID  32902
#define NVIDIA_ID 4318

namespace oneapi {
namespace mkl {

inline oneapi::mkl::device get_device_id(cl::sycl::queue &queue) {
    oneapi::mkl::device device_id;
    if (queue.is_host())
        device_id = device::x86cpu;
    else if (queue.get_device().is_cpu())
        device_id = device::x86cpu;
    else if (queue.get_device().is_gpu()) {
        unsigned int vendor_id = static_cast<unsigned int>(
            queue.get_device().get_info<cl::sycl::info::device::vendor_id>());

        if (vendor_id == INTEL_ID)
            device_id = device::intelgpu;
        else if (vendor_id == NVIDIA_ID)
            device_id = device::nvidiagpu;
        else {
            throw unsupported_device("", "", queue.get_device());
        }
    }
    else {
        throw unsupported_device("", "", queue.get_device());
    }
    return device_id;
}

} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_GET_DEVICE_ID_HPP_

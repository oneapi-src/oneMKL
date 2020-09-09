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

#ifndef _ONEMKL_BACKEND_SELECTOR_HPP_
#define _ONEMKL_BACKEND_SELECTOR_HPP_

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/detail/backends.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi {
namespace mkl {

template <backend Backend>
class backend_selector {
public:
    explicit backend_selector(sycl::queue queue) : queue_(queue) {
        if ((queue.is_host() || queue.get_device().is_cpu())) {
            if (Backend != backend::mklcpu) {
                throw unsupported_device("", "backend_selector<" + backend_map[Backend] + ">",
                                         queue.get_device());
            }
        }
        else if (queue.get_device().is_gpu()) {
            unsigned int vendor_id = static_cast<unsigned int>(
                queue.get_device().get_info<cl::sycl::info::device::vendor_id>());
            if (vendor_id == INTEL_ID) {
                if (Backend != backend::mklgpu) {
                    throw unsupported_device("", "backend_selector<" + backend_map[Backend] + ">",
                                             queue.get_device());
                }
            }
            else if (vendor_id == NVIDIA_ID) {
                if (Backend != backend::cublas) {
                    throw unsupported_device("", "backend_selector<" + backend_map[Backend] + ">",
                                             queue.get_device());
                }
            }
            else {
                throw unsupported_device("", "backend_selector<" + backend_map[Backend] + ">",
                                         queue.get_device());
            }
        }
        else {
            throw unsupported_device("", "backend_selector<" + backend_map[Backend] + ">",
                                     queue.get_device());
        }
    }
    sycl::queue& get_queue() {
        return queue_;
    }

private:
    sycl::queue queue_;
};

} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_BACKEND_SELECTOR_HPP_

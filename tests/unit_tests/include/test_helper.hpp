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

#ifndef _TEST_HELPER_HPP_
#define _TEST_HELPER_HPP_

#include <gtest/gtest.h>

#ifdef ENABLE_MKLCPU_BACKEND
    #define TEST_RUN_INTELCPU(q, func, args) \
        func<onemkl::library::intelmkl, onemkl::backend::intelcpu> args
#else
    #define TEST_RUN_INTELCPU(q, func, args)
#endif

#ifdef ENABLE_MKLGPU_BACKEND
    #define TEST_RUN_INTELGPU(q, func, args) \
        func<onemkl::library::intelmkl, onemkl::backend::intelgpu> args
#else
    #define TEST_RUN_INTELGPU(q, func, args)
#endif

#ifdef ENABLE_CUBLAS_BACKEND
    #define TEST_RUN_NVIDIAGPU(q, func, args) \
        func<onemkl::library::cublas, onemkl::backend::nvidiagpu> args
#else
    #define TEST_RUN_NVIDIAGPU(q, func, args)
#endif

#define TEST_RUN_CT(q, func, args)                                             \
    do {                                                                       \
        if (q.is_host() || q.get_device().is_cpu())                            \
            TEST_RUN_INTELCPU(q, func, args);                                  \
        else if (q.get_device().is_gpu()) {                                    \
            unsigned int vendor_id = static_cast<unsigned int>(                \
                q.get_device().get_info<cl::sycl::info::device::vendor_id>()); \
            if (vendor_id == INTEL_ID)                                         \
                TEST_RUN_INTELGPU(q, func, args);                              \
            else if (vendor_id == NVIDIA_ID)                                   \
                TEST_RUN_NVIDIAGPU(q, func, args);                             \
        }                                                                      \
    } while (0);

class DeviceNamePrint {
public:
    std::string operator()(testing::TestParamInfo<cl::sycl::device> dev) const {
        if (dev.param.is_cpu())
            return std::string("CPU");
        if (dev.param.is_host())
            return std::string("HOST");
        if (dev.param.is_gpu()) {
            unsigned int vendor_id =
                static_cast<unsigned int>(dev.param.get_info<cl::sycl::info::device::vendor_id>());
            switch (vendor_id) {
                case INTEL_ID:
                    return std::string("INTELGPU");
                case NVIDIA_ID:
                    return std::string("NVIDIAGPU");
            }
        }
        if (dev.param.is_accelerator())
            return std::string("ACCELERATOR");
        return std::string("UNKNOWN");
    }
};

#endif // _TEST_HELPER_HPP_

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
#include "onemkl/detail/config.hpp"

#ifdef _WIN64
    #include <malloc.h>
#else
    #incude < stdlib.h>
#endif

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
        std::string dev_name = dev.param.get_info<cl::sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        return dev_name;
    }
};

/* to accomodate Windows and Linux differences between alligned_alloc and 
   _aligned_malloc calls use onemkl::aligned_alloc and onemkl::aligned_free instead */
namespace onemkl {

static inline void *aligned_alloc(size_t align, size_t size) {
#ifdef _WIN64
    return _aligned_malloc(size, align);
#else
    return aligned_alloc(align, size);
#endif
}

static inline void aligned_free(void *p) {
#ifdef _WIN64
    _aligned_free(p);
#else
    free(p);
#endif
}
} // namespace onemkl

#endif // _TEST_HELPER_HPP_

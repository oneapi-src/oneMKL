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

#ifndef _RNG_ROCRAND_COMMON_HPP_
#define _RNG_ROCRAND_COMMON_HPP_

#include <CL/sycl.hpp>
#include "rocrand_helper.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace rocrand {

#ifdef __HIPSYCL__
template <typename H, typename F>
static inline void host_task_internal(H &cgh, rocrand_generator engine, F f, long) {
    cgh.hipSYCL_enqueue_custom_operation([f, engine](cl::sycl::interop_handle ih) {
        rocrand_status status;
        auto stream = ih.get_native_queue<cl::sycl::backend::hip>();
        ROCRAND_CALL(rocrand_set_stream, status, engine, stream);
        f(ih);
    });
}
#else
template <typename H, typename F>
static inline void host_task_internal(H& cgh, rocrand_generator& engine, F f, long) {
    //cgh.template single_task(f);
    cgh.host_task(f);
}

#endif
template <typename H, typename F>
static inline void host_task(H &cgh, rocrand_generator &engine, F f) {
    (void)host_task_internal(cgh, engine, f, 0);
}

template <typename Engine, typename Distr>
class kernel_name {};

template <typename Engine, typename Distr>
class kernel_name_usm {};

} // namespace rocrand
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_RNG_ROCRAND_COMMON_HPP_

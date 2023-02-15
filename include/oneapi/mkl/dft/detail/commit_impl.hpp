/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef _ONEMKL_DFT_COMMIT_IMPL_HPP_
#define _ONEMKL_DFT_COMMIT_IMPL_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

class commit_impl {
public:
    commit_impl(sycl::queue queue) : queue_(queue), status(false) {}

    commit_impl(const commit_impl& other) : queue_(other.queue_), status(other.status) {}

    virtual ~commit_impl() {}

    sycl::queue& get_queue() {
        return queue_;
    }

protected:
    bool status;
    sycl::queue queue_;
};


} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_COMMIT_IMPL_HPP_


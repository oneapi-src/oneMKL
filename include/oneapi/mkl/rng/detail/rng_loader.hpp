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

#ifndef _ONEMKL_RNG_LOADER_HPP_
#define _ONEMKL_RNG_LOADER_HPP_

#include <cstdint>
#include <CL/sycl.hpp>

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

#include "oneapi/mkl/rng/detail/engine_impl.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace detail {

ONEMKL_EXPORT engine_impl* create_philox4x32x10(oneapi::mkl::device libkey, cl::sycl::queue queue,
                                                std::uint64_t seed);

ONEMKL_EXPORT engine_impl* create_philox4x32x10(oneapi::mkl::device libkey, cl::sycl::queue queue,
                                                std::initializer_list<std::uint64_t> seed);

ONEMKL_EXPORT engine_impl* create_mrg32k3a(oneapi::mkl::device libkey, cl::sycl::queue queue,
                                           std::uint32_t seed);

ONEMKL_EXPORT engine_impl* create_mrg32k3a(oneapi::mkl::device libkey, cl::sycl::queue queue,
                                           std::initializer_list<std::uint32_t> seed);

} // namespace detail
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_RNG_LOADER_HPP_

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

#ifndef _ONEMKL_RNG_PREDICATES_HPP_
#define _ONEMKL_RNG_PREDICATES_HPP_

#include <cstdint>
#include <CL/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace rng {

// Buffer APIs

template <typename Distr, typename Engine>
inline void generate_precondition(const Distr& distr, Engine& engine, std::int64_t n,
                                  sycl::buffer<typename Distr::result_type, 1>& r) {
#ifndef ONEMKL_DISABLE_PREDICATES
    if (n < 0 || n > r.get_count()) {
        throw oneapi::mkl::invalid_argument("rng", "generate", "n");
    }
#endif
}

// USM APIs

template <typename Distr, typename Engine>
inline void generate_precondition(const Distr& distr, Engine& engine, std::int64_t n,
                                  typename Distr::result_type* r,
                                  const cl::sycl::vector_class<cl::sycl::event>& dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    if (n < 0) {
        throw oneapi::mkl::invalid_argument("rng", "generate", "n");
    }
    if (r == nullptr) {
        throw oneapi::mkl::invalid_argument("rng", "generate", "r is nullptr");
    }
#endif
}

} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_RNG_PREDICATES_HPP_

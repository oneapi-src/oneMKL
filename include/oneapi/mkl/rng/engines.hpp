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

#ifndef _ONEMKL_RNG_ENGINES_HPP_
#define _ONEMKL_RNG_ENGINES_HPP_

#include <cstdint>
#include <limits>
#include <memory>
#include <CL/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"

#include "oneapi/mkl/rng/detail/engine_impl.hpp"

#ifdef ENABLE_MKLCPU_BACKEND
#include "oneapi/mkl/rng/detail/mklcpu/onemkl_rng_mklcpu.hpp"
#endif
#ifdef ENABLE_MKLGPU_BACKEND
#include "oneapi/mkl/rng/detail/mklgpu/onemkl_rng_mklgpu.hpp"
#endif

namespace oneapi {
namespace mkl {
namespace rng {

// Class oneapi::mkl::rng::philox4x32x10
//
// Represents Philox4x32-10 counter-based pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
class philox4x32x10 {
public:
    static constexpr std::uint64_t default_seed = 0;

    philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed)
            : pimpl_(detail::create_philox4x32x10(get_device_id(queue), queue, seed)) {}

    philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : pimpl_(detail::create_philox4x32x10(get_device_id(queue), queue, seed)) {}

#ifdef ENABLE_MKLCPU_BACKEND
    philox4x32x10(backend_selector<backend::mklcpu> selector, std::uint64_t seed = default_seed)
            : pimpl_(mklcpu::create_philox4x32x10(selector.get_queue(), seed)) {}

    philox4x32x10(backend_selector<backend::mklcpu> selector,
                  std::initializer_list<std::uint64_t> seed)
            : pimpl_(mklcpu::create_philox4x32x10(selector.get_queue(), seed)) {}
#endif

#ifdef ENABLE_MKLGPU_BACKEND
    philox4x32x10(backend_selector<backend::mklgpu> selector, std::uint64_t seed = default_seed)
            : pimpl_(mklgpu::create_philox4x32x10(selector.get_queue(), seed)) {}

    philox4x32x10(backend_selector<backend::mklgpu> selector,
                  std::initializer_list<std::uint64_t> seed)
            : pimpl_(mklgpu::create_philox4x32x10(selector.get_queue(), seed)) {}
#endif

    philox4x32x10(const philox4x32x10& other) {
        pimpl_.reset(other.pimpl_.get()->copy_state());
    }

    philox4x32x10(philox4x32x10&& other) {
        pimpl_ = std::move(other.pimpl_);
    }

    philox4x32x10& operator=(const philox4x32x10& other) {
        if (this == &other)
            return *this;
        pimpl_.reset(other.pimpl_.get()->copy_state());
        return *this;
    }

    philox4x32x10& operator=(philox4x32x10&& other) {
        if (this == &other)
            return *this;
        pimpl_ = std::move(other.pimpl_);
        return *this;
    }

private:
    std::unique_ptr<detail::engine_impl> pimpl_;

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename Distr, typename Engine>
    friend void generate(const Distr& distr, Engine& engine, std::int64_t n,
                         sycl::buffer<typename Distr::result_type, 1>& r);

    template <typename Distr, typename Engine>
    friend sycl::event generate(const Distr& distr, Engine& engine, std::int64_t n,
                                typename Distr::result_type* r,
                                const sycl::vector_class<sycl::event>& dependencies);
};

} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_RNG_ENGINES_HPP_

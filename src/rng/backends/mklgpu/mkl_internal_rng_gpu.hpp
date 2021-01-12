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

#ifndef _MKL_INTERNAL_RNG_GPU_HPP_
#define _MKL_INTERNAL_RNG_GPU_HPP_

#include <CL/sycl.hpp>

namespace oneapi {
namespace mkl {
namespace rng {
namespace detail {

template <typename EngineType>
class engine_base_impl;

namespace gpu {

template <typename EngineType>
engine_base_impl<EngineType>* create_engine(sycl::queue& queue, std::uint64_t seed);

template <typename EngineType>
engine_base_impl<EngineType>* create_engine(sycl::queue& queue, std::int64_t n,
                                            const unsigned int* seed_ptr);

template <typename EngineType>
engine_base_impl<EngineType>* create_engine(sycl::queue& queue,
                                            engine_base_impl<EngineType>* other_impl);

template <typename EngineType>
void skip_ahead(sycl::queue& queue, engine_base_impl<EngineType>* impl, std::uint64_t num_to_skip);

template <typename EngineType>
void skip_ahead(sycl::queue& queue, engine_base_impl<EngineType>* impl,
                std::initializer_list<std::uint64_t> num_to_skip);

template <typename EngineType>
void leapfrog(sycl::queue& queue, engine_base_impl<EngineType>* impl, std::uint64_t idx,
              std::uint64_t stride);

template <typename EngineType>
void delete_engine(sycl::queue& queue, engine_base_impl<EngineType>* impl);

template <typename EngineType, typename DistrType>
sycl::event generate(sycl::queue& queue, const DistrType& distr,
                     engine_base_impl<EngineType>* engine, std::int64_t n,
                     sycl::buffer<typename DistrType::result_type, 1>& r);

template <typename EngineType, typename DistrType>
sycl::event generate(sycl::queue& queue, const DistrType& distr,
                     engine_base_impl<EngineType>* engine, std::int64_t n,
                     typename DistrType::result_type* r,
                     const sycl::vector_class<sycl::event>& dependencies = {});

} // namespace gpu
} // namespace detail
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_MKL_INTERNAL_RNG_GPU_HPP_

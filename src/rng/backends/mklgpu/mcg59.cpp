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

#include <iostream>
#include <CL/sycl.hpp>

#include "mkl_version.h"

#include "oneapi/mkl/rng/detail/engine_impl.hpp"
#include "oneapi/mkl/rng/engines.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/detail/mklgpu/onemkl_rng_mklgpu.hpp"

#include "mkl_internal_rng_gpu.hpp"

namespace oneapi::mkl::rng::mklgpu {

#if !defined(_WIN64)
class mcg59_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    mcg59_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        engine_ =
            oneapi::mkl::rng::detail::gpu::create_engine<oneapi::mkl::rng::mcg59>(queue, seed);
    }

    mcg59_impl(const mcg59_impl* other) : oneapi::mkl::rng::detail::engine_impl(*other) {
        cl::sycl::queue queue(other->queue_);
        engine_ = oneapi::mkl::rng::detail::gpu::create_engine<oneapi::mkl::rng::mcg59>(
            queue, other->engine_);
    }

    // Buffers API

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const oneapi::mkl::rng::uniform<
                              std::int32_t, oneapi::mkl::rng::uniform_method::standard>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              float, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              double, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              float, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              double, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t>& r) override {
        oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r);
    }

    // USM APIs

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<std::int32_t, oneapi::mkl::rng::uniform_method::standard>&
            distr,
        std::int64_t n, std::int32_t* r,
        const std::vector<cl::sycl::event>& dependencies) override {
        ;
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                                     std::int64_t n, std::int32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                                     std::int64_t n, std::uint32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(const bits<std::uint32_t>& distr, std::int64_t n,
                                     std::uint32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        return oneapi::mkl::rng::detail::gpu::generate(queue_, distr, engine_, n, r, dependencies);
    }

    virtual oneapi::mkl::rng::detail::engine_impl* copy_state() override {
        return new mcg59_impl(this);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        oneapi::mkl::rng::detail::gpu::skip_ahead(queue_, engine_, num_to_skip);
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 skipAheadEx");
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        oneapi::mkl::rng::detail::gpu::leapfrog(queue_, engine_, idx, stride);
    }

    virtual ~mcg59_impl() override {
        oneapi::mkl::rng::detail::gpu::delete_engine(queue_, engine_);
    }

private:
    oneapi::mkl::rng::detail::engine_base_impl<oneapi::mkl::rng::mcg59>* engine_;
};
#else // GPU backend is not supported for Windows OS currently
class mcg59_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    mcg59_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    mcg59_impl(cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    mcg59_impl(const mcg59_impl* other) : oneapi::mkl::rng::detail::engine_impl(*other) {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    // Buffers API

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const oneapi::mkl::rng::uniform<
                              std::int32_t, oneapi::mkl::rng::uniform_method::standard>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              float, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              double, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              float, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              double, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t>& r) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    // USM APIs

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<std::int32_t, oneapi::mkl::rng::uniform_method::standard>&
            distr,
        std::int64_t n, std::int32_t* r,
        const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, float* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, double* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                                     std::int64_t n, std::int32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                                     std::int64_t n, std::uint32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(const bits<std::uint32_t>& distr, std::int64_t n,
                                     std::uint32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return cl::sycl::event{};
    }

    virtual oneapi::mkl::rng::detail::engine_impl* copy_state() override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
        return nullptr;
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 engine");
    }

    virtual ~mcg59_impl() override {}
};
#endif

oneapi::mkl::rng::detail::engine_impl* create_mcg59(cl::sycl::queue queue, std::uint64_t seed) {
    return new mcg59_impl(queue, seed);
}

oneapi::mkl::rng::detail::engine_impl* create_mcg59(cl::sycl::queue queue,
                                                    std::initializer_list<std::uint64_t> seed) {
    throw oneapi::mkl::unimplemented("rng", "mcg59 skipAheadEx");
}

} // namespace oneapi::mkl::rng::mklgpu

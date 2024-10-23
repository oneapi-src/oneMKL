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

#include <iostream>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// Intel(R) oneMKL headers
#include <mkl_version.h>
#include <oneapi/mkl/rng.hpp>

#include "oneapi/math/rng/detail/engine_impl.hpp"
#include "oneapi/math/rng/engines.hpp"
#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/rng/detail/mklgpu/onemath_rng_mklgpu.hpp"

#include "onemkl_distribution_conversion.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace mklgpu {

class philox4x32x10_impl : public oneapi::math::rng::detail::engine_impl {
public:
    philox4x32x10_impl(sycl::queue queue, std::uint64_t seed)
            : oneapi::math::rng::detail::engine_impl(queue),
              engine_((oneapi::mkl::rng::philox4x32x10(queue, seed))) {}

    philox4x32x10_impl(sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : oneapi::math::rng::detail::engine_impl(queue),
              engine_((oneapi::mkl::rng::philox4x32x10(queue, seed))) {}

    philox4x32x10_impl(const philox4x32x10_impl* other)
            : oneapi::math::rng::detail::engine_impl(*other),
              engine_((oneapi::mkl::rng::philox4x32x10(other->engine_))) {}

    // Buffers API

    virtual void generate(
        const oneapi::math::rng::uniform<float, oneapi::math::rng::uniform_method::standard>& distr,
        std::int64_t n, sycl::buffer<float, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::uniform<
                              double, oneapi::math::rng::uniform_method::standard>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::uniform<
                              std::int32_t, oneapi::math::rng::uniform_method::standard>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(
        const oneapi::math::rng::uniform<float, oneapi::math::rng::uniform_method::accurate>& distr,
        std::int64_t n, sycl::buffer<float, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::uniform<
                              double, oneapi::math::rng::uniform_method::accurate>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::gaussian<
                              float, oneapi::math::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<float, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::gaussian<
                              double, oneapi::math::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(
        const oneapi::math::rng::gaussian<float, oneapi::math::rng::gaussian_method::icdf>& distr,
        std::int64_t n, sycl::buffer<float, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(
        const oneapi::math::rng::gaussian<double, oneapi::math::rng::gaussian_method::icdf>& distr,
        std::int64_t n, sycl::buffer<double, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::lognormal<
                              float, oneapi::math::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<float, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::lognormal<
                              double, oneapi::math::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(
        const oneapi::math::rng::lognormal<float, oneapi::math::rng::lognormal_method::icdf>& distr,
        std::int64_t n, sycl::buffer<float, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const oneapi::math::rng::lognormal<
                              double, oneapi::math::rng::lognormal_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<std::uint32_t, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, sycl::buffer<std::uint32_t, 1>& r) override {
        RETHROW_ONEMKL_EXCEPTIONS(
            oneapi::mkl::rng::generate(detail::get_onemkl_distribution(distr), engine_, n, r));
    }

    virtual void generate(const bits<std::uint32_t>& /*distr*/, std::int64_t /*n*/,
                          sycl::buffer<std::uint32_t, 1>& /*r*/) override {
        throw unimplemented("rng/mklgpu", "philox4x32x10::generate",
                            "bits distribution is not supported");
    }

    // USM APIs

    virtual sycl::event generate(
        const oneapi::math::rng::uniform<float, oneapi::math::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::uniform<double, oneapi::math::rng::uniform_method::standard>&
            distr,
        std::int64_t n, double* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::uniform<std::int32_t, oneapi::math::rng::uniform_method::standard>&
            distr,
        std::int64_t n, std::int32_t* r, const std::vector<sycl::event>& dependencies) override {
        ;
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::uniform<float, oneapi::math::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::uniform<double, oneapi::math::rng::uniform_method::accurate>&
            distr,
        std::int64_t n, double* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::gaussian<float, oneapi::math::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, float* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::gaussian<double, oneapi::math::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, double* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::gaussian<float, oneapi::math::rng::gaussian_method::icdf>& distr,
        std::int64_t n, float* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::gaussian<double, oneapi::math::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::lognormal<float, oneapi::math::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, float* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::lognormal<double,
                                           oneapi::math::rng::lognormal_method::box_muller2>& distr,
        std::int64_t n, double* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::lognormal<float, oneapi::math::rng::lognormal_method::icdf>& distr,
        std::int64_t n, float* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const oneapi::math::rng::lognormal<double, oneapi::math::rng::lognormal_method::icdf>&
            distr,
        std::int64_t n, double* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                                 std::int64_t n, std::int32_t* r,
                                 const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                                 std::int64_t n, std::uint32_t* r,
                                 const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const std::vector<sycl::event>& dependencies) override {
        RETHROW_ONEMKL_EXCEPTIONS_RET(oneapi::mkl::rng::generate(
            detail::get_onemkl_distribution(distr), engine_, n, r, dependencies));
    }

    virtual sycl::event generate(const bits<std::uint32_t>& /*distr*/, std::int64_t /*n*/,
                                 std::uint32_t* /*r*/,
                                 const std::vector<sycl::event>& /*dependencies*/) override {
        throw unimplemented("rng/mklgpu", "philox4x32x10::generate",
                            "bits distribution is not supported");
        return {};
    }

    virtual oneapi::math::rng::detail::engine_impl* copy_state() override {
        return new philox4x32x10_impl(this);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::rng::skip_ahead(engine_, num_to_skip));
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        RETHROW_ONEMKL_EXCEPTIONS(oneapi::mkl::rng::skip_ahead(engine_, num_to_skip));
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::math::unimplemented("rng", "leapfrog");
    }

    virtual ~philox4x32x10_impl() override {}

private:
    oneapi::mkl::rng::philox4x32x10 engine_;
};

oneapi::math::rng::detail::engine_impl* create_philox4x32x10(sycl::queue queue,
                                                             std::uint64_t seed) {
    return new philox4x32x10_impl(queue, seed);
}

oneapi::math::rng::detail::engine_impl* create_philox4x32x10(
    sycl::queue queue, std::initializer_list<std::uint64_t> seed) {
    return new philox4x32x10_impl(queue, seed);
}

} // namespace mklgpu
} // namespace rng
} // namespace math
} // namespace oneapi

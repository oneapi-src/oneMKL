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
#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

#include "oneapi/mkl/rng/detail/engine_impl.hpp"
#include "oneapi/mkl/rng/engines.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/detail/curand/onemkl_rng_curand.hpp"

#include "curand_helper.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace curand {

#if !defined(_WIN64)
/*
 * Note that cuRAND consists of two pieces: a host (CPU) API and a device (GPU) API.
 * The host API acts like any standard library; the `curand.h' header is included and the functions
 * can be called as usual. The generator is instantiated on the host and random numbers can be
 * generated on either the host CPU or device. For device-side generation, calls to the library happen
 * on the host, but the actual work of RNG is done on the device. In this case, the resulting random
 * numbers are stored in global memory on the device. These random numbers can then be used in other
 * kernels or be copied back to the host for further processing. For host-side generation, everything
 * is done on the host, and the random numbers are stored in host memory.
 * 
 * The second piece is the device header, `curand_kernel.h'. Using this file permits setting up random
 * number generator states and generating sequences of random numbers. This allows random numbers to
 * be generated and immediately consumed in other kernels without requiring the random numbers to be
 * written to, and read from, global memory.
 * 
 * Here we utilize the host API since this is most aligned with how oneMKL generates random numbers.
 * 
*/
class philox4x32x10_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    philox4x32x10_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        curandStatus_t status;
        CURAND_CALL(curandCreateGenerator, status, &engine_, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    }

    philox4x32x10_impl(cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    philox4x32x10_impl(const philox4x32x10_impl* other)
            : oneapi::mkl::rng::detail::engine_impl(*other) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    // Buffers API

    inline void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                // cgh.interop_task([=](cl::sycl::interop_handler ih) {
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<float*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniform, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
        // Post-processing on device (currently crashes with `OPENCL_INVALID_BINARY')
        // curandGenerateUniform generates uniform random numbers on [0, 1).
        // We need to convert to range [distr.a(), distr.b()).
        // queue_.wait_and_throw();
        // queue_
        //     .submit([&](cl::sycl::handler& cgh) {
        //         auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
        //         cgh.parallel_for<class philox4x32x10_impl>(
        //             cl::sycl::range<1>(n), [=](cl::sycl::id<1> id) {
        //                 acc[id] = acc[id] * (distr.b() - distr.a()) + distr.a();
        //             });
        //     })
        //     .wait_and_throw();
        // queue_.wait_and_throw();
        // Post-processing on host (CPU)
        // curandGenerateUniform generates uniform random numbers on [0, 1).
        // We need to convert to range [distr.a(), distr.b()).
        // auto acc = r.get_access<cl::sycl::access::mode::read_write>();
        // for (unsigned int i = 0; i < n; ++i) {
        //     acc[i] = acc[i] * (distr.b() - distr.a()) + distr.a();
        // }
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        queue_.submit([&](cl::sycl::handler& cgh) {
            auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
            cgh.interop_task([=](cl::sycl::interop_handler ih) {
                auto r_ptr = reinterpret_cast<double*>(ih.get_mem<cl::sycl::backend::cuda>(acc));
                curandStatus_t status;
                CURAND_CALL(curandGenerateUniformDouble, status, engine_, r_ptr, n);
            });
        });
        // Post-processing
        // curandGenerateUniform generates uniform random numbers on [0, 1).
        // We need to convert to range [distr.a(), distr.b()).
        auto acc = r.get_access<cl::sycl::access::mode::read_write>();
        for (unsigned int i = 0; i < n; ++i) {
            acc[i] = acc[i] * (distr.b() - distr.a()) + distr.a();
        }
    }

    virtual void generate(const oneapi::mkl::rng::uniform<
                              std::int32_t, oneapi::mkl::rng::uniform_method::standard>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              float, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              double, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              float, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              double, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

private:
    // The engine (CUDA "generator")
    curandGenerator_t engine_;

    // USM APIs

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<std::int32_t, oneapi::mkl::rng::uniform_method::standard>&
            distr,
        std::int64_t n, std::int32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        ;
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const bernoulli<std::int32_t, bernoulli_method::icdf>& distr, std::int64_t n,
        std::int32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr, std::int64_t n,
        std::uint32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual cl::sycl::event generate(
        const bits<std::uint32_t>& distr, std::int64_t n, std::uint32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual oneapi::mkl::rng::detail::engine_impl* copy_state() override {
        return new philox4x32x10_impl(this);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "skip_ahead");
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "skip_ahead");
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::mkl::unimplemented("rng", "leapfrog");
    }

    virtual ~philox4x32x10_impl() override {
        oneapi::mkl::unimplemented("rng", "~philox4x32x10_impl");
    }

private:
    // oneapi::mkl::rng::detail::engine_base_impl<oneapi::mkl::rng::philox4x32x10>* engine_;
};
#else // cuRAND backend is currently not supported on Windows
class philox4x32x10_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    philox4x32x10_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    philox4x32x10_impl(cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    philox4x32x10_impl(const philox4x32x10_impl* other)
            : oneapi::mkl::rng::detail::engine_impl(*other) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    // Buffers API

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::uniform<
                              std::int32_t, oneapi::mkl::rng::uniform_method::standard>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              float, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              double, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              float, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              double, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    // USM APIs

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<std::int32_t, oneapi::mkl::rng::uniform_method::standard>&
            distr,
        std::int64_t n, std::int32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const bernoulli<std::int32_t, bernoulli_method::icdf>& distr, std::int64_t n,
        std::int32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr, std::int64_t n,
        std::uint32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const bits<std::uint32_t>& distr, std::int64_t n, std::uint32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return cl::sycl::event{};
    }

    virtual oneapi::mkl::rng::detail::engine_impl* copy_state() override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
        return nullptr;
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine");
    }

    virtual ~philox4x32x10_impl() override {}
};
#endif

oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(sycl::queue queue, std::uint64_t seed) {
    return new philox4x32x10_impl(queue, seed);
}

oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(
    cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed) {
    return new philox4x32x10_impl(queue, seed);
}

} // namespace curand
} // namespace rng
} // namespace mkl
} // namespace oneapi

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

#include "mkl_vsl.h"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/detail/engine_impl.hpp"
#include "oneapi/mkl/rng/detail/mklcpu/onemkl_rng_mklcpu.hpp"

#include "cpu_common.hpp"

namespace oneapi::mkl::rng::mklcpu {

class mcg59_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    mcg59_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        vslNewStream(&stream_, VSL_BRNG_MCG59, seed);
        state_size_ = vslGetStreamSize(stream_);
    }

    mcg59_impl(const mcg59_impl* other) : oneapi::mkl::rng::detail::engine_impl(*other) {
        vslCopyStream(&stream_, other->stream_);
        state_size_ = vslGetStreamSize(stream_);
    }

    // Buffers APIs

    virtual void generate(const uniform<float, uniform_method::standard>& distr, std::int64_t n,
                          cl::sycl::buffer<float>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             acc_r.get_pointer(), distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<double, uniform_method::standard>& distr, std::int64_t n,
                          cl::sycl::buffer<double>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             acc_r.get_pointer(), distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<std::int32_t, uniform_method::standard>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             acc_r.get_pointer(), distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<float, uniform_method::accurate>& distr, std::int64_t n,
                          cl::sycl::buffer<float>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             acc_r.get_pointer(), distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<double, uniform_method::accurate>& distr, std::int64_t n,
                          cl::sycl::buffer<double>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             acc_r.get_pointer(), distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const gaussian<float, gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                              static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                              acc_r.get_pointer(), distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const gaussian<double, gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                              static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                              acc_r.get_pointer(), distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const gaussian<float, gaussian_method::icdf>& distr, std::int64_t n,
                          cl::sycl::buffer<float>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                              static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                              acc_r.get_pointer(), distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const gaussian<double, gaussian_method::icdf>& distr, std::int64_t n,
                          cl::sycl::buffer<double>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                              static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                              acc_r.get_pointer(), distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const lognormal<float, lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2,
                               static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                               acc_r.get_pointer(), distr.m(), distr.s(), distr.displ(),
                               distr.scale());
            });
        });
    }

    virtual void generate(const lognormal<double, lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2,
                               static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                               acc_r.get_pointer(), distr.m(), distr.s(), distr.displ(),
                               distr.scale());
            });
        });
    }

    virtual void generate(const lognormal<float, lognormal_method::icdf>& distr, std::int64_t n,
                          cl::sycl::buffer<float>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF,
                               static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                               acc_r.get_pointer(), distr.m(), distr.s(), distr.displ(),
                               distr.scale());
            });
        });
    }

    virtual void generate(const lognormal<double, lognormal_method::icdf>& distr, std::int64_t n,
                          cl::sycl::buffer<double>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF,
                               static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                               acc_r.get_pointer(), distr.m(), distr.s(), distr.displ(),
                               distr.scale());
            });
        });
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF,
                               static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                               acc_r.get_pointer(), distr.p());
            });
        });
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                std::uint32_t* r_ptr = acc_r.get_pointer();
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF,
                               static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                               reinterpret_cast<std::int32_t*>(r_ptr), distr.p());
            });
        });
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             acc_r.get_pointer(), distr.lambda());
            });
        });
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                std::uint32_t* r_ptr = acc_r.get_pointer();
                viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM,
                             static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                             reinterpret_cast<std::int32_t*>(r_ptr), distr.lambda());
            });
        });
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t>& r) override {
        cl::sycl::buffer<char> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::accessor acc_stream{ stream_buf, cgh, cl::sycl::read_write };
            cl::sycl::accessor acc_r{ r, cgh, cl::sycl::read_write };
            host_task<kernel_name<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD,
                                 static_cast<VSLStreamStatePtr>(acc_stream.get_pointer()), n,
                                 acc_r.get_pointer());
            });
        });
    }

    // USM APIs

    virtual cl::sycl::event generate(const uniform<float, uniform_method::standard>& distr,
                                     std::int64_t n, float* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, distr.a(), distr.b());
            });
        });
    }

    virtual cl::sycl::event generate(const uniform<double, uniform_method::standard>& distr,
                                     std::int64_t n, double* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, distr.a(), distr.b());
            });
        });
    }

    virtual cl::sycl::event generate(const uniform<std::int32_t, uniform_method::standard>& distr,
                                     std::int64_t n, std::int32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, distr.a(), distr.b());
            });
        });
    }

    virtual cl::sycl::event generate(const uniform<float, uniform_method::accurate>& distr,
                                     std::int64_t n, float* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, n, r, distr.a(),
                             distr.b());
            });
        });
    }

    virtual cl::sycl::event generate(const uniform<double, uniform_method::accurate>& distr,
                                     std::int64_t n, double* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, n, r, distr.a(),
                             distr.b());
            });
        });
    }

    virtual cl::sycl::event generate(const gaussian<float, gaussian_method::box_muller2>& distr,
                                     std::int64_t n, float* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual cl::sycl::event generate(const gaussian<double, gaussian_method::box_muller2>& distr,
                                     std::int64_t n, double* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual cl::sycl::event generate(const gaussian<float, gaussian_method::icdf>& distr,
                                     std::int64_t n, float* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual cl::sycl::event generate(const gaussian<double, gaussian_method::icdf>& distr,
                                     std::int64_t n, double* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual cl::sycl::event generate(const lognormal<float, lognormal_method::box_muller2>& distr,
                                     std::int64_t n, float* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream, n, r, distr.m(),
                               distr.s(), distr.displ(), distr.scale());
            });
        });
    }

    virtual cl::sycl::event generate(const lognormal<double, lognormal_method::box_muller2>& distr,
                                     std::int64_t n, double* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream, n, r, distr.m(),
                               distr.s(), distr.displ(), distr.scale());
            });
        });
    }

    virtual cl::sycl::event generate(const lognormal<float, lognormal_method::icdf>& distr,
                                     std::int64_t n, float* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF, stream, n, r, distr.m(), distr.s(),
                               distr.displ(), distr.scale());
            });
        });
    }

    virtual cl::sycl::event generate(const lognormal<double, lognormal_method::icdf>& distr,
                                     std::int64_t n, double* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF, stream, n, r, distr.m(), distr.s(),
                               distr.displ(), distr.scale());
            });
        });
    }

    virtual cl::sycl::event generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                                     std::int64_t n, std::int32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, n, r, distr.p());
            });
        });
    }

    virtual cl::sycl::event generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                                     std::int64_t n, std::uint32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, n,
                               reinterpret_cast<int32_t*>(r), distr.p());
            });
        });
    }

    virtual cl::sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, stream, n, r, distr.lambda());
            });
        });
    }

    virtual cl::sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, stream, n,
                             reinterpret_cast<int32_t*>(r), distr.lambda());
            });
        });
    }

    virtual cl::sycl::event generate(const bits<std::uint32_t>& distr, std::int64_t n,
                                     std::uint32_t* r,
                                     const std::vector<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mcg59_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD, stream, n,
                                 reinterpret_cast<uint32_t*>(r));
            });
        });
    }

    virtual oneapi::mkl::rng::detail::engine_impl* copy_state() override {
        return new mcg59_impl(this);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        vslSkipAheadStream(stream_, num_to_skip);
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "mcg59 skipAheadEx");
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        vslLeapfrogStream(stream_, idx, stride);
    }

    virtual ~mcg59_impl() override {
        vslDeleteStream(&stream_);
    }

private:
    VSLStreamStatePtr stream_;
    std::int32_t state_size_;
};

oneapi::mkl::rng::detail::engine_impl* create_mcg59(cl::sycl::queue queue, std::uint64_t seed) {
    return new mcg59_impl(queue, seed);
}

oneapi::mkl::rng::detail::engine_impl* create_mcg59(cl::sycl::queue queue,
                                                    std::initializer_list<std::uint64_t> seed) {
    throw oneapi::mkl::unimplemented("rng", "mcg59 skipAheadEx");
}

} // namespace oneapi::mkl::rng::mklcpu

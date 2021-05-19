/*******************************************************************************
 * cuRAND back-end Copyright (c) 2021, The Regents of the University of
 * California, through Lawrence Berkeley National Laboratory (subject to receipt
 * of any required approvals from the U.S. Dept. of Energy). All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * (1) Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * (2) Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * (3) Neither the name of the University of California, Lawrence Berkeley
 * National Laboratory, U.S. Dept. of Energy nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * You are under no obligation whatsoever to provide any bug fixes, patches,
 * or upgrades to the features, functionality or performance of the source
 * code ("Enhancements") to anyone; however, if you choose to make your
 * Enhancements available either publicly, or directly to Lawrence Berkeley
 * National Laboratory, without imposing a separate written license agreement
 * for such Enhancements, then you hereby grant the following license: a
 * non-exclusive, royalty-free perpetual license to install, use, modify,
 * prepare derivative works, incorporate into other computer software,
 * distribute, and sublicense such enhancements or derivative works thereof,
 * in binary and source code form.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Intellectual Property Office at
 * IPO@lbl.gov.
 *
 * NOTICE.  This Software was developed under funding from the U.S. Department
 * of Energy and the U.S. Government consequently retains certain rights.  As
 * such, the U.S. Government has been granted for itself and others acting on
 * its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, distribute copies to the public, prepare derivative
 * works, and perform publicly and display publicly, and to permit others to do
 * so.
 ******************************************************************************/

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <iostream>

#include "oneapi/mkl/rng/detail/engine_impl.hpp"
// #include "oneapi/mkl/rng/engines.hpp"
#include "curand_helper.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/detail/curand/onemkl_rng_curand.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace curand {

#if !defined(_WIN64)
/*
 * Note that cuRAND consists of two pieces: a host (CPU) API and a device (GPU)
 * API. The host API acts like any standard library; the `curand.h' header is
 * included and the functions can be called as usual. The generator is
 * instantiated on the host and random numbers can be generated on either the
 * host CPU or device. For device-side generation, calls to the library happen
 * on the host, but the actual work of RNG is done on the device. In this case,
 * the resulting random numbers are stored in global memory on the device. These
 * random numbers can then be used in other kernels or be copied back to the
 * host for further processing. For host-side generation, everything is done on
 * the host, and the random numbers are stored in host memory.
 *
 * The second piece is the device header, `curand_kernel.h'. Using this file
 * permits setting up random number generator states and generating sequences of
 * random numbers. This allows random numbers to be generated and immediately
 * consumed in other kernels without requiring the random numbers to be written
 * to, and read from, global memory.
 *
 * Here we utilize the host API since this is most aligned with how oneMKL
 * generates random numbers.
 *
 */
class philox4x32x10_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    philox4x32x10_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        curandStatus_t status;
        CURAND_CALL(curandCreateGenerator, status, &engine_, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed, status, engine_, (unsigned long long)seed);
    }

    philox4x32x10_impl(cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : oneapi::mkl::rng::detail::engine_impl(queue) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine",
                                         "multi-seed unsupported by cuRAND backend");
    }

    philox4x32x10_impl(const philox4x32x10_impl* other)
            : oneapi::mkl::rng::detail::engine_impl(*other) {
        throw oneapi::mkl::unimplemented("rng", "philox4x32x10 engine",
                                         "copy construction unsupported by cuRAND backend");
    }

    // Buffers API

    virtual inline void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<float*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniform, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
        range_transform_fp<float>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<double*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniformDouble, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
        range_transform_fp<double>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual void generate(const oneapi::mkl::rng::uniform<
                              std::int32_t, oneapi::mkl::rng::uniform_method::standard>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        cl::sycl::buffer<std::uint32_t, 1> ib(n);
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = ib.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr = reinterpret_cast<std::uint32_t*>(
                        ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerate, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
        range_transform_int<std::int32_t>(queue_, distr.a(), distr.b(), n, ib, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<float*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniform, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
        range_transform_fp_accurate<float>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<double*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniformDouble, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
        range_transform_fp_accurate<double>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              float, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<float*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateNormal, status, engine_, r_ptr, n, distr.mean(),
                                distr.stddev());
                });
            })
            .wait_and_throw();
    }

    virtual void generate(const oneapi::mkl::rng::gaussian<
                              double, oneapi::mkl::rng::gaussian_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<double*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateNormalDouble, status, engine_, r_ptr, n, distr.mean(),
                                distr.stddev());
                });
            })
            .wait_and_throw();
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              float, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<float*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateLogNormal, status, engine_, r_ptr, n, distr.m(),
                                distr.s());
                });
            })
            .wait_and_throw();
    }

    virtual void generate(const oneapi::mkl::rng::lognormal<
                              double, oneapi::mkl::rng::lognormal_method::box_muller2>& distr,
                          std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr =
                        reinterpret_cast<double*>(ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateLogNormalDouble, status, engine_, r_ptr, n, distr.m(),
                                distr.s());
                });
            })
            .wait_and_throw();
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<float, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, cl::sycl::buffer<double, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, cl::sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t, 1>& r) override {
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                auto acc = r.template get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    auto r_ptr = reinterpret_cast<std::uint32_t*>(
                        ih.get_native_mem<cl::sycl::backend::cuda>(acc));
                    curandStatus_t status;
                    CURAND_CALL(curandGenerate, status, engine_, r_ptr, n);
                });
            })
            .wait_and_throw();
    }

    // USM APIs

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniform, status, engine_, r, n);
                });
            })
            .wait_and_throw();
        return range_transform_fp<float>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniformDouble, status, engine_, r, n);
                });
            })
            .wait_and_throw();
        return range_transform_fp<double>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<std::int32_t, oneapi::mkl::rng::uniform_method::standard>&
            distr,
        std::int64_t n, std::int32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        std::uint32_t* ib = (std::uint32_t*)malloc_device(
            n * sizeof(std::uint32_t), queue_.get_device(), queue_.get_context());
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    curandStatus_t status;
                    CURAND_CALL(curandGenerate, status, engine_, ib, n);
                });
            })
            .wait_and_throw();
        return range_transform_int(queue_, distr.a(), distr.b(), n, ib, r);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniform, status, engine_, r, n);
                });
            })
            .wait_and_throw();
        return range_transform_fp_accurate<float>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        queue_
            .submit([&](cl::sycl::handler& cgh) {
                cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                    curandStatus_t status;
                    CURAND_CALL(curandGenerateUniformDouble, status, engine_, r, n);
                });
            })
            .wait_and_throw();
        return range_transform_fp_accurate<double>(queue_, distr.a(), distr.b(), n, r);
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                curandStatus_t status;
                CURAND_CALL(curandGenerateNormal, status, engine_, r, n, distr.mean(),
                            distr.stddev());
            });
        });
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2>&
            distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                curandStatus_t status;
                CURAND_CALL(curandGenerateNormalDouble, status, engine_, r, n, distr.mean(),
                            distr.stddev());
            });
        });
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                curandStatus_t status;
                CURAND_CALL(curandGenerateLogNormal, status, engine_, r, n, distr.m(), distr.s());
            });
        });
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::box_muller2>&
            distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                curandStatus_t status;
                CURAND_CALL(curandGenerateLogNormalDouble, status, engine_, r, n, distr.m(),
                            distr.s());
            });
        });
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<float, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const oneapi::mkl::rng::lognormal<double, oneapi::mkl::rng::lognormal_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const bernoulli<std::int32_t, bernoulli_method::icdf>& distr, std::int64_t n,
        std::int32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr, std::int64_t n,
        std::uint32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        throw oneapi::mkl::unimplemented(
            "rng", "philox4x32x10 engine",
            "ICDF method not used for pseudorandom generators in cuRAND backend");
        return cl::sycl::event{};
    }

    virtual cl::sycl::event generate(
        const bits<std::uint32_t>& distr, std::int64_t n, std::uint32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        cl::sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](cl::sycl::handler& cgh) {
            cgh.codeplay_host_task([=](cl::sycl::interop_handle ih) {
                curandStatus_t status;
                CURAND_CALL(curandGenerate, status, engine_, r, n);
            });
        });
    }

    virtual oneapi::mkl::rng::detail::engine_impl* copy_state() override {
        return new philox4x32x10_impl(this);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        curandStatus_t status;
        CURAND_CALL(curandSetGeneratorOffset, status, engine_, num_to_skip);
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        throw oneapi::mkl::unimplemented("rng", "skip_ahead",
                                         "initializer list unsupported by cuRAND backend");
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::mkl::unimplemented("rng", "leapfrog", "unsupported by cuRAND backend");
    }

    virtual ~philox4x32x10_impl() override {
        curandDestroyGenerator(engine_);
    }

private:
    curandGenerator_t engine_;
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

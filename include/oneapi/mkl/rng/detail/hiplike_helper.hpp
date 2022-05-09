/*******************************************************************************
 * Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) 
 * and Computing Centre (URZ)
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
#ifndef _MKL_RNG_HIPLIKE_HELPER_HPP_
#define _MKL_RNG_HIPLIKE_HELPER_HPP_

namespace oneapi {
namespace mkl {
namespace rng {

// Static template functions oneapi::mkl::rng::curand::range_transform_fp for
// Buffer and USM APIs
//
// cuRAND has no built-in functionality to specify a custom range for sampling
// random numbers; `curandGenerateUniform' generates uniform random numbers on
// [0, 1). This function is used to convert to range [a, b).
//
// Supported types:
//      float
//      double
//
// Input arguments:
//      queue - the queue to submit the kernel to
//      a     - range lower bound (inclusive)
//      b     - range upper bound (exclusive)
//      r     - buffer to store transformed random numbers
template <typename T>
static inline void range_transform_fp(cl::sycl::queue& queue, T a, T b, std::int64_t n,
                                      cl::sycl::buffer<T, 1>& r) {
    queue.submit([&](cl::sycl::handler& cgh) {
        auto acc = r.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(cl::sycl::range<1>(n),
                         [=](cl::sycl::id<1> id) { acc[id[0]] = acc[id[0]] * (b - a) + a; });
    });
}
template <typename T>
static inline cl::sycl::event range_transform_fp(cl::sycl::queue& queue, T a, T b, std::int64_t n,
                                                 T* r) {
    return queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(cl::sycl::range<1>(n),
                         [=](cl::sycl::id<1> id) { r[id[0]] = r[id[0]] * (b - a) + a; });
    });
}
template <typename T>
static inline void range_transform_fp_accurate(cl::sycl::queue& queue, T a, T b, std::int64_t n,
                                               cl::sycl::buffer<T, 1>& r) {
    queue.submit([&](cl::sycl::handler& cgh) {
        auto acc = r.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(cl::sycl::range<1>(n), [=](cl::sycl::id<1> id) {
            acc[id[0]] = acc[id[0]] * (b - a) + a;
            if (acc[id[0]] < a) {
                acc[id[0]] = a;
            }
            else if (acc[id[0]] > b) {
                acc[id[0]] = b;
            }
        });
    });
}
template <typename T>
static inline cl::sycl::event range_transform_fp_accurate(cl::sycl::queue& queue, T a, T b,
                                                          std::int64_t n, T* r) {
    return queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(cl::sycl::range<1>(n), [=](cl::sycl::id<1> id) {
            r[id[0]] = r[id[0]] * (b - a) + a;
            if (r[id[0]] < a) {
                r[id[0]] = a;
            }
            else if (r[id[0]] > b) {
                r[id[0]] = b;
            }
        });
    });
}

// Static template functions oneapi::mkl::rng::curand::range_transform_int for
// Buffer and USM APIs
//
// cuRAND has no built-in functionality to specify a custom range for sampling
// random numbers; `curandGenerateUniform' generates uniform random numbers on
// [0, 1). This function is used to convert to range [a, b).
//
// Supported types:
//      std::int32_t
//      std::uint32_t
//
// Input arguments:
//      queue - the queue to submit the kernel to
//      a     - range lower bound (inclusive)
//      b     - range upper bound (exclusive)
//      r     - buffer to store transformed random numbers
template <typename T>
inline void range_transform_int(cl::sycl::queue& queue, T a, T b, std::int64_t n,
                                cl::sycl::buffer<std::uint32_t, 1>& in,
                                cl::sycl::buffer<T, 1>& out) {
    queue.submit([&](cl::sycl::handler& cgh) {
        auto acc_in = in.template get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_out = out.template get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for(cl::sycl::range<1>(n),
                         [=](cl::sycl::id<1> id) { acc_out[id[0]] = a + acc_in[id[0]] % (b - a); });
    });
}
template <typename T>
inline cl::sycl::event range_transform_int(cl::sycl::queue& queue, T a, T b, std::int64_t n,
                                           std::uint32_t* in, T* out) {
    return queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(cl::sycl::range<1>(n),
                         [=](cl::sycl::id<1> id) { out[id[0]] = a + in[id[0]] % (b - a); });
    });
}

// Static template functions oneapi::mkl::rng::curand::sample_bernoulli for
// Buffer and USM APIs
//
// cuRAND has no built-in functionality to sample from a Bernoulli distribution.
// The implementation here uses uniformly-generated random numbers and returns
// the corresponding Bernoulli distribution based on a probability.
//
// Supported types:
//      std::int32_t
//      std::uint32_t
//
// Input arguments:
//      queue - the queue to submit the kernel to
//      p     - success probablity of a trial
//      in    - buffer containing uniformly-generated random numbers
//      out   - buffer to store Bernoulli
template <typename T>
static inline void sample_bernoulli_from_uniform(cl::sycl::queue& queue, float p, std::int64_t n,
                                                 cl::sycl::buffer<float, 1> in,
                                                 cl::sycl::buffer<T, 1>& out) {
    queue.submit([&](cl::sycl::handler& cgh) {
        auto acc_in = in.template get_access<cl::sycl::access::mode::read>(cgh);
        auto acc_out = out.template get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for(cl::sycl::range<1>(n),
                         [=](cl::sycl::id<1> id) { acc_out[id[0]] = acc_in[id[0]] < p; });
    });
}
template <typename T>
static inline cl::sycl::event sample_bernoulli_from_uniform(cl::sycl::queue& queue, float p,
                                                            std::int64_t n, float* in, T* out) {
    return queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(cl::sycl::range<1>(n),
                         [=](cl::sycl::id<1> id) { out[id[0]] = in[id[0]] < p; });
    });
}

} // namespace rng
} // namespace mkl
} // namespace oneapi

//namespace oneapi::mkl::rng::rocrand::detail = oneapi::mkl::rng::curand::detail;

#endif // _MKL_RNG_HIPLIKE_HELPER_HPP_

/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

/*
When only a specific backend library is required (eg. libonemkl_dft_<BACKEND>) 
it may be preferenable to only link to that specific backend library without
the requirement that the main OneMKL library also be linked.

To enable this, function signatures from the main dft library are duplicated
here, forwarding directly to the backend implementation instead of the function
table lookup mechanism.

This file should be included for each backend, with <BACKEND> defined to match
the namespace of the backend's implementation.
*/

#include "oneapi/mkl/dft/forward.hpp"
#include "oneapi/mkl/dft/backward.hpp"

namespace oneapi {
namespace mkl {
namespace dft {

#define ONEAPI_MKL_DFT_SIGNATURES(EXT, PRECISION, DOMAIN, T_REAL, T_FORWARD, T_BACKWARD)                \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(             \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout) {           \
        oneapi::mkl::dft::BACKEND::compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>,          \
                                                   T_REAL>(desc, inout);                                \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex*/                                                                    \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(         \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & inout) {       \
        oneapi::mkl::dft::BACKEND::compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>,          \
                                                   T_BACKWARD>(desc, inout);                            \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(             \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout_re,          \
        sycl::buffer<T_REAL, 1> & inout_im) {                                                           \
        oneapi::mkl::dft::BACKEND::compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>,          \
                                                   T_REAL>(desc, inout_re, inout_im);                   \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(                 \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_FORWARD, 1> & in,             \
        sycl::buffer<T_BACKWARD, 1> & out) {                                                            \
        oneapi::mkl::dft::BACKEND::compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>,          \
                                                   T_FORWARD, T_BACKWARD>(desc, in, out);               \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform - real*/                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in,                \
        sycl::buffer<T_REAL, 1> & out) {                                                                \
        oneapi::mkl::dft::BACKEND::compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>,          \
                                                   T_REAL, T_REAL>(desc, in, out);                      \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in_re,             \
        sycl::buffer<T_REAL, 1> & in_im, sycl::buffer<T_REAL, 1> & out_re,                              \
        sycl::buffer<T_REAL, 1> & out_im) {                                                             \
        oneapi::mkl::dft::BACKEND::compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>,          \
                                                   T_REAL, T_REAL>(desc, in_re, in_im, out_re,          \
                                                                   out_im);                             \
    }                                                                                                   \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(      \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout,                              \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                              \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(desc, inout, dependencies);             \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex*/                                                                    \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                            \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,                          \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                              \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(desc, inout, dependencies);         \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(      \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re, T_REAL * inout_im,        \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                              \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(desc, inout_re, inout_im,               \
                                                                dependencies);                          \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(                 \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_FORWARD * in, T_BACKWARD * out,            \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                              \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(desc, in, out,           \
                                                                               dependencies);           \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in, T_REAL * out,                   \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                              \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(desc, in, out,                  \
                                                                        dependencies);                  \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re, T_REAL * in_im,              \
        T_REAL * out_re, T_REAL * out_im, const std::vector<sycl::event>& dependencies) {               \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                              \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                                \
            desc, in_re, in_im, out_re, out_im, dependencies);                                          \
    }                                                                                                   \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(            \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout) {           \
        oneapi::mkl::dft::BACKEND::compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>,         \
                                                    T_REAL>(desc, inout);                               \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex */                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & inout) {       \
        oneapi::mkl::dft::BACKEND::compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>,         \
                                                    T_BACKWARD>(desc, inout);                           \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(            \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout_re,          \
        sycl::buffer<T_REAL, 1> & inout_im) {                                                           \
        oneapi::mkl::dft::BACKEND::compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>,         \
                                                    T_REAL>(desc, inout_re, inout_im);                  \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(                \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & in,            \
        sycl::buffer<T_FORWARD, 1> & out) {                                                             \
        oneapi::mkl::dft::BACKEND::compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>,         \
                                                    T_BACKWARD, T_FORWARD>(desc, in, out);              \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform - real*/                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in,                \
        sycl::buffer<T_REAL, 1> & out) {                                                                \
        oneapi::mkl::dft::BACKEND::compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>,         \
                                                    T_REAL, T_REAL>(desc, in, out);                     \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in_re,             \
        sycl::buffer<T_REAL, 1> & in_im, sycl::buffer<T_REAL, 1> & out_re,                              \
        sycl::buffer<T_REAL, 1> & out_im) {                                                             \
        oneapi::mkl::dft::BACKEND::compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>,         \
                                                    T_REAL, T_REAL>(desc, in_re, in_im, out_re,         \
                                                                    out_im);                            \
    }                                                                                                   \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(                               \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout,                              \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                             \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(desc, inout, dependencies);             \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex*/                                                                    \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                           \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,                          \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                             \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(desc, inout, dependencies);         \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(                               \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re, T_REAL * inout_im,        \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                             \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(desc, inout_re, inout_im,               \
                                                                dependencies);                          \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(                \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * in, T_FORWARD * out,            \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                             \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(desc, in, out,           \
                                                                               dependencies);           \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform - real*/                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in, T_REAL * out,                   \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                             \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(desc, in, out,                  \
                                                                        dependencies);                  \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re, T_REAL * in_im,              \
        T_REAL * out_re, T_REAL * out_im, const std::vector<sycl::event>& dependencies) {               \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                             \
            dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                                \
            desc, in_re, in_im, out_re, out_im, dependencies);                                          \
    }

// Signatures with forward_t=complex, backwards_t=complex are already instantiated for complex domain
// but not real domain.
#define ONEAPI_MKL_DFT_REAL_ONLY_SIGNATURES(EXT, PRECISION, T_COMPLEX)                            \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, detail::domain::REAL>,  \
                                       T_COMPLEX, T_COMPLEX>(                                     \
        dft::detail::descriptor<PRECISION, detail::domain::REAL> & desc,                          \
        sycl::buffer<T_COMPLEX, 1> & in, sycl::buffer<T_COMPLEX, 1> & out) {                      \
        oneapi::mkl::dft::BACKEND::compute_forward<                                               \
            dft::detail::descriptor<PRECISION, detail::domain::REAL>, T_COMPLEX, T_COMPLEX>(      \
            desc, in, out);                                                                       \
    }                                                                                             \
                                                                                                  \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT sycl::event compute_forward<                                                    \
        dft::detail::descriptor<PRECISION, detail::domain::REAL>, T_COMPLEX, T_COMPLEX>(          \
        dft::detail::descriptor<PRECISION, detail::domain::REAL> & desc, T_COMPLEX * in,          \
        T_COMPLEX * out, const std::vector<sycl::event>& dependencies) {                          \
        return oneapi::mkl::dft::BACKEND::compute_forward<                                        \
            dft::detail::descriptor<PRECISION, detail::domain::REAL>, T_COMPLEX, T_COMPLEX>(      \
            desc, in, out, dependencies);                                                         \
    }                                                                                             \
                                                                                                  \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, detail::domain::REAL>, \
                                        T_COMPLEX, T_COMPLEX>(                                    \
        dft::detail::descriptor<PRECISION, detail::domain::REAL> & desc,                          \
        sycl::buffer<T_COMPLEX, 1> & in, sycl::buffer<T_COMPLEX, 1> & out) {                      \
        oneapi::mkl::dft::BACKEND::compute_backward<                                              \
            dft::detail::descriptor<PRECISION, detail::domain::REAL>, T_COMPLEX, T_COMPLEX>(      \
            desc, in, out);                                                                       \
    }                                                                                             \
                                                                                                  \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT sycl::event compute_backward<                                                   \
        dft::detail::descriptor<PRECISION, detail::domain::REAL>, T_COMPLEX, T_COMPLEX>(          \
        dft::detail::descriptor<PRECISION, detail::domain::REAL> & desc, T_COMPLEX * in,          \
        T_COMPLEX * out, const std::vector<sycl::event>& dependencies) {                          \
        return oneapi::mkl::dft::BACKEND::compute_backward<                                       \
            dft::detail::descriptor<PRECISION, detail::domain::REAL>, T_COMPLEX, T_COMPLEX>(      \
            desc, in, out, dependencies);                                                         \
    }

ONEAPI_MKL_DFT_SIGNATURES(f, dft::detail::precision::SINGLE, dft::detail::domain::REAL, float,
                          float, std::complex<float>)
ONEAPI_MKL_DFT_REAL_ONLY_SIGNATURES(f, dft::detail::precision::SINGLE, std::complex<float>)
ONEAPI_MKL_DFT_SIGNATURES(c, dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX, float,
                          std::complex<float>, std::complex<float>)
ONEAPI_MKL_DFT_SIGNATURES(d, dft::detail::precision::DOUBLE, dft::detail::domain::REAL, double,
                          double, std::complex<double>)
ONEAPI_MKL_DFT_REAL_ONLY_SIGNATURES(d, dft::detail::precision::DOUBLE, std::complex<double>)
ONEAPI_MKL_DFT_SIGNATURES(z, dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX, double,
                          std::complex<double>, std::complex<double>)
#undef ONEAPI_MKL_DFT_SIGNATURES

} // namespace dft
} // namespace mkl
} // namespace oneapi

/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/mkl/dft/detail/dft_loader.hpp"
#include "oneapi/mkl/dft/forward.hpp"
#include "oneapi/mkl/dft/backward.hpp"

#include "function_table_initializer.hpp"
#include "dft/function_table.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

static oneapi::mkl::detail::table_initializer<mkl::domain::dft, dft_function_table_t>
    function_tables;

template <>
commit_impl* create_commit<precision::SINGLE, domain::COMPLEX>(
    const descriptor<precision::SINGLE, domain::COMPLEX>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[libkey].create_commit_sycl_fz(desc, sycl_queue);
}

template <>
commit_impl* create_commit<precision::DOUBLE, domain::COMPLEX>(
    const descriptor<precision::DOUBLE, domain::COMPLEX>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[libkey].create_commit_sycl_dz(desc, sycl_queue);
}

template <>
commit_impl* create_commit<precision::SINGLE, domain::REAL>(
    const descriptor<precision::SINGLE, domain::REAL>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[libkey].create_commit_sycl_fr(desc, sycl_queue);
}

template <>
commit_impl* create_commit<precision::DOUBLE, domain::REAL>(
    const descriptor<precision::DOUBLE, domain::REAL>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[libkey].create_commit_sycl_dr(desc, sycl_queue);
}

template <precision prec, domain dom>
inline oneapi::mkl::device get_device(descriptor<prec, dom>& desc, const char* func_name) {
    config_value is_committed{ config_value::UNCOMMITTED };
    desc.get_value(config_param::COMMIT_STATUS, &is_committed);
    if (is_committed != config_value::COMMITTED) {
        throw mkl::invalid_argument("DFT", func_name, "Descriptor not committed.");
    }
    // Committed means that the commit pointer is not null.
    return get_device_id(get_commit(desc)->get_queue());
}

} // namespace detail

#define ONEAPI_MKL_DFT_SIGNATURES(EXT, PRECISION, DOMAIN, T_REAL, T_FORWARD, T_BACKWARD)                \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(             \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout) {           \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                            \
            .compute_forward_buffer_inplace_real_##EXT(desc, inout);                                    \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex*/                                                                    \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(         \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & inout) {       \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                            \
            .compute_forward_buffer_inplace_complex_##EXT(desc, inout);                                 \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(             \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout_re,          \
        sycl::buffer<T_REAL, 1> & inout_im) {                                                           \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                            \
            .compute_forward_buffer_inplace_split_##EXT(desc, inout_re, inout_im);                      \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(                 \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_FORWARD, 1> & in,             \
        sycl::buffer<T_BACKWARD, 1> & out) {                                                            \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                            \
            .compute_forward_buffer_outofplace_##EXT(desc, in, out);                                    \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform - real*/                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in,                \
        sycl::buffer<T_REAL, 1> & out) {                                                                \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                            \
            .compute_forward_buffer_outofplace_real_##EXT(desc, in, out);                               \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in_re,             \
        sycl::buffer<T_REAL, 1> & in_im, sycl::buffer<T_REAL, 1> & out_re,                              \
        sycl::buffer<T_REAL, 1> & out_im) {                                                             \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                            \
            .compute_forward_buffer_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im);         \
    }                                                                                                   \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(      \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout,                              \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]                     \
            .compute_forward_usm_inplace_real_##EXT(desc, inout, dependencies);                         \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex*/                                                                    \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                            \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,                          \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]                     \
            .compute_forward_usm_inplace_complex_##EXT(desc, inout, dependencies);                      \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(      \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re, T_REAL * inout_im,        \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]                     \
            .compute_forward_usm_inplace_split_##EXT(desc, inout_re, inout_im, dependencies);           \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(                 \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_FORWARD * in, T_BACKWARD * out,            \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]                     \
            .compute_forward_usm_outofplace_##EXT(desc, in, out, dependencies);                         \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in, T_REAL * out,                   \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]                     \
            .compute_forward_usm_outofplace_real_##EXT(desc, in, out, dependencies);                    \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_forward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re, T_REAL * in_im,              \
        T_REAL * out_re, T_REAL * out_im, const std::vector<sycl::event>& dependencies) {               \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]                     \
            .compute_forward_usm_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im,             \
                                                        dependencies);                                  \
    }                                                                                                   \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform - real*/                                                                       \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(            \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout) {           \
        detail::function_tables[detail::get_device(desc, "compute_backward")]                           \
            .compute_backward_buffer_inplace_real_##EXT(desc, inout);                                   \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex */                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(        \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & inout) {       \
        detail::function_tables[detail::get_device(desc, "compute_backward")]                           \
            .compute_backward_buffer_inplace_complex_##EXT(desc, inout);                                \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT void compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(            \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout_re,          \
        sycl::buffer<T_REAL, 1> & inout_im) {                                                           \
        detail::function_tables[detail::get_device(desc, "compute_backward")]                           \
            .compute_backward_buffer_inplace_split_##EXT(desc, inout_re, inout_im);                     \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(                \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & in,            \
        sycl::buffer<T_FORWARD, 1> & out) {                                                             \
        detail::function_tables[detail::get_device(desc, "compute_backward")]                           \
            .compute_backward_buffer_outofplace_##EXT(desc, in, out);                                   \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform - real*/                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in,                \
        sycl::buffer<T_REAL, 1> & out) {                                                                \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_buffer_outofplace_real_##EXT(desc, in, out);                              \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT void                                                                                  \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in_re,             \
        sycl::buffer<T_REAL, 1> & in_im, sycl::buffer<T_REAL, 1> & out_re,                              \
        sycl::buffer<T_REAL, 1> & out_im) {                                                             \
        detail::function_tables[detail::get_device(desc, "compute_backward")]                           \
            .compute_backward_buffer_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im);        \
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
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_usm_inplace_real_##EXT(desc, inout, dependencies);                        \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform - complex*/                                                                    \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                           \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,                          \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_usm_inplace_complex_##EXT(desc, inout, dependencies);                     \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>(                               \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re, T_REAL * inout_im,        \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_usm_inplace_split_##EXT(desc, inout_re, inout_im, dependencies);          \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(                \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * in, T_FORWARD * out,            \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_usm_outofplace_##EXT(desc, in, out, dependencies);                        \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform - real*/                                                                   \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in, T_REAL * out,                   \
        const std::vector<sycl::event>& dependencies) {                                                 \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_usm_outofplace_real_##EXT(desc, in, out, dependencies);                   \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    ONEMKL_EXPORT sycl::event                                                                           \
    compute_backward<dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                       \
        dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re, T_REAL * in_im,              \
        T_REAL * out_re, T_REAL * out_im, const std::vector<sycl::event>& dependencies) {               \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]                    \
            .compute_backward_usm_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im,            \
                                                         dependencies);                                 \
    }

// Signatures with forward_t=complex, backwards_t=complex are already instantiated for complex domain
// but not real domain.
#define ONEAPI_MKL_DFT_REAL_ONLY_SIGNATURES(EXT, PRECISION, T_COMPLEX)                            \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT void                                                                            \
    compute_forward<dft::detail::descriptor<PRECISION, domain::REAL>, T_COMPLEX, T_COMPLEX>(      \
        dft::detail::descriptor<PRECISION, domain::REAL> & desc, sycl::buffer<T_COMPLEX, 1> & in, \
        sycl::buffer<T_COMPLEX, 1> & out) {                                                       \
        detail::function_tables[detail::get_device(desc, "compute_forward")]                      \
            .compute_forward_buffer_outofplace_complex_##EXT(desc, in, out);                      \
    }                                                                                             \
                                                                                                  \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT sycl::event                                                                     \
    compute_forward<dft::detail::descriptor<PRECISION, domain::REAL>, T_COMPLEX, T_COMPLEX>(      \
        dft::detail::descriptor<PRECISION, domain::REAL> & desc, T_COMPLEX * in, T_COMPLEX * out, \
        const std::vector<sycl::event>& dependencies) {                                           \
        return detail::function_tables[detail::get_device(desc, "compute_forward")]               \
            .compute_forward_usm_outofplace_complex_##EXT(desc, in, out, dependencies);           \
    }                                                                                             \
                                                                                                  \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT void                                                                            \
    compute_backward<dft::detail::descriptor<PRECISION, domain::REAL>, T_COMPLEX, T_COMPLEX>(     \
        dft::detail::descriptor<PRECISION, domain::REAL> & desc, sycl::buffer<T_COMPLEX, 1> & in, \
        sycl::buffer<T_COMPLEX, 1> & out) {                                                       \
        detail::function_tables[detail::get_device(desc, "compute_backward")]                     \
            .compute_backward_buffer_outofplace_complex_##EXT(desc, in, out);                     \
    }                                                                                             \
                                                                                                  \
    /*Out-of-place transform - complex*/                                                          \
    template <>                                                                                   \
    ONEMKL_EXPORT sycl::event                                                                     \
    compute_backward<dft::detail::descriptor<PRECISION, domain::REAL>, T_COMPLEX, T_COMPLEX>(     \
        dft::detail::descriptor<PRECISION, domain::REAL> & desc, T_COMPLEX * in, T_COMPLEX * out, \
        const std::vector<sycl::event>& dependencies) {                                           \
        return detail::function_tables[detail::get_device(desc, "compute_backward")]              \
            .compute_backward_usm_outofplace_complex_##EXT(desc, in, out, dependencies);          \
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

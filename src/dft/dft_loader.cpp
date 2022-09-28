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

#include "oneapi/mkl/dft.hpp"

#include "function_table_initializer.hpp"
#include "dft/function_table.hpp"

#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi {
namespace mkl {
namespace dft {

namespace detail {
static oneapi::mkl::detail::table_initializer<mkl::domain::dft, dft_function_table_t>
    function_tables;
} // namespace detail

#define ONEAPI_MKL_DFT_SIGNATURES(EXT, PRECISION, DOMAIN, T_REAL, T_FORWARD, T_BACKWARD)                \
                                                                                                        \
    template <>                                                                                         \
    void descriptor<PRECISION, DOMAIN>::commit(sycl::queue &queue) {                                    \
        this->queue_ = queue;                                                                           \
        detail::function_tables[get_device_id(queue)].commit_##EXT(*this, queue);                       \
    }                                                                                                   \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform*/                                                                              \
    template <>                                                                                         \
    void compute_forward<descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                                    \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & inout) {                    \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_forward_buffer_inplace_##EXT(desc, inout);                                         \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    void compute_forward<descriptor<PRECISION, DOMAIN>, T_REAL>(                                        \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout_re,                       \
        sycl::buffer<T_REAL, 1> & inout_im) {                                                           \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_forward_buffer_inplace_split_##EXT(desc, inout_re, inout_im);                      \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    void compute_forward<descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(                         \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_FORWARD, 1> & in,                          \
        sycl::buffer<T_BACKWARD, 1> & out) {                                                            \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_forward_buffer_outofplace_##EXT(desc, in, out);                                    \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    void compute_forward<descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                                \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in_re,                          \
        sycl::buffer<T_REAL, 1> & in_im, sycl::buffer<T_REAL, 1> & out_re,                              \
        sycl::buffer<T_REAL, 1> & out_im) {                                                             \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_forward_buffer_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im);         \
    }                                                                                                   \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform*/                                                                              \
    template <>                                                                                         \
    sycl::event compute_forward<descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                             \
        descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,                                       \
        const std::vector<sycl::event> &dependencies) {                                                 \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_forward_usm_inplace_##EXT(desc, inout, dependencies);                              \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    sycl::event compute_forward<descriptor<PRECISION, DOMAIN>, T_REAL>(                                 \
        descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re, T_REAL * inout_im,                     \
        const std::vector<sycl::event> &dependencies) {                                                 \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_forward_usm_inplace_split_##EXT(desc, inout_re, inout_im, dependencies);           \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    sycl::event compute_forward<descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>(                  \
        descriptor<PRECISION, DOMAIN> & desc, T_FORWARD * in, T_BACKWARD * out,                         \
        const std::vector<sycl::event> &dependencies) {                                                 \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_forward_usm_outofplace_##EXT(desc, in, out, dependencies);                         \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    sycl::event compute_forward<descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                         \
        descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re, T_REAL * in_im, T_REAL * out_re,          \
        T_REAL * out_im, const std::vector<sycl::event> &dependencies) {                                \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_forward_usm_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im,             \
                                                        dependencies);                                  \
    }                                                                                                   \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform*/                                                                              \
    template <>                                                                                         \
    void compute_backward<descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                                   \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & inout) {                    \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_backward_buffer_inplace_##EXT(desc, inout);                                        \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    void compute_backward<descriptor<PRECISION, DOMAIN>, T_REAL>(                                       \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & inout_re,                       \
        sycl::buffer<T_REAL, 1> & inout_im) {                                                           \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_backward_buffer_inplace_split_##EXT(desc, inout_re, inout_im);                     \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    void compute_backward<descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(                        \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_BACKWARD, 1> & in,                         \
        sycl::buffer<T_FORWARD, 1> & out) {                                                             \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_backward_buffer_outofplace_##EXT(desc, in, out);                                   \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    void compute_backward<descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                               \
        descriptor<PRECISION, DOMAIN> & desc, sycl::buffer<T_REAL, 1> & in_re,                          \
        sycl::buffer<T_REAL, 1> & in_im, sycl::buffer<T_REAL, 1> & out_re,                              \
        sycl::buffer<T_REAL, 1> & out_im) {                                                             \
        detail::function_tables[get_device_id(desc.get_queue())]                                        \
            .compute_backward_buffer_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im);        \
    }                                                                                                   \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform*/                                                                              \
    template <>                                                                                         \
    sycl::event compute_backward<descriptor<PRECISION, DOMAIN>, T_BACKWARD>(                            \
        descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,                                       \
        const std::vector<sycl::event> &dependencies) {                                                 \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_backward_usm_inplace_##EXT(desc, inout, dependencies);                             \
    }                                                                                                   \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    template <>                                                                                         \
    sycl::event compute_backward<descriptor<PRECISION, DOMAIN>, T_REAL>(                                \
        descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re, T_REAL * inout_im,                     \
        const std::vector<sycl::event> &dependencies) {                                                 \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_backward_usm_inplace_split_##EXT(desc, inout_re, inout_im, dependencies);          \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    template <>                                                                                         \
    sycl::event compute_backward<descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>(                 \
        descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * in, T_FORWARD * out,                         \
        const std::vector<sycl::event> &dependencies) {                                                 \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_backward_usm_outofplace_##EXT(desc, in, out, dependencies);                        \
    }                                                                                                   \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    template <>                                                                                         \
    sycl::event compute_backward<descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>(                        \
        descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re, T_REAL * in_im, T_REAL * out_re,          \
        T_REAL * out_im, const std::vector<sycl::event> &dependencies) {                                \
        return detail::function_tables[get_device_id(desc.get_queue())]                                 \
            .compute_backward_usm_outofplace_split_##EXT(desc, in_re, in_im, out_re, out_im,            \
                                                         dependencies);                                 \
    }

ONEAPI_MKL_DFT_SIGNATURES(f, precision::SINGLE, domain::REAL, float, float, std::complex<float>)
ONEAPI_MKL_DFT_SIGNATURES(c, precision::SINGLE, domain::COMPLEX, float, std::complex<float>,
                          std::complex<float>)
ONEAPI_MKL_DFT_SIGNATURES(d, precision::DOUBLE, domain::REAL, double, double, std::complex<double>)
ONEAPI_MKL_DFT_SIGNATURES(z, precision::DOUBLE, domain::COMPLEX, double, std::complex<double>,
                          std::complex<double>)

#undef ONEAPI_MKL_DFT_SIGNATURES

} // namespace dft
} // namespace mkl
} // namespace oneapi

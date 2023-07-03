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

#ifndef _ONEMKL_DFT_COMMIT_IMPL_HPP_
#define _ONEMKL_DFT_COMMIT_IMPL_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "descriptor_impl.hpp"

namespace oneapi::mkl {
enum class backend;
}

namespace oneapi::mkl::dft::detail {

template <precision prec, domain dom>
class dft_values;

template <precision prec, domain dom>
class commit_impl {
    sycl::queue queue_;
    mkl::backend backend_;

public:
    using descriptor_type = typename oneapi::mkl::dft::detail::descriptor<prec, dom>;
    using fwd_type = typename descriptor_info<descriptor_type>::forward_type;
    using bwd_type = typename descriptor_info<descriptor_type>::backward_type;
    using scalar_type = typename descriptor_info<descriptor_type>::scalar_type;

    commit_impl(sycl::queue queue, mkl::backend backend) : queue_(queue), backend_(backend) {}

    // rule of three
    commit_impl(const commit_impl &other) = delete;
    commit_impl &operator=(const commit_impl &other) = delete;
    virtual ~commit_impl() = default;

    sycl::queue &get_queue() noexcept {
        return queue_;
    }

    mkl::backend get_backend() const noexcept {
        return backend_;
    }

    virtual void *get_handle() noexcept = 0;

    virtual void commit(const dft_values<prec, dom> &) = 0;

    virtual void forward_ip_cc(descriptor_type &desc, sycl::buffer<fwd_type, 1> &inout) = 0;
    virtual void forward_ip_rr(descriptor_type &desc, sycl::buffer<scalar_type, 1> &inout_re,
                               sycl::buffer<scalar_type, 1> &inout_im) = 0;
    virtual void forward_op_cc(descriptor_type &desc, sycl::buffer<fwd_type, 1> &in,
                               sycl::buffer<bwd_type, 1> &out) = 0;
    virtual void forward_op_rr(descriptor_type &desc, sycl::buffer<scalar_type, 1> &in_re,
                               sycl::buffer<scalar_type, 1> &in_im,
                               sycl::buffer<scalar_type, 1> &out_re,
                               sycl::buffer<scalar_type, 1> &out_im) = 0;

    virtual sycl::event forward_ip_cc(descriptor_type &desc, fwd_type *inout,
                                      const std::vector<sycl::event> &dependencies) = 0;
    virtual sycl::event forward_ip_rr(descriptor_type &desc, scalar_type *inout_re,
                                      scalar_type *inout_im,
                                      const std::vector<sycl::event> &dependencies) = 0;
    virtual sycl::event forward_op_cc(descriptor_type &desc, fwd_type *in, bwd_type *out,
                                      const std::vector<sycl::event> &dependencies) = 0;
    virtual sycl::event forward_op_rr(descriptor_type &desc, scalar_type *in_re, scalar_type *in_im,
                                      scalar_type *out_re, scalar_type *out_im,
                                      const std::vector<sycl::event> &dependencies) = 0;

    virtual void backward_ip_cc(descriptor_type &desc, sycl::buffer<fwd_type, 1> &inout) = 0;
    virtual void backward_ip_rr(descriptor_type &desc, sycl::buffer<scalar_type, 1> &inout_re,
                                sycl::buffer<scalar_type, 1> &inout_im) = 0;
    virtual void backward_op_cc(descriptor_type &desc, sycl::buffer<bwd_type, 1> &in,
                                sycl::buffer<fwd_type, 1> &out) = 0;
    virtual void backward_op_rr(descriptor_type &desc, sycl::buffer<scalar_type, 1> &in_re,
                                sycl::buffer<scalar_type, 1> &in_im,
                                sycl::buffer<scalar_type, 1> &out_re,
                                sycl::buffer<scalar_type, 1> &out_im) = 0;

    virtual sycl::event backward_ip_cc(descriptor_type &desc, fwd_type *inout,
                                       const std::vector<sycl::event> &dependencies) = 0;
    virtual sycl::event backward_ip_rr(descriptor_type &desc, scalar_type *inout_re,
                                       scalar_type *inout_im,
                                       const std::vector<sycl::event> &dependencies) = 0;
    virtual sycl::event backward_op_cc(descriptor_type &desc, bwd_type *in, fwd_type *out,
                                       const std::vector<sycl::event> &dependencies) = 0;
    virtual sycl::event backward_op_rr(descriptor_type &desc, scalar_type *in_re,
                                       scalar_type *in_im, scalar_type *out_re, scalar_type *out_im,
                                       const std::vector<sycl::event> &dependencies) = 0;
};

} // namespace oneapi::mkl::dft::detail

#endif //_ONEMKL_DFT_COMMIT_IMPL_HPP_

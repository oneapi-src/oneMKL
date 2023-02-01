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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/detail/exceptions.hpp"

#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

#include "dft/backends/mklcpu/mklcpu_helpers.hpp"

// MKLCPU header
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {
namespace detail {

// Forward a MKLCPU DFT call to the backend, checking that the commit impl is valid.
// assumes the parameter pack has same types
template <dft::detail::precision prec, dft::detail::domain dom, typename... ArgTs>
inline auto compute_forward(dft::detail::descriptor<prec, dom> &desc, ArgTs &&... args) {
    using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;
    using commit_t = dft::detail::commit_impl;

    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::mklcpu) {
        throw mkl::invalid_argument("DFT", "computer_forward",
                                    "DFT descriptor has not been commited for MKLCPU");
    }

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};

    MKL_LONG commit_status{ DFTI_UNCOMMITTED };
    DftiGetValue(mklcpu_desc, DFTI_COMMIT_STATUS, &commit_status);
    if (commit_status != DFTI_COMMITTED) {
        throw mkl::invalid_argument("DFT", "compute_forward",
                                    "MKLCPU DFT descriptor was not successfully committed.");
    }
}

// Throw an mkl::invalid_argument if the runtime param in the descriptor does not match
// the expected value.
template <dft::detail::config_param Param, dft::detail::config_value Expected, typename DescT>
inline auto expect_config(DescT &desc, const char *message) {
    dft::detail::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw mkl::invalid_argument("DFT", "compute_forward", message);
    }
}
} // namespace detail

using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;
using commit_t = dft::detail::commit_impl;
//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};

    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        auto inout_acc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), inout_acc.get_pointer());
        });
    }).wait();
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout_re,
                                   sycl::buffer<data_type, 1> &inout_im) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE, dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");

    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};

    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        auto re_acc = inout_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto im_acc = inout_im.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), re_acc.get_pointer(), im_acc.get_pointer());
        });
    }).wait();
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<input_type, 1> &in,
                                   sycl::buffer<output_type, 1> &out) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::NOT_INPLACE>( desc, "Unexpected value for placement");

    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};
    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        auto in_acc = in.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_acc = out.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), in_acc.get_pointer(), out_acc.get_pointer());
        });
    }).wait();
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<input_type, 1> &in_re,
                                   sycl::buffer<input_type, 1> &in_im,
                                   sycl::buffer<output_type, 1> &out_re,
                                   sycl::buffer<output_type, 1> &out_im) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");

    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};
    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        auto inre_acc = in_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto inim_acc = in_im.template get_access<sycl::access::mode::read_write>(cgh);
        auto outre_acc = out_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto outim_acc = out_im.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), inre_acc.get_pointer(), inim_acc.get_pointer(),
                                                        outre_acc.get_pointer(), outim_acc.get_pointer());
        });
    }).wait();
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, data_type *inout,
                                          const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");

    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};

    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), inout);
        });
    });
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, data_type *inout_re,
                                          data_type *inout_im,
                                          const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE, dft::detail::config_value::REAL_REAL>(
            desc, "Unexpected value for complex storage");
    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};

    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), inout_re, inout_im);
        });
    });
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, input_type *in, output_type *out,
                                          const std::vector<sycl::event> &dependencies) {
    // Check: inplace, complex storage
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::NOT_INPLACE>(
        desc, "Unexpected value for placement");

    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};
    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), in, out);
        });
    });
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, input_type *in_re,
                                          input_type *in_im, output_type *out_re,
                                          output_type *out_im,
                                          const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE, dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    auto commit_handle = dft::detail::get_commit(desc);
    auto mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    sycl::queue &cpu_queue{ commit_handle->get_queue() };
    sycl::buffer<mklcpu_desc_t, 1> mklcpu_desc_buffer {&mklcpu_desc, sycl::range<1>{1}};
    return cpu_queue.submit([&](sycl::handler& cgh){
        auto desc_acc = mklcpu_desc_buffer.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            DftiComputeForward(*desc_acc.get_pointer(), in_re, in_im, out_re, out_im);
        });
    });
}

// Template function instantiations
#include "dft/backends/backend_forward_instantiations.cxx"

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

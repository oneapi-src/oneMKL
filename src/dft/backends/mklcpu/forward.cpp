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

    MKL_LONG commit_status{ DFTI_UNCOMMITTED };
    DftiGetValue(mklcpu_desc, DFTI_COMMIT_STATUS, &commit_status);
    if (commit_status != DFTI_COMMITTED) {
        throw mkl::invalid_argument("DFT", "compute_forward",
                                    "MKLCPU DFT descriptor was not successfully committed.");
    }
    if (sizeof...(args) == 1) {
        std::cout << "inplace transform" << std::endl;
    }
    else if (sizeof...(args) == 2) {
        std::cout << "out of place transform" << std::endl;
        auto in = std::get<0>(std::forward_as_tuple(args...));
        auto out = std::get<1>(std::forward_as_tuple(args...));

        class vector_addition;
        // if the input is a sycl buffer; could be complex or real
        if constexpr (is_buffer_v<decltype(in)> && is_buffer_v<decltype(out)>) {
            std::cout << "out of place transform" << std::endl;
        }
    }
    else {
        throw mkl::invalid_argument("DFT", "compute_forward", "invalid number of args.");
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

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout_re,
                                   sycl::buffer<data_type, 1> &inout_im) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<input_type, 1> &in,
                                   sycl::buffer<output_type, 1> &out) {
    using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;
    if constexpr (!std::is_same_v<input_type, output_type>) {
        throw mkl::unimplemented(
            "DFT", "compute_forward",
            "MKLCPU does not support out-of-place FFT with different input and output types.");
    }
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::NOT_INPLACE>( desc, "Unexpected value for placement");

    auto commit_handle = dft::detail::get_commit(desc);
    auto &cpu_queue = commit_handle->get_queue();
    mklcpu_desc_t mklcpu_desc = reinterpret_cast<mklcpu_desc_t>(commit_handle->get_handle());

    cpu_queue.submit([&](sycl::handler& cgh){
        auto in_acc = in.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_acc = out.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
            std::cout << "in ker" << std::endl;
            std::cout << "value : " << (*in_acc.get_pointer()) << std::endl;
            MKL_LONG status = 0;

            // DFTI_DESCRIPTOR_HANDLE handle = nullptr;
            // status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)10);
            status = DftiCommitDescriptor(mklcpu_desc);
            status = DftiComputeForward(mklcpu_desc, in_acc.get_pointer());
        });
    }).wait();

   std::cout << "kernel module end" << std::endl; 
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<input_type, 1> &in_re,
                                   sycl::buffer<input_type, 1> &in_im,
                                   sycl::buffer<output_type, 1> &out_re,
                                   sycl::buffer<output_type, 1> &out_im) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, data_type *inout,
                                          const std::vector<sycl::event> &dependencies) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
    return sycl::event{};
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, data_type *inout_re,
                                          data_type *inout_im,
                                          const std::vector<sycl::event> &dependencies) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
    return sycl::event{};
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, input_type *in, output_type *out,
                                          const std::vector<sycl::event> &dependencies) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
    return sycl::event{};
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, input_type *in_re,
                                          input_type *in_im, output_type *out_re,
                                          output_type *out_im,
                                          const std::vector<sycl::event> &dependencies) {
    throw mkl::unimplemented("DFT", "compute_forward", "Not implemented for MKLCPU");
    return sycl::event{};
}

// Template function instantiations
#include "dft/backends/backend_forward_instantiations.cxx"

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

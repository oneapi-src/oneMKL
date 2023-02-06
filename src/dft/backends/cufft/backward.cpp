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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/cufft/onemkl_dft_cufft.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"

// cuFFT headers
#include <cuda_runtime.h>
#include <cufft.h>

namespace oneapi::mkl::dft::cufft {
namespace detail {

template <dft::detail::precision prec, dft::detail::domain dom>
inline dft::detail::commit_impl *get_commit(dft::detail::descriptor<prec, dom> &desc) {
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::cufft) {
        throw mkl::invalid_argument("DFT", "compute_backward",
                                    "DFT descriptor has not been commited for cuFFT");
    }
    return commit_handle;
}

/// Throw an mkl::invalid_argument if the runtime param in the descriptor does not match
/// the expected value.
template <dft::detail::config_param Param, dft::detail::config_value Expected, typename DescT>
inline auto expect_config(DescT &desc, const char *message) {
    dft::detail::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw mkl::invalid_argument("DFT", "compute_backward", message);
    }
}
} // namespace detail

// BUFFER version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout) {
    if constexpr (descriptor_type::precision == dft::detail::precision::SINGLE &&
                  !std::is_floating_point_v<data_type>) {
        detail::expect_config<dft::detail::config_param::PLACEMENT,
                              dft::detail::config_value::INPLACE>(desc,
                                                                  "Unexpected value for placement");
        auto commit = get_commit(desc);
        auto queue = commit->get_queue();
        auto plan = *static_cast<cufftHandle *>(commit->get_handle());

        cufftResult result = CUFFT_NOT_SUPPORTED;
        cufftResult *result_ptr = &result;

        queue.submit([&](sycl::handler &cgh) {
            auto inout_acc = inout.template get_access<sycl::access::mode::read_write>(cgh);

            cgh.host_task([=](sycl::interop_handle ih) {
                auto inout_native = reinterpret_cast<cufftComplex *>(
                    ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(inout_acc));
                auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
                *result_ptr = cufftSetStream(plan, stream);
                if (*result_ptr == CUFFT_SUCCESS) {
                    *result_ptr = cufftExecC2C(plan, inout_native, inout_native, CUFFT_INVERSE);
                }
            });
        }).wait();

        if (result != CUFFT_SUCCESS) {
            throw oneapi::mkl::exception("DFT", "compute_backward(desc, inout)",
                                         "cuFFTResult value of " + std::to_string(result));
        }
    }
    else {
        throw oneapi::mkl::unimplemented("DFT", "compute_backward", "not yet implemented");
    }
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout_re,
                                    sycl::buffer<data_type, 1> &inout_im) {
    throw oneapi::mkl::unimplemented("DFT", "compute_backward(desc, inout_re, inout_im)",
                                     "cuFFT does not support real-real complex storage.");
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<input_type, 1> &in,
                                    sycl::buffer<output_type, 1> &out) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    throw oneapi::mkl::unimplemented("DFT", "compute_backward(desc, in, out)",
                                     "not yet implemented");
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<input_type, 1> &in_re,
                                    sycl::buffer<input_type, 1> &in_im,
                                    sycl::buffer<output_type, 1> &out_re,
                                    sycl::buffer<output_type, 1> &out_im) {
    throw oneapi::mkl::unimplemented("DFT", "compute_backward(desc, in_re, in_im, out_re, out_im)",
                                     "cuFFT does not support real-real complex storage.");
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, data_type *inout,
                                           const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    throw oneapi::mkl::unimplemented("DFT", "compute_backward(desc, inout, dependencies)",
                                     "not yet implemented");
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, data_type *inout_re,
                                           data_type *inout_im,
                                           const std::vector<sycl::event> &dependencies) {
    throw oneapi::mkl::unimplemented("DFT",
                                     "compute_backward(desc, inout_re, inout_im, dependencies)",
                                     "cuFFT does not support real-real complex storage.");
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, input_type *in, output_type *out,
                                           const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    throw oneapi::mkl::unimplemented("DFT", "compute_backward(desc, in, out, dependencies)",
                                     "not yet implemented");
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, input_type *in_re,
                                           input_type *in_im, output_type *out_re,
                                           output_type *out_im,
                                           const std::vector<sycl::event> &dependencies) {
    throw oneapi::mkl::unimplemented("DFT",
                                     "compute_backward(desc, in_re, in_im, out_re, out_im, deps)",
                                     "cuFFT does not support real-real complex storage.");
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace oneapi::mkl::dft::cufft

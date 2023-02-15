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

#include "oneapi/mkl/dft/detail/mklgpu/onemkl_dft_mklgpu.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "dft/backends/mklgpu/mklgpu_helpers.hpp"

// MKLGPU header
#include "oneapi/mkl/dfti.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklgpu {
namespace detail {

/// Forward a MKLGPU DFT call to the backend, checking that the commit impl is valid.
/// Assumes backend descriptor values match those of the frontend.
template <dft::detail::precision prec, dft::detail::domain dom, typename... ArgTs>
inline auto compute_backward(dft::detail::descriptor<prec, dom> &desc, ArgTs &&... args) {
    using mklgpu_desc_t = dft::descriptor<to_mklgpu(prec), to_mklgpu(dom)>;
    using commit_t = dft::detail::commit_impl;
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::mklgpu) {
        throw mkl::invalid_argument("DFT", "compute_backward",
                                    "DFT descriptor has not been commited for MKLGPU");
    }
    auto mklgpu_desc = reinterpret_cast<mklgpu_desc_t *>(commit_handle->get_handle());
    int commit_status{ DFTI_UNCOMMITTED };
    mklgpu_desc->get_value(dft::config_param::COMMIT_STATUS, &commit_status);
    if (commit_status != DFTI_COMMITTED) {
        throw mkl::invalid_argument("DFT", "compute_backward",
                                    "MKLGPU DFT descriptor was not successfully committed.");
    }
    // The MKLGPU backend's iterface contains fewer function signatures than in this
    // open-source library. Consequently, it is not required to forward template arguments
    // to resolve to the correct function.
    return dft::compute_backward(*mklgpu_desc, std::forward<ArgTs>(args)...);
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
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    return detail::compute_backward(desc, inout);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout_re,
                                    sycl::buffer<data_type, 1> &inout_im) {
    throw mkl::unimplemented("DFT", "compute_backward",
                             "MKLGPU does not support compute_backward(desc, inout_re, inout_im).");
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<input_type, 1> &in,
                                    sycl::buffer<output_type, 1> &out) {
    if constexpr (!std::is_same_v<input_type, output_type>) {
        throw mkl::unimplemented(
            "DFT", "compute_backward",
            "MKLGPU does not support out-of-place FFT with different input and output types.");
    }
    else {
        detail::expect_config<dft::detail::config_param::PLACEMENT,
                              dft::detail::config_value::NOT_INPLACE>(
            desc, "Unexpected value for placement");
        return detail::compute_backward(desc, in, out);
    }
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<input_type, 1> &in_re,
                                    sycl::buffer<input_type, 1> &in_im,
                                    sycl::buffer<output_type, 1> &out_re,
                                    sycl::buffer<output_type, 1> &out_im) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    throw oneapi::mkl::unimplemented(
        "DFT", "compute_backward(desc, in_re, in_im, out_re, out_im)",
        "MKLGPU does not support out-of-place FFT with real-real complex storage.");
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, data_type *inout,
                                           const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    return detail::compute_backward(desc, inout, dependencies);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, data_type *inout_re,
                                           data_type *inout_im,
                                           const std::vector<sycl::event> &dependencies) {
    throw mkl::unimplemented("DFT", "compute_backward",
                             "MKLGPU does not support compute_backward(desc, inout_re, inout_im).");
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, input_type *in, output_type *out,
                                           const std::vector<sycl::event> &dependencies) {
    if constexpr (!std::is_same_v<input_type, output_type>) {
        throw mkl::unimplemented(
            "DFT", "compute_backward",
            "MKLGPU does not support out-of-place FFT with different input and output types.");
    }
    else {
        detail::expect_config<dft::detail::config_param::PLACEMENT,
                              dft::detail::config_value::NOT_INPLACE>(
            desc, "Unexpected value for placement");
        return detail::compute_backward(desc, in, out, dependencies);
    }
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, input_type *in_re,
                                           input_type *in_im, output_type *out_re,
                                           output_type *out_im,
                                           const std::vector<sycl::event> &dependencies) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    throw oneapi::mkl::unimplemented(
        "DFT", "compute_backward(desc, in_re, in_im, out_re, out_im, deps)",
        "MKLGPU does not support out-of-place FFT with real-real complex storage.");
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace mklgpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

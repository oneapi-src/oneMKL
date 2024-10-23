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

#include "oneapi/math/exceptions.hpp"

#include "oneapi/math/dft/detail/mklgpu/onemath_dft_mklgpu.hpp"
#include "oneapi/math/dft/detail/descriptor_impl.hpp"

#include "common_onemkl_conversion.hpp"
#include "mklgpu_helpers.hpp"

// Intel(R) oneMKL headers
#include <mkl_version.h>
#if INTEL_MKL_VERSION < 20250000
#include <oneapi/mkl/dfti.hpp>
#else
#include <oneapi/mkl/dft.hpp>
#endif

namespace oneapi::math::dft::mklgpu {
namespace detail {

/// Forward a MKLGPU DFT call to the backend, checking that the commit impl is valid.
/// Assumes backend descriptor values match those of the frontend.
template <dft::detail::precision prec, dft::detail::domain dom, typename... ArgTs>
inline auto compute_backward(dft::detail::descriptor<prec, dom>& desc, ArgTs&&... args) {
    using mklgpu_desc_t = oneapi::mkl::dft::descriptor<to_mklgpu(prec), to_mklgpu(dom)>;
    using desc_shptr_t = std::shared_ptr<mklgpu_desc_t>;
    using handle_t = std::pair<desc_shptr_t, desc_shptr_t>;
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::mklgpu) {
        throw math::invalid_argument("DFT", "compute_backward",
                                     "DFT descriptor has not been commited for MKLGPU");
    }
    auto handle = reinterpret_cast<handle_t*>(commit_handle->get_handle());
    auto mklgpu_desc = handle->second; // Second because backward DFT.
    int commit_status{ DFTI_UNCOMMITTED };
    mklgpu_desc->get_value(oneapi::mkl::dft::config_param::COMMIT_STATUS, &commit_status);
    if (commit_status != DFTI_COMMITTED) {
        throw math::invalid_argument("DFT", "compute_backward",
                                     "MKLGPU DFT descriptor was not successfully committed.");
    }
    // The MKLGPU backend's interface contains fewer function signatures than in this
    // open-source library. Consequently, it is not required to forward template arguments
    // to resolve to the correct function.
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        oneapi::mkl::dft::compute_backward(*mklgpu_desc, std::forward<ArgTs>(args)...));
}

/// Throw an math::invalid_argument if the runtime param in the descriptor does not match
/// the expected value.
template <dft::detail::config_param Param, dft::detail::config_value Expected, typename DescT>
inline auto expect_config(DescT& desc, const char* message) {
    dft::detail::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw math::invalid_argument("DFT", "compute_backward", message);
    }
}
} // namespace detail

// BUFFER version

//In-place transform
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<fwd<descriptor_type>, 1>& inout) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    return detail::compute_backward(desc, inout);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& /*desc*/,
                                     sycl::buffer<scalar<descriptor_type>, 1>& /*inout_re*/,
                                     sycl::buffer<scalar<descriptor_type>, 1>& /*inout_im*/) {
    throw math::unimplemented(
        "DFT", "compute_backward",
        "MKLGPU does not support compute_backward(desc, inout_re, inout_im).");
}

//Out-of-place transform
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<bwd<descriptor_type>, 1>& in,
                                     sycl::buffer<fwd<descriptor_type>, 1>& out) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    return detail::compute_backward(desc, in, out);
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<scalar<descriptor_type>, 1>& /*in_re*/,
                                     sycl::buffer<scalar<descriptor_type>, 1>& /*in_im*/,
                                     sycl::buffer<scalar<descriptor_type>, 1>& /*out_re*/,
                                     sycl::buffer<scalar<descriptor_type>, 1>& /*out_im*/) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    throw oneapi::math::unimplemented(
        "DFT", "compute_backward(desc, in_re, in_im, out_re, out_im)",
        "MKLGPU does not support out-of-place FFT with real-real complex storage.");
}

//USM version

//In-place transform
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, fwd<descriptor_type>* inout,
                                            const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    return detail::compute_backward(desc, inout, dependencies);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& /*desc*/,
                                            scalar<descriptor_type>* /*inout_re*/,
                                            scalar<descriptor_type>* /*inout_im*/,
                                            const std::vector<sycl::event>& /*dependencies*/) {
    throw math::unimplemented(
        "DFT", "compute_backward",
        "MKLGPU does not support compute_backward(desc, inout_re, inout_im, dependencies).");
}

//Out-of-place transform
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, bwd<descriptor_type>* in,
                                            fwd<descriptor_type>* out,
                                            const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    return detail::compute_backward(desc, in, out, dependencies);
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc,
                                            scalar<descriptor_type>* /*in_re*/,
                                            scalar<descriptor_type>* /*in_im*/,
                                            scalar<descriptor_type>* /*out_re*/,
                                            scalar<descriptor_type>* /*out_im*/,
                                            const std::vector<sycl::event>& /*dependencies*/) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    throw oneapi::math::unimplemented(
        "DFT", "compute_backward(desc, in_re, in_im, out_re, out_im, deps)",
        "MKLGPU does not support out-of-place FFT with real-real complex storage.");
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace oneapi::math::dft::mklgpu

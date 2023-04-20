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

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

// BUFFER version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& /*desc*/,
                                    sycl::buffer<data_type, 1>& /*inout*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& /*desc*/,
                                    sycl::buffer<data_type, 1>& /*inout_re*/,
                                    sycl::buffer<data_type, 1>& /*inout_im*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& /*desc*/, sycl::buffer<input_type, 1>& /*in*/,
                                    sycl::buffer<output_type, 1>& /*out*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& /*desc*/,
                                    sycl::buffer<input_type, 1>& /*in_re*/,
                                    sycl::buffer<input_type, 1>& /*in_im*/,
                                    sycl::buffer<output_type, 1>& /*out_re*/,
                                    sycl::buffer<output_type, 1>& /*out_im*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& /*desc*/, data_type* /*inout*/,
                                           const std::vector<sycl::event>& /*dependencies*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
    return sycl::event{};
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& /*desc*/, data_type* /*inout_re*/,
                                           data_type* /*inout_im*/,
                                           const std::vector<sycl::event>& /*dependencies*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
    return sycl::event{};
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& /*desc*/, input_type* /*in*/,
                                           output_type* /*out*/,
                                           const std::vector<sycl::event>& /*dependencies*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
    return sycl::event{};
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& /*desc*/, input_type* /*in_re*/,
                                           input_type* /*in_im*/, output_type* /*out_re*/,
                                           output_type* /*out_im*/,
                                           const std::vector<sycl::event>& /*dependencies*/) {
    throw mkl::unimplemented("DFT", "compute_backward", "Not implemented for MKLCPU");
    return sycl::event{};
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

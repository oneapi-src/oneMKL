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
#include "oneapi/mkl/detail/backends.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"

#include "dft/backends/mklcpu/mklcpu_helpers.hpp"
#include "mkl_service.h"
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {
namespace detail {

template <dft::detail::precision prec, dft::detail::domain dom>
commit_derived_impl<prec, dom>::commit_derived_impl(
    sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values)
        : oneapi::mkl::dft::detail::commit_impl<prec, dom>(queue, backend::mklcpu) {
    // create the descriptor once for the lifetime of the descriptor class
    DFT_ERROR status = DFTI_BAD_DESCRIPTOR;

    if (config_values.dimensions.size() == 1)
        status = DftiCreateDescriptor(&device_handle, mklcpu_prec,
                                            mklcpu_dom, 1, config_values.dimensions[0]);
    else
        status = DftiCreateDescriptor(&device_handle, mklcpu_prec,
                                            mklcpu_dom, config_values.dimensions.size(),
                                            config_values.dimensions.data());

    if (status != DFTI_NO_ERROR)
    {
        throw oneapi::mkl::exception("dft/backends/mklcpu", "create_descriptor",
                                        "DftiCreateDescriptor falied");
    }
}

template <dft::detail::precision prec, dft::detail::domain dom>
commit_derived_impl<prec, dom>::~commit_derived_impl() {
    DftiFreeDescriptor(&device_handle);
}

template <dft::detail::precision prec, dft::detail::domain dom>
void commit_derived_impl<prec, dom>::commit(const dft::detail::dft_values<prec, dom>& config_values) {
    DFT_ERROR status = DFTI_BAD_DESCRIPTOR;
    sycl::buffer<DFT_ERROR, 1> status_buffer{ &status, sycl::range<1>{ 1 } };

    set_value(device_handle, config_values);

    this->get_queue().submit([&](sycl::handler& cgh) {
        auto handle_obj =
            handle_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        auto status_obj =
            status_buffer.template get_access<sycl::access::mode::read_write>(cgh);

        host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=]() {
            std::int64_t local_err = DFTI_BAD_DESCRIPTOR;

            local_err = DftiCommitDescriptor(*handle_obj.get_pointer());
            *status_obj.get_pointer() = local_err;
        });
    });

    status = status_buffer.template get_access<sycl::access::mode::read>()[0];
    device_handle = handle_buffer.template get_access<sycl::access::mode::read>()[0];

    if(!device_handle || status != DFTI_NO_ERROR)
        throw oneapi::mkl::exception("dft/backends/mklcpu", "commit",
                                        "DftiCommitDescriptor failed");
}

template <dft::detail::precision prec, dft::detail::domain dom>
void* commit_derived_impl<prec, dom>::get_handle() noexcept {
    return reinterpret_cast<void*>(device_handle);
}


template <dft::detail::precision prec, dft::detail::domain dom>
template <typename... Args>
void commit_derived_impl<prec, dom>::set_value_item(mklcpu_desc_t hand, enum DFTI_CONFIG_PARAM name, Args... args) {
    DFT_ERROR value_err = DftiSetValue(hand, name, args...);
    if (value_err != DFTI_NO_ERROR) {
        throw oneapi::mkl::exception("dft/backends/mklcpu", "set_value_item",
                                        std::to_string(name));
    }
}

template <dft::detail::precision prec, dft::detail::domain dom>
void commit_derived_impl<prec, dom>::set_value(mklcpu_desc_t descHandle, const dft::detail::dft_values<prec, dom>& config) {
    set_value_item(descHandle, DFTI_INPUT_STRIDES, config.input_strides.data());
    set_value_item(descHandle, DFTI_OUTPUT_STRIDES, config.output_strides.data());
    set_value_item(descHandle, DFTI_BACKWARD_SCALE, config.bwd_scale);
    set_value_item(descHandle, DFTI_FORWARD_SCALE, config.fwd_scale);
    set_value_item(descHandle, DFTI_NUMBER_OF_TRANSFORMS, config.number_of_transforms);
    set_value_item(descHandle, DFTI_INPUT_DISTANCE, config.fwd_dist);
    set_value_item(descHandle, DFTI_OUTPUT_DISTANCE, config.bwd_dist);
    set_value_item(descHandle, DFTI_COMPLEX_STORAGE,
                    to_mklcpu<config_param::COMPLEX_STORAGE>(config.complex_storage));
    set_value_item(descHandle, DFTI_REAL_STORAGE,
                    to_mklcpu<config_param::REAL_STORAGE>(config.real_storage));
    set_value_item(descHandle, DFTI_CONJUGATE_EVEN_STORAGE,
                    to_mklcpu<config_param::CONJUGATE_EVEN_STORAGE>(config.conj_even_storage));
    set_value_item(descHandle, DFTI_PLACEMENT,
                    to_mklcpu<config_param::PLACEMENT>(config.placement));
    set_value_item(descHandle, DFTI_PACKED_FORMAT,
                    to_mklcpu<config_param::PACKED_FORMAT>(config.packed_format));
    // Setting the workspace causes an FFT_INVALID_DESCRIPTOR.
    if (config.workspace != config_value::ALLOW)
        throw mkl::invalid_argument("dft/backends/mklcpu", "commit",
                                    "MKLCPU only supports workspace set to allow");
    // Setting the ordering causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
    if (config.ordering != dft::detail::config_value::ORDERED) {
        throw mkl::invalid_argument("dft/backends/mklcpu", "commit",
                                    "MKLCPU only supports ordered ordering.");
    }
    // Setting the transpose causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
    if (config.transpose != false) {
        throw mkl::invalid_argument("dft/backends/mklcpu", "commit",
                                    "MKLCPU only supports non-transposed.");
    }
}

template <dft::detail::precision prec, dft::detail::domain dom>
sycl::buffer<mklcpu_desc_t, 1> commit_derived_impl<prec, dom>::get_handle_buffer() noexcept {
    return handle_buffer;
}

} // namespace detail

template <dft::detail::precision prec, dft::detail::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                        sycl::queue& sycl_queue) {
    return new detail::commit_derived_impl<prec, dom>(sycl_queue, desc.get_values());
}

template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

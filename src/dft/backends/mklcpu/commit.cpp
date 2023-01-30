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

/// Commit impl class specialization for MKLCPU.
template <dft::detail::precision prec, dft::detail::domain dom>
class commit_derived_impl : public dft::detail::commit_impl {
private:
    static constexpr DFTI_CONFIG_VALUE mklcpu_prec = to_mklcpu(prec);
    static constexpr DFTI_CONFIG_VALUE mklcpu_dom = to_mklcpu(dom);
    using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;
public:
    commit_derived_impl(sycl::queue queue, dft::detail::dft_values<prec, dom> config_values)
            : dft::detail::commit_impl(queue, backend::mklcpu), status(DFTI_NOTSET) {

        MKL_LONG err;
        sycl::buffer<MKL_LONG, 1> err_buffer{&err, sycl::range<1>{1}};

        set_value(handle, config_values);

        queue.submit([&](sycl::handler &cgh) {
            auto handle_obj = handle_buffer.template get_access<sycl::access::mode::read_write>(cgh);
            auto error_obj = handle_buffer.template get_access<sycl::access::mode::read_write>(cgh);
            std::int64_t local_err = DFTI_MEMORY_ERROR;

            host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=](){
                std::cout << "in commit kernel" << std::endl;
                if (config_values.rank == 1)
                    local_error = DftiCreateDescriptor(&handle_obj, mklcpu_prec, mklcpu_dom, config_values.rank, config_values.dimensions[0]);
                else 
                    local_error = DftiCreateDescriptor(&handle_obj, mklcpu_prec, mklcpu_dom, config_values.rank, config_values.dimensions.data());

                if (status != DFTI_NO_ERROR) {
                    throw oneapi::mkl::exception("dft", "commit", "DftiCreateDescriptor failed");
                }

                local_err = DftiCommitDescriptor(handle_obj);
                (*err_buffer.get_pointer()) = local_err;
            });
        });

        if (status != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft", "commit", "DftiCommitDescriptor failed");

        }
    }

    virtual void* get_handle() override {
        return handle_;
    }

    virtual ~commit_derived_impl() override {
        DftiFreeDescriptor((mklcpu_dest_t*)&handle);
    }

    DFT_ERROR status;
    sycl::buffer<mklcpu_dest_t, 1> handle_buffer = nullptr;

    template <typename... Args>
    DFT_ERROR set_value_item(mklcpu_dest_t hand, enum DFTI_CONFIG_PARAM name,
                             Args... args) {
        DFT_ERROR value_err = DFTI_NOTSET;
        value_err = DftiSetValue(hand, name, args...);
        if (value_err != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft", "set_value_item", std::to_string(name));
        }

        return value_err;
    }

    void set_value(mklcpu_dest_t& descHandle, dft::detail::dft_values<prec, dom> config) {
        set_value_item(descHandle, DFTI_INPUT_STRIDES, config.input_strides.data());
        set_value_item(descHandle, DFTI_OUTPUT_STRIDES, config.output_strides.data());
        set_value_item(descHandle, DFTI_BACKWARD_SCALE, config.bwd_scale);
        set_value_item(descHandle, DFTI_FORWARD_SCALE, config.fwd_scale);
        set_value_item(descHandle, DFTI_NUMBER_OF_TRANSFORMS, config.number_of_transforms);
        set_value_item(descHandle, DFTI_INPUT_DISTANCE, config.fwd_dist);
        set_value_item(descHandle, DFTI_OUTPUT_DISTANCE, config.bwd_dist);
        set_value_item(
            descHandle, DFTI_PLACEMENT,
            (config.placement == config_value::INPLACE) ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    }
};
} // namespace detail

template <dft::detail::precision prec, dft::detail::domain dom>
dft::detail::commit_impl* create_commit(dft::detail::descriptor<prec, dom>& desc,
                                        sycl::queue& sycl_queue) {
    return new detail::commit_derived_impl<prec, dom>(sycl_queue, desc.get_values());
}

template dft::detail::commit_impl* create_commit(
    dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl* create_commit(
    dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);
template dft::detail::commit_impl* create_commit(
    dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl* create_commit(
    dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

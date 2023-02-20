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
#include "mkl_service.h"
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

template <precision prec, domain dom>
class commit_derived_impl final : public detail::commit_impl {
public:
    commit_derived_impl(sycl::queue queue, const detail::dft_values<prec, dom>& config_values)
            : detail::commit_impl(queue, backend::mklcpu) {
        DFT_ERROR status = DFT_NOTSET;
        if (config_values.dimensions.size() == 1) {
            status = DftiCreateDescriptor(&handle, get_precision(prec), get_domain(dom),
                                          config_values.rank, config_values.dimensions[0]);
        }
        else {
            status = DftiCreateDescriptor(&handle, get_precision(prec), get_domain(dom),
                                          config_values.rank, config_values.dimensions.data());
        }
        if (status != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft/backends/mklcpu", "commit", "DftiCreateDescriptor failed");
        }

        set_value(handle, config_values);

        status = DftiCommitDescriptor(handle);
        if (status != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft/backends/mklcpu", "commit", "DftiCommitDescriptor failed");
        }
    }

    virtual void* get_handle() noexcept override {
        return handle;
    }

    virtual ~commit_derived_impl() override {
        DftiFreeDescriptor((DFTI_DESCRIPTOR_HANDLE*)&handle);
    }

private:
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    constexpr DFTI_CONFIG_VALUE get_domain(domain d) {
        if (d == domain::COMPLEX) {
            return DFTI_COMPLEX;
        }
        else {
            return DFTI_REAL;
        }
    }

    constexpr DFTI_CONFIG_VALUE get_precision(precision p) {
        if (p == precision::SINGLE) {
            return DFTI_SINGLE;
        }
        else {
            return DFTI_DOUBLE;
        }
    }

    template <typename... Args>
    void set_value_item(DFTI_DESCRIPTOR_HANDLE hand, enum DFTI_CONFIG_PARAM name,
                             Args... args) {
        if (auto ret = DftiSetValue(hand, name, args...); ret != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception(
                "dft/backends/mklcpu", "set_value_item",
                "name: " + std::to_string(name) + " error: " + std::to_string(ret));
        }
    }

    void set_value(DFTI_DESCRIPTOR_HANDLE& descHandle, const detail::dft_values<prec, dom>& config) {
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

template <precision prec, domain dom>
detail::commit_impl* create_commit(const descriptor<prec, dom>& desc, sycl::queue& sycl_queue) {
    return new commit_derived_impl<prec, dom>(sycl_queue, desc.get_values());
}

template detail::commit_impl* create_commit(const descriptor<precision::SINGLE, domain::REAL>&,
                                            sycl::queue&);
template detail::commit_impl* create_commit(const descriptor<precision::SINGLE, domain::COMPLEX>&,
                                            sycl::queue&);
template detail::commit_impl* create_commit(const descriptor<precision::DOUBLE, domain::REAL>&,
                                            sycl::queue&);
template detail::commit_impl* create_commit(const descriptor<precision::DOUBLE, domain::COMPLEX>&,
                                            sycl::queue&);

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

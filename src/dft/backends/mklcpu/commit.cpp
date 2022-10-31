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

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "mkl_service.h"
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

class commit_derived_impl : public oneapi::mkl::dft::detail::commit_impl {
public:
    commit_derived_impl(sycl::queue queue, dft_values config_values)
            : oneapi::mkl::dft::detail::commit_impl(queue),
              status(0) {
        logf("CPU impl, handle->%p", &handle);

        DFTI_DESCRIPTOR_HANDLE local_handle = nullptr;

        std::cout << config_values << std::endl;
        if (config_values.rank == 1) {
            status = DftiCreateDescriptor(&local_handle, precision_map[config_values.precision],
                                          domain_map[config_values.domain], config_values.rank,
                                          config_values.dimension[0]);
        }
        else {
            status = DftiCreateDescriptor(&local_handle, precision_map[config_values.precision],
                                          domain_map[config_values.domain], config_values.rank,
                                          &config_values.dimension[0]);
        }
        if(status != DFTI_NO_ERROR) throw oneapi::mkl::exception("dft", "commit", "DftiCreateDescriptor failed");

        set_value(local_handle, config_values);

        status = DftiCommitDescriptor(local_handle);
        if(status != DFTI_NO_ERROR) throw oneapi::mkl::exception("dft", "commit", "DftiCommitDescriptor failed");

        // commit_impl (pimpl_->handle) should return this handle
        handle = local_handle;
    }

    commit_derived_impl(const commit_derived_impl* other)
            : oneapi::mkl::dft::detail::commit_impl(*other) { }

    virtual oneapi::mkl::dft::detail::commit_impl* copy_state() override {
        return new commit_derived_impl(this);
    }

    virtual ~commit_derived_impl() override { }

private:
    bool status;
    std::unordered_map<oneapi::mkl::dft::precision, DFTI_CONFIG_VALUE> precision_map{
        { oneapi::mkl::dft::precision::SINGLE, DFTI_SINGLE },
        { oneapi::mkl::dft::precision::DOUBLE, DFTI_DOUBLE }
    };
    std::unordered_map<oneapi::mkl::dft::domain, DFTI_CONFIG_VALUE> domain_map{
        { oneapi::mkl::dft::domain::REAL, DFTI_REAL },
        { oneapi::mkl::dft::domain::COMPLEX, DFTI_COMPLEX }
    };

    void set_value(DFTI_DESCRIPTOR_HANDLE& descHandle, dft_values config) {
        logf("address of cpu handle->%p", &descHandle);
        logf("handle is_null? %s", (descHandle == nullptr) ? "yes" : "no");

        // TODO : add complex storage and workspace
        if (config.set_input_strides)
            status |= DftiSetValue(descHandle, DFTI_INPUT_STRIDES, &config.input_strides[0]);
        if (config.set_output_strides)
            status |= DftiSetValue(descHandle, DFTI_OUTPUT_STRIDES, &config.output_strides[0]);
        if (config.set_bwd_scale)
            status |= DftiSetValue(descHandle, DFTI_BACKWARD_SCALE, config.bwd_scale);
        if (config.set_fwd_scale)
            status |= DftiSetValue(descHandle, DFTI_BACKWARD_SCALE, config.fwd_scale);
        if (config.set_number_of_transforms)
            status |= DftiSetValue(descHandle, DFTI_NUMBER_OF_TRANSFORMS, config.number_of_transforms);
        if (config.set_fwd_dist)
            status |= DftiSetValue(descHandle, DFTI_FWD_DISTANCE, config.fwd_dist);
        if (config.set_bwd_dist)
            status |= DftiSetValue(descHandle, DFTI_BWD_DISTANCE, config.bwd_dist);
        if (config.set_placement)
            status |= DftiSetValue(descHandle, DFTI_PLACEMENT,
                         (config.placement == oneapi::mkl::dft::config_value::INPLACE)
                             ? DFTI_INPLACE
                             : DFTI_NOT_INPLACE);

        if(status != DFTI_NO_ERROR) throw oneapi::mkl::exception("dft", "commit", "DftiSetValue failed");
    }
};

oneapi::mkl::dft::detail::commit_impl* create_commit(sycl::queue queue, dft_values values) {
    return new commit_derived_impl(queue, values);
}

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

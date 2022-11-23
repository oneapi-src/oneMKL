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
#include "oneapi/mkl/dft/descriptor.hpp"

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
              status(-1) {

        if (config_values.rank == 1) {
            status = DftiCreateDescriptor(&handle, get_precision(config_values.precision),
                                          get_domain(config_values.domain), config_values.rank,
                                          config_values.dimensions[0]);
        } else {
            status = DftiCreateDescriptor(&handle, get_precision(config_values.precision),
                                          get_domain(config_values.domain), config_values.rank,
                                          config_values.dimensions.data());
        }
        if(status != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft", "commit", "DftiCreateDescriptor failed");
        }

        set_value(handle, config_values);

        status = DftiCommitDescriptor(handle);
        if(status != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft", "commit", "DftiCommitDescriptor failed");
        }
    }

    commit_derived_impl(const commit_derived_impl* other)
            : oneapi::mkl::dft::detail::commit_impl(*other) { }

    virtual ~commit_derived_impl() override { 
        DftiFreeDescriptor((DFTI_DESCRIPTOR_HANDLE *) &handle);
    }

private:
    DFT_ERROR status;
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    constexpr DFTI_CONFIG_VALUE get_domain(oneapi::mkl::dft::domain dom) {
        if (dom == oneapi::mkl::dft::domain::COMPLEX) {
            return DFTI_COMPLEX;
        } else {
            return DFTI_REAL;
        }
    }

    constexpr DFTI_CONFIG_VALUE get_precision(oneapi::mkl::dft::precision prec) {
        if (prec == oneapi::mkl::dft::precision::SINGLE) {
            return DFTI_SINGLE;
        } else {
            return DFTI_DOUBLE;
        }
    }

    void set_value(DFTI_DESCRIPTOR_HANDLE& descHandle, dft_values config) {
            // TODO : add complex storage and workspace, fix error handling
            status |= DftiSetValue(descHandle, DFTI_INPUT_STRIDES, config.input_strides.data());
            status |= DftiSetValue(descHandle, DFTI_OUTPUT_STRIDES, config.output_strides.data());
            status |= DftiSetValue(descHandle, DFTI_BACKWARD_SCALE, config.bwd_scale);
            status |= DftiSetValue(descHandle, DFTI_FORWARD_SCALE, config.fwd_scale);
            status |= DftiSetValue(descHandle, DFTI_NUMBER_OF_TRANSFORMS, config.number_of_transforms);
            status |= DftiSetValue(descHandle, DFTI_INPUT_DISTANCE, config.fwd_dist);
            status |= DftiSetValue(descHandle, DFTI_OUTPUT_DISTANCE, config.bwd_dist);
            status |= DftiSetValue(descHandle, DFTI_PLACEMENT,
                         (config.placement == oneapi::mkl::dft::config_value::INPLACE)
                             ? DFTI_INPLACE
                             : DFTI_NOT_INPLACE);

        if(status != DFTI_NO_ERROR) {
            throw oneapi::mkl::exception("dft", "commit", "DftiSetValue failed");
        }
    }
};

oneapi::mkl::dft::detail::commit_impl* create_commit(
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX> &desc
) {
    return new commit_derived_impl(desc.get_queue(), desc.get_values());
}
oneapi::mkl::dft::detail::commit_impl* create_commit(
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX> &desc
) {
    return new commit_derived_impl(desc.get_queue(), desc.get_values());
}
oneapi::mkl::dft::detail::commit_impl* create_commit(
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL> &desc
) {
    return new commit_derived_impl(desc.get_queue(), desc.get_values());
}
oneapi::mkl::dft::detail::commit_impl* create_commit(
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> &desc
) {
    return new commit_derived_impl(desc.get_queue(), desc.get_values());
}

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

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
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/detail/backends.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/detail/cufft/onemkl_dft_cufft.hpp"

// cuFFT headers
#include <cuda_runtime.h>
#include <cufft.h>

namespace oneapi::mkl::dft::cufft {
namespace detail {

/// Commit impl class specialization for cuFFT.
template <dft::detail::precision prec, dft::detail::domain dom>
class cufft_commit final : public dft::detail::commit_impl {
private:
    cufftHandle plan;

public:
    cufft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::mkl::dft::detail::commit_impl(queue, backend::cufft) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw mkl::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
        cufftCreate(&plan);

        // The cudaStream for the plan is set a execution time so the interop handler can pick the stream.

        const cufftType type = CUFFT_C2C;

        constexpr std::size_t max_supported_dims = 3;
        std::array<int, max_supported_dims> n_copy;
        std::copy(config_values.dimensions.begin(), config_values.dimensions.end(), n_copy.data());

        cufftPlanMany(&plan, // plan
                      static_cast<int>(config_values.dimensions.size()), // rank
                      n_copy.data(), // n
                      /*TODO*/ nullptr, // inembed
                      /*TODO*/ 1, // istride
                      /*TODO*/ 1, // idist
                      /*TODO*/ nullptr, // onembed
                      /*TODO*/ 1, // ostride
                      /*TODO*/ 1, // odist
                      /*TODO*/ type, // type
                      /*TODO*/ 1 // batch
        );
    }

    ~cufft_commit() override {
        cufftDestroy(plan);
    }

    virtual void* get_handle() override {
        return &plan;
    }
};
} // namespace detail

template <dft::detail::precision prec, dft::detail::domain dom>
dft::detail::commit_impl* create_commit(dft::detail::descriptor<prec, dom>& desc,
                                        sycl::queue& sycl_queue) {
    return new detail::cufft_commit<prec, dom>(sycl_queue, desc.get_values());
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

} // namespace oneapi::mkl::dft::cufft

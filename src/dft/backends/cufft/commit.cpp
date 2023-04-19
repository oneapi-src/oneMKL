/*******************************************************************************
* Copyright Codeplay Software Ltd
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

#include <array>
#include <algorithm>
#include <optional>

#include <cufft.h>
#include <cuda.h>

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/exceptions.hpp"

namespace oneapi::mkl::dft::cufft {
namespace detail {

/// Commit impl class specialization for cuFFT.
template <dft::precision prec, dft::domain dom>
class cufft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    // For real to complex transforms, the "type" arg also encodes the direction (e.g. CUFFT_R2C vs CUFFT_C2R) in the plan so we must have one for each direction.
    // We also need this because oneMKL uses a directionless "FWD_DISTANCE" and "BWD_DISTANCE" while cuFFT uses a directional "idist" and "odist".
    // plans[0] is forward, plans[1] is backward
    std::optional<std::array<cufftHandle, 2>> plans = std::nullopt;

public:
    cufft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::mkl::dft::detail::commit_impl<prec, dom>(queue, backend::cufft) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw mkl::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
    }

    void clean_plans() {
        if (plans) {
            if (cufftDestroy(plans.value()[0]) != CUFFT_SUCCESS) {
                throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                     "Failed to destroy forward cuFFT plan.");
            }
            if (cufftDestroy(plans.value()[1]) != CUFFT_SUCCESS) {
                throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                     "Failed to destroy backward cuFFT plan.");
            }
            // cufftDestroy changes the context so change it back.
            CUcontext interopContext =
                sycl::get_native<sycl::backend::ext_oneapi_cuda>(this->get_queue().get_context());
            if (cuCtxSetCurrent(interopContext) != CUDA_SUCCESS) {
                throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                     "Failed to change cuda context.");
            }
            plans = std::nullopt;
        }
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // this could be a recommit
        clean_plans();

        // The cudaStream for the plan is set at execution time so the interop handler can pick the stream.
        constexpr cufftType fwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_C2C;
                }
                else {
                    return CUFFT_Z2Z;
                }
            }
            else {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_R2C;
                }
                else {
                    return CUFFT_D2Z;
                }
            }
        }();
        constexpr cufftType bwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_C2C;
                }
                else {
                    return CUFFT_Z2Z;
                }
            }
            else {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_C2R;
                }
                else {
                    return CUFFT_Z2D;
                }
            }
        }();

        constexpr std::size_t max_supported_dims = 3;
        std::array<int, max_supported_dims> n_copy;
        std::copy(config_values.dimensions.begin(), config_values.dimensions.end(), n_copy.data());
        const int rank = static_cast<int>(config_values.dimensions.size());
        const int istride = static_cast<int>(config_values.input_strides.back());
        const int ostride = static_cast<int>(config_values.output_strides.back());
        const int batch = static_cast<int>(config_values.number_of_transforms);
        const int fwd_dist = static_cast<int>(config_values.fwd_dist);
        const int bwd_dist = static_cast<int>(config_values.bwd_dist);
        std::array<int, max_supported_dims> inembed;
        if (rank == 2) {
            inembed[1] = config_values.input_strides[1];
        }
        else if (rank == 3) {
            inembed[2] = config_values.input_strides[2];
            inembed[1] = config_values.input_strides[1] / inembed[2];
        }
        std::array<int, max_supported_dims> onembed;
        if (rank == 2) {
            onembed[1] = config_values.output_strides[1];
        }
        else if (rank == 3) {
            onembed[2] = config_values.output_strides[2];
            onembed[1] = config_values.output_strides[1] / onembed[2];
        }

        cufftHandle fwd_plan, bwd_plan;

        // forward plan
        auto res = cufftPlanMany(&fwd_plan, // plan
                                 rank, // rank
                                 n_copy.data(), // n
                                 inembed.data(), // inembed
                                 istride, // istride
                                 fwd_dist, // idist
                                 onembed.data(), // onembed
                                 ostride, // ostride
                                 bwd_dist, // odist
                                 fwd_type, // type
                                 batch // batch
        );

        if (res != CUFFT_SUCCESS) {
            throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                 "Failed to create forward cuFFT plan.");
        }

        // flip fwd_distance and bwd_distance because cuFFt uses input distance and output distance.
        // backward plan
        res = cufftPlanMany(&bwd_plan, // plan
                            rank, // rank
                            n_copy.data(), // n
                            inembed.data(), // inembed
                            istride, // istride
                            bwd_dist, // idist
                            onembed.data(), // onembed
                            ostride, // ostride
                            fwd_dist, // odist
                            bwd_type, // type
                            batch // batch
        );
        if (res != CUFFT_SUCCESS) {
            throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                 "Failed to create backward cuFFT plan.");
        }
        plans = { fwd_plan, bwd_plan };
    }

    ~cufft_commit() override {
        clean_plans();
    }

    void* get_handle() noexcept override {
        return plans.value().data();
    }
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::cufft_commit<prec, dom>(sycl_queue, desc.get_values());
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

} // namespace oneapi::mkl::dft::cufft

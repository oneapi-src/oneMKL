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

template<typename T>
void print(std::vector<T> v){
    for(T a : v){
        std::cout << a << ", ";
    }
    std::cout << std::endl;
}

/// Commit impl class specialization for cuFFT.
template <dft::precision prec, dft::domain dom>
class cufft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    // For real to complex transforms, the "type" arg also encodes the direction (e.g. CUFFT_R2C vs CUFFT_C2R) in the plan so we must have one for each direction.
    // We also need this because oneMKL uses a directionless "FWD_DISTANCE" and "BWD_DISTANCE" while cuFFT uses a directional "idist" and "odist".
    // plans[0] is forward, plans[1] is backward
    std::array<std::optional<cufftHandle>, 2> plans = { std::nullopt, std::nullopt };
    int64_t in_offset;
    int64_t out_offset;

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
        auto fix_context = plans[0].has_value() || plans[1].has_value();
        if (plans[0]) {
            if (cufftDestroy(plans[0].value()) != CUFFT_SUCCESS) {
                throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                     "Failed to destroy forward cuFFT plan.");
            }
            plans[0] = std::nullopt;
        }
        if (plans[1]) {
            if (cufftDestroy(plans[1].value()) != CUFFT_SUCCESS) {
                throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                     "Failed to destroy backward cuFFT plan.");
            }
            plans[1] = std::nullopt;
        }
        if (fix_context) {
            // cufftDestroy changes the context so change it back.
            CUcontext interopContext =
                sycl::get_native<sycl::backend::ext_oneapi_cuda>(this->get_queue().get_context());
            if (cuCtxSetCurrent(interopContext) != CUDA_SUCCESS) {
                throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                     "Failed to change cuda context.");
            }
        }
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // this could be a recommit
        clean_plans();

        if (config_values.fwd_scale != 1.0 || config_values.bwd_scale != 1.0) {
            throw mkl::unimplemented(
                "dft/backends/cufft", __FUNCTION__,
                "cuFFT does not support values other than 1 for FORWARD/BACKWARD_SCALE");
        }

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
        std::vector<int> inembed = {config_values.input_strides.begin(), config_values.input_strides.end()};
        std::vector<int> onembed = {config_values.output_strides.begin(), config_values.output_strides.end()};
        std::cout << "inembed "; 
        print(inembed);
        auto i_min = std::min_element(inembed.begin()+1, inembed.end());
        auto o_min = std::min_element(onembed.begin()+1, onembed.end());
        std::cout << "i_min " << *i_min << std::endl; 
        if constexpr(dom == dft::domain::REAL){
            if(i_min != inembed.begin() + rank){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                     "cufft requires the last input stride to be the the smallest one for real transforms!");
            }
            if(o_min != onembed.begin() + rank){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                     "cufft requires the last output stride to be the the smallest one for real transforms!");
            }
        } else{
            if(o_min - onembed.begin() != i_min - inembed.begin()){
                    throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                        "cufft requires that if the strides are ordered by stride length, the order is the same for input and output strides!");
            }
        }
        const int istride = static_cast<int>(*i_min);
        const int ostride = static_cast<int>(*o_min);
        inembed.erase(i_min);
        onembed.erase(o_min);
        if(o_min - onembed.begin() != rank){
            // swap dimensions to have the last one have the smallest stride
            std::swap(n_copy[o_min - onembed.begin()-1], n_copy[rank-1]);
        }
        for(int i=0;i<rank-1;i++){
            std::cout << "inembed stride div " << i << ": " << inembed[i+1] << " / " << istride << std::endl;
            if(inembed[i+1] % istride != 0){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                     "cufft requires an input stride to be divisible by all smaller input strides!");
            }
            inembed[i+1] /= istride;
            if(onembed[i+1] % ostride != 0){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                     "cufft requires an output stride to be divisible by all smaller output strides!");
            }
            onembed[i+1] /= ostride;
        }
        if(rank>2){
            if(inembed[1]>inembed[2] && onembed[1]<onembed[2]){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                    "cufft requires that if the strides are ordered by stride length, the order is the same for input and output strides!");
            } else if(inembed[1]<inembed[2] && onembed[1]<onembed[2]){
                // swap dimensions to have the first one have the biggest stride
                std::swap(inembed[1],inembed[2]);
                std::swap(onembed[1],onembed[2]);
                std::swap(n_copy[0],n_copy[1]);
            }
            if(inembed[1] % inembed[2] != 0){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                     "cufft requires an input stride to be divisible by all smaller input strides!");
            }
            if(onembed[1] % onembed[2] != 0){
                throw mkl::unimplemented("dft/backends/cufft", __FUNCTION__,
                                     "cufft requires an output stride to be divisible by all smaller output strides!");
            }
            std::cout << "inembed inembed div " << inembed[1] << " / " << inembed[2] << std::endl;
            inembed[1] /= inembed[2];
            onembed[1] /= onembed[2];
        }
        in_offset = config_values.input_strides[0];
        out_offset = config_values.output_strides[0];
        const int batch = static_cast<int>(config_values.number_of_transforms);
        const int fwd_dist = static_cast<int>(config_values.fwd_dist);
        const int bwd_dist = static_cast<int>(config_values.bwd_dist);
        std::cout << "rank: " << rank << std::endl;
        std::cout << "dimensions: " << n_copy[0] << " " << n_copy[1] << " " << n_copy[2] << std::endl; 
        std::cout << "embeds1: " << inembed[1] << " " << onembed[1];
        if(rank>2){
          std::cout << "embeds2: " << inembed[2] << " " << onembed[2];
        }
        std::cout << " strides: " << istride << " " << ostride << " distances " << fwd_dist << " " << bwd_dist << std::endl;
        /*std::array<int, max_supported_dims> inembed;
        if (rank == 2) {
            inembed[1] = config_values.input_strides[1] / istride;
        }
        else if (rank == 3) {
            inembed[2] = config_values.input_strides[2] / istride;
            inembed[1] = config_values.input_strides[1] / inembed[2] / istride;
        }
        std::array<int, max_supported_dims> onembed;
        if (rank == 2) {
            onembed[1] = config_values.output_strides[1] / ostride;
        }
        else if (rank == 3) {
            onembed[2] = config_values.output_strides[2] / ostride;
            onembed[1] = config_values.output_strides[1] / onembed[2] / ostride;
        }*/

        // When creating real-complex descriptions, the strides will always be wrong for one of the directions.
        // This is because the least significant dimension is symmetric.
        // If the strides are invalid (too small to fit) then just don't bother creating the plan.
        const bool ignore_strides = dom == dft::domain::COMPLEX || rank == 1;
        const bool valid_forward =
            ignore_strides || (n_copy[rank - 1] <= inembed[rank - 1] &&
                               (n_copy[rank - 1] / 2 + 1) <= onembed[rank - 1]);
        const bool valid_backward =
            ignore_strides || (n_copy[rank - 1] <= onembed[rank - 1] &&
                               (n_copy[rank - 1] / 2 + 1) <= inembed[rank - 1]);

        if (!valid_forward && !valid_backward) {
            throw mkl::exception("dft/backends/cufft", __FUNCTION__,
                                    "Invalid strides.");
        }

        if (valid_forward) {
            cufftHandle fwd_plan;
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

            plans[0] = fwd_plan;
        }

        if (valid_backward) {
            cufftHandle bwd_plan;

            // flip fwd_distance and bwd_distance because cuFFt uses input distance and output distance.
            auto res = cufftPlanMany(&bwd_plan, // plan
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
            plans[1] = bwd_plan;
        }
    }

    ~cufft_commit() override {
        clean_plans();
    }

    void* get_handle() noexcept override {
        return plans.data();
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

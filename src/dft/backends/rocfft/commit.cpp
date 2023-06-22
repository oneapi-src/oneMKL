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

#include <rocfft.h>
#include <hip/hip_runtime_api.h>

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "rocfft_handle.hpp"

namespace oneapi::mkl::dft::rocfft {
namespace detail {

// rocfft has global setup and cleanup functions which use some global state internally.
// Each can be called multiple times in an application, but due to the global nature, they always need to alternate.
// I don't believe its possible to avoid the user calling rocfft_cleanup in their own code,
// breaking our code, but we can try avoid it for them.
// rocfft_cleanup internally uses some singletons, so it is very difficult to decide if this is safe due to
// the static initialisation order fiasco.
class rocfft_singleton {
    rocfft_singleton() {
        const auto result = rocfft_setup();
        if (result != rocfft_status_success) {
            throw mkl::exception(
                "DFT", "rocfft",
                "Failed to setup rocfft. returned status " + std::to_string(result));
        }
    }

    ~rocfft_singleton() {
        (void)rocfft_cleanup();
    }

    // no copies or moves allowed
    rocfft_singleton(const rocfft_singleton& other) = delete;
    rocfft_singleton(rocfft_singleton&& other) noexcept = delete;
    rocfft_singleton& operator=(const rocfft_singleton& other) = delete;
    rocfft_singleton& operator=(rocfft_singleton&& other) noexcept = delete;

public:
    static void init() {
        static rocfft_singleton instance;
        (void)instance;
    }
};

/// Commit impl class specialization for rocFFT.
template <dft::precision prec, dft::domain dom>
class rocfft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    // For real to complex transforms, the "transform_type" arg also encodes the direction (e.g. rocfft_transform_type_*_forward vs rocfft_transform_type_*_backward)
    // in the plan so we must have one for each direction.
    // We also need this because oneMKL uses a directionless "FWD_DISTANCE" and "BWD_DISTANCE" while rocFFT uses a directional "in_distance" and "out_distance".
    // The same is also true for "FORWARD_SCALE" and "BACKWARD_SCALE".
    // handles[0] is forward, handles[1] is backward
    std::array<rocfft_handle, 2> handles{};

public:
    rocfft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::mkl::dft::detail::commit_impl<prec, dom>(queue, backend::rocfft) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw mkl::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
        // initialise the rocFFT global state
        rocfft_singleton::init();
    }

    void clean_plans() {
        if (handles[0].plan) {
            if (rocfft_plan_destroy(handles[0].plan.value()) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to destroy forward plan.");
            }
            handles[0].plan = std::nullopt;
        }
        if (handles[1].plan) {
            if (rocfft_plan_destroy(handles[1].plan.value()) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to destroy backward plan.");
            }
            handles[1].plan = std::nullopt;
        }

        if (handles[0].info) {
            if (rocfft_execution_info_destroy(handles[0].info.value()) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to destroy forward execution info .");
            }
            handles[0].info = std::nullopt;
        }
        if (handles[1].info) {
            if (rocfft_execution_info_destroy(handles[1].info.value()) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to destroy backward execution info .");
            }
            handles[1].info = std::nullopt;
        }
        if (handles[0].buffer) {
            if (hipFree(handles[0].buffer.value()) != hipSuccess) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to free forward buffer.");
            }
            handles[0].buffer = std::nullopt;
        }
        if (handles[1].buffer) {
            if (hipFree(handles[1].buffer.value()) != hipSuccess) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to free backward buffer.");
            }
            handles[1].buffer = std::nullopt;
        }
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // this could be a recommit
        clean_plans();

        const rocfft_result_placement placement =
            (config_values.placement == dft::config_value::INPLACE) ? rocfft_placement_inplace
                                                                    : rocfft_placement_notinplace;

        constexpr rocfft_transform_type fwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                return rocfft_transform_type_complex_forward;
            }
            else {
                return rocfft_transform_type_real_forward;
            }
        }();
        constexpr rocfft_transform_type bwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                return rocfft_transform_type_complex_inverse;
            }
            else {
                return rocfft_transform_type_real_inverse;
            }
        }();

        constexpr rocfft_precision precision = [] {
            if constexpr (prec == dft::precision::SINGLE) {
                return rocfft_precision_single;
            }
            else {
                return rocfft_precision_double;
            }
        }();

        const std::size_t dimensions = config_values.dimensions.size();

        constexpr std::size_t max_supported_dims = 3;
        std::array<std::size_t, max_supported_dims> lengths;
        // rocfft does dimensions in the reverse order to oneMKL
        std::copy(config_values.dimensions.crbegin(), config_values.dimensions.crend(),
                  lengths.data());

        const std::size_t number_of_transforms =
            static_cast<std::size_t>(config_values.number_of_transforms);

        const std::size_t fwd_distance = static_cast<std::size_t>(config_values.fwd_dist);
        const std::size_t bwd_distance = static_cast<std::size_t>(config_values.bwd_dist);

        const rocfft_array_type fwd_array_ty = [&config_values]() {
            if constexpr (dom == dft::domain::COMPLEX) {
                if (config_values.complex_storage == dft::config_value::COMPLEX_COMPLEX) {
                    return rocfft_array_type_complex_interleaved;
                }
                else {
                    return rocfft_array_type_complex_planar;
                }
            }
            else {
                return rocfft_array_type_real;
            }
        }();
        const rocfft_array_type bwd_array_ty = [&config_values]() {
            if constexpr (dom == dft::domain::COMPLEX) {
                if (config_values.complex_storage == dft::config_value::COMPLEX_COMPLEX) {
                    return rocfft_array_type_complex_interleaved;
                }
                else {
                    return rocfft_array_type_complex_planar;
                }
            }
            else {
                if (config_values.conj_even_storage != dft::config_value::COMPLEX_COMPLEX) {
                    throw mkl::exception(
                        "dft/backends/rocfft", __FUNCTION__,
                        "only COMPLEX_COMPLEX conjugate_even_storage is supported");
                }
                return rocfft_array_type_hermitian_interleaved;
            }
        }();

        std::array<std::size_t, 2> in_offsets{
            static_cast<std::size_t>(config_values.input_strides[0]),
            static_cast<std::size_t>(config_values.input_strides[0])
        };
        std::array<std::size_t, 2> out_offsets{
            static_cast<std::size_t>(config_values.output_strides[0]),
            static_cast<std::size_t>(config_values.output_strides[0])
        };

        std::array<std::size_t, 3> in_strides;
        std::array<std::size_t, 3> out_strides;

        for (std::size_t i = 0; i != dimensions; ++i) {
            in_strides[i] = config_values.input_strides[dimensions - i];
            out_strides[i] = config_values.output_strides[dimensions - i];
        }

        rocfft_plan_description plan_desc;
        if (rocfft_plan_description_create(&plan_desc) != rocfft_status_success) {
            throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                 "Failed to create plan description.");
        }

        // plan_description can be destroyed afted plan_create
        auto description_destroy = [](rocfft_plan_description p) {
            if (rocfft_plan_description_destroy(p) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to destroy plan description.");
            }
        };
        std::unique_ptr<rocfft_plan_description_t, decltype(description_destroy)>
            description_destroyer(plan_desc, description_destroy);

        // When creating real-complex descriptions, the strides will always be wrong for one of the directions.
        // This is because the least significant dimension is symmetric.
        // If the strides are invalid (too small to fit) then just don't bother creating the plan.
        const bool ignore_strides = dom == dft::domain::COMPLEX || dimensions == 1;
        const bool valid_forward =
            ignore_strides || (lengths[0] <= in_strides[1] && lengths[0] / 2 + 1 <= out_strides[1]);
        const bool valid_backward =
            ignore_strides || (lengths[0] <= out_strides[1] && lengths[0] / 2 + 1 <= in_strides[1]);

        if (valid_forward) {
            auto res =
                rocfft_plan_description_set_data_layout(plan_desc, fwd_array_ty, bwd_array_ty,
                                                        in_offsets.data(), // in offsets
                                                        out_offsets.data(), // out offsets
                                                        dimensions,
                                                        in_strides.data(), //in strides
                                                        fwd_distance, // in distance
                                                        dimensions,
                                                        out_strides.data(), // out strides
                                                        bwd_distance // out distance
                );
            if (res != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to set forward data layout.");
            }

            if (rocfft_plan_description_set_scale_factor(plan_desc, config_values.fwd_scale) !=
                rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to set forward scale factor.");
            }

            rocfft_plan fwd_plan;
            res = rocfft_plan_create(&fwd_plan, placement, fwd_type, precision, dimensions,
                                     lengths.data(), number_of_transforms, plan_desc);

            if (res != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to create forward plan.");
            }

            handles[0].plan = fwd_plan;

            rocfft_execution_info fwd_info;
            if (rocfft_execution_info_create(&fwd_info) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to create forward execution info.");
            }
            handles[0].info = fwd_info;

            // plan work buffer
            std::size_t work_buf_size;
            if (rocfft_plan_get_work_buffer_size(fwd_plan, &work_buf_size) !=
                rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to get forward work buffer size.");
            }
            if (work_buf_size != 0) {
                void* work_buf;
                if (hipMalloc(&work_buf, work_buf_size) != hipSuccess) {
                    throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                         "Failed to get allocate forward work buffer.");
                }
                handles[0].buffer = work_buf;
                if (rocfft_execution_info_set_work_buffer(fwd_info, work_buf, work_buf_size) !=
                    rocfft_status_success) {
                    throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                         "Failed to set forward work buffer.");
                }
            }
        }

        if (valid_backward) {
            auto res =
                rocfft_plan_description_set_data_layout(plan_desc, bwd_array_ty, fwd_array_ty,
                                                        in_offsets.data(), // in offsets
                                                        out_offsets.data(), // out offsets
                                                        dimensions,
                                                        in_strides.data(), //in strides
                                                        bwd_distance, // in distance
                                                        dimensions,
                                                        out_strides.data(), // out strides
                                                        fwd_distance // out distance
                );
            if (res != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to set backward data layout.");
            }

            if (rocfft_plan_description_set_scale_factor(plan_desc, config_values.bwd_scale) !=
                rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to set backward scale factor.");
            }

            rocfft_plan bwd_plan;
            res = rocfft_plan_create(&bwd_plan, placement, bwd_type, precision, dimensions,
                                     lengths.data(), number_of_transforms, plan_desc);
            if (res != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to create backward rocFFT plan.");
            }
            handles[1].plan = bwd_plan;

            rocfft_execution_info bwd_info;
            if (rocfft_execution_info_create(&bwd_info) != rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to create backward execution info.");
            }
            handles[1].info = bwd_info;

            std::size_t work_buf_size;
            if (rocfft_plan_get_work_buffer_size(bwd_plan, &work_buf_size) !=
                rocfft_status_success) {
                throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                     "Failed to get backward work buffer size.");
            }

            if (work_buf_size != 0) {
                void* work_buf;
                if (hipMalloc(&work_buf, work_buf_size) != hipSuccess) {
                    throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                         "Failed to get allocate backward work buffer.");
                }
                handles[1].buffer = work_buf;

                if (rocfft_execution_info_set_work_buffer(bwd_info, work_buf, work_buf_size) !=
                    rocfft_status_success) {
                    throw mkl::exception("dft/backends/rocfft", __FUNCTION__,
                                         "Failed to set backward work buffer.");
                }
            }
        }
    }

    ~rocfft_commit() override {
        clean_plans();
    }

    // Rule of three. Copying could lead to memory safety issues.
    rocfft_commit(const rocfft_commit& other) = delete;
    rocfft_commit& operator=(const rocfft_commit& other) = delete;

    void* get_handle() noexcept override {
        return handles.data();
    }
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::rocfft_commit<prec, dom>(sycl_queue, desc.get_values());
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

} // namespace oneapi::mkl::dft::rocfft

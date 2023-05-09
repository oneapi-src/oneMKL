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
#include "oneapi/mkl/dft/detail/mklgpu/onemkl_dft_mklgpu.hpp"

#include "dft/backends/mklgpu/mklgpu_helpers.hpp"

// MKLGPU header
#include "oneapi/mkl/dfti.hpp"

/**
Note that in this file, the Intel oneMKL closed-source library's interface mirrors the interface
of this OneMKL open-source library. Consequently, the types under dft::TYPE are closed-source oneMKL types,
and types under dft::detail::TYPE are from this library.
**/

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklgpu {
namespace detail {

/// Commit impl class specialization for MKLGPU.
template <dft::detail::precision prec, dft::detail::domain dom>
class commit_derived_impl final : public dft::detail::commit_impl<prec, dom> {
private:
    // Equivalent MKLGPU precision and domain from OneMKL's precision / domain.
    static constexpr dft::precision mklgpu_prec = to_mklgpu(prec);
    static constexpr dft::domain mklgpu_dom = to_mklgpu(dom);
    using mklgpu_descriptor_t = dft::descriptor<mklgpu_prec, mklgpu_dom>;

public:
    commit_derived_impl(sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::mkl::dft::detail::commit_impl<prec, dom>(queue, backend::mklgpu),
              handle(config_values.dimensions) {
        // MKLGPU does not throw an informative exception for the following:
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw mkl::exception("dft/backends/mklgpu", "commit",
                                     "Device does not support double precision.");
            }
        }
    }

    virtual void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        set_value(handle, config_values);
        try {
            handle.commit(this->get_queue());
        }
        catch (const std::exception& mkl_exception) {
            // Catching the real MKL exception causes headaches with naming.
            throw mkl::exception("dft/backends/mklgpu", "commit", mkl_exception.what());
        }
    }

    void* get_handle() noexcept override {
        return &handle;
    }

    ~commit_derived_impl() override = default;

private:
    // The native MKLGPU class.
    mklgpu_descriptor_t handle;

    void set_value(mklgpu_descriptor_t& desc, const dft::detail::dft_values<prec, dom>& config) {
        using onemkl_param = dft::detail::config_param;
        using backend_param = dft::config_param;

        // The following are read-only:
        // Dimension, forward domain, precision, commit status.
        // Lengths are supplied at descriptor construction time.
        desc.set_value(backend_param::FORWARD_SCALE, config.fwd_scale);
        desc.set_value(backend_param::BACKWARD_SCALE, config.bwd_scale);
        desc.set_value(backend_param::NUMBER_OF_TRANSFORMS, config.number_of_transforms);
        desc.set_value(backend_param::COMPLEX_STORAGE,
                       to_mklgpu<onemkl_param::COMPLEX_STORAGE>(config.complex_storage));
        if (config.real_storage != dft::detail::config_value::REAL_REAL) {
            throw mkl::invalid_argument("dft/backends/mklgpu", "commit",
                                        "MKLGPU only supports real-real real storage.");
        }
        desc.set_value(backend_param::CONJUGATE_EVEN_STORAGE,
                       to_mklgpu<onemkl_param::CONJUGATE_EVEN_STORAGE>(config.conj_even_storage));
        desc.set_value(backend_param::PLACEMENT,
                       to_mklgpu<onemkl_param::PLACEMENT>(config.placement));
        desc.set_value(backend_param::INPUT_STRIDES, config.input_strides.data());
        desc.set_value(backend_param::OUTPUT_STRIDES, config.output_strides.data());
        desc.set_value(backend_param::FWD_DISTANCE, config.fwd_dist);
        desc.set_value(backend_param::BWD_DISTANCE, config.bwd_dist);
        // Setting the workspace causes an FFT_INVALID_DESCRIPTOR.
        // Setting the ordering causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.ordering != dft::detail::config_value::ORDERED) {
            throw mkl::invalid_argument("dft/backends/mklgpu", "commit",
                                        "MKLGPU only supports ordered ordering.");
        }
        // Setting the transpose causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.transpose != false) {
            throw mkl::invalid_argument("dft/backends/mklgpu", "commit",
                                        "MKLGPU only supports non-transposed.");
        }
    }
};
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

} // namespace mklgpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

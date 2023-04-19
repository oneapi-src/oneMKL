/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#ifndef _ONEMKL_DFT_SRC_MKLCPU_HELPERS_HPP_
#define _ONEMKL_DFT_SRC_MKLCPU_HELPERS_HPP_

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"

// MKLCPU header
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {
namespace detail {

// host_task automatically uses run_on_host_intel if it is supported by the
//  compiler. Otherwise, it falls back to single_task.
template <typename K, typename H, typename F>
static inline auto host_task_internal(H &cgh, F f, int) -> decltype(cgh.host_task(f)) {
    return cgh.host_task(f);
}

template <typename K, typename H, typename F>
static inline void host_task(H &cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

template <typename Desc>
class kernel_name {};

/// Convert domain to equivalent backend native value.
inline constexpr DFTI_CONFIG_VALUE to_mklcpu(dft::detail::domain dom) {
    if (dom == dft::detail::domain::REAL) {
        return DFTI_REAL;
    }
    else {
        return DFTI_COMPLEX;
    }
}

/// Convert precision to equivalent backend native value.
inline constexpr DFTI_CONFIG_VALUE to_mklcpu(dft::detail::precision dom) {
    if (dom == dft::detail::precision::SINGLE) {
        return DFTI_SINGLE;
    }
    else {
        return DFTI_DOUBLE;
    }
}

/// Convert a config_param to equivalent backend native value.
inline constexpr DFTI_CONFIG_PARAM to_mklcpu(dft::detail::config_param param) {
    using iparam = dft::detail::config_param;
    switch (param) {
        case iparam::FORWARD_DOMAIN: return DFTI_FORWARD_DOMAIN;
        case iparam::DIMENSION: return DFTI_DIMENSION;
        case iparam::LENGTHS: return DFTI_LENGTHS;
        case iparam::PRECISION: return DFTI_PRECISION;
        case iparam::FORWARD_SCALE: return DFTI_FORWARD_SCALE;
        case iparam::NUMBER_OF_TRANSFORMS: return DFTI_NUMBER_OF_TRANSFORMS;
        case iparam::COMPLEX_STORAGE: return DFTI_COMPLEX_STORAGE;
        case iparam::REAL_STORAGE: return DFTI_REAL_STORAGE;
        case iparam::CONJUGATE_EVEN_STORAGE: return DFTI_CONJUGATE_EVEN_STORAGE;
        case iparam::INPUT_STRIDES: return DFTI_INPUT_STRIDES;
        case iparam::OUTPUT_STRIDES: return DFTI_OUTPUT_STRIDES;
        case iparam::FWD_DISTANCE: return DFTI_FWD_DISTANCE;
        case iparam::BWD_DISTANCE: return DFTI_BWD_DISTANCE;
        case iparam::WORKSPACE: return DFTI_WORKSPACE;
        case iparam::ORDERING: return DFTI_ORDERING;
        case iparam::TRANSPOSE: return DFTI_TRANSPOSE;
        case iparam::PACKED_FORMAT: return DFTI_PACKED_FORMAT;
        case iparam::COMMIT_STATUS: return DFTI_COMMIT_STATUS;
        default:
            throw mkl::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                        "Invalid config param.");
            return static_cast<DFTI_CONFIG_PARAM>(0);
    }
}

/** Convert a config_value to the backend's native value. Throw on invalid input.
 * @tparam Param The config param the value is for.
 * @param value The config value to convert.
**/
template <dft::detail::config_param Param>
inline constexpr int to_mklcpu(dft::detail::config_value value);

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::COMPLEX_STORAGE>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::COMPLEX_COMPLEX) {
        return DFTI_COMPLEX_COMPLEX;
    }
    else if (value == dft::detail::config_value::REAL_REAL) {
        return DFTI_REAL_REAL;
    }
    else {
        throw mkl::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                    "Invalid config value for complex storage.");
        return 0;
    }
}

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::REAL_STORAGE>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::REAL_REAL) {
        return DFTI_REAL_REAL;
    }
    else {
        throw mkl::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                    "Invalid config value for real storage.");
        return 0;
    }
}
template <>
inline constexpr int to_mklcpu<dft::detail::config_param::CONJUGATE_EVEN_STORAGE>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::COMPLEX_COMPLEX) {
        return DFTI_COMPLEX_COMPLEX;
    }
    else {
        throw mkl::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                    "Invalid config value for conjugate even storage.");
        return 0;
    }
}

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::PLACEMENT>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::INPLACE) {
        return DFTI_INPLACE;
    }
    else if (value == dft::detail::config_value::NOT_INPLACE) {
        return DFTI_NOT_INPLACE;
    }
    else {
        throw mkl::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                    "Invalid config value for inplace.");
        return 0;
    }
}

template <>
inline constexpr int to_mklcpu<dft::detail::config_param::PACKED_FORMAT>(
    dft::detail::config_value value) {
    if (value == dft::detail::config_value::CCE_FORMAT) {
        return DFTI_CCE_FORMAT;
    }
    else {
        throw mkl::invalid_argument("dft", "MKLCPU descriptor set_value()",
                                    "Invalid config value for packed format.");
        return 0;
    }
}

enum DIR { fwd, bwd };

template <dft::detail::precision prec, dft::detail::domain dom>
class commit_derived_impl : public dft::detail::commit_impl<prec, dom> {
private:
    static constexpr DFTI_CONFIG_VALUE mklcpu_prec = to_mklcpu(prec);
    static constexpr DFTI_CONFIG_VALUE mklcpu_dom = to_mklcpu(dom);
    using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;

public:
    commit_derived_impl(sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values);

    virtual void commit(const dft::detail::dft_values<prec, dom>& config_values) override;

    void* get_handle() noexcept override;

    virtual ~commit_derived_impl() override;

    virtual sycl::buffer<std::vector<mklcpu_desc_t>, 1> get_handle_buffer() noexcept;

private:
    mklcpu_desc_t device_handle = nullptr;
    sycl::buffer<mklcpu_desc_t, 1> handle_buffer{ &device_handle, sycl::range<1>{ 1 } };

    // ------- new api
    std::vector<mklcpu_desc_t> bidirection_handle {nullptr, nullptr};
    sycl::buffer<std::vector<mklcpu_desc_t>, 1> bidirection_buffer { &bidirection_handle, sycl::range<1>{ 1 } };

    template <typename... Args>
    void set_value_item(mklcpu_desc_t hand, enum DFTI_CONFIG_PARAM name, Args... args);

    void set_value(mklcpu_desc_t* descHandle, const dft::detail::dft_values<prec, dom>& config);

};

using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;

template<dft::detail::precision prec, dft::detail::domain dom>
using commit_t = dft::detail::commit_impl<prec, dom>;

template<dft::detail::precision prec, dft::detail::domain dom>
using commit_derived_t = detail::commit_derived_impl<prec, dom>;




} // namespace detail
} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_DFT_SRC_MKLCPU_HELPERS_HPP_

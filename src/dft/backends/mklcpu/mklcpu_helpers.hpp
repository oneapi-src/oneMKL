/*******************************************************************************
* Copyright 2022 Codeplay Software Ltd.
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

#include "oneapi/mkl/detail/exceptions.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"

// MKLCPU header
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {
namespace detail {

template<typename T> struct is_buffer : std::false_type {};
template<> struct is_buffer<sycl::buffer<float>> : std::true_type {};
template<> struct is_buffer<sycl::buffer<std::complex<float>>> : std::true_type {};
template<> struct is_buffer<sycl::buffer<double>> : std::true_type {};
template<> struct is_buffer<sycl::buffer<std::complex<double>>> : std::true_type {};

template< class T>
inline constexpr bool is_buffer_v = is_buffer<T>::value;

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
inline constexpr dft::config_param to_mklcpu(dft::detail::config_param param) {
    using iparam = dft::detail::config_param;
    using oparam = dft::config_param;
    switch (param) {
        case iparam::FORWARD_DOMAIN: return oparam::FORWARD_DOMAIN;
        case iparam::DIMENSION: return oparam::DIMENSION;
        case iparam::LENGTHS: return oparam::LENGTHS;
        case iparam::PRECISION: return oparam::PRECISION;
        case iparam::FORWARD_SCALE: return oparam::FORWARD_SCALE;
        case iparam::NUMBER_OF_TRANSFORMS: return oparam::NUMBER_OF_TRANSFORMS;
        case iparam::COMPLEX_STORAGE: return oparam::COMPLEX_STORAGE;
        case iparam::REAL_STORAGE: return oparam::REAL_STORAGE;
        case iparam::CONJUGATE_EVEN_STORAGE: return oparam::CONJUGATE_EVEN_STORAGE;
        case iparam::INPUT_STRIDES: return oparam::INPUT_STRIDES;
        case iparam::OUTPUT_STRIDES: return oparam::OUTPUT_STRIDES;
        case iparam::FWD_DISTANCE: return oparam::FWD_DISTANCE;
        case iparam::BWD_DISTANCE: return oparam::BWD_DISTANCE;
        case iparam::WORKSPACE: return oparam::WORKSPACE;
        case iparam::ORDERING: return oparam::ORDERING;
        case iparam::TRANSPOSE: return oparam::TRANSPOSE;
        case iparam::PACKED_FORMAT: return oparam::PACKED_FORMAT;
        case iparam::COMMIT_STATUS: return oparam::COMMIT_STATUS;
        default:
            throw mkl::invalid_argument("dft", "MKLcPU descriptor set_value()",
                                        "Invalid config param.");
            return static_cast<oparam>(0);
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
    else {
        throw mkl::unimplemented("dft", "MKLcPU descriptor set_value()",
                                 "MKLcPU only supports complex-complex for complex storage.");
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
        throw mkl::invalid_argument("dft", "MKLcPU descriptor set_value()",
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
        throw mkl::invalid_argument("dft", "MKLcPU descriptor set_value()",
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
        throw mkl::invalid_argument("dft", "MKLcPU descriptor set_value()",
                                    "Invalid config value for packed format.");
        return 0;
    }
}
} // namespace detail
} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_DFT_SRC_MKLCPU_HELPERS_HPP_


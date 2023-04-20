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

#ifndef _ONEMKL_DETAIL_DESCRIPTOR_CONFIG_HELPER_HPP_
#define _ONEMKL_DETAIL_DESCRIPTOR_CONFIG_HELPER_HPP_

#include <cstdint>
#include <type_traits>

#include "oneapi/mkl/dft/descriptor.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

/// Helper to get real type from precision.
template <precision Prec>
struct real_helper;

template <>
struct real_helper<precision::SINGLE> {
    using type = float;
};

template <>
struct real_helper<precision::DOUBLE> {
    using type = double;
};

template <precision Prec>
using real_helper_t = typename real_helper<Prec>::type;

/** Helper to get the argument type for a config param.
 * @tparam RealT The real type for the DFT.
 * @tparam Param The config param to get the arg for.
**/
template <typename RealT, config_param Param>
struct param_type_helper;

template <typename RealT, config_param Param>
using param_type_helper_t = typename param_type_helper<RealT, Param>::type;

#define PARAM_TYPE_HELPER(param, param_type) \
    template <typename RealT>                \
    struct param_type_helper<RealT, param> { \
        using type = param_type;             \
    };
PARAM_TYPE_HELPER(config_param::FORWARD_DOMAIN, domain)
PARAM_TYPE_HELPER(config_param::DIMENSION, std::int64_t)
PARAM_TYPE_HELPER(config_param::LENGTHS, std::int64_t*)
PARAM_TYPE_HELPER(config_param::PRECISION, precision)
PARAM_TYPE_HELPER(config_param::FORWARD_SCALE, RealT)
PARAM_TYPE_HELPER(config_param::BACKWARD_SCALE, RealT)
PARAM_TYPE_HELPER(config_param::NUMBER_OF_TRANSFORMS, std::int64_t)
PARAM_TYPE_HELPER(config_param::COMPLEX_STORAGE, config_value)
PARAM_TYPE_HELPER(config_param::REAL_STORAGE, config_value)
PARAM_TYPE_HELPER(config_param::CONJUGATE_EVEN_STORAGE, config_value)
PARAM_TYPE_HELPER(config_param::PLACEMENT, config_value)
PARAM_TYPE_HELPER(config_param::INPUT_STRIDES, std::int64_t*)
PARAM_TYPE_HELPER(config_param::OUTPUT_STRIDES, std::int64_t*)
PARAM_TYPE_HELPER(config_param::FWD_DISTANCE, std::int64_t)
PARAM_TYPE_HELPER(config_param::BWD_DISTANCE, std::int64_t)
PARAM_TYPE_HELPER(config_param::WORKSPACE, config_value)
PARAM_TYPE_HELPER(config_param::ORDERING, config_value)
PARAM_TYPE_HELPER(config_param::TRANSPOSE, bool)
PARAM_TYPE_HELPER(config_param::PACKED_FORMAT, config_value)
PARAM_TYPE_HELPER(config_param::COMMIT_STATUS, config_value)
#undef PARAM_TYPE_HELPER

/** Set a value in dft_values, throwing on invalid args.
 * @tparam Param The config param to set.
 * @tparam prec The precision of the DFT.
 * @tparam dom The domain of the DFT.
 * @param vals The struct to update the value in.
 * @param set_val The value to set Param to.
**/
template <config_param Param, precision prec, domain dom>
void set_value(dft_values<prec, dom>& vals,
               param_type_helper_t<real_helper_t<prec>, Param>&& set_val) {
    if constexpr (Param == config_param::LENGTHS) {
        if (set_val == nullptr) {
            throw mkl::invalid_argument("DFT", "set_value", "Given nullptr.");
        }
        for (std::size_t i{ 0 }; i < vals.dimensions.size(); ++i) {
            if (set_val[i] <= 0) {
                throw mkl::invalid_argument("DFT", "set_value",
                                            "Invalid length value (negative or 0).");
            }
        }
        std::copy(set_val, set_val + vals.dimensions.size(), vals.dimensions.begin());
    }
    else if constexpr (Param == config_param::PRECISION) {
        throw mkl::invalid_argument("DFT", "set_value", "Read-only parameter.");
    }
    else if constexpr (Param == config_param::FORWARD_SCALE) {
        vals.fwd_scale = set_val;
    }
    else if constexpr (Param == config_param::BACKWARD_SCALE) {
        vals.bwd_scale = set_val;
    }
    else if constexpr (Param == config_param::NUMBER_OF_TRANSFORMS) {
        if (set_val <= 0) {
            throw mkl::invalid_argument("DFT", "set_value",
                                        "Number of transforms must be positive.");
        }
        vals.number_of_transforms = set_val;
    }
    else if constexpr (Param == config_param::COMPLEX_STORAGE) {
        if (set_val == config_value::COMPLEX_COMPLEX || set_val == config_value::REAL_REAL) {
            vals.complex_storage = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value",
                                        "Complex storage must be complex_complex or real_real.");
        }
    }
    else if constexpr (Param == config_param::REAL_STORAGE) {
        if (set_val == config_value::REAL_REAL) {
            vals.real_storage = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value", "Real storage must be real_real.");
        }
    }
    else if constexpr (Param == config_param::CONJUGATE_EVEN_STORAGE) {
        if (set_val == config_value::COMPLEX_COMPLEX) {
            vals.conj_even_storage = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value",
                                        "Conjugate even storage must be complex_complex.");
        }
    }
    else if constexpr (Param == config_param::PLACEMENT) {
        if (set_val == config_value::INPLACE || set_val == config_value::NOT_INPLACE) {
            vals.placement = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value",
                                        "Placement must be inplace or not inplace.");
        }
    }
    else if constexpr (Param == config_param::INPUT_STRIDES) {
        if (set_val == nullptr) {
            throw mkl::invalid_argument("DFT", "set_value", "Given nullptr.");
        }
        std::copy(set_val, set_val + vals.dimensions.size() + 1, vals.input_strides.begin());
    }
    else if constexpr (Param == config_param::OUTPUT_STRIDES) {
        if (set_val == nullptr) {
            throw mkl::invalid_argument("DFT", "set_value", "Given nullptr.");
        }
        std::copy(set_val, set_val + vals.dimensions.size() + 1, vals.output_strides.begin());
    }
    else if constexpr (Param == config_param::FWD_DISTANCE) {
        vals.fwd_dist = set_val;
    }
    else if constexpr (Param == config_param::BWD_DISTANCE) {
        vals.bwd_dist = set_val;
    }
    else if constexpr (Param == config_param::WORKSPACE) {
        if (set_val == config_value::ALLOW || set_val == config_value::AVOID) {
            vals.workspace = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value", "Workspace must be allow or avoid.");
        }
    }
    else if constexpr (Param == config_param::ORDERING) {
        if (set_val == config_value::ORDERED || set_val == config_value::BACKWARD_SCRAMBLED) {
            vals.ordering = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value",
                                        "Ordering must be ordered or backwards scrambled.");
        }
    }
    else if constexpr (Param == config_param::TRANSPOSE) {
        vals.transpose = set_val;
    }
    else if constexpr (Param == config_param::PACKED_FORMAT) {
        if (set_val == config_value::CCE_FORMAT) {
            vals.packed_format = set_val;
        }
        else {
            throw mkl::invalid_argument("DFT", "set_value", "Packed format must be CCE.");
        }
    }
}

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DETAIL_DESCRIPTOR_CONFIG_HELPER_HPP_

/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef _ONEMKL_DETAIL_TYPES_IMPL_HPP_
#define _ONEMKL_DETAIL_TYPES_IMPL_HPP_

#include <cstdint>
#include <vector>
#include <type_traits>

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

typedef long DFT_ERROR;

enum class precision { SINGLE, DOUBLE };

template <precision prec>
struct precision_t {
    using real_t = std::conditional_t<prec == precision::SINGLE, float, double>;
};

enum class domain { REAL, COMPLEX };
enum class config_param {
    FORWARD_DOMAIN,
    DIMENSION,
    LENGTHS,
    PRECISION,

    FORWARD_SCALE,
    BACKWARD_SCALE,

    NUMBER_OF_TRANSFORMS,

    COMPLEX_STORAGE,
    REAL_STORAGE,
    CONJUGATE_EVEN_STORAGE,

    PLACEMENT,

    INPUT_STRIDES,
    OUTPUT_STRIDES,

    FWD_DISTANCE,
    BWD_DISTANCE,

    WORKSPACE,
    ORDERING,
    TRANSPOSE,
    PACKED_FORMAT,
    COMMIT_STATUS
};

enum class config_value {
    // for config_param::COMMIT_STATUS
    COMMITTED,
    UNCOMMITTED,

    // for config_param::COMPLEX_STORAGE,
    //     config_param::REAL_STORAGE and
    //     config_param::CONJUGATE_EVEN_STORAGE
    COMPLEX_COMPLEX,
    REAL_COMPLEX,
    REAL_REAL,

    // for config_param::PLACEMENT
    INPLACE,
    NOT_INPLACE,

    // for config_param::ORDERING
    ORDERED,
    BACKWARD_SCRAMBLED,

    // Allow/avoid certain usages
    ALLOW,
    AVOID,
    NONE,

    // for config_param::PACKED_FORMAT for storing conjugate-even finite sequence in real containers
    CCE_FORMAT
};

template <precision prec, domain dom>
class dft_values {
private:
    using real_t = typename precision_t<prec>::real_t;

public:
    std::vector<std::int64_t> input_strides;
    std::vector<std::int64_t> output_strides;
    real_t bwd_scale;
    real_t fwd_scale;
    std::int64_t number_of_transforms;
    std::int64_t fwd_dist;
    std::int64_t bwd_dist;
    config_value placement;
    config_value complex_storage;
    config_value real_storage;
    config_value conj_even_storage;
    config_value workspace;
    config_value ordering;
    bool transpose;
    config_value packed_format;
    std::vector<std::int64_t> dimensions;
};

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DETAIL_TYPES_IMPL_HPP_

/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef _ONEMKL_DFT_TYPES_HPP_
#define _ONEMKL_DFT_TYPES_HPP_

#include "oneapi/mkl/bfloat16.hpp"
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#ifdef NDEBUG
#define logf(...)
#else
#define logf(...)                                   \
    printf("%s - (%s) : ", __FILE__, __FUNCTION__); \
    printf(__VA_ARGS__);                            \
    printf("\n");
#endif

template <typename S>
std::ostream& operator<<(std::ostream& os, const std::vector<S>& vector) {
    if (vector.empty()) return os;
    os.put('[');
    for (auto element : vector) {
        os << element << ", ";
    }
    return os << "\b\b]";
}

namespace oneapi {
namespace mkl {
namespace dft {

enum class precision { SINGLE, DOUBLE };
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

static std::unordered_map<precision, std::string> prec_map{ { precision::SINGLE, "SINGLE" },
                                                            { precision::DOUBLE, "DOUBLE" } };

static std::unordered_map<domain, std::string> dom_map{ { domain::REAL, "REAL" },
                                                        { domain::COMPLEX, "COMPLEX" } };

struct dft_values {
    std::vector<std::int64_t> input_strides;
    std::vector<std::int64_t> output_strides;
    double bwd_scale;
    double fwd_scale;
    std::int64_t number_of_transforms;
    std::int64_t fwd_dist;
    std::int64_t bwd_dist;
    config_value placement;
    config_value complex_storage;
    config_value conj_even_storage;

    bool set_input_strides = false;
    bool set_output_strides = false;
    bool set_bwd_scale = false;
    bool set_fwd_scale = false;
    bool set_number_of_transforms = false;
    bool set_fwd_dist = false;
    bool set_bwd_dist = false;
    bool set_placement = false;
    bool set_complex_storage = false;
    bool set_conj_even_storage = false;

    std::vector<std::int64_t> dimension;
    std::int64_t rank;
    domain domain;
    precision precision;
    friend auto operator<<(std::ostream& os, dft_values const& val) -> std::ostream& {
        os << "------------- oneAPI Descriptor ------------\n";
        os << "input_strides        : " << val.input_strides << "\n";
        os << "output_strides       : " << val.output_strides << "\n";
        os << "bwd_scale            : " << val.bwd_scale << "\n";
        os << "fwd_scale            : " << val.fwd_scale << "\n";
        os << "number_of_transforms : " << val.number_of_transforms << "\n";
        os << "fwd_dist             : " << val.fwd_dist << "\n";
        os << "bwd_dist             : " << val.bwd_dist << "\n";
        os << "placement            : " << (int) val.placement << "\n";
        os << "complex_storage      : " << (int) val.complex_storage << "\n";
        os << "conj_even_storage    : " << (int) val.conj_even_storage << "\n";
        os << "dimension            : " << val.dimension << "\n";
        os << "rank                 : " << val.rank << "\n";
        os << "domain               : " << dom_map[val.domain] << "\n";
        os << "precision            : " << prec_map[val.precision];
        return os;
    }
};
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_TYPES_HPP_
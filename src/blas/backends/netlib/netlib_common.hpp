/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _NETLIB_COMMON_HPP_
#define _NETLIB_COMMON_HPP_

#include <CL/sycl.hpp>
#include <complex>

#include "cblas.h"

#include "oneapi/mkl/blas/detail/netlib/onemkl_blas_netlib.hpp"
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace netlib {

typedef enum { CblasFixOffset = 101, CblasColOffset = 102, CblasRowOffset = 103 } CBLAS_OFFSET;

/**
 * Helper methods for converting between onemkl types and their BLAS
 * equivalents.
 */

inline CBLAS_TRANSPOSE convert_to_cblas_trans(transpose trans) {
    if (trans == transpose::trans)
        return CBLAS_TRANSPOSE::CblasTrans;
    else if (trans == transpose::conjtrans)
        return CBLAS_TRANSPOSE::CblasConjTrans;
    else
        return CBLAS_TRANSPOSE::CblasNoTrans;
}

inline CBLAS_UPLO convert_to_cblas_uplo(uplo is_upper) {
    return is_upper == uplo::upper ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
}

inline CBLAS_DIAG convert_to_cblas_diag(diag is_unit) {
    return is_unit == diag::unit ? CBLAS_DIAG::CblasUnit : CBLAS_DIAG::CblasNonUnit;
}

inline CBLAS_SIDE convert_to_cblas_side(side is_left) {
    return is_left == side::left ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight;
}

inline CBLAS_OFFSET convert_to_cblas_offset(offset offsetc) {
    if (offsetc == offset::fix)
        return CBLAS_OFFSET::CblasFixOffset;
    else if (offsetc == offset::column)
        return CBLAS_OFFSET::CblasColOffset;
    else
        return CBLAS_OFFSET::CblasRowOffset;
}

// host_task automatically uses run_on_host_intel if it is supported by the
//  compiler. Otherwise, it falls back to single_task.
template <typename K, typename H, typename F>
static inline auto host_task_internal(H &cgh, F f, int) -> decltype(cgh.run_on_host_intel(f)) {
    return cgh.run_on_host_intel(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal(H &cgh, F f, long) {
    cgh.template single_task<K>(f);
}

template <typename K, typename H, typename F>
static inline void host_task(H &cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

} // namespace netlib
} // namespace blas
} // namespace mkl
} // namespace oneapi

#endif //_NETLIB_COMMON_HPP_

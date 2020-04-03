/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef ONEMKL_BLAS_HELPER_HPP
#define ONEMKL_BLAS_HELPER_HPP

#include "cblas.h"

#include "onemkl/types.hpp"

/**
 * Helper methods for converting between onemkl types and their BLAS
 * equivalents.
 */

inline CBLAS_TRANSPOSE convert_to_cblas_trans(onemkl::transpose trans) {
    if (trans == onemkl::transpose::trans)
        return CBLAS_TRANSPOSE::CblasTrans;
    else if (trans == onemkl::transpose::conjtrans)
        return CBLAS_TRANSPOSE::CblasConjTrans;
    else
        return CBLAS_TRANSPOSE::CblasNoTrans;
}

inline CBLAS_UPLO convert_to_cblas_uplo(onemkl::uplo is_upper) {
    return is_upper == onemkl::uplo::upper ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
}

inline CBLAS_DIAG convert_to_cblas_diag(onemkl::diag is_unit) {
    return is_unit == onemkl::diag::unit ? CBLAS_DIAG::CblasUnit : CBLAS_DIAG::CblasNonUnit;
}

inline CBLAS_SIDE convert_to_cblas_side(onemkl::side is_left) {
    return is_left == onemkl::side::left ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight;
}

#endif // ONEMKL_BLAS_HELPER_HPP

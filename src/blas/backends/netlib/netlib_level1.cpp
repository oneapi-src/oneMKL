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

#include <CL/sycl.hpp>

#include "netlib_common.hpp"
#include "oneapi/mkl/blas/detail/netlib/onemkl_blas_netlib.hpp"

extern "C" {
void csrot_(const int *N, void *X, const int *incX, void *Y, const int *incY, const float *c,
            const float *s);

void zdrot_(const int *N, void *X, const int *incX, void *Y, const int *incY, const double *c,
            const double *s);

void crotg_(void *a, void *b, const float *c, void *s);

void zrotg_(void *a, void *b, const double *c, void *s);
}

inline float abs_val(float val) {
    return std::abs(val);
}

inline double abs_val(double val) {
    return std::abs(val);
}

inline float abs_val(std::complex<float> val) {
    return std::abs(val.real()) + std::abs(val.imag());
}

inline double abs_val(std::complex<double> val) {
    return std::abs(val.real()) + std::abs(val.imag());
}

int cblas_isamin(int n, const float *x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < n; ++logical_i) {
        int i             = logical_i * std::abs(incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

int cblas_idamin(int n, const double *x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < n; ++logical_i) {
        int i             = logical_i * std::abs(incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

int cblas_icamin(int n, const std::complex<float> *x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < n; ++logical_i) {
        int i             = logical_i * std::abs(incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

int cblas_izamin(int n, const std::complex<double> *x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < n; ++logical_i) {
        int i             = logical_i * std::abs(incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

namespace oneapi {
namespace mkl {
namespace netlib {
namespace column_major {

#include "netlib_level1.cxx"

} // namespace column_major
namespace row_major {

#include "netlib_level1.cxx"

} // namespace row_major
} // namespace netlib
} // namespace mkl
} // namespace oneapi

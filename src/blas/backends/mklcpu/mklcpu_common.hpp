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

#ifndef _MKLCPU_COMMON_HPP_
#define _MKLCPU_COMMON_HPP_

#define MKL_Complex8  std::complex<float>
#define MKL_Complex16 std::complex<double>

#include <CL/sycl.hpp>
#include <complex>

#include "mkl_blas.h"
#include "mkl_cblas.h"

#include "oneapi/mkl/blas/detail/mklcpu/onemkl_blas_mklcpu.hpp"
#include "oneapi/mkl/types.hpp"
#include "runtime_support_helper.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace mklcpu {

// host_task automatically uses run_on_host_intel if it is supported by the
//  compiler. Otherwise, it falls back to single_task.
template <typename K, typename H, typename F>
static inline auto host_task_internal(H &cgh, F f, int) -> decltype(cgh.host_task(f)) {
    return cgh.host_task(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal(H &cgh, F f, long) {
#ifndef __SYCL_DEVICE_ONLY__
    cgh.template single_task<K>(f);
#endif
}

template <typename K, typename H, typename F>
static inline void host_task(H &cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

// Conversion functions to traditional Fortran characters.
inline const char *fortran_char(transpose t) {
    if (t == transpose::nontrans)
        return "N";
    if (t == transpose::trans)
        return "T";
    if (t == transpose::conjtrans)
        return "C";
    return "N";
}

inline const char *fortran_char(offset t) {
    if (t == offset::fix)
        return "F";
    if (t == offset::row)
        return "R";
    if (t == offset::column)
        return "C";
    return "N";
}

inline const char *fortran_char(uplo u) {
    if (u == uplo::upper)
        return "U";
    if (u == uplo::lower)
        return "L";
    return "U";
}

inline const char *fortran_char(diag d) {
    if (d == diag::nonunit)
        return "N";
    if (d == diag::unit)
        return "U";
    return "N";
}

inline const char *fortran_char(side s) {
    if (s == side::left)
        return "L";
    if (s == side::right)
        return "R";
    return "L";
}

// Conversion functions to CBLAS enums.
inline CBLAS_TRANSPOSE cblas_convert(transpose t) {
    if (t == transpose::nontrans)
        return CblasNoTrans;
    if (t == transpose::trans)
        return CblasTrans;
    if (t == transpose::conjtrans)
        return CblasConjTrans;
    return CblasNoTrans;
}

inline CBLAS_UPLO cblas_convert(uplo u) {
    if (u == uplo::upper)
        return CblasUpper;
    if (u == uplo::lower)
        return CblasLower;
    return CblasUpper;
}

inline CBLAS_DIAG cblas_convert(diag d) {
    if (d == diag::nonunit)
        return CblasNonUnit;
    if (d == diag::unit)
        return CblasUnit;
    return CblasNonUnit;
}

inline CBLAS_SIDE cblas_convert(side s) {
    if (s == side::left)
        return CblasLeft;
    if (s == side::right)
        return CblasRight;
    return CblasLeft;
}

inline CBLAS_OFFSET cblas_convert(oneapi::mkl::offset o) {
    if (o == oneapi::mkl::offset::fix)
        return CblasFixOffset;
    if (o == oneapi::mkl::offset::column)
        return CblasColOffset;
    return CblasRowOffset;
}

template <typename transpose_type>
inline bool isNonTranspose(transpose_type trans) {
    return true;
}

template <>
inline bool isNonTranspose(transpose trans) {
    return trans == transpose::nontrans;
}

template <typename offset_type>
inline offset offset_convert(offset_type off_kind) {
    return offset::F;
}

template <>
inline offset offset_convert(CBLAS_OFFSET off_kind) {
    if (off_kind == CblasFixOffset)
        return offset::F;
    if (off_kind == CblasRowOffset)
        return offset::R;
    return offset::C;
}

template <typename T_src, typename T_dest, typename transpose_type>
static inline void copy_mat(T_src &src, MKL_LAYOUT layout, transpose_type trans, int64_t row,
                            int64_t col, int64_t ld, T_dest off, T_dest *&dest) {
    int64_t i, j, Iend, Jend;
    if (layout == MKL_COL_MAJOR) {
        Jend = isNonTranspose(trans) ? col : row;
        Iend = isNonTranspose(trans) ? row : col;
    }
    else {
        Jend = isNonTranspose(trans) ? row : col;
        Iend = isNonTranspose(trans) ? col : row;
    }
    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            dest[i + ld * j] = (T_dest)src[i + ld * j] - off;
        }
    }
}

template <typename T_src, typename T_dest, typename T_off, typename offset_type>
static inline void copy_mat(T_src *src, MKL_LAYOUT layout, int64_t row, int64_t col, int64_t ld,
                            offset_type off_kind, T_off off, T_dest *dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    int64_t i, j;
    T_data tmp;

    int64_t Jend = (layout == MKL_COL_MAJOR) ? col : row;
    int64_t Iend = (layout == MKL_COL_MAJOR) ? row : col;

    if (offset_convert(off_kind) == offset::F) {
        tmp = off[0];
        for (j = 0; j < Jend; j++) {
            for (i = 0; i < Iend; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (((offset_convert(off_kind) == offset::C) && (layout == MKL_COL_MAJOR)) ||
             ((offset_convert(off_kind) == offset::R) && (layout == MKL_ROW_MAJOR))) {
        for (j = 0; j < Jend; j++) {
            for (i = 0; i < Iend; i++) {
                tmp = off[i];
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else {
        for (j = 0; j < Jend; j++) {
            tmp = off[j];
            for (i = 0; i < Iend; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
}

inline offset column_to_row(offset o) {
    return (o == offset::C) ? offset::R : (o == offset::R) ? offset::C : offset::F;
}

static inline bool is_int8(int v) {
    return (v >= -128) && (v < 128);
}

} // namespace mklcpu
} // namespace blas
} // namespace mkl
} // namespace oneapi

#endif //_MKLCPU_COMMON_HPP_

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

#ifndef _SPARSE_REFERENCE_HPP__
#define _SPARSE_REFERENCE_HPP__

#include <stdexcept>
#include <string>
#include <tuple>

#include "oneapi/mkl.hpp"

#include "test_common.hpp"

template <typename T>
inline T conjugate(T) {
    static_assert(false, "Unsupported type");
}
template <>
inline float conjugate(float t) {
    return t;
}
template <>
inline double conjugate(double t) {
    return t;
}
template <>
inline std::complex<float> conjugate(std::complex<float> t) {
    return std::conj(t);
}
template <>
inline std::complex<double> conjugate(std::complex<double> t) {
    return std::conj(t);
}

template <typename T>
inline T opVal(const T t, const bool isConj) {
    return (isConj ? conjugate(t) : t);
};

template <typename fpType, typename intType, typename accIntType, typename accFpType>
void do_csr_transpose(const oneapi::mkl::transpose opA, intType *ia_t, intType *ja_t, fpType *a_t,
                      intType a_nrows, intType a_ncols, intType a_ind, accIntType &ia,
                      accIntType &ja, accFpType &a, const bool structOnlyFlag = false) {
    const bool isConj = (opA == oneapi::mkl::transpose::conjtrans);

    // initialize ia_t to zero
    for (intType i = 0; i < a_ncols + 1; ++i) {
        ia_t[i] = 0;
    }

    // fill ia_t with counts of columns
    for (intType i = 0; i < a_nrows; ++i) {
        const intType st = ia[i] - a_ind;
        const intType en = ia[i + 1] - a_ind;
        for (intType j = st; j < en; ++j) {
            const intType col = ja[j] - a_ind;
            ia_t[col + 1]++;
        }
    }
    // prefix sum to get official ia_t counts
    ia_t[0] = a_ind;
    for (intType i = 0; i < a_ncols; ++i) {
        ia_t[i + 1] += ia_t[i];
    }

    // second pass through data to fill transpose structure
    for (intType i = 0; i < a_nrows; ++i) {
        const intType st = ia[i] - a_ind;
        const intType en = ia[i + 1] - a_ind;
        for (intType j = st; j < en; ++j) {
            const intType col = ja[j] - a_ind;
            const intType j_in_a_t = ia_t[col] - a_ind;
            ia_t[col]++;
            ja_t[j_in_a_t] = i + a_ind;
            if (!structOnlyFlag) {
                const fpType val = a[j];
                a_t[j_in_a_t] = opVal(val, isConj);
            }
        }
    }

    // adjust ia_t back to original state after filling structure
    for (intType i = a_ncols; i > 0; --i) {
        ia_t[i] = ia_t[i - 1];
    }
    ia_t[0] = a_ind;
}

// Transpose the given sparse matrix if needed
template <typename fpType, typename intType>
auto sparse_transpose_if_needed(const intType *ia, const intType *ja, const fpType *a,
                                intType a_nrows, intType a_ncols, std::size_t nnz, intType a_ind,
                                oneapi::mkl::transpose transpose_val) {
    std::vector<intType> iopa;
    std::vector<intType> jopa;
    std::vector<fpType> opa;
    if (transpose_val == oneapi::mkl::transpose::nontrans) {
        iopa.assign(ia, ia + a_nrows + 1);
        jopa.assign(ja, ja + nnz);
        opa.assign(a, a + nnz);
    }
    else if (transpose_val == oneapi::mkl::transpose::trans ||
             transpose_val == oneapi::mkl::transpose::conjtrans) {
        iopa.resize(static_cast<std::size_t>(a_ncols + 1));
        jopa.resize(nnz);
        opa.resize(nnz);
        do_csr_transpose(transpose_val, iopa.data(), jopa.data(), opa.data(), a_nrows, a_ncols,
                         a_ind, ia, ja, a);
    }
    else {
        throw std::runtime_error("unsupported transpose_val=" +
                                 std::to_string(static_cast<char>(transpose_val)));
    }
    return std::make_tuple(iopa, jopa, opa);
}

template <typename fpType>
auto dense_transpose_if_needed(const fpType *x, std::size_t outer_size, std::size_t inner_size,
                               std::size_t ld, oneapi::mkl::transpose transpose_val) {
    std::vector<fpType> opx;
    if (transpose_val == oneapi::mkl::transpose::nontrans) {
        opx.assign(x, x + outer_size * ld);
    }
    else {
        opx.resize(outer_size * ld);
        for (std::size_t i = 0; i < outer_size; ++i) {
            for (std::size_t j = 0; j < inner_size; ++j) {
                opx[i + j * ld] = x[i * ld + j];
            }
        }
    }
    return opx;
}

/// Return the dense matrix A in row major layout.
/// Diagonal values are overwritten with 1s if diag_val is unit.
template <typename fpType, typename intType>
std::vector<fpType> sparse_to_dense(const intType *ia, const intType *ja, const fpType *a,
                                    std::size_t a_nrows, std::size_t a_ncols, intType a_ind,
                                    oneapi::mkl::transpose transpose_val,
                                    oneapi::mkl::diag diag_val) {
    std::vector<fpType> dense_a(a_nrows * a_ncols, fpType(0));
    for (std::size_t row = 0; row < a_nrows; row++) {
        for (intType i = ia[row] - a_ind; i < ia[row + 1] - a_ind; i++) {
            std::size_t iu = static_cast<std::size_t>(i);
            std::size_t col = static_cast<std::size_t>(ja[iu] - a_ind);
            std::size_t dense_a_idx = transpose_val != oneapi::mkl::transpose::nontrans
                                          ? col * a_nrows + row
                                          : row * a_ncols + col;
            fpType val = a[iu];
            if constexpr (complex_info<fpType>::is_complex) {
                if (transpose_val == oneapi::mkl::transpose::conjtrans) {
                    val = std::conj(val);
                }
            }
            dense_a[dense_a_idx] = val;
        }
    }
    if (diag_val == oneapi::mkl::diag::unit) {
        for (std::size_t i = 0; i < a_nrows; ++i) {
            dense_a[i * a_ncols + i] = set_fp_value<fpType>()(1.f, 0.f);
        }
    }
    return dense_a;
}

template <typename fpType, typename intType>
void prepare_reference_gemv_data(const intType *ia, const intType *ja, const fpType *a,
                                 intType a_nrows, intType a_ncols, intType a_nnz, intType a_ind,
                                 oneapi::mkl::transpose opA, fpType alpha, fpType beta,
                                 const fpType *x, fpType *y_ref) {
    std::size_t opa_nrows =
        static_cast<std::size_t>((opA == oneapi::mkl::transpose::nontrans) ? a_nrows : a_ncols);
    const std::size_t nnz = static_cast<std::size_t>(a_nnz);
    auto [iopa, jopa, opa] =
        sparse_transpose_if_needed(ia, ja, a, a_nrows, a_ncols, nnz, a_ind, opA);

    //
    // do GEMV operation
    //
    //  y_ref <- alpha * op(A) * x + beta * y_ref
    //
    for (std::size_t row = 0; row < opa_nrows; row++) {
        fpType tmp = 0;
        for (intType i = iopa[row] - a_ind; i < iopa[row + 1] - a_ind; i++) {
            std::size_t iu = static_cast<std::size_t>(i);
            std::size_t x_ind = static_cast<std::size_t>(jopa[iu] - a_ind);
            tmp += opa[iu] * x[x_ind];
        }

        y_ref[row] = alpha * tmp + beta * y_ref[row];
    }
}

template <typename fpType, typename intType>
void prepare_reference_gemm_data(const intType *ia, const intType *ja, const fpType *a,
                                 intType a_nrows, intType a_ncols, intType c_ncols, intType a_nnz,
                                 intType a_ind, oneapi::mkl::layout dense_matrix_layout,
                                 oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                                 fpType alpha, fpType beta, intType ldb, intType ldc,
                                 const fpType *b, fpType *c_ref) {
    std::size_t opa_nrows =
        static_cast<std::size_t>((opA == oneapi::mkl::transpose::nontrans) ? a_nrows : a_ncols);
    std::size_t opa_ncols =
        static_cast<std::size_t>((opA == oneapi::mkl::transpose::nontrans) ? a_ncols : a_nrows);
    const std::size_t nnz = static_cast<std::size_t>(a_nnz);
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    auto [iopa, jopa, opa] =
        sparse_transpose_if_needed(ia, ja, a, a_nrows, a_ncols, nnz, a_ind, opA);

    std::size_t b_outer_size = static_cast<std::size_t>(opa_ncols);
    std::size_t b_inner_size = static_cast<std::size_t>(c_ncols);
    if (dense_matrix_layout == oneapi::mkl::layout::col_major) {
        std::swap(b_outer_size, b_inner_size);
    }
    auto opb = dense_transpose_if_needed(b, b_outer_size, b_inner_size, ldb_u, opB);

    //
    // do GEMM operation
    //
    //  C <- alpha * opA(A) * opB(B) + beta * C
    //
    if (dense_matrix_layout == oneapi::mkl::layout::row_major) {
        for (std::size_t row = 0; row < opa_nrows; row++) {
            for (std::size_t col = 0; col < static_cast<std::size_t>(c_ncols); col++) {
                fpType tmp = 0;
                for (std::size_t i = static_cast<std::size_t>(iopa[row] - a_ind);
                     i < static_cast<std::size_t>(iopa[row + 1] - a_ind); i++) {
                    tmp += opa[i] * opb[static_cast<std::size_t>(jopa[i] - a_ind) * ldb_u + col];
                }
                fpType &c = c_ref[row * ldc_u + col];
                c = alpha * tmp + beta * c;
            }
        }
    }
    else {
        for (std::size_t col = 0; col < static_cast<std::size_t>(c_ncols); col++) {
            for (std::size_t row = 0; row < opa_nrows; row++) {
                fpType tmp = 0;
                for (std::size_t i = static_cast<std::size_t>(iopa[row] - a_ind);
                     i < static_cast<std::size_t>(iopa[row + 1] - a_ind); i++) {
                    tmp += opa[i] * opb[static_cast<std::size_t>(jopa[i] - a_ind) + col * ldb_u];
                }
                fpType &c = c_ref[row + col * ldc_u];
                c = alpha * tmp + beta * c;
            }
        }
    }
}

template <typename fpType, typename intType>
void prepare_reference_trsv_data(const intType *ia, const intType *ja, const fpType *a, intType m,
                                 intType a_ind, oneapi::mkl::uplo uplo_val,
                                 oneapi::mkl::transpose opA, oneapi::mkl::diag diag_val,
                                 const fpType *x, fpType *y_ref) {
    std::size_t mu = static_cast<std::size_t>(m);
    auto dense_a = sparse_to_dense(ia, ja, a, mu, mu, a_ind, opA, diag_val);

    //
    // do TRSV operation
    //
    //  y_ref <- op(A)^-1 * x
    //
    // Compute each element of the reference one after the other starting from 0 (resp. the end) for a lower (resp. upper) triangular matrix.
    // A matrix is considered lowered if it is lower and not transposed or upper and transposed.
    const bool is_lower =
        (uplo_val == oneapi::mkl::uplo::lower) == (opA == oneapi::mkl::transpose::nontrans);
    for (std::size_t row = 0; row < mu; row++) {
        std::size_t uplo_row = is_lower ? row : (mu - 1 - row);
        fpType rhs = x[uplo_row];
        for (std::size_t col = 0; col < row; col++) {
            std::size_t uplo_col = is_lower ? col : (mu - 1 - col);
            rhs -= dense_a[uplo_row * mu + uplo_col] * y_ref[uplo_col];
        }
        y_ref[uplo_row] = rhs / dense_a[uplo_row * mu + uplo_row];
    }
}

#endif // _SPARSE_REFERENCE_HPP__

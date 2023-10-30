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

#include "oneapi/mkl.hpp"

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

template <typename fpType, typename intType>
void prepare_reference_gemv_data(const intType *ia, const intType *ja, const fpType *a,
                                 intType a_nrows, intType a_ncols, intType a_nnz, intType a_ind,
                                 oneapi::mkl::transpose opA, fpType alpha, fpType beta,
                                 const fpType *x, fpType *y_ref) {
    std::size_t opa_nrows =
        static_cast<std::size_t>((opA == oneapi::mkl::transpose::nontrans) ? a_nrows : a_ncols);
    const std::size_t nnz = static_cast<std::size_t>(a_nnz);

    // prepare op(A) locally
    std::vector<intType> iopa;
    std::vector<intType> jopa;
    std::vector<fpType> opa;
    if (opA == oneapi::mkl::transpose::nontrans) {
        iopa.assign(ia, ia + a_nrows + 1);
        jopa.assign(ja, ja + nnz);
        opa.assign(a, a + nnz);
    }
    else if (opA == oneapi::mkl::transpose::trans || opA == oneapi::mkl::transpose::conjtrans) {
        iopa.resize(opa_nrows + 1);
        jopa.resize(nnz);
        opa.resize(nnz);
        do_csr_transpose(opA, iopa.data(), jopa.data(), opa.data(), a_nrows, a_ncols, a_ind, ia, ja,
                         a);
    }
    else {
        throw std::runtime_error(
            "unsupported transpose_val (opA) in prepare_reference_gemv_data()");
    }

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

#endif // _SPARSE_REFERENCE_HPP__

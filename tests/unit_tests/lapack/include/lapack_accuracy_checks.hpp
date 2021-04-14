/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef _LAPACK_ACCURACY_CHECKS_TPP_
#define _LAPACK_ACCURACY_CHECKS_TPP_

#include <cstdio>
#include <complex>
#include <vector>

#include "lapack_common.hpp"
#include "reference_lapack_wrappers.hpp"

/* computes |A - Ref| / (|Ref| min(m,n) eps) < threshold */
template <typename fp>
bool rel_mat_err_check(int64_t m, int64_t n, const fp* A, int64_t lda, const fp* Ref, int64_t ldr,
                       float threshold = 10.0, char norm_type = '1') {
    using fp_real = typename complex_info<fp>::real_type;

    std::vector<fp> residual(m * n);
    for (int64_t col = 0; col < n; col++) {
        for (int64_t row = 0; row < m; row++) {
            residual[row + col * m] = A[row + col * lda] - Ref[row + col * ldr];
        }
    }

    /* Compute norm of residual and check if it is within tolerance threshold */
    auto norm_residual = reference::lange(norm_type, m, n, residual.data(), m);
    auto norm_Ref = reference::lange(norm_type, m, n, Ref, ldr);
    auto ulp = reference::lamch<fp_real>('P');
    auto denom = norm_Ref * std::min(m, n) * ulp;
    denom = denom > 0.0 ? denom : ulp;

    auto rel_err = norm_residual / denom;

    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "|A - Ref| / (|Ref| min(m,n) eps) = |%e| / (|%e| %d * %e) = %e\n", norm_residual,
                 norm_Ref, static_cast<int>(std::min(m, n)), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }
    return result;
}

/* computes |A - I| / (|I| n eps) < threshold */
template <typename fp>
bool rel_id_err_check(int64_t n, const fp* A, int64_t lda, float threshold = 10.0,
                      char norm_type = '1') {
    using fp_real = typename complex_info<fp>::real_type;

    std::vector<fp> residual(n * n);
    reference::lacpy('F', n, n, A, lda, residual.data(), n);
    for (int64_t diag = 0; diag < n; diag++) {
        residual[diag + diag * n] -= static_cast<fp_real>(1.0);
    }

    /* Compute norm of residual and check if it is within tolerance threshold */
    auto norm_residual = reference::lange(norm_type, n, n, residual.data(), n);
    auto ulp = reference::lamch<fp_real>('P');
    auto denom = n * n * ulp;
    denom = denom > 0.0 ? denom : ulp;
    auto rel_err = norm_residual / denom;

    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "|A - I| / (|I| n eps) = |%e| / (|%d| %d * %e) = %e\n", norm_residual,
                 static_cast<int>(n), static_cast<int>(n), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }
    return result;
}

/* computes |V - Ref| / (|Ref| eps) < threshold */
template <typename fp>
bool rel_vec_err_check(int64_t n, const fp* A, const fp* Ref, float threshold = 10.0,
                       char norm_type = '1') {
    using fp_real = typename complex_info<fp>::real_type;

    std::vector<fp> residual(n);
    for (int64_t row = 0; row < n; row++) {
        residual[row] = A[row] - Ref[row];
    }

    /* Compute norm of residual and check if it is within tolerance threshold */
    auto norm_residual = reference::lange(norm_type, n, 1, residual.data(), n);
    auto norm_Ref = reference::lange(norm_type, n, 1, Ref, n);
    auto ulp = reference::lamch<fp_real>('P');
    auto denom = norm_Ref * ulp;
    denom = denom > 0.0 ? denom : ulp;
    auto rel_err = norm_residual / denom;

    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "|V - Ref| / (|Ref| eps) = |%e| / (|%e| %e) = %e\n", norm_residual, norm_Ref, ulp,
                 rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }
    return result;
}

template <typename fp>
bool check_or_un_gbr_accuracy(oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k,
                              const fp* Q, int64_t ldq) {
    bool result = true;

    if (vect == oneapi::mkl::generate::Q) {
        int64_t rows_Q = m;
        int64_t cols_Q = (m >= k) ? n : m;

        /* | I - Q'Q | < m O(eps) */
        std::vector<fp> QQ(cols_Q * cols_Q);
        int64_t ldqq = cols_Q;
        reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, cols_Q,
                            rows_Q, 1.0, Q, ldq, 0.0, QQ.data(), ldqq);
        hermitian_to_full(oneapi::mkl::uplo::upper, cols_Q, QQ, ldqq);

        if (!rel_id_err_check(cols_Q, QQ.data(), ldqq)) {
            global::log << "Q Orthogonality check failed" << std::endl;
            result = false;
        }
    }
    else { /* vect == oneapi::mkl::generate::P */
        auto& P = Q;
        auto& ldp = ldq;
        int64_t rows_P = (k < n) ? m : n;
        int64_t cols_P = n;

        /* | I - (P')(P')' | < m O(eps) */
        std::vector<fp> PP(rows_P * rows_P);
        int64_t ldpp = rows_P;
        reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, rows_P,
                            cols_P, 1.0, P, ldp, 0.0, PP.data(), ldpp);
        hermitian_to_full(oneapi::mkl::uplo::upper, rows_P, PP, ldpp);
        if (!rel_id_err_check(rows_P, PP.data(), ldpp)) {
            global::log << "P^t Orthogonality check failed" << std::endl;
            result = false;
        }
    }
    return result;
}

template <typename fp>
bool check_or_un_gqr_accuracy(int64_t m, int64_t n, const fp* Q, int64_t ldq) {
    bool result = true;

    /* | I - Q'Q | < m O(eps) */
    std::vector<fp> QQ(n * n);
    reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, n, m, 1.0, Q,
                        ldq, 0.0, QQ.data(), n);
    hermitian_to_full(oneapi::mkl::uplo::upper, n, QQ, n);

    if (!rel_id_err_check(n, QQ.data(), n)) {
        global::log << "Orthogonality check failed" << std::endl;
        result = false;
    }
    return result;
}

template <typename fp>
bool check_or_un_gtr_accuracy(int64_t n, const fp* Q, int64_t ldq) {
    bool result = true;

    /* | I - Q'Q | < m O(eps) */
    std::vector<fp> QQ(n * n);
    reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, n, n, 1.0, Q,
                        ldq, 0.0, QQ.data(), n);
    hermitian_to_full(oneapi::mkl::uplo::upper, n, QQ, n);

    if (!rel_id_err_check(n, QQ.data(), n)) {
        global::log << "Orthogonality check failed" << std::endl;
        result = false;
    }
    return result;
}

template <typename fp>
bool check_geqrf_accuracy(const fp* A, const fp* A_initial, const fp* tau, int64_t m, int64_t n,
                          int64_t lda) {
    bool result = true;
    /* |A - Q R| < |A| O(eps) */
    std::vector<fp> R(m * n);
    int64_t ldr = m;
    reference::laset(oneapi::mkl::uplo::lower, m, n, 0.0, 0.0, R.data(), ldr);
    reference::lacpy(oneapi::mkl::uplo::upper, m, n, A, lda, R.data(), ldr);
    auto info = reference::or_un_mqr(oneapi::mkl::side::left, oneapi::mkl::transpose::nontrans, m,
                                     n, std::min(m, n), A, lda, tau, R.data(), ldr);
    if (0 != info) {
        global::log << "reference ormqr/unmqr failed with info = " << info << std::endl;
        return false;
    }
    const auto& QR = R;
    auto ldqr = ldr;
    if (!rel_mat_err_check(m, n, QR.data(), ldqr, A_initial, lda)) {
        global::log << "Factorization check failed" << std::endl;
        result = false;
    }

    /* | I - Q Q' | < n O(eps) */
    std::vector<fp> Q(m * m);
    int64_t ldq = m;
    reference::lacpy('L', m - 1, n, A + 1, lda, Q.data() + 1, ldq);
    info = reference::or_un_gqr(m, m, std::min(m, n), Q.data(), ldq, tau);
    if (0 != info) {
        global::log << "reference org/ungqr failed with info = " << info << std::endl;
        return false;
    }
    std::vector<fp> QQ(m * m);
    int64_t ldqq = m;
    reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, m, m, 1.0,
                        Q.data(), ldq, 0.0, QQ.data(), ldqq);
    hermitian_to_full(oneapi::mkl::uplo::upper, m, QQ, m);
    if (!rel_id_err_check(m, QQ.data(), m)) {
        global::log << "Orthogonality check failed" << std::endl;
        result = false;
    }

    return result;
}

template <typename fp>
bool check_getrf_accuracy(int64_t m, int64_t n, const fp* A, int64_t lda, const int64_t* ipiv,
                          const fp* A_initial) {
    using fp_real = typename complex_info<fp>::real_type;

    std::vector<fp> residual(m * n);

    /* Compute P L U */
    reference::laset('A', m, n, 0.0, 0.0, residual.data(), m);
    if (m < n) {
        reference::lacpy(oneapi::mkl::uplo::upper, m, n, A, lda, residual.data(), m);
        reference::trmm(oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                        oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, m, n, 1.0, A,
                        lda, residual.data(), m);
    }
    else {
        reference::lacpy(oneapi::mkl::uplo::lower, m, n, A, lda, residual.data(), m);
        for (int64_t diag = 0; diag < n; diag++)
            residual[diag + diag * m] = 1.0;
        reference::trmm(oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                        oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, m, n, 1.0, A,
                        lda, residual.data(), m);
    }
    reference::laswp(n, residual.data(), m, 1, std::min(m, n), ipiv, -1);

    /* Compute | L U - A | / ( |A| min(m,n) ulp ) */
    for (int64_t col = 0; col < n; col++) {
        for (int64_t row = 0; row < m; row++) {
            residual[row + col * m] -= A_initial[row + col * lda];
        }
    }
    auto norm_residual = reference::lange('1', m, n, residual.data(), m);
    auto norm_A = reference::lange('1', m, n, A_initial, lda);
    auto ulp = reference::lamch<fp_real>('P');
    auto denom = norm_A * std::min(m, n) * ulp;
    denom = denom > 0.0 ? denom : ulp;
    auto rel_err = norm_residual / denom;

    fp_real threshold = 30.0;
    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "| L * U - A | / ( |A| * min(m,n) * ulp ) = |%e| / (|%e| %d * %e) = %e",
                 norm_residual, norm_A, static_cast<int>(std::min(m, n)), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }

    return result;
}

template <typename fp>
bool check_potrf_accuracy(const fp* init, const fp* sol, oneapi::mkl::uplo uplo, int64_t n,
                          int64_t lda) {
    using fp_real = typename complex_info<fp>::real_type;

    std::vector<fp> ref(init, init + lda * n);
    reference::potrf(uplo, n, ref.data(), lda);

    fp_real eps = reference::lamch<fp_real>('e');
    fp_real error, max_error = 0;
    bool lower =
        (uplo == oneapi::mkl::uplo::
                     upper); // lower for row-major (which is this source) is upper for column major
    bool result = true;
    // Check solution values are inside allowed error bounds derived in:
    //   J. W. Demmel, On floating point errors in Cholesky, LAPACK Working Note 14 CS-89–87,
    //   Department of Computer Science, University of Tennessee, Knoxville, TN, USA, 1989.
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < i + 1; j++) {
            fp exact = lower ? ref[i * lda + j] : ref[j * lda + i];
            fp solve = lower ? sol[i * lda + j] : sol[j * lda + i];
            error = std::abs(solve - exact);
            if (error > ((n + 1) * eps / (1 - (n + 1) * eps)) *
                            std::sqrt(std::abs(init[i * lda + i] * init[j * lda + j]))) {
                result = false;
            }
            if (error > max_error)
                max_error = error;
        }
    }
    if (!result)
        global::log << "Tolerance exceded, max_error = " << max_error << std::endl;

    return result;
}

template <typename fp>
bool check_getri_accuracy(int64_t n, fp* A, int64_t lda, int64_t* ipiv, fp* A_initial) {
    using fp_real = typename complex_info<fp>::real_type;

    // Norms of original matrix A matrix and inv(A) for error analysis
    fp_real norm_A = reference::lange('1', n, n, A_initial, lda);
    fp_real norm_invA = reference::lange('1', n, n, A, lda);
    fp_real ulp = reference::lamch<fp_real>('P');
    std::vector<fp> residual(n * n + n);
    fp_real threshold = 30.0;

    /* denom = ( |A| * |inv(A)| * n * ulp )  */
    fp_real denom = n * ulp * norm_A * norm_invA;
    denom = denom > 0.0 ? denom : ulp;

    /* Compute | I - inv(A)*A |. Store in residual array */
    reference::laset('A', n, n, 0.0, 1.0, residual.data(), n);
    reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n,
                    -1.0, A, lda, A_initial, lda, 1.0, residual.data(), n);

    /* | I - inv(A)*A | / ( |A| * |inv(A)| * n * ulp ) */
    fp_real norm_residual = reference::lange('1', n, n, residual.data(), n);
    auto rel_err = norm_residual / denom;
    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "| I - inv(A) A | / ( |A| |inv(A)| n ulp ) = |%e| / ( |%e| |%e| %d * %e ) = %e",
                 norm_residual, norm_A, norm_invA, static_cast<int>(n), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }

    /* Compute | I - A*inv(A) |. Store in residual */
    reference::laset('A', n, n, 0.0, 1.0, residual.data(), n);
    reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n,
                    -1.0, A_initial, lda, A, lda, 1.0, residual.data(), n);

    /* | I - A*inv(A) | / ( |A| * |inv(A)| * n * ulp ) */
    norm_residual = reference::lange('1', n, n, residual.data(), n);
    rel_err = norm_residual / denom;
    result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "| I - inv(A) A | / ( |A| |inv(A)| n ulp ) = |%e | / ( |%e| |%e| %d * %e) = %e",
                 norm_residual, norm_A, norm_invA, static_cast<int>(n), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }

    return result;
}

template <typename fp>
bool check_getrs_accuracy(oneapi::mkl::transpose transa, int64_t n, int64_t nrhs, fp* A,
                          int64_t lda, int64_t* ipiv, fp* B, int64_t ldb, fp* A_initial,
                          fp* B_initial) {
    using fp_real = typename complex_info<fp>::real_type;

    // Compute A*X - B. Store result in B_initial
    reference::gemm(transa, oneapi::mkl::transpose::nontrans, n, nrhs, n, -1.0, A_initial, lda, B,
                    ldb, 1.0, B_initial, ldb);

    // Compute norm residual |A*X - B|
    fp_real norm_residual = reference::lange('1', n, nrhs, B_initial, ldb);

    // Norms of original matrix A matrix and solution matrix B for error analysis
    fp_real norm_A = reference::lange('1', n, n, A_initial, lda);
    fp_real norm_B = reference::lange('1', n, nrhs, B, ldb);
    fp_real ulp = reference::lamch<fp_real>('P');
    fp_real denom = n * ulp * norm_A * norm_B;
    denom = denom > 0.0 ? denom : ulp;
    auto rel_err = norm_residual / denom;

    fp_real threshold = 30.0;
    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "| AX - B | / ( |A| |X| n ulp ) = |%e| / ( |%e| |%e| %d * %e ) = %e",
                 norm_residual, norm_A, norm_B, static_cast<int>(n), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }

    return result;
}

template <typename fp>
bool check_potrs_accuracy(oneapi::mkl::uplo uplo, int64_t n, int64_t nrhs, fp* A, int64_t lda,
                          fp* B, int64_t ldb, fp* A_initial, fp* B_initial) {
    using fp_real = typename complex_info<fp>::real_type;

    hermitian_to_full(uplo, n, A_initial, lda);
    // Compute A*X - B. Store result in B_initial
    reference::gemm(oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, nrhs, n,
                    -1.0, A_initial, lda, B, ldb, 1.0, B_initial, ldb);

    // Compute norm residual |A*X - B|
    fp_real norm_residual = reference::lange('1', n, nrhs, B_initial, ldb);

    // Norms of original matrix A matrix and solution matrix B for error analysis
    fp_real norm_A = reference::lange('1', n, n, A_initial, lda);
    fp_real norm_B = reference::lange('1', n, nrhs, B, ldb);
    fp_real ulp = reference::lamch<fp_real>('P');
    fp_real denom = n * ulp * norm_A * norm_B;
    denom = denom > 0.0 ? denom : ulp;
    auto rel_err = norm_residual / denom;

    fp_real threshold = 30.0;
    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "| AX - B | / ( |A| |X| n ulp ) = |%e| / ( |%e| |%e| %d * %e ) = %e",
                 norm_residual, norm_A, norm_B, static_cast<int>(n), ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
    }

    return result;
}

template <typename fp>
bool check_gerqf_accuracy(fp* A, fp* A_initial, fp* tau, int64_t m, int64_t n, int64_t lda) {
    bool result = true;

    /* |A - R Q| < |A| O(eps) */
    if (m >= n) {
        std::vector<fp> R(m * n);
        int64_t ldr = m;
        reference::lacpy('A', m, n, A, lda, R.data(), ldr);
        reference::laset(oneapi::mkl::uplo::lower, n - 1, n - 1, 0.0, 0.0,
                         R.data() + ((m - n + 1) + 0 * ldr), ldr);

        std::vector<fp> Q(lda * n);
        int64_t ldq = n;
        reference::lacpy('A', n, n, A + ((m - n) + 0 * lda), lda, Q.data(), ldq);

        auto info = reference::or_un_mrq(oneapi::mkl::side::right, oneapi::mkl::transpose::nontrans,
                                         m, n, std::min(m, n), Q.data(), ldq, tau, R.data(), ldr);
        /* auto info = reference::or_un_mrq(oneapi::mkl::side::right, oneapi::mkl::transpose::nontrans, m, n, std::min(m,n), A+std::max<int64_t>(0,(m-n+0*lda)), lda, tau, R.data(), ldr); */
        if (0 != info) {
            global::log << "reference ormqr/unmqr failed with info = " << info << std::endl;
            return false;
        }
        if (!rel_mat_err_check(m, n, R.data(), ldr, A_initial, lda)) {
            global::log << "Factorization check failed" << std::endl;
            result = false;
        }
    }
    else {
        std::vector<fp> R(m * n);
        int64_t ldr = m;
        reference::laset(oneapi::mkl::uplo::lower, m, m, 0.0, 0.0, R.data(), ldr);
        reference::lacpy(oneapi::mkl::uplo::upper, m, m, A + (0 + (n - m) * lda), lda,
                         R.data() + (0 + (n - m) * ldr), ldr);

        std::vector<fp> Q(n * n);
        int64_t ldq = n;
        reference::lacpy('A', m, n, A, lda, Q.data() + (n - m + 0 * ldq), ldq);

        std::vector<fp> tau2(n);
        for (int64_t i = 0; i < std::min(m, n); i++)
            tau2[n - m + i] = tau[i];
        auto info = reference::or_un_mrq(oneapi::mkl::side::right, oneapi::mkl::transpose::nontrans,
                                         m, n, n, Q.data(), ldq, tau2.data(), R.data(), ldr);
        if (0 != info) {
            global::log << "reference ormqr/unmqr failed with info = " << info << std::endl;
            return false;
        }
        if (!rel_mat_err_check(m, n, R.data(), ldr, A_initial, lda)) {
            global::log << "Factorization check failed" << std::endl;
            result = false;
        }
    }

    /* | I - Q Q' | < n O(eps) */
    std::vector<fp> Q(std::min(m, n) * n);
    int64_t ldq = std::min(m, n);
    if (m <= n)
        reference::lacpy('A', m, n, A, lda, Q.data(), ldq);
    else
        reference::lacpy('A', n, n, A + ((m - n) + 0 * lda), lda, Q.data(), ldq);
    auto info = reference::or_un_grq(std::min(m, n), n, std::min(m, n), Q.data(), ldq, tau);
    if (0 != info) {
        global::log << "reference orgqr/ungqr failed with info = " << info << std::endl;
        return false;
    }

    std::vector<fp> I(std::min(m, n) * std::min(m, n));
    int64_t ldi = std::min(m, n);
    reference::sy_he_rk(oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, std::min(m, n),
                        n, 1.0, Q.data(), ldq, 0.0, I.data(), ldi);
    hermitian_to_full(oneapi::mkl::uplo::upper, std::min(m, n), I, ldi);

    if (!rel_id_err_check(std::min(m, n), I.data(), ldi)) {
        global::log << "Orthogonality check failed" << std::endl;
        result = false;
    }

    return result;
}

template <typename fp>
bool check_trtrs_accuracy(oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                          oneapi::mkl::diag diag, int64_t n, int64_t nrhs, fp* A, int64_t lda,
                          const fp* B, int64_t ldb, const fp* B_initial) {
    using fp_real = typename complex_info<fp>::real_type;
    fp_real threshold = 10.0;

    /* |A x - b| = |A (x-x_0)| < |A| |x-x0| < |A| |x| cond(A) O(eps) */
    if (diag == oneapi::mkl::diag::unit)
        for (int64_t d = 0; d < n; d++)
            A[d + d * lda] = 1.0;
    if (uplo == oneapi::mkl::uplo::upper)
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = col + 1; row < n; row++)
                A[row + col * lda] = 0.0;
    else
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = 0; row < col; row++)
                A[row + col * lda] = 0.0;

    auto norm_A = reference::lange('I', n, n, A, lda);
    auto norm_x = reference::lange('I', n, nrhs, B, ldb);

    fp_real cond_A;
    if (diag == oneapi::mkl::diag::unit)
        cond_A = 1.0;
    else {
        fp_real min = std::abs(A[0]);
        fp_real max = std::abs(A[0]);
        for (int64_t d = 0; d < n; d++) {
            auto val = std::abs(A[d + d * lda]);
            min = (val < min) ? val : min;
            max = (val > max) ? val : max;
        }
        cond_A = max / min;
    }

    auto ulp = reference::lamch<fp_real>('P');
    auto denom = norm_A * norm_x * cond_A * ulp;
    denom = denom > 0.0 ? denom : ulp;

    std::vector<fp> residual(n * nrhs);
    int64_t ldr = n;
    reference::gemm(trans, oneapi::mkl::transpose::nontrans, n, nrhs, n, 1.0, A, lda, B, ldb, 0.0,
                    residual.data(), ldr);
    for (int64_t col = 0; col < nrhs; col++)
        for (int64_t row = 0; row < n; row++)
            residual[row + col * ldr] -= B_initial[row + col * ldb];

    auto norm_residual = reference::lange('I', n, nrhs, residual.data(), ldr);
    auto rel_err = norm_residual / denom;

    bool result = rel_err < threshold;
    if (!result) {
        snprintf(global::buffer.data(), global::buffer.size(),
                 "|Ax - b| / (|A| |x| cond(A) eps) = |%e| / (|%e| |%e| %e * %e) = %e\n",
                 norm_residual, norm_A, norm_x, cond_A, ulp, rel_err);
        global::log << global::buffer.data();
        global::log << "threshold = " << threshold << std::endl;
        global::log << "Solve check failed" << std::endl;
    }

    return result;
}

#endif // _LAPACK_ACCURACY_CHECKS_TPP_

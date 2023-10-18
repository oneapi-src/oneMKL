/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
*
* (*Licensed under the Apache License, Version 2.0 )(the "License");
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

#ifndef _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_
#define _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_

#include "oneapi/mkl/sparse_blas/types.hpp"
#include "sparse_blas/macros.hpp"

#define DEFINE_SET_CSR_DATA(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                        \
    void (*set_csr_data_buffer##FP_SUFFIX##INT_SUFFIX)(                                      \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t handle, INT_TYPE num_rows, \
        INT_TYPE num_cols, INT_TYPE nnz, oneapi::mkl::index_base index,                      \
        sycl::buffer<INT_TYPE, 1> & row_ptr, sycl::buffer<INT_TYPE, 1> & col_ind,            \
        sycl::buffer<FP_TYPE, 1> & val);                                                     \
    sycl::event (*set_csr_data_usm##FP_SUFFIX##INT_SUFFIX)(                                  \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t handle, INT_TYPE num_rows, \
        INT_TYPE num_cols, INT_TYPE nnz, oneapi::mkl::index_base index, INT_TYPE * row_ptr,  \
        INT_TYPE * col_ind, FP_TYPE * val, const std::vector<sycl::event> &dependencies)

#define DEFINE_GEMV(FP_TYPE, FP_SUFFIX)                                                      \
    void (*gemv_buffer##FP_SUFFIX)(                                                          \
        sycl::queue & queue, oneapi::mkl::transpose transpose_val, const FP_TYPE alpha,      \
        oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x,          \
        const FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &y);                                    \
    sycl::event (*gemv_usm##FP_SUFFIX)(                                                      \
        sycl::queue & queue, oneapi::mkl::transpose transpose_val, const FP_TYPE alpha,      \
        oneapi::mkl::sparse::matrix_handle_t A_handle, const FP_TYPE *x, const FP_TYPE beta, \
        FP_TYPE *y, const std::vector<sycl::event> &dependencies)

#define DEFINE_TRSV(FP_TYPE, FP_SUFFIX)                                                        \
    void (*trsv_buffer##FP_SUFFIX)(                                                            \
        sycl::queue & queue, oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose transpose_val, \
        oneapi::mkl::diag diag_val, oneapi::mkl::sparse::matrix_handle_t A_handle,             \
        sycl::buffer<FP_TYPE, 1> & x, sycl::buffer<FP_TYPE, 1> & y);                           \
    sycl::event (*trsv_usm##FP_SUFFIX)(                                                        \
        sycl::queue & queue, oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose transpose_val, \
        oneapi::mkl::diag diag_val, oneapi::mkl::sparse::matrix_handle_t A_handle,             \
        const FP_TYPE *x, FP_TYPE *y, const std::vector<sycl::event> &dependencies)

#define DEFINE_GEMM(FP_TYPE, FP_SUFFIX)                                                       \
    void (*gemm_buffer##FP_SUFFIX)(                                                           \
        sycl::queue & queue, oneapi::mkl::layout dense_matrix_layout,                         \
        oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,               \
        const FP_TYPE alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,                   \
        sycl::buffer<FP_TYPE, 1> &B, const std::int64_t columns, const std::int64_t ldb,      \
        const FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &C, const std::int64_t ldc);             \
    sycl::event (*gemm_usm##FP_SUFFIX)(                                                       \
        sycl::queue & queue, oneapi::mkl::layout dense_matrix_layout,                         \
        oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,               \
        const FP_TYPE alpha, oneapi::mkl::sparse::matrix_handle_t A_handle, const FP_TYPE *B, \
        const std::int64_t columns, const std::int64_t ldb, const FP_TYPE beta, FP_TYPE *C,   \
        const std::int64_t ldc, const std::vector<sycl::event> &dependencies)

typedef struct {
    int version;
    void (*init_matrix_handle)(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *p_handle);

    sycl::event (*release_matrix_handle)(sycl::queue &queue,
                                         oneapi::mkl::sparse::matrix_handle_t *p_handle,
                                         const std::vector<sycl::event> &dependencies);

    FOR_EACH_FP_AND_INT_TYPE(DEFINE_SET_CSR_DATA);

    // optimize_*
    sycl::event (*optimize_gemm)(sycl::queue &queue, oneapi::mkl::transpose transpose_A,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);
    sycl::event (*optimize_gemv)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);
    sycl::event (*optimize_trsv)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                                 oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);

    FOR_EACH_FP_TYPE(DEFINE_GEMV);
    FOR_EACH_FP_TYPE(DEFINE_TRSV);
    FOR_EACH_FP_TYPE(DEFINE_GEMM);
} sparse_blas_function_table_t;

#undef DEFINE_SET_CSR_DATA
#undef DEFINE_GEMV
#undef DEFINE_TRSV
#undef DEFINE_GEMM

#endif // _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_

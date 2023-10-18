/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
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

#include "oneapi/mkl/sparse_blas/detail/sparse_blas_rt.hpp"

#include "function_table_initializer.hpp"
#include "sparse_blas/function_table.hpp"
#include "sparse_blas/macros.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi::mkl::sparse {

static oneapi::mkl::detail::table_initializer<mkl::domain::sparse_blas,
                                              sparse_blas_function_table_t>
    function_tables;

void init_matrix_handle(sycl::queue &queue, matrix_handle_t *p_handle) {
    auto libkey = get_device_id(queue);
    function_tables[libkey].init_matrix_handle(queue, p_handle);
}

sycl::event release_matrix_handle(sycl::queue &queue, matrix_handle_t *p_handle,
                                  const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].release_matrix_handle(queue, p_handle, dependencies);
}

#define DEFINE_SET_CSR_DATA(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                              \
    template <>                                                                                    \
    void set_csr_data(sycl::queue &queue, matrix_handle_t handle, INT_TYPE num_rows,               \
                      INT_TYPE num_cols, INT_TYPE nnz, index_base index,                           \
                      sycl::buffer<INT_TYPE, 1> &row_ptr, sycl::buffer<INT_TYPE, 1> &col_ind,      \
                      sycl::buffer<FP_TYPE, 1> &val) {                                             \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[libkey].set_csr_data_buffer##FP_SUFFIX##INT_SUFFIX(                        \
            queue, handle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val);                 \
    }                                                                                              \
    template <>                                                                                    \
    sycl::event set_csr_data(sycl::queue &queue, matrix_handle_t handle, INT_TYPE num_rows,        \
                             INT_TYPE num_cols, INT_TYPE nnz, index_base index, INT_TYPE *row_ptr, \
                             INT_TYPE *col_ind, FP_TYPE *val,                                      \
                             const std::vector<sycl::event> &dependencies) {                       \
        auto libkey = get_device_id(queue);                                                        \
        return function_tables[libkey].set_csr_data_usm##FP_SUFFIX##INT_SUFFIX(                    \
            queue, handle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val, dependencies);   \
    }

FOR_EACH_FP_AND_INT_TYPE(DEFINE_SET_CSR_DATA)
#undef DEFINE_SET_CSR_DATA

sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_gemm(queue, transpose_A, handle, dependencies);
}

sycl::event optimize_gemv(sycl::queue &queue, transpose transpose_val, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_gemv(queue, transpose_val, handle, dependencies);
}

sycl::event optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                          matrix_handle_t handle, const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_trsv(queue, uplo_val, transpose_val, diag_val, handle,
                                                 dependencies);
}

#define DEFINE_GEMV(FP_TYPE, FP_SUFFIX)                                                           \
    template <>                                                                                   \
    void gemv(sycl::queue &queue, transpose transpose_val, const FP_TYPE alpha,                   \
              matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x, const FP_TYPE beta,          \
              sycl::buffer<FP_TYPE, 1> &y) {                                                      \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[libkey].gemv_buffer##FP_SUFFIX(queue, transpose_val, alpha, A_handle, x,  \
                                                       beta, y);                                  \
    }                                                                                             \
    template <>                                                                                   \
    sycl::event gemv(sycl::queue &queue, transpose transpose_val, const FP_TYPE alpha,            \
                     matrix_handle_t A_handle, const FP_TYPE *x, const FP_TYPE beta, FP_TYPE *y,  \
                     const std::vector<sycl::event> &dependencies) {                              \
        auto libkey = get_device_id(queue);                                                       \
        return function_tables[libkey].gemv_usm##FP_SUFFIX(queue, transpose_val, alpha, A_handle, \
                                                           x, beta, y, dependencies);             \
    }

FOR_EACH_FP_TYPE(DEFINE_GEMV)
#undef DEFINE_GEMV

#define DEFINE_TRSV(FP_TYPE, FP_SUFFIX)                                                          \
    template <>                                                                                  \
    void trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,         \
              matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x,                             \
              sycl::buffer<FP_TYPE, 1> &y) {                                                     \
        auto libkey = get_device_id(queue);                                                      \
        function_tables[libkey].trsv_buffer##FP_SUFFIX(queue, uplo_val, transpose_val, diag_val, \
                                                       A_handle, x, y);                          \
    }                                                                                            \
    template <>                                                                                  \
    sycl::event trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,  \
                     matrix_handle_t A_handle, const FP_TYPE *x, FP_TYPE *y,                     \
                     const std::vector<sycl::event> &dependencies) {                             \
        auto libkey = get_device_id(queue);                                                      \
        return function_tables[libkey].trsv_usm##FP_SUFFIX(                                      \
            queue, uplo_val, transpose_val, diag_val, A_handle, x, y, dependencies);             \
    }

FOR_EACH_FP_TYPE(DEFINE_TRSV)
#undef DEFINE_TRSV

#define DEFINE_GEMM(FP_TYPE, FP_SUFFIX)                                                          \
    template <>                                                                                  \
    void gemm(sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A,             \
              transpose transpose_B, const FP_TYPE alpha, matrix_handle_t A_handle,              \
              sycl::buffer<FP_TYPE, 1> &B, const std::int64_t columns, const std::int64_t ldb,   \
              const FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &C, const std::int64_t ldc) {         \
        auto libkey = get_device_id(queue);                                                      \
        function_tables[libkey].gemm_buffer##FP_SUFFIX(queue, dense_matrix_layout, transpose_A,  \
                                                       transpose_B, alpha, A_handle, B, columns, \
                                                       ldb, beta, C, ldc);                       \
    }                                                                                            \
    template <>                                                                                  \
    sycl::event gemm(sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A,      \
                     transpose transpose_B, const FP_TYPE alpha, matrix_handle_t A_handle,       \
                     const FP_TYPE *B, const std::int64_t columns, const std::int64_t ldb,       \
                     const FP_TYPE beta, FP_TYPE *C, const std::int64_t ldc,                     \
                     const std::vector<sycl::event> &dependencies) {                             \
        auto libkey = get_device_id(queue);                                                      \
        return function_tables[libkey].gemm_usm##FP_SUFFIX(                                      \
            queue, dense_matrix_layout, transpose_A, transpose_B, alpha, A_handle, B, columns,   \
            ldb, beta, C, ldc, dependencies);                                                    \
    }

FOR_EACH_FP_TYPE(DEFINE_GEMM)
#undef DEFINE_GEMM

} // namespace oneapi::mkl::sparse

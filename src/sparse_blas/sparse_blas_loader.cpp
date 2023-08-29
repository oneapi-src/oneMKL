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

#include "oneapi/mkl/sparse_blas/sparse_blas.hpp"

#include "function_table_initializer.hpp"
#include "sparse_blas/function_table.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi::mkl::sparse {

static oneapi::mkl::detail::table_initializer<mkl::domain::sparse_blas,
                                              sparse_blas_function_table_t>
    function_tables;

void init_matrix_handle(sycl::queue &queue, matrix_handle_t *handle) {
    auto libkey = get_device_id(queue);
    function_tables[libkey].init_matrix_handle(queue, handle);
}

sycl::event release_matrix_handle(sycl::queue &queue, matrix_handle_t *handle,
                                  const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].release_matrix_handle(queue, handle, dependencies);
}

#define DEFINE_SET_CSR_DATA(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                          \
    template <>                                                                                \
    void set_csr_data(sycl::queue &queue, matrix_handle_t handle, INT_TYPE num_rows,           \
                      INT_TYPE num_cols, index_base index, sycl::buffer<INT_TYPE, 1> &row_ptr, \
                      sycl::buffer<INT_TYPE, 1> &col_ind, sycl::buffer<FP_TYPE, 1> &val) {     \
        auto libkey = get_device_id(queue);                                                    \
        function_tables[libkey].set_csr_data_buffer##FP_SUFFIX##INT_SUFFIX(                    \
            queue, handle, num_rows, num_cols, index, row_ptr, col_ind, val);                  \
    }                                                                                          \
    template <>                                                                                \
    sycl::event set_csr_data(sycl::queue &queue, matrix_handle_t handle, INT_TYPE num_rows,    \
                             INT_TYPE num_cols, index_base index, INT_TYPE *row_ptr,           \
                             INT_TYPE *col_ind, FP_TYPE *val,                                  \
                             const std::vector<sycl::event> &dependencies) {                   \
        auto libkey = get_device_id(queue);                                                    \
        return function_tables[libkey].set_csr_data_usm##FP_SUFFIX##INT_SUFFIX(                \
            queue, handle, num_rows, num_cols, index, row_ptr, col_ind, val, dependencies);    \
    }

DEFINE_SET_CSR_DATA(float, _sf, std::int32_t, _i32)
DEFINE_SET_CSR_DATA(double, _sd, std::int32_t, _i32)
DEFINE_SET_CSR_DATA(std::complex<float>, _cf, std::int32_t, _i32)
DEFINE_SET_CSR_DATA(std::complex<double>, _cd, std::int32_t, _i32)
DEFINE_SET_CSR_DATA(float, _sf, std::int64_t, _i64)
DEFINE_SET_CSR_DATA(double, _sd, std::int64_t, _i64)
DEFINE_SET_CSR_DATA(std::complex<float>, _cf, std::int64_t, _i64)
DEFINE_SET_CSR_DATA(std::complex<double>, _cd, std::int64_t, _i64)
#undef DEFINE_SET_CSR_DATA

sycl::event optimize_gemv(sycl::queue &queue, transpose transpose_val, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_gemv(queue, transpose_val, handle, dependencies);
}

sycl::event optimize_trmv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                          matrix_handle_t handle, const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_trmv(queue, uplo_val, transpose_val, diag_val, handle,
                                                 dependencies);
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
                     matrix_handle_t A_handle, const FP_TYPE *x, const FP_TYPE beta,              \
                     const FP_TYPE *y, const std::vector<sycl::event> &dependencies) {            \
        auto libkey = get_device_id(queue);                                                       \
        return function_tables[libkey].gemv_usm##FP_SUFFIX(queue, transpose_val, alpha, A_handle, \
                                                           x, beta, y, dependencies);             \
    }

DEFINE_GEMV(float, _sf)
DEFINE_GEMV(double, _sd)
DEFINE_GEMV(std::complex<float>, _cf)
DEFINE_GEMV(std::complex<double>, _cd)
#undef DEFINE_GEMV

#define DEFINE_GEMVDOT(FP_TYPE, FP_SUFFIX)                                                       \
    template <>                                                                                  \
    void gemvdot(sycl::queue &queue, transpose transpose_val, FP_TYPE alpha,                     \
                 matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x, FP_TYPE beta,            \
                 sycl::buffer<FP_TYPE, 1> &y, sycl::buffer<FP_TYPE, 1> &d) {                     \
        auto libkey = get_device_id(queue);                                                      \
        function_tables[libkey].gemvdot_buffer##FP_SUFFIX(queue, transpose_val, alpha, A_handle, \
                                                          x, beta, y, d);                        \
    }                                                                                            \
    template <>                                                                                  \
    sycl::event gemvdot(sycl::queue &queue, transpose transpose_val, FP_TYPE alpha,              \
                        matrix_handle_t A_handle, FP_TYPE *x, FP_TYPE beta, FP_TYPE *y,          \
                        FP_TYPE *d, const std::vector<sycl::event> &dependencies) {              \
        auto libkey = get_device_id(queue);                                                      \
        return function_tables[libkey].gemvdot_usm##FP_SUFFIX(                                   \
            queue, transpose_val, alpha, A_handle, x, beta, y, d, dependencies);                 \
    }

DEFINE_GEMVDOT(float, _sf)
DEFINE_GEMVDOT(double, _sd)
DEFINE_GEMVDOT(std::complex<float>, _cf)
DEFINE_GEMVDOT(std::complex<double>, _cd)
#undef DEFINE_GEMVDOT

#define DEFINE_SYMV(FP_TYPE, FP_SUFFIX)                                                           \
    template <>                                                                                   \
    void symv(sycl::queue &queue, uplo uplo_val, FP_TYPE alpha, matrix_handle_t A_handle,         \
              sycl::buffer<FP_TYPE, 1> &x, FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &y) {           \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[libkey].symv_buffer##FP_SUFFIX(queue, uplo_val, alpha, A_handle, x, beta, \
                                                       y);                                        \
    }                                                                                             \
    template <>                                                                                   \
    sycl::event symv(sycl::queue &queue, uplo uplo_val, FP_TYPE alpha, matrix_handle_t A_handle,  \
                     FP_TYPE *x, FP_TYPE beta, FP_TYPE *y,                                        \
                     const std::vector<sycl::event> &dependencies) {                              \
        auto libkey = get_device_id(queue);                                                       \
        return function_tables[libkey].symv_usm##FP_SUFFIX(queue, uplo_val, alpha, A_handle, x,   \
                                                           beta, y, dependencies);                \
    }

DEFINE_SYMV(float, _sf)
DEFINE_SYMV(double, _sd)
DEFINE_SYMV(std::complex<float>, _cf)
DEFINE_SYMV(std::complex<double>, _cd)
#undef DEFINE_SYMV

#define DEFINE_TRMV(FP_TYPE, FP_SUFFIX)                                                           \
    template <>                                                                                   \
    void trmv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,          \
              FP_TYPE alpha, matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x, FP_TYPE beta, \
              sycl::buffer<FP_TYPE, 1> &y) {                                                      \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[libkey].trmv_buffer##FP_SUFFIX(queue, uplo_val, transpose_val, diag_val,  \
                                                       alpha, A_handle, x, beta, y);              \
    }                                                                                             \
    template <>                                                                                   \
    sycl::event trmv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,   \
                     FP_TYPE alpha, matrix_handle_t A_handle, FP_TYPE *x, FP_TYPE beta,           \
                     FP_TYPE *y, const std::vector<sycl::event> &dependencies) {                  \
        auto libkey = get_device_id(queue);                                                       \
        return function_tables[libkey].trmv_usm##FP_SUFFIX(                                       \
            queue, uplo_val, transpose_val, diag_val, alpha, A_handle, x, beta, y, dependencies); \
    }

DEFINE_TRMV(float, _sf)
DEFINE_TRMV(double, _sd)
DEFINE_TRMV(std::complex<float>, _cf)
DEFINE_TRMV(std::complex<double>, _cd)
#undef DEFINE_TRMV

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
                     matrix_handle_t A_handle, FP_TYPE *x, FP_TYPE *y,                           \
                     const std::vector<sycl::event> &dependencies) {                             \
        auto libkey = get_device_id(queue);                                                      \
        return function_tables[libkey].trsv_usm##FP_SUFFIX(                                      \
            queue, uplo_val, transpose_val, diag_val, A_handle, x, y, dependencies);             \
    }

DEFINE_TRSV(float, _sf)
DEFINE_TRSV(double, _sd)
DEFINE_TRSV(std::complex<float>, _cf)
DEFINE_TRSV(std::complex<double>, _cd)
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
                     const FP_TYPE beta, const FP_TYPE *C, const std::int64_t ldc,               \
                     const std::vector<sycl::event> &dependencies) {                             \
        auto libkey = get_device_id(queue);                                                      \
        return function_tables[libkey].gemm_usm##FP_SUFFIX(                                      \
            queue, dense_matrix_layout, transpose_A, transpose_B, alpha, A_handle, B, columns,   \
            ldb, beta, C, ldc, dependencies);                                                    \
    }

DEFINE_GEMM(float, _sf)
DEFINE_GEMM(double, _sd)
DEFINE_GEMM(std::complex<float>, _cf)
DEFINE_GEMM(std::complex<double>, _cd)
#undef DEFINE_GEMM

} // namespace oneapi::mkl::sparse

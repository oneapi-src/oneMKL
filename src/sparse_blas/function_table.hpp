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

typedef struct {
    int version;
    void (*init_matrix_handle)(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *handle);

    sycl::event (*release_matrix_handle)(sycl::queue &queue,
                                         oneapi::mkl::sparse::matrix_handle_t *handle,
                                         const std::vector<sycl::event> &dependencies);

    // set_csr_data
    void (*set_csr_data_buffer_sf_i32)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int32_t num_rows, std::int32_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int32_t, 1> &row_ptr,
                                       sycl::buffer<std::int32_t, 1> &col_ind,
                                       sycl::buffer<float, 1> &val);
    void (*set_csr_data_buffer_sd_i32)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int32_t num_rows, std::int32_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int32_t, 1> &row_ptr,
                                       sycl::buffer<std::int32_t, 1> &col_ind,
                                       sycl::buffer<double, 1> &val);
    void (*set_csr_data_buffer_cf_i32)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int32_t num_rows, std::int32_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int32_t, 1> &row_ptr,
                                       sycl::buffer<std::int32_t, 1> &col_ind,
                                       sycl::buffer<std::complex<float>, 1> &val);
    void (*set_csr_data_buffer_cd_i32)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int32_t num_rows, std::int32_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int32_t, 1> &row_ptr,
                                       sycl::buffer<std::int32_t, 1> &col_ind,
                                       sycl::buffer<std::complex<double>, 1> &val);
    void (*set_csr_data_buffer_sf_i64)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int64_t, 1> &row_ptr,
                                       sycl::buffer<std::int64_t, 1> &col_ind,
                                       sycl::buffer<float, 1> &val);
    void (*set_csr_data_buffer_sd_i64)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int64_t, 1> &row_ptr,
                                       sycl::buffer<std::int64_t, 1> &col_ind,
                                       sycl::buffer<double, 1> &val);
    void (*set_csr_data_buffer_cf_i64)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int64_t, 1> &row_ptr,
                                       sycl::buffer<std::int64_t, 1> &col_ind,
                                       sycl::buffer<std::complex<float>, 1> &val);
    void (*set_csr_data_buffer_cd_i64)(sycl::queue &queue,
                                       oneapi::mkl::sparse::matrix_handle_t handle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       oneapi::mkl::index_base index,
                                       sycl::buffer<std::int64_t, 1> &row_ptr,
                                       sycl::buffer<std::int64_t, 1> &col_ind,
                                       sycl::buffer<std::complex<double>, 1> &val);
    sycl::event (*set_csr_data_usm_sf_i32)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int32_t num_rows, std::int32_t num_cols,
                                           oneapi::mkl::index_base index, std::int32_t *row_ptr,
                                           std::int32_t *col_ind, float *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_sd_i32)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int32_t num_rows, std::int32_t num_cols,
                                           oneapi::mkl::index_base index, std::int32_t *row_ptr,
                                           std::int32_t *col_ind, double *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_cf_i32)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int32_t num_rows, std::int32_t num_cols,
                                           oneapi::mkl::index_base index, std::int32_t *row_ptr,
                                           std::int32_t *col_ind, std::complex<float> *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_cd_i32)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int32_t num_rows, std::int32_t num_cols,
                                           oneapi::mkl::index_base index, std::int32_t *row_ptr,
                                           std::int32_t *col_ind, std::complex<double> *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_sf_i64)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int64_t num_rows, std::int64_t num_cols,
                                           oneapi::mkl::index_base index, std::int64_t *row_ptr,
                                           std::int64_t *col_ind, float *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_sd_i64)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int64_t num_rows, std::int64_t num_cols,
                                           oneapi::mkl::index_base index, std::int64_t *row_ptr,
                                           std::int64_t *col_ind, double *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_cf_i64)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int64_t num_rows, std::int64_t num_cols,
                                           oneapi::mkl::index_base index, std::int64_t *row_ptr,
                                           std::int64_t *col_ind, std::complex<float> *val,
                                           const std::vector<sycl::event> &dependencies);
    sycl::event (*set_csr_data_usm_cd_i64)(sycl::queue &queue,
                                           oneapi::mkl::sparse::matrix_handle_t handle,
                                           std::int64_t num_rows, std::int64_t num_cols,
                                           oneapi::mkl::index_base index, std::int64_t *row_ptr,
                                           std::int64_t *col_ind, std::complex<double> *val,
                                           const std::vector<sycl::event> &dependencies);

    // optimize_*
    sycl::event (*optimize_gemv)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);
    sycl::event (*optimize_trmv)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                                 oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);
    sycl::event (*optimize_trsv)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                                 oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);

    // gemv
    void (*gemv_buffer_sf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                           const float alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<float, 1> &x, const float beta, sycl::buffer<float, 1> &y);
    void (*gemv_buffer_sd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                           const double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<double, 1> &x, const double beta,
                           sycl::buffer<double, 1> &y);
    void (*gemv_buffer_cf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                           const std::complex<float> alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<float>, 1> &x, const std::complex<float> beta,
                           sycl::buffer<std::complex<float>, 1> &y);
    void (*gemv_buffer_cd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                           const std::complex<double> alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<double>, 1> &x,
                           const std::complex<double> beta,
                           sycl::buffer<std::complex<double>, 1> &y);
    sycl::event (*gemv_usm_sf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                               const float alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                               const float *x, const float beta, const float *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*gemv_usm_sd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                               const double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                               const double *x, const double beta, const double *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*gemv_usm_cf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                               const std::complex<float> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               const std::complex<float> *x, const std::complex<float> beta,
                               const std::complex<float> *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*gemv_usm_cd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                               const std::complex<double> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               const std::complex<double> *x, const std::complex<double> beta,
                               const std::complex<double> *y,
                               const std::vector<sycl::event> &dependencies);

    // gemvdot
    void (*gemvdot_buffer_sf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val, float alpha,
                              oneapi::mkl::sparse::matrix_handle_t A_handle,
                              sycl::buffer<float, 1> &x, float beta, sycl::buffer<float, 1> &y,
                              sycl::buffer<float, 1> &d);
    void (*gemvdot_buffer_sd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                              double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                              sycl::buffer<double, 1> &x, double beta, sycl::buffer<double, 1> &y,
                              sycl::buffer<double, 1> &d);
    void (*gemvdot_buffer_cf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                              std::complex<float> alpha,
                              oneapi::mkl::sparse::matrix_handle_t A_handle,
                              sycl::buffer<std::complex<float>, 1> &x, std::complex<float> beta,
                              sycl::buffer<std::complex<float>, 1> &y,
                              sycl::buffer<std::complex<float>, 1> &d);
    void (*gemvdot_buffer_cd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                              std::complex<double> alpha,
                              oneapi::mkl::sparse::matrix_handle_t A_handle,
                              sycl::buffer<std::complex<double>, 1> &x, std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &y,
                              sycl::buffer<std::complex<double>, 1> &d);
    sycl::event (*gemvdot_usm_sf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                  float alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  float *x, float beta, float *y, float *d,
                                  const std::vector<sycl::event> &dependencies);
    sycl::event (*gemvdot_usm_sd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                  double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  double *x, double beta, double *y, double *d,
                                  const std::vector<sycl::event> &dependencies);
    sycl::event (*gemvdot_usm_cf)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                  std::complex<float> alpha,
                                  oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  std::complex<float> *x, std::complex<float> beta,
                                  std::complex<float> *y, std::complex<float> *d,
                                  const std::vector<sycl::event> &dependencies);
    sycl::event (*gemvdot_usm_cd)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                  std::complex<double> alpha,
                                  oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  std::complex<double> *x, std::complex<double> beta,
                                  std::complex<double> *y, std::complex<double> *d,
                                  const std::vector<sycl::event> &dependencies);

    // symv
    void (*symv_buffer_sf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val, float alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<float, 1> &x,
                           float beta, sycl::buffer<float, 1> &y);
    void (*symv_buffer_sd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val, double alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<double, 1> &x, double beta, sycl::buffer<double, 1> &y);
    void (*symv_buffer_cf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           std::complex<float> alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<float>, 1> &x, std::complex<float> beta,
                           sycl::buffer<std::complex<float>, 1> &y);
    void (*symv_buffer_cd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           std::complex<double> alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<double>, 1> &x, std::complex<double> beta,
                           sycl::buffer<std::complex<double>, 1> &y);
    sycl::event (*symv_usm_sf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val, float alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle, float *x, float beta,
                               float *y, const std::vector<sycl::event> &dependencies);
    sycl::event (*symv_usm_sd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val, double alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle, double *x,
                               double beta, double *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*symv_usm_cf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               std::complex<float> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               std::complex<float> *x, std::complex<float> beta,
                               std::complex<float> *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*symv_usm_cd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               std::complex<double> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               std::complex<double> *x, std::complex<double> beta,
                               std::complex<double> *y,
                               const std::vector<sycl::event> &dependencies);

    // trmv
    void (*trmv_buffer_sf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           float alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<float, 1> &x, float beta, sycl::buffer<float, 1> &y);
    void (*trmv_buffer_sd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<double, 1> &x, double beta, sycl::buffer<double, 1> &y);
    void (*trmv_buffer_cf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           std::complex<float> alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<float>, 1> &x, std::complex<float> beta,
                           sycl::buffer<std::complex<float>, 1> &y);
    void (*trmv_buffer_cd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           std::complex<double> alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<double>, 1> &x, std::complex<double> beta,
                           sycl::buffer<std::complex<double>, 1> &y);
    sycl::event (*trmv_usm_sf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               float alpha, oneapi::mkl::sparse::matrix_handle_t A_handle, float *x,
                               float beta, float *y, const std::vector<sycl::event> &dependencies);
    sycl::event (*trmv_usm_sd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                               double *x, double beta, double *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*trmv_usm_cf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               std::complex<float> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               std::complex<float> *x, std::complex<float> beta,
                               std::complex<float> *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*trmv_usm_cd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               std::complex<double> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               std::complex<double> *x, std::complex<double> beta,
                               std::complex<double> *y,
                               const std::vector<sycl::event> &dependencies);

    // trsv
    void (*trsv_buffer_sf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<float, 1> &x,
                           sycl::buffer<float, 1> &y);
    void (*trsv_buffer_sd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<double, 1> &x, sycl::buffer<double, 1> &y);
    void (*trsv_buffer_cf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<float>, 1> &x,
                           sycl::buffer<std::complex<float>, 1> &y);
    void (*trsv_buffer_cd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                           oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<double>, 1> &x,
                           sycl::buffer<std::complex<double>, 1> &y);
    sycl::event (*trsv_usm_sf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               oneapi::mkl::sparse::matrix_handle_t A_handle, float *x, float *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*trsv_usm_sd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               oneapi::mkl::sparse::matrix_handle_t A_handle, double *x, double *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*trsv_usm_cf)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               std::complex<float> *x, std::complex<float> *y,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*trsv_usm_cd)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                               oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               std::complex<double> *x, std::complex<double> *y,
                               const std::vector<sycl::event> &dependencies);

    // gemm
    void (*gemm_buffer_sf)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                           oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,
                           const float alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<float, 1> &B, const std::int64_t columns,
                           const std::int64_t ldb, const float beta, sycl::buffer<float, 1> &C,
                           const std::int64_t ldc);
    void (*gemm_buffer_sd)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                           oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,
                           const double alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<double, 1> &B, const std::int64_t columns,
                           const std::int64_t ldb, const double beta, sycl::buffer<double, 1> &C,
                           const std::int64_t ldc);
    void (*gemm_buffer_cf)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                           oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,
                           const std::complex<float> alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<float>, 1> &B, const std::int64_t columns,
                           const std::int64_t ldb, const std::complex<float> beta,
                           sycl::buffer<std::complex<float>, 1> &C, const std::int64_t ldc);
    void (*gemm_buffer_cd)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                           oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,
                           const std::complex<double> alpha,
                           oneapi::mkl::sparse::matrix_handle_t A_handle,
                           sycl::buffer<std::complex<double>, 1> &B, const std::int64_t columns,
                           const std::int64_t ldb, const std::complex<double> beta,
                           sycl::buffer<std::complex<double>, 1> &C, const std::int64_t ldc);
    sycl::event (*gemm_usm_sf)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                               oneapi::mkl::transpose transpose_A,
                               oneapi::mkl::transpose transpose_B, const float alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle, const float *B,
                               const std::int64_t columns, const std::int64_t ldb, const float beta,
                               const float *C, const std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*gemm_usm_sd)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                               oneapi::mkl::transpose transpose_A,
                               oneapi::mkl::transpose transpose_B, const double alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle, const double *B,
                               const std::int64_t columns, const std::int64_t ldb,
                               const double beta, const double *C, const std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*gemm_usm_cf)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                               oneapi::mkl::transpose transpose_A,
                               oneapi::mkl::transpose transpose_B, const std::complex<float> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               const std::complex<float> *B, const std::int64_t columns,
                               const std::int64_t ldb, const std::complex<float> beta,
                               const std::complex<float> *C, const std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies);
    sycl::event (*gemm_usm_cd)(sycl::queue &queue, oneapi::mkl::layout dense_matrix_layout,
                               oneapi::mkl::transpose transpose_A,
                               oneapi::mkl::transpose transpose_B, const std::complex<double> alpha,
                               oneapi::mkl::sparse::matrix_handle_t A_handle,
                               const std::complex<double> *B, const std::int64_t columns,
                               const std::int64_t ldb, const std::complex<double> beta,
                               const std::complex<double> *C, const std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies);
} sparse_blas_function_table_t;

#endif // _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_
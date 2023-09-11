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

// MKLCPU backend includes
// These includes define their own oneapi::mkl::sparse namespace with some of the types that are used here: matrix_handle_t, index_base.
#include <oneapi/mkl/spblas.hpp>

#include "oneapi/mkl/sparse_blas/detail/helper_types.hpp"
#include "oneapi/mkl/sparse_blas/detail/mklcpu/onemkl_sparse_blas_mklcpu.hpp"

// Includes are set up so that oneapi::mkl::sparse namespace refers to the MKLCPU backend namespace (oneMKL product)
// in this file.
// oneapi::mkl::sparse::detail namespace refers to the oneMKL interface namespace.

namespace oneapi::mkl::sparse::detail {

auto get_handle(detail::matrix_handle **handle) {
    return reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t *>(handle);
}
auto get_handle(detail::matrix_handle *handle) {
    return reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t>(handle);
}

} // namespace oneapi::mkl::sparse::detail

namespace oneapi::mkl::sparse::mklcpu {

void init_matrix_handle(sycl::queue & /*queue*/, detail::matrix_handle **p_handle) {
    oneapi::mkl::sparse::init_matrix_handle(detail::get_handle(p_handle));
}

sycl::event release_matrix_handle(sycl::queue &queue, detail::matrix_handle **p_handle,
                                  const std::vector<sycl::event> &dependencies) {
    return oneapi::mkl::sparse::release_matrix_handle(queue, detail::get_handle(p_handle),
                                                      dependencies);
}

template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>> set_csr_data(
    sycl::queue &queue, detail::matrix_handle *handle, intType num_rows, intType num_cols,
    index_base index, sycl::buffer<intType, 1> &row_ptr, sycl::buffer<intType, 1> &col_ind,
    sycl::buffer<fpType, 1> &val) {
    oneapi::mkl::sparse::set_csr_data(queue, detail::get_handle(handle), num_rows, num_cols, index,
                                      row_ptr, col_ind, val);
}

template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>, sycl::event> set_csr_data(
    sycl::queue &queue, detail::matrix_handle *handle, intType num_rows, intType num_cols,
    index_base index, intType *row_ptr, intType *col_ind, fpType *val,
    const std::vector<sycl::event> &dependencies) {
    return oneapi::mkl::sparse::set_csr_data(queue, detail::get_handle(handle), num_rows, num_cols,
                                             index, row_ptr, col_ind, val, dependencies);
}

#define INSTANTIATE_SET_CSR_DATA(FP_TYPE, INT_TYPE)                                                \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    set_csr_data<FP_TYPE, INT_TYPE>(                                                               \
        sycl::queue & queue, detail::matrix_handle * handle, INT_TYPE num_rows, INT_TYPE num_cols, \
        index_base index, sycl::buffer<INT_TYPE, 1> & row_ptr,                                     \
        sycl::buffer<INT_TYPE, 1> & col_ind, sycl::buffer<FP_TYPE, 1> & val);                      \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>, sycl::event>      \
    set_csr_data<FP_TYPE, INT_TYPE>(sycl::queue & queue, detail::matrix_handle * handle,           \
                                    INT_TYPE num_rows, INT_TYPE num_cols, index_base index,        \
                                    INT_TYPE * row_ptr, INT_TYPE * col_ind, FP_TYPE * val,         \
                                    const std::vector<sycl::event> &dependencies)

INSTANTIATE_SET_CSR_DATA(float, std::int32_t);
INSTANTIATE_SET_CSR_DATA(double, std::int32_t);
INSTANTIATE_SET_CSR_DATA(std::complex<float>, std::int32_t);
INSTANTIATE_SET_CSR_DATA(std::complex<double>, std::int32_t);
INSTANTIATE_SET_CSR_DATA(float, std::int64_t);
INSTANTIATE_SET_CSR_DATA(double, std::int64_t);
INSTANTIATE_SET_CSR_DATA(std::complex<float>, std::int64_t);
INSTANTIATE_SET_CSR_DATA(std::complex<double>, std::int64_t);
#undef INSTANTIATE_SET_CSR_DATA

} // namespace oneapi::mkl::sparse::mklcpu

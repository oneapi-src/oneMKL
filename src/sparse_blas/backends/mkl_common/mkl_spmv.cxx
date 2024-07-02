/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

// The operation descriptor is not needed as long as the backend does not have an equivalent type and does not support external workspace.
using spmv_descr = void *;

void init_spmv_descr(sycl::queue & /*queue*/, oneapi::mkl::sparse::spmv_descr_t *p_spmv_descr) {
    *p_spmv_descr = nullptr;
}

sycl::event release_spmv_descr(sycl::queue &queue, oneapi::mkl::sparse::spmv_descr_t /*spmv_descr*/,
                               const std::vector<sycl::event> &dependencies) {
    return detail::collapse_dependencies(queue, dependencies);
}

void check_valid_spmv(const std::string function_name, sycl::queue &queue,
                      oneapi::mkl::transpose opA, oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t y_handle, const void *alpha,
                      const void *beta) {
    THROW_IF_NULLPTR(function_name, A_handle);
    THROW_IF_NULLPTR(function_name, x_handle);
    THROW_IF_NULLPTR(function_name, y_handle);

    auto internal_A_handle = detail::get_internal_handle(A_handle);
    detail::check_all_containers_compatible(function_name, internal_A_handle, x_handle, y_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::check_ptr_is_host_accessible("spmv", "alpha", queue, alpha);
        detail::check_ptr_is_host_accessible("spmv", "beta", queue, beta);
    }
    if (detail::is_ptr_accessible_on_host(queue, alpha) !=
        detail::is_ptr_accessible_on_host(queue, beta)) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "Alpha and beta must both be placed on host memory or device memory.");
    }
    if (A_view.type_view == oneapi::mkl::sparse::matrix_descr::diagonal) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix view's type cannot be diagonal.");
    }

    if (A_view.type_view != oneapi::mkl::sparse::matrix_descr::triangular &&
        A_view.diag_view == oneapi::mkl::diag::unit) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "`unit` diag_view can only be used with a triangular type_view.");
    }

    if ((A_view.type_view == oneapi::mkl::sparse::matrix_descr::symmetric ||
         A_view.type_view == oneapi::mkl::sparse::matrix_descr::hermitian) &&
        opA == oneapi::mkl::transpose::conjtrans) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "Symmetric or Hermitian matrix cannot be conjugated with `conjtrans`.");
    }
}

void spmv_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t x_handle, const void *beta,
                      oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                      oneapi::mkl::sparse::spmv_alg /*alg*/,
                      oneapi::mkl::sparse::spmv_descr_t /*spmv_descr*/,
                      std::size_t &temp_buffer_size) {
    // TODO: Add support for external workspace once the close-source oneMKL backend supports it.
    check_valid_spmv(__func__, queue, opA, A_view, A_handle, x_handle, y_handle, alpha, beta);
    temp_buffer_size = 0;
}

void spmv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                   oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t x_handle, const void *beta,
                   oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                   oneapi::mkl::sparse::spmv_alg alg,
                   oneapi::mkl::sparse::spmv_descr_t /*spmv_descr*/,
                   sycl::buffer<std::uint8_t, 1> /*workspace*/) {
    check_valid_spmv(__func__, queue, opA, A_view, A_handle, x_handle, y_handle, alpha, beta);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (!internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (alg == oneapi::mkl::sparse::spmv_alg::no_optimize_alg) {
        return;
    }
    sycl::event event;
    internal_A_handle->can_be_reset = false;
    if (A_view.type_view == matrix_descr::triangular) {
        event = oneapi::mkl::sparse::optimize_trmv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                                   internal_A_handle->backend_handle);
    }
    else if (A_view.type_view == matrix_descr::symmetric ||
             A_view.type_view == matrix_descr::hermitian) {
        // No optimize_symv currently
        return;
    }
    else {
        event = oneapi::mkl::sparse::optimize_gemv(queue, opA, internal_A_handle->backend_handle);
    }
    // spmv_optimize is not asynchronous for buffers as the backend optimize functions don't take buffers.
    event.wait_and_throw();
}

sycl::event spmv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t x_handle, const void *beta,
                          oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                          oneapi::mkl::sparse::spmv_alg alg,
                          oneapi::mkl::sparse::spmv_descr_t /*spmv_descr*/, void * /*workspace*/,
                          const std::vector<sycl::event> &dependencies) {
    check_valid_spmv(__func__, queue, opA, A_view, A_handle, x_handle, y_handle, alpha, beta);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (alg == oneapi::mkl::sparse::spmv_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    internal_A_handle->can_be_reset = false;
    if (A_view.type_view == matrix_descr::triangular) {
        return oneapi::mkl::sparse::optimize_trmv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                                  internal_A_handle->backend_handle, dependencies);
    }
    else if (A_view.type_view == matrix_descr::symmetric ||
             A_view.type_view == matrix_descr::hermitian) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    else {
        return oneapi::mkl::sparse::optimize_gemv(queue, opA, internal_A_handle->backend_handle,
                                                  dependencies);
    }
}

template <typename T>
sycl::event internal_spmv(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t x_handle, const void *beta,
                          oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                          oneapi::mkl::sparse::spmv_alg /*alg*/,
                          oneapi::mkl::sparse::spmv_descr_t /*spmv_descr*/,
                          const std::vector<sycl::event> &dependencies) {
    T host_alpha = detail::get_scalar_on_host(queue, static_cast<const T *>(alpha));
    T host_beta = detail::get_scalar_on_host(queue, static_cast<const T *>(beta));
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    internal_A_handle->can_be_reset = false;
    auto backend_handle = internal_A_handle->backend_handle;
    if (internal_A_handle->all_use_buffer()) {
        auto x_buffer = x_handle->get_buffer<T>();
        auto y_buffer = y_handle->get_buffer<T>();
        if (A_view.type_view == matrix_descr::triangular) {
            oneapi::mkl::sparse::trmv(queue, A_view.uplo_view, opA, A_view.diag_view, host_alpha,
                                      backend_handle, x_buffer, host_beta, y_buffer);
        }
        else if (A_view.type_view == matrix_descr::symmetric ||
                 A_view.type_view == matrix_descr::hermitian) {
            oneapi::mkl::sparse::symv(queue, A_view.uplo_view, host_alpha, backend_handle, x_buffer,
                                      host_beta, y_buffer);
        }
        else {
            oneapi::mkl::sparse::gemv(queue, opA, host_alpha, backend_handle, x_buffer, host_beta,
                                      y_buffer);
        }
        // Dependencies are not used for buffers
        return {};
    }
    else {
        auto x_usm = x_handle->get_usm_ptr<T>();
        auto y_usm = y_handle->get_usm_ptr<T>();
        if (A_view.type_view == matrix_descr::triangular) {
            return oneapi::mkl::sparse::trmv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                             host_alpha, backend_handle, x_usm, host_beta, y_usm,
                                             dependencies);
        }
        else if (A_view.type_view == matrix_descr::symmetric ||
                 A_view.type_view == matrix_descr::hermitian) {
            return oneapi::mkl::sparse::symv(queue, A_view.uplo_view, host_alpha, backend_handle,
                                             x_usm, host_beta, y_usm, dependencies);
        }
        else {
            return oneapi::mkl::sparse::gemv(queue, opA, host_alpha, backend_handle, x_usm,
                                             host_beta, y_usm, dependencies);
        }
    }
}

sycl::event spmv(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                 oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_vector_handle_t x_handle, const void *beta,
                 oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                 oneapi::mkl::sparse::spmv_alg alg, oneapi::mkl::sparse::spmv_descr_t spmv_descr,
                 const std::vector<sycl::event> &dependencies) {
    check_valid_spmv(__func__, queue, opA, A_view, A_handle, x_handle, y_handle, alpha, beta);
    auto value_type = detail::get_internal_handle(A_handle)->get_value_type();
    DISPATCH_MKL_OPERATION("spmv", value_type, internal_spmv, queue, opA, alpha, A_view, A_handle,
                           x_handle, beta, y_handle, alg, spmv_descr, dependencies);
}

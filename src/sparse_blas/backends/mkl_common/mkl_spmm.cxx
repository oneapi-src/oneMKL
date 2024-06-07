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
using spmm_descr = void *;

void init_spmm_descr(sycl::queue & /*queue*/, oneapi::mkl::sparse::spmm_descr_t *p_spmm_descr) {
    *p_spmm_descr = nullptr;
}

sycl::event release_spmm_descr(sycl::queue &queue, oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                               const std::vector<sycl::event> &dependencies) {
    return detail::collapse_dependencies(queue, dependencies);
}

void check_valid_spmm(const std::string function_name, sycl::queue &queue,
                      oneapi::mkl::transpose opA, oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle, const void *alpha,
                      const void *beta) {
    THROW_IF_NULLPTR(function_name, A_handle);
    THROW_IF_NULLPTR(function_name, B_handle);
    THROW_IF_NULLPTR(function_name, C_handle);

    auto internal_A_handle = detail::get_internal_handle(A_handle);
    detail::check_all_containers_compatible(function_name, internal_A_handle, B_handle, C_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::check_ptr_is_host_accessible("spmm", "alpha", queue, alpha);
        detail::check_ptr_is_host_accessible("spmm", "beta", queue, beta);
    }
    if (B_handle->dense_layout != C_handle->dense_layout) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "B and C matrices must used the same layout.");
    }

    if (A_view.type_view != oneapi::mkl::sparse::matrix_descr::general) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix view's type must be `matrix_descr::general`.");
    }

    if (A_view.diag_view != oneapi::mkl::diag::nonunit) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix's diag_view must be `nonunit`.");
    }

#if BACKEND == gpu
    if (opA == oneapi::mkl::transpose::conjtrans &&
        internal_A_handle->has_matrix_property(oneapi::mkl::sparse::matrix_property::symmetric)) {
        throw mkl::unimplemented("sparse_blas/mklgpu", function_name,
                                 "spmm does not support conjtrans with the symmetric property.");
    }
#else
    (void)opA;
#endif // BACKEND
}

void spmm_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA,
                      oneapi::mkl::transpose /*opB*/, const void *alpha,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                      oneapi::mkl::sparse::spmm_alg /*alg*/,
                      oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                      std::size_t &temp_buffer_size) {
    // TODO: Add support for external workspace once the close-source oneMKL backend supports it.
    check_valid_spmm(__FUNCTION__, queue, opA, A_view, A_handle, B_handle, C_handle, alpha, beta);
    temp_buffer_size = 0;
}

void spmm_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose /*opB*/,
                   const void *alpha, oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                   oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                   oneapi::mkl::sparse::spmm_alg alg,
                   oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                   sycl::buffer<std::uint8_t, 1> /*workspace*/) {
    check_valid_spmm(__FUNCTION__, queue, opA, A_view, A_handle, B_handle, C_handle, alpha, beta);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (!internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__FUNCTION__);
    }
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg) {
        return;
    }
    internal_A_handle->can_be_reset = false;
    // TODO: Add support for spmm_optimize once the close-source oneMKL backend supports it.
}

sycl::event spmm_optimize(sycl::queue &queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose /*opB*/, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                          oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                          oneapi::mkl::sparse::spmm_alg alg,
                          oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/, void * /*workspace*/,
                          const std::vector<sycl::event> &dependencies) {
    check_valid_spmm(__FUNCTION__, queue, opA, A_view, A_handle, B_handle, C_handle, alpha, beta);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__FUNCTION__);
    }
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    internal_A_handle->can_be_reset = false;
    // TODO: Add support for spmm_optimize once the close-source oneMKL backend supports it.
    return detail::collapse_dependencies(queue, dependencies);
}

template <typename T>
sycl::event internal_spmm(sycl::queue &queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void *alpha,
                          oneapi::mkl::sparse::matrix_view /*A_view*/,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                          oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                          oneapi::mkl::sparse::spmm_alg /*alg*/,
                          oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                          const std::vector<sycl::event> &dependencies) {
    T cast_alpha = *static_cast<const T *>(alpha);
    T cast_beta = *static_cast<const T *>(beta);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    internal_A_handle->can_be_reset = false;
    auto layout = B_handle->dense_layout;
    auto columns = C_handle->num_cols;
    auto ldb = B_handle->ld;
    auto ldc = C_handle->ld;
    if (internal_A_handle->all_use_buffer()) {
        oneapi::mkl::sparse::gemm(queue, layout, opA, opB, cast_alpha,
                                  internal_A_handle->backend_handle, B_handle->get_buffer<T>(),
                                  columns, ldb, cast_beta, C_handle->get_buffer<T>(), ldc);
        // Dependencies are not used for buffers
        return {};
    }
    else {
        return oneapi::mkl::sparse::gemm(queue, layout, opA, opB, cast_alpha,
                                         internal_A_handle->backend_handle,
                                         B_handle->get_usm_ptr<T>(), columns, ldb, cast_beta,
                                         C_handle->get_usm_ptr<T>(), ldc, dependencies);
    }
}

sycl::event spmm(sycl::queue &queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void *alpha, oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                 oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                 oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                 const std::vector<sycl::event> &dependencies) {
    check_valid_spmm(__FUNCTION__, queue, opA, A_view, A_handle, B_handle, C_handle, alpha, beta);
    auto value_type = detail::get_internal_handle(A_handle)->get_value_type();
    DISPATCH_MKL_OPERATION("spmm", value_type, internal_spmm, queue, opA, opB, alpha, A_view,
                           A_handle, B_handle, beta, C_handle, alg, spmm_descr, dependencies);
}

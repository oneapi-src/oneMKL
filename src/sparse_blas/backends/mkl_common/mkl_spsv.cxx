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
using spsv_descr = void *;

void init_spsv_descr(sycl::queue & /*queue*/, oneapi::mkl::sparse::spsv_descr_t *p_spsv_descr) {
    *p_spsv_descr = nullptr;
}

sycl::event release_spsv_descr(sycl::queue &queue, oneapi::mkl::sparse::spsv_descr_t /*spsv_descr*/,
                               const std::vector<sycl::event> &dependencies) {
    return detail::collapse_dependencies(queue, dependencies);
}

void check_valid_spsv(const std::string &function_name, oneapi::mkl::transpose opA,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                      bool is_alpha_host_accessible, oneapi::mkl::sparse::spsv_alg alg) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    detail::check_valid_spsv_common(function_name, A_view, internal_A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);

    if (alg == oneapi::mkl::sparse::spsv_alg::no_optimize_alg &&
        !internal_A_handle->has_matrix_property(oneapi::mkl::sparse::matrix_property::sorted)) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support `no_optimize_alg` unless A_handle has the property `matrix_property::sorted`.");
    }

#if BACKEND == gpu
    detail::data_type data_type = internal_A_handle->get_value_type();
    if ((data_type == detail::data_type::complex_fp32 ||
         data_type == detail::data_type::complex_fp64) &&
        opA == oneapi::mkl::transpose::conjtrans) {
        throw mkl::unimplemented("sparse_blas", function_name,
                                 "The backend does not support spsv using conjtrans.");
    }
#else
    (void)opA;
#endif // BACKEND
}

void spsv_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                      oneapi::mkl::sparse::spsv_alg alg,
                      oneapi::mkl::sparse::spsv_descr_t /*spsv_descr*/,
                      std::size_t &temp_buffer_size) {
    // TODO: Add support for external workspace once the close-source oneMKL backend supports it.
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     alg);
    temp_buffer_size = 0;
}

void spsv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                   oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                   oneapi::mkl::sparse::spsv_alg alg,
                   oneapi::mkl::sparse::spsv_descr_t /*spsv_descr*/,
                   sycl::buffer<std::uint8_t, 1> /*workspace*/) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     alg);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (!internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (alg == oneapi::mkl::sparse::spsv_alg::no_optimize_alg) {
        return;
    }
    internal_A_handle->can_be_reset = false;
    oneapi::mkl::sparse::optimize_trsv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                       internal_A_handle->backend_handle);
}

sycl::event spsv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                          oneapi::mkl::sparse::spsv_alg alg,
                          oneapi::mkl::sparse::spsv_descr_t /*spsv_descr*/, void * /*workspace*/,
                          const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     alg);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (alg == oneapi::mkl::sparse::spsv_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    internal_A_handle->can_be_reset = false;
    return oneapi::mkl::sparse::optimize_trsv(queue, A_view.uplo_view, opA, A_view.diag_view,
                                              internal_A_handle->backend_handle, dependencies);
}

template <typename T>
sycl::event internal_spsv(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                          oneapi::mkl::sparse::spsv_alg /*alg*/,
                          oneapi::mkl::sparse::spsv_descr_t /*spsv_descr*/,
                          const std::vector<sycl::event> &dependencies,
                          bool is_alpha_host_accessible) {
    T host_alpha =
        detail::get_scalar_on_host(queue, static_cast<const T *>(alpha), is_alpha_host_accessible);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    internal_A_handle->can_be_reset = false;
    if (internal_A_handle->all_use_buffer()) {
        oneapi::mkl::sparse::trsv(queue, A_view.uplo_view, opA, A_view.diag_view, host_alpha,
                                  internal_A_handle->backend_handle, x_handle->get_buffer<T>(),
                                  y_handle->get_buffer<T>());
        // Dependencies are not used for buffers
        return {};
    }
    else {
        return oneapi::mkl::sparse::trsv(queue, A_view.uplo_view, opA, A_view.diag_view, host_alpha,
                                         internal_A_handle->backend_handle,
                                         x_handle->get_usm_ptr<T>(), y_handle->get_usm_ptr<T>(),
                                         dependencies);
    }
}

sycl::event spsv(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                 oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                 oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                 oneapi::mkl::sparse::spsv_alg alg, oneapi::mkl::sparse::spsv_descr_t spsv_descr,
                 const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     alg);
    auto value_type = detail::get_internal_handle(A_handle)->get_value_type();
    DISPATCH_MKL_OPERATION("spsv", value_type, internal_spsv, queue, opA, alpha, A_view, A_handle,
                           x_handle, y_handle, alg, spsv_descr, dependencies,
                           is_alpha_host_accessible);
}

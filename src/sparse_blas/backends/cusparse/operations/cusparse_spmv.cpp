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

#include "oneapi/mkl/sparse_blas/detail/cusparse/onemkl_sparse_blas_cusparse.hpp"

#include "sparse_blas/backends/cusparse/cusparse_error.hpp"
#include "sparse_blas/backends/cusparse/cusparse_helper.hpp"
#include "sparse_blas/backends/cusparse/cusparse_task.hpp"
#include "sparse_blas/backends/cusparse/cusparse_handles.hpp"
#include "sparse_blas/common_op_verification.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/matrix_view_comparison.hpp"
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of the incomplete type
struct spmv_descr {
    detail::generic_container workspace;
    std::size_t temp_buffer_size = 0;
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    matrix_view last_optimized_A_view;
    matrix_handle_t last_optimized_A_handle;
    dense_vector_handle_t last_optimized_x_handle;
    dense_vector_handle_t last_optimized_y_handle;
    spmv_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::cusparse {

void init_spmv_descr(sycl::queue & /*queue*/, spmv_descr_t *p_spmv_descr) {
    *p_spmv_descr = new spmv_descr();
}

sycl::event release_spmv_descr(sycl::queue &queue, spmv_descr_t spmv_descr,
                               const std::vector<sycl::event> &dependencies) {
    return detail::submit_release(queue, spmv_descr, dependencies);
}

inline auto get_cuda_spmv_alg(spmv_alg alg) {
    switch (alg) {
        case spmv_alg::coo_alg1: return CUSPARSE_SPMV_COO_ALG1;
        case spmv_alg::coo_alg2: return CUSPARSE_SPMV_COO_ALG2;
        case spmv_alg::csr_alg1: return CUSPARSE_SPMV_CSR_ALG1;
        case spmv_alg::csr_alg2: return CUSPARSE_SPMV_CSR_ALG2;
        default: return CUSPARSE_SPMV_ALG_DEFAULT;
    }
}

void check_valid_spmv(const std::string &function_name, oneapi::mkl::transpose opA,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      dense_vector_handle_t y_handle, bool is_alpha_host_accessible,
                      bool is_beta_host_accessible) {
    detail::check_valid_spmv_common(function_name, opA, A_view, A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);
    check_valid_matrix_properties(function_name, A_handle);
    if (A_view.type_view != matrix_descr::general) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support spmv with a `type_view` other than `matrix_descr::general`.");
    }
}

void spmv_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                      matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      const void *beta, dense_vector_handle_t y_handle, spmv_alg alg,
                      spmv_descr_t spmv_descr, std::size_t &temp_buffer_size) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    auto functor = [=, &temp_buffer_size](CusparseScopedContextHandler &sc) {
        auto cu_handle = sc.get_handle(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_x = x_handle->backend_handle;
        auto cu_y = y_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        auto cu_op = get_cuda_operation(type, opA);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spmv_alg(alg);
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpMV_bufferSize(cu_handle, cu_op, alpha, cu_a, cu_x, beta, cu_y,
                                              cu_type, cu_alg, &temp_buffer_size);
        check_status(status, __func__);
    };
    auto event = dispatch_submit(__func__, queue, functor, A_handle, x_handle, y_handle);
    event.wait_and_throw();
    spmv_descr->temp_buffer_size = temp_buffer_size;
    spmv_descr->buffer_size_called = true;
}

inline void common_spmv_optimize(oneapi::mkl::transpose opA, bool is_alpha_host_accessible,
                                 matrix_view A_view, matrix_handle_t A_handle,
                                 dense_vector_handle_t x_handle, bool is_beta_host_accessible,
                                 dense_vector_handle_t y_handle, spmv_alg alg,
                                 spmv_descr_t spmv_descr) {
    check_valid_spmv("spmv_optimize", opA, A_view, A_handle, x_handle, y_handle,
                     is_alpha_host_accessible, is_beta_host_accessible);
    if (!spmv_descr->buffer_size_called) {
        throw mkl::uninitialized("sparse_blas", "spmv_optimize",
                                 "spmv_buffer_size must be called before spmv_optimize.");
    }
    spmv_descr->optimized_called = true;
    spmv_descr->last_optimized_opA = opA;
    spmv_descr->last_optimized_A_view = A_view;
    spmv_descr->last_optimized_A_handle = A_handle;
    spmv_descr->last_optimized_x_handle = x_handle;
    spmv_descr->last_optimized_y_handle = y_handle;
    spmv_descr->last_optimized_alg = alg;
}

#if CUSPARSE_VERSION >= 12300
// cusparseSpMV_preprocess was added in cuSPARSE 12.3.0.142 (CUDA 12.4)
void spmv_optimize_impl(cusparseHandle_t cu_handle, oneapi::mkl::transpose opA, const void *alpha,
                        matrix_handle_t A_handle, dense_vector_handle_t x_handle, const void *beta,
                        dense_vector_handle_t y_handle, spmv_alg alg, void *workspace_ptr,
                        bool is_alpha_host_accessible) {
    auto cu_a = A_handle->backend_handle;
    auto cu_x = x_handle->backend_handle;
    auto cu_y = y_handle->backend_handle;
    auto type = A_handle->value_container.data_type;
    auto cu_op = get_cuda_operation(type, opA);
    auto cu_type = get_cuda_value_type(type);
    auto cu_alg = get_cuda_spmv_alg(alg);
    set_pointer_mode(cu_handle, is_alpha_host_accessible);
    auto status = cusparseSpMV_preprocess(cu_handle, cu_op, alpha, cu_a, cu_x, beta, cu_y, cu_type,
                                          cu_alg, workspace_ptr);
    check_status(status, "spmv_optimize");
}
#endif

void spmv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                   matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                   const void *beta, dense_vector_handle_t y_handle, spmv_alg alg,
                   spmv_descr_t spmv_descr, sycl::buffer<std::uint8_t, 1> workspace) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmv_optimize(opA, is_alpha_host_accessible, A_view, A_handle, x_handle,
                         is_beta_host_accessible, y_handle, alg, spmv_descr);
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spmv_descr->workspace.set_buffer_untyped(workspace);
    if (alg == spmv_alg::no_optimize_alg) {
        return;
    }

#if CUSPARSE_VERSION < 12300
    // cusparseSpMV_preprocess was added in cuSPARSE 12.3.0.142 (CUDA 12.4)
    return;
#else
    if (spmv_descr->temp_buffer_size > 0) {
        auto functor = [=](CusparseScopedContextHandler &sc,
                           sycl::accessor<std::uint8_t> workspace_acc) {
            auto cu_handle = sc.get_handle(queue);
            auto workspace_ptr = sc.get_mem(workspace_acc);
            spmv_optimize_impl(cu_handle, opA, alpha, A_handle, x_handle, beta, y_handle, alg,
                               workspace_ptr, is_alpha_host_accessible);
        };

        // The accessor can only be created if the buffer size is greater than 0
        dispatch_submit(__func__, queue, functor, A_handle, workspace, x_handle, y_handle);
    }
    else {
        auto functor = [=](CusparseScopedContextHandler &sc) {
            auto cu_handle = sc.get_handle(queue);
            spmv_optimize_impl(cu_handle, opA, alpha, A_handle, x_handle, beta, y_handle, alg,
                               nullptr, is_alpha_host_accessible);
        };
        dispatch_submit(__func__, queue, functor, A_handle, x_handle, y_handle);
    }
#endif
}

sycl::event spmv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                          matrix_view A_view, matrix_handle_t A_handle,
                          dense_vector_handle_t x_handle, const void *beta,
                          dense_vector_handle_t y_handle, spmv_alg alg, spmv_descr_t spmv_descr,
                          void *workspace, const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmv_optimize(opA, is_alpha_host_accessible, A_view, A_handle, x_handle,
                         is_beta_host_accessible, y_handle, alg, spmv_descr);
    spmv_descr->workspace.usm_ptr = workspace;
    if (alg == spmv_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }

#if CUSPARSE_VERSION < 12300
    // cusparseSpMV_preprocess was added in cuSPARSE 12.3.0.142 (CUDA 12.4)
    return detail::collapse_dependencies(queue, dependencies);
#else
    auto functor = [=](CusparseScopedContextHandler &sc) {
        auto cu_handle = sc.get_handle(queue);
        spmv_optimize_impl(cu_handle, opA, alpha, A_handle, x_handle, beta, y_handle, alg,
                           workspace, is_alpha_host_accessible);
    };
    return dispatch_submit(__func__, queue, dependencies, functor, A_handle, x_handle, y_handle);
#endif
}

sycl::event spmv(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                 matrix_view A_view, matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                 const void *beta, dense_vector_handle_t y_handle, spmv_alg alg,
                 spmv_descr_t spmv_descr, const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmv(__func__, opA, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    if (A_handle->all_use_buffer() != spmv_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }

    if (!spmv_descr->optimized_called) {
        throw mkl::uninitialized("sparse_blas", __func__,
                                 "spmv_optimize must be called before spmv.");
    }
    CHECK_DESCR_MATCH(spmv_descr, opA, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, A_view, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, A_handle, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, x_handle, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, y_handle, "spmv_optimize");
    CHECK_DESCR_MATCH(spmv_descr, alg, "spmv_optimize");

    auto compute_functor = [=](CusparseScopedContextHandler &sc, void *workspace_ptr) {
        auto [cu_handle, cu_stream] = sc.get_handle_and_stream(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_x = x_handle->backend_handle;
        auto cu_y = y_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        auto cu_op = get_cuda_operation(type, opA);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spmv_alg(alg);
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpMV(cu_handle, cu_op, alpha, cu_a, cu_x, beta, cu_y, cu_type, cu_alg,
                                   workspace_ptr);
        check_status(status, __func__);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
#endif
    };
    if (A_handle->all_use_buffer() && spmv_descr->temp_buffer_size > 0) {
        // The accessor can only be created if the buffer size is greater than 0
        auto functor_buffer = [=](CusparseScopedContextHandler &sc,
                                  sycl::accessor<std::uint8_t> workspace_acc) {
            auto workspace_ptr = sc.get_mem(workspace_acc);
            compute_functor(sc, workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, functor_buffer, A_handle,
                                          spmv_descr->workspace.get_buffer<std::uint8_t>(),
                                          x_handle, y_handle);
    }
    else {
        // The same dispatch_submit can be used for USM or buffers if no
        // workspace accessor is needed, workspace_ptr will be a nullptr in the
        // latter case.
        auto workspace_ptr = spmv_descr->workspace.usm_ptr;
        auto functor_usm = [=](CusparseScopedContextHandler &sc) {
            compute_functor(sc, workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, dependencies, functor_usm, A_handle,
                                          x_handle, y_handle);
    }
}

} // namespace oneapi::mkl::sparse::cusparse

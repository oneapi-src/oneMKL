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
struct spmm_descr {
    detail::generic_container workspace;
    std::size_t temp_buffer_size = 0;
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    oneapi::mkl::transpose last_optimized_opB;
    oneapi::mkl::sparse::matrix_view last_optimized_A_view;
    oneapi::mkl::sparse::matrix_handle_t last_optimized_A_handle;
    oneapi::mkl::sparse::dense_matrix_handle_t last_optimized_B_handle;
    oneapi::mkl::sparse::dense_matrix_handle_t last_optimized_C_handle;
    oneapi::mkl::sparse::spmm_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::cusparse {

void init_spmm_descr(sycl::queue& /*queue*/, spmm_descr_t* p_spmm_descr) {
    *p_spmm_descr = new spmm_descr();
}

sycl::event release_spmm_descr(sycl::queue& queue, spmm_descr_t spmm_descr,
                               const std::vector<sycl::event>& dependencies) {
    return detail::submit_release(queue, spmm_descr, dependencies);
}

inline auto get_cuda_spmm_alg(spmm_alg alg) {
    switch (alg) {
        case spmm_alg::coo_alg1: return CUSPARSE_SPMM_COO_ALG1;
        case spmm_alg::coo_alg2: return CUSPARSE_SPMM_COO_ALG2;
        case spmm_alg::coo_alg3: return CUSPARSE_SPMM_COO_ALG3;
        case spmm_alg::coo_alg4: return CUSPARSE_SPMM_COO_ALG4;
        case spmm_alg::csr_alg1: return CUSPARSE_SPMM_CSR_ALG1;
        case spmm_alg::csr_alg2: return CUSPARSE_SPMM_CSR_ALG2;
        case spmm_alg::csr_alg3: return CUSPARSE_SPMM_CSR_ALG3;
        default: return CUSPARSE_SPMM_ALG_DEFAULT;
    }
}

inline void fallback_alg_if_needed(oneapi::mkl::sparse::spmm_alg& alg, oneapi::mkl::transpose opA,
                                   oneapi::mkl::transpose opB) {
    if (alg == oneapi::mkl::sparse::spmm_alg::csr_alg3 &&
        (opA != oneapi::mkl::transpose::nontrans || opB == oneapi::mkl::transpose::conjtrans)) {
        // Avoid warnings printed on std::cerr
        alg = oneapi::mkl::sparse::spmm_alg::default_alg;
    }
}

void spmm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                      const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                      oneapi::mkl::sparse::spmm_alg alg,
                      oneapi::mkl::sparse::spmm_descr_t spmm_descr, std::size_t& temp_buffer_size) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    detail::check_valid_spmm_common(__func__, A_view, A_handle, B_handle, C_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);
    fallback_alg_if_needed(alg, opA, opB);
    auto functor = [=, &temp_buffer_size](CusparseScopedContextHandler& sc) {
        auto cu_handle = sc.get_handle(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_b = B_handle->backend_handle;
        auto cu_c = C_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        auto cu_op_a = get_cuda_operation(type, opA);
        auto cu_op_b = get_cuda_operation(type, opB);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spmm_alg(alg);
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpMM_bufferSize(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta,
                                              cu_c, cu_type, cu_alg, &temp_buffer_size);
        check_status(status, __func__);
    };
    auto event = dispatch_submit(__func__, queue, functor, A_handle, B_handle, C_handle);
    event.wait_and_throw();
    spmm_descr->temp_buffer_size = temp_buffer_size;
    spmm_descr->buffer_size_called = true;
}

inline void common_spmm_optimize(
    oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, bool is_alpha_host_accessible,
    oneapi::mkl::sparse::matrix_view A_view, oneapi::mkl::sparse::matrix_handle_t A_handle,
    oneapi::mkl::sparse::dense_matrix_handle_t B_handle, bool is_beta_host_accessible,
    oneapi::mkl::sparse::dense_matrix_handle_t C_handle, oneapi::mkl::sparse::spmm_alg alg,
    oneapi::mkl::sparse::spmm_descr_t spmm_descr) {
    detail::check_valid_spmm_common(__func__, A_view, A_handle, B_handle, C_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);
    if (!spmm_descr->buffer_size_called) {
        throw mkl::uninitialized("sparse_blas", __func__,
                                 "spmm_buffer_size must be called before spmm_optimize.");
    }
    spmm_descr->optimized_called = true;
    spmm_descr->last_optimized_opA = opA;
    spmm_descr->last_optimized_opB = opB;
    spmm_descr->last_optimized_A_view = A_view;
    spmm_descr->last_optimized_A_handle = A_handle;
    spmm_descr->last_optimized_B_handle = B_handle;
    spmm_descr->last_optimized_C_handle = C_handle;
    spmm_descr->last_optimized_alg = alg;
}

void spmm_optimize_impl(cusparseHandle_t cu_handle, oneapi::mkl::transpose opA,
                        oneapi::mkl::transpose opB, const void* alpha,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                        oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                        oneapi::mkl::sparse::spmm_alg alg, void* workspace_ptr,
                        bool is_alpha_host_accessible) {
    auto cu_a = A_handle->backend_handle;
    auto cu_b = B_handle->backend_handle;
    auto cu_c = C_handle->backend_handle;
    auto type = A_handle->value_container.data_type;
    auto cu_op_a = get_cuda_operation(type, opA);
    auto cu_op_b = get_cuda_operation(type, opB);
    auto cu_type = get_cuda_value_type(type);
    auto cu_alg = get_cuda_spmm_alg(alg);
    set_pointer_mode(cu_handle, is_alpha_host_accessible);
    auto status = cusparseSpMM_preprocess(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta,
                                          cu_c, cu_type, cu_alg, workspace_ptr);
    check_status(status, "optimize_spmm");
}

void spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                   const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                   oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                   oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                   sycl::buffer<std::uint8_t, 1> workspace) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmm_optimize(opA, opB, is_alpha_host_accessible, A_view, A_handle, B_handle,
                         is_beta_host_accessible, C_handle, alg, spmm_descr);
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spmm_descr->workspace.set_buffer_untyped(workspace);
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg || workspace.size() == 0) {
        // cusparseSpMM_preprocess cannot be called if the workspace is empty
        return;
    }
    fallback_alg_if_needed(alg, opA, opB);
    auto functor = [=](CusparseScopedContextHandler& sc,
                       sycl::accessor<std::uint8_t> workspace_acc) {
        auto cu_handle = sc.get_handle(queue);
        auto workspace_ptr = sc.get_mem(workspace_acc);
        spmm_optimize_impl(cu_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle, alg,
                           workspace_ptr, is_alpha_host_accessible);
    };

    sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc(workspace);
    dispatch_submit(__func__, queue, functor, A_handle, workspace_placeholder_acc, B_handle,
                    C_handle);
}

sycl::event spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void* alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                          oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                          oneapi::mkl::sparse::spmm_alg alg,
                          oneapi::mkl::sparse::spmm_descr_t spmm_descr, void* workspace,
                          const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    common_spmm_optimize(opA, opB, is_alpha_host_accessible, A_view, A_handle, B_handle,
                         is_beta_host_accessible, C_handle, alg, spmm_descr);
    spmm_descr->workspace.usm_ptr = workspace;
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg || workspace == nullptr) {
        // cusparseSpMM_preprocess cannot be called if the workspace is empty
        return detail::collapse_dependencies(queue, dependencies);
    }
    fallback_alg_if_needed(alg, opA, opB);
    auto functor = [=](CusparseScopedContextHandler& sc) {
        auto cu_handle = sc.get_handle(queue);
        spmm_optimize_impl(cu_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle, alg,
                           workspace, is_alpha_host_accessible);
    };

    return dispatch_submit_native_ext(__func__, queue, dependencies, functor, A_handle, B_handle,
                                      C_handle);
}

sycl::event spmm(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                 oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                 oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                 const std::vector<sycl::event>& dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    detail::check_valid_spmm_common(__func__, A_view, A_handle, B_handle, C_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);
    if (A_handle->all_use_buffer() != spmm_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }

    if (!spmm_descr->optimized_called) {
        throw mkl::uninitialized("sparse_blas", __func__,
                                 "spmm_optimize must be called before spmm.");
    }
    CHECK_DESCR_MATCH(spmm_descr, opA, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, opB, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, A_view, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, A_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, B_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, C_handle, "spmm_optimize");
    CHECK_DESCR_MATCH(spmm_descr, alg, "spmm_optimize");

    fallback_alg_if_needed(alg, opA, opB);
    auto compute_functor = [=](CusparseScopedContextHandler& sc, void* workspace_ptr) {
        auto [cu_handle, cu_stream] = sc.get_handle_and_stream(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_b = B_handle->backend_handle;
        auto cu_c = C_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        auto cu_op_a = get_cuda_operation(type, opA);
        auto cu_op_b = get_cuda_operation(type, opB);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spmm_alg(alg);
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpMM(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta, cu_c,
                                   cu_type, cu_alg, workspace_ptr);
        check_status(status, __func__);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
#endif
    };
    if (A_handle->all_use_buffer() && spmm_descr->temp_buffer_size > 0) {
        // The accessor can only be bound to the cgh if the buffer size is
        // greater than 0
        auto functor_buffer = [=](CusparseScopedContextHandler& sc,
                                  sycl::accessor<std::uint8_t> workspace_acc) {
            auto workspace_ptr = sc.get_mem(workspace_acc);
            compute_functor(sc, workspace_ptr);
        };
        sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc(
            spmm_descr->workspace.get_buffer<std::uint8_t>());
        return dispatch_submit_native_ext(__func__, queue, functor_buffer, A_handle,
                                          workspace_placeholder_acc, B_handle, C_handle);
    }
    else {
        // The same dispatch_submit can be used for USM or buffers if no
        // workspace accessor is needed, workspace_ptr will be a nullptr in the
        // latter case.
        auto workspace_ptr = spmm_descr->workspace.usm_ptr;
        auto functor_usm = [=](CusparseScopedContextHandler& sc) {
            compute_functor(sc, workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, dependencies, functor_usm, A_handle,
                                          B_handle, C_handle);
    }
}

} // namespace oneapi::mkl::sparse::cusparse

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

#include "oneapi/mkl/sparse_blas/detail/rocsparse/onemkl_sparse_blas_rocsparse.hpp"

#include "sparse_blas/backends/rocsparse/rocsparse_error.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_helper.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_task.hpp"
#include "sparse_blas/backends/rocsparse/rocsparse_handles.hpp"
#include "sparse_blas/common_op_verification.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/matrix_view_comparison.hpp"
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of the incomplete type
struct spsv_descr {
    detail::generic_container workspace;
    std::size_t temp_buffer_size = 0;
    bool buffer_size_called = false;
    bool optimized_called = false;
    oneapi::mkl::transpose last_optimized_opA;
    oneapi::mkl::sparse::matrix_view last_optimized_A_view;
    oneapi::mkl::sparse::matrix_handle_t last_optimized_A_handle;
    oneapi::mkl::sparse::dense_vector_handle_t last_optimized_x_handle;
    oneapi::mkl::sparse::dense_vector_handle_t last_optimized_y_handle;
    oneapi::mkl::sparse::spsv_alg last_optimized_alg;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::rocsparse {

void init_spsv_descr(sycl::queue & /*queue*/, spsv_descr_t *p_spsv_descr) {
    *p_spsv_descr = new spsv_descr();
}

sycl::event release_spsv_descr(sycl::queue &queue, spsv_descr_t spsv_descr,
                               const std::vector<sycl::event> &dependencies) {
    return detail::submit_release(queue, spsv_descr, dependencies);
}

inline auto get_roc_spsv_alg(spsv_alg /*alg*/) {
    return rocsparse_spsv_alg_default;
}

void check_valid_spsv(const std::string &function_name, matrix_view A_view,
                      matrix_handle_t A_handle, dense_vector_handle_t x_handle,
                      dense_vector_handle_t y_handle, bool is_alpha_host_accessible) {
    detail::check_valid_spsv_common(function_name, A_view, A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);
    A_handle->check_valid_handle(function_name);
}

void spsv_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                      oneapi::mkl::sparse::spsv_alg alg,
                      oneapi::mkl::sparse::spsv_descr_t spsv_descr, std::size_t &temp_buffer_size) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    check_valid_spsv(__func__, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible);
    auto functor = [=, &temp_buffer_size](RocsparseScopedContextHandler &sc) {
        auto [roc_handle, roc_stream] = sc.get_handle_and_stream(queue);
        auto roc_a = A_handle->backend_handle;
        auto roc_x = x_handle->backend_handle;
        auto roc_y = y_handle->backend_handle;
        set_matrix_attributes(__func__, roc_a, A_view);
        auto roc_op = get_roc_operation(opA);
        auto roc_type = get_roc_value_type(A_handle->value_container.data_type);
        auto roc_alg = get_roc_spsv_alg(alg);
        set_pointer_mode(roc_handle, is_alpha_host_accessible);
        auto status =
            rocsparse_spsv(roc_handle, roc_op, alpha, roc_a, roc_x, roc_y, roc_type, roc_alg,
                           rocsparse_spsv_stage_buffer_size, &temp_buffer_size, nullptr);
        check_status(status, __func__);
        HIP_ERROR_FUNC(hipStreamSynchronize, roc_stream);
    };
    auto event = dispatch_submit(__func__, queue, functor, A_handle, x_handle, y_handle);
    event.wait_and_throw();
    spsv_descr->temp_buffer_size = temp_buffer_size;
    spsv_descr->buffer_size_called = true;
}

inline void common_spsv_optimize(oneapi::mkl::transpose opA, bool is_alpha_host_accessible,
                                 oneapi::mkl::sparse::matrix_view A_view,
                                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                                 oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                                 oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                                 oneapi::mkl::sparse::spsv_alg alg,
                                 oneapi::mkl::sparse::spsv_descr_t spsv_descr) {
    check_valid_spsv("spsv_optimize", A_view, A_handle, x_handle, y_handle,
                     is_alpha_host_accessible);
    if (!spsv_descr->buffer_size_called) {
        throw mkl::uninitialized(
            "sparse_blas", "spsv_optimize",
            "spsv_buffer_size must be called with the same arguments before spsv_optimize.");
    }
    spsv_descr->optimized_called = true;
    spsv_descr->last_optimized_opA = opA;
    spsv_descr->last_optimized_A_view = A_view;
    spsv_descr->last_optimized_A_handle = A_handle;
    spsv_descr->last_optimized_x_handle = x_handle;
    spsv_descr->last_optimized_y_handle = y_handle;
    spsv_descr->last_optimized_alg = alg;
}

void spsv_optimize_impl(rocsparse_handle roc_handle, oneapi::mkl::transpose opA, const void *alpha,
                        oneapi::mkl::sparse::matrix_view A_view,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                        oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                        oneapi::mkl::sparse::spsv_alg alg, std::size_t buffer_size,
                        void *workspace_ptr, bool is_alpha_host_accessible) {
    auto roc_a = A_handle->backend_handle;
    auto roc_x = x_handle->backend_handle;
    auto roc_y = y_handle->backend_handle;
    set_matrix_attributes("spsv_optimize", roc_a, A_view);
    auto roc_op = get_roc_operation(opA);
    auto roc_type = get_roc_value_type(A_handle->value_container.data_type);
    auto roc_alg = get_roc_spsv_alg(alg);
    set_pointer_mode(roc_handle, is_alpha_host_accessible);
    // rocsparse_spsv_stage_preprocess stage is blocking
    auto status = rocsparse_spsv(roc_handle, roc_op, alpha, roc_a, roc_x, roc_y, roc_type, roc_alg,
                                 rocsparse_spsv_stage_preprocess, &buffer_size, workspace_ptr);
    check_status(status, "spsv_optimize");
}

void spsv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                   oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                   oneapi::mkl::sparse::spsv_alg alg, oneapi::mkl::sparse::spsv_descr_t spsv_descr,
                   sycl::buffer<std::uint8_t, 1> workspace) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    A_handle->check_valid_handle(__func__);
    common_spsv_optimize(opA, is_alpha_host_accessible, A_view, A_handle, x_handle, y_handle, alg,
                         spsv_descr);
    // Ignore spsv_alg::no_optimize_alg as this step is mandatory for rocSPARSE
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spsv_descr->workspace.set_buffer_untyped(workspace);
    std::size_t buffer_size = spsv_descr->temp_buffer_size;
    // The accessor can only be created if the buffer size is greater than 0
    if (buffer_size > 0) {
        auto functor = [=](RocsparseScopedContextHandler &sc,
                           sycl::accessor<std::uint8_t> workspace_acc) {
            auto roc_handle = sc.get_handle(queue);
            auto workspace_ptr = sc.get_mem(workspace_acc);
            spsv_optimize_impl(roc_handle, opA, alpha, A_view, A_handle, x_handle, y_handle, alg,
                               buffer_size, workspace_ptr, is_alpha_host_accessible);
        };

        dispatch_submit(__func__, queue, functor, A_handle, workspace, x_handle, y_handle);
    }
    else {
        auto functor = [=](RocsparseScopedContextHandler &sc) {
            auto roc_handle = sc.get_handle(queue);
            spsv_optimize_impl(roc_handle, opA, alpha, A_view, A_handle, x_handle, y_handle, alg,
                               buffer_size, nullptr, is_alpha_host_accessible);
        };

        dispatch_submit(__func__, queue, functor, A_handle, x_handle, y_handle);
    }
}

sycl::event spsv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                          oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                          oneapi::mkl::sparse::spsv_alg alg,
                          oneapi::mkl::sparse::spsv_descr_t spsv_descr, void *workspace,
                          const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    A_handle->check_valid_handle(__func__);
    common_spsv_optimize(opA, is_alpha_host_accessible, A_view, A_handle, x_handle, y_handle, alg,
                         spsv_descr);
    spsv_descr->workspace.usm_ptr = workspace;
    // Ignore spsv_alg::no_optimize_alg as this step is mandatory for rocSPARSE
    std::size_t buffer_size = spsv_descr->temp_buffer_size;
    auto functor = [=](RocsparseScopedContextHandler &sc) {
        auto roc_handle = sc.get_handle(queue);
        spsv_optimize_impl(roc_handle, opA, alpha, A_view, A_handle, x_handle, y_handle, alg,
                           buffer_size, workspace, is_alpha_host_accessible);
    };

    return dispatch_submit(__func__, queue, dependencies, functor, A_handle, x_handle, y_handle);
}

sycl::event spsv(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                 oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                 oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                 oneapi::mkl::sparse::spsv_alg alg, oneapi::mkl::sparse::spsv_descr_t spsv_descr,
                 const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    if (A_handle->all_use_buffer() != spsv_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    check_valid_spsv(__func__, A_view, A_handle, x_handle, y_handle, is_alpha_host_accessible);

    if (!spsv_descr->optimized_called) {
        throw mkl::uninitialized(
            "sparse_blas", __func__,
            "spsv_optimize must be called with the same arguments before spsv.");
    }
    CHECK_DESCR_MATCH(spsv_descr, opA, "spsv_optimize");
    CHECK_DESCR_MATCH(spsv_descr, A_view, "spsv_optimize");
    CHECK_DESCR_MATCH(spsv_descr, A_handle, "spsv_optimize");
    CHECK_DESCR_MATCH(spsv_descr, x_handle, "spsv_optimize");
    CHECK_DESCR_MATCH(spsv_descr, y_handle, "spsv_optimize");
    CHECK_DESCR_MATCH(spsv_descr, alg, "spsv_optimize");

    A_handle->mark_used();
    auto &buffer_size = spsv_descr->temp_buffer_size;
    auto compute_functor = [=, &buffer_size](RocsparseScopedContextHandler &sc,
                                             void *workspace_ptr) {
        auto [roc_handle, roc_stream] = sc.get_handle_and_stream(queue);
        auto roc_a = A_handle->backend_handle;
        auto roc_x = x_handle->backend_handle;
        auto roc_y = y_handle->backend_handle;
        set_matrix_attributes(__func__, roc_a, A_view);
        auto roc_op = get_roc_operation(opA);
        auto roc_type = get_roc_value_type(A_handle->value_container.data_type);
        auto roc_alg = get_roc_spsv_alg(alg);
        set_pointer_mode(roc_handle, is_alpha_host_accessible);
        auto status =
            rocsparse_spsv(roc_handle, roc_op, alpha, roc_a, roc_x, roc_y, roc_type, roc_alg,
                           rocsparse_spsv_stage_compute, &buffer_size, workspace_ptr);
        check_status(status, __func__);
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
        HIP_ERROR_FUNC(hipStreamSynchronize, roc_stream);
#endif
    };
    // The accessor can only be created if the buffer size is greater than 0
    if (A_handle->all_use_buffer() && buffer_size > 0) {
        auto functor_buffer = [=](RocsparseScopedContextHandler &sc,
                                  sycl::accessor<std::uint8_t> workspace_acc) {
            auto workspace_ptr = sc.get_mem(workspace_acc);
            compute_functor(sc, workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, functor_buffer, A_handle,
                                          spsv_descr->workspace.get_buffer<std::uint8_t>(),
                                          x_handle, y_handle);
    }
    else {
        // The same dispatch_submit can be used for USM or buffers if no
        // workspace accessor is needed.
        // workspace_ptr will be a nullptr in the latter case.
        auto workspace_ptr = spsv_descr->workspace.usm_ptr;
        auto functor_usm = [=](RocsparseScopedContextHandler &sc) {
            compute_functor(sc, workspace_ptr);
        };
        return dispatch_submit_native_ext(__func__, queue, dependencies, functor_usm, A_handle,
                                          x_handle, y_handle);
    }
}

} // namespace oneapi::mkl::sparse::rocsparse

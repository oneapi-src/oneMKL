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
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of the incomplete type
struct spsv_descr {
    cusparseSpSVDescr_t cu_descr;
    detail::generic_container workspace;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::cusparse {

void init_spsv_descr(sycl::queue & /*queue*/, spsv_descr_t *p_spsv_descr) {
    *p_spsv_descr = new spsv_descr();
    CUSPARSE_ERR_FUNC(cusparseSpSV_createDescr, &(*p_spsv_descr)->cu_descr);
}

sycl::event release_spsv_descr(sycl::queue &queue, spsv_descr_t spsv_descr,
                               const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {
            CUSPARSE_ERR_FUNC(cusparseSpSV_destroyDescr, spsv_descr->cu_descr);
            delete spsv_descr;
        });
    });
}

inline auto get_cuda_spsv_alg(spsv_alg /*alg*/) {
    return CUSPARSE_SPSV_ALG_DEFAULT;
}

void spsv_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                      oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                      oneapi::mkl::sparse::spsv_alg alg,
                      oneapi::mkl::sparse::spsv_descr_t spsv_descr, std::size_t &temp_buffer_size) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    detail::check_valid_spsv_common(__func__, A_view, A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);
    auto functor = [=, &temp_buffer_size](CusparseScopedContextHandler &sc) {
        auto cu_handle = sc.get_handle(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_x = x_handle->backend_handle;
        auto cu_y = y_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        set_matrix_attributes(__func__, cu_a, A_view);
        auto cu_op = get_cuda_operation(type, opA);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spsv_alg(alg);
        auto cu_descr = spsv_descr->cu_descr;
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpSV_bufferSize(cu_handle, cu_op, alpha, cu_a, cu_x, cu_y, cu_type,
                                              cu_alg, cu_descr, &temp_buffer_size);
        check_status(status, __func__);
    };
    auto event = dispatch_submit(__func__, queue, functor, A_handle, x_handle, y_handle);
    event.wait_and_throw();
}

void spsv_optimize_impl(cusparseHandle_t cu_handle, oneapi::mkl::transpose opA, const void *alpha,
                        oneapi::mkl::sparse::matrix_view A_view,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                        oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                        oneapi::mkl::sparse::spsv_alg alg,
                        oneapi::mkl::sparse::spsv_descr_t spsv_descr, void *workspace_ptr,
                        bool is_alpha_host_accessible) {
    auto cu_a = A_handle->backend_handle;
    auto cu_x = x_handle->backend_handle;
    auto cu_y = y_handle->backend_handle;
    auto type = A_handle->value_container.data_type;
    set_matrix_attributes("optimize_spsv", cu_a, A_view);
    auto cu_op = get_cuda_operation(type, opA);
    auto cu_type = get_cuda_value_type(type);
    auto cu_alg = get_cuda_spsv_alg(alg);
    auto cu_descr = spsv_descr->cu_descr;
    set_pointer_mode(cu_handle, is_alpha_host_accessible);
    auto status = cusparseSpSV_analysis(cu_handle, cu_op, alpha, cu_a, cu_x, cu_y, cu_type, cu_alg,
                                        cu_descr, workspace_ptr);
    check_status(status, "optimize_spsv");
}

void spsv_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, const void *alpha,
                   oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                   oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                   oneapi::mkl::sparse::spsv_alg alg, oneapi::mkl::sparse::spsv_descr_t spsv_descr,
                   sycl::buffer<std::uint8_t, 1> workspace) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    detail::check_valid_spsv_common(__func__, A_view, A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    // Ignore spsv_alg::no_optimize_alg as this step is mandatory for cuSPARSE
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spsv_descr->workspace.set_buffer_untyped(workspace);

    if (workspace.size() > 0) {
        auto functor = [=](CusparseScopedContextHandler &sc,
                           sycl::accessor<std::uint8_t> workspace_acc) {
            auto cu_handle = sc.get_handle(queue);
            auto workspace_ptr = sc.get_mem(workspace_acc);
            spsv_optimize_impl(cu_handle, opA, alpha, A_view, A_handle, x_handle, y_handle, alg,
                               spsv_descr, workspace_ptr, is_alpha_host_accessible);
        };

        // The accessor can only be bound to the cgh if the buffer size is
        // greater than 0
        sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc(workspace);
        auto event = dispatch_submit(__func__, queue, functor, A_handle, workspace_placeholder_acc,
                                     x_handle, y_handle);
        event.wait_and_throw();
    }
    else {
        auto functor = [=](CusparseScopedContextHandler &sc) {
            auto cu_handle = sc.get_handle(queue);
            spsv_optimize_impl(cu_handle, opA, alpha, A_view, A_handle, x_handle, y_handle, alg,
                               spsv_descr, nullptr, is_alpha_host_accessible);
        };

        auto event = dispatch_submit(__func__, queue, functor, A_handle, x_handle, y_handle);
        event.wait_and_throw();
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
    detail::check_valid_spsv_common(__func__, A_view, A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    // Ignore spsv_alg::no_optimize_alg as this step is mandatory for cuSPARSE
    auto functor = [=](CusparseScopedContextHandler &sc) {
        auto cu_handle = sc.get_handle(queue);
        spsv_optimize_impl(cu_handle, opA, alpha, A_view, A_handle, x_handle, y_handle, alg,
                           spsv_descr, workspace, is_alpha_host_accessible);
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
    detail::check_valid_spsv_common(__func__, A_view, A_handle, x_handle, y_handle,
                                    is_alpha_host_accessible);
    if (A_handle->all_use_buffer() != spsv_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    auto functor = [=](CusparseScopedContextHandler &sc) {
        auto [cu_handle, cu_stream] = sc.get_handle_and_stream(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_x = x_handle->backend_handle;
        auto cu_y = y_handle->backend_handle;
        auto type = A_handle->value_container.data_type;
        set_matrix_attributes(__func__, cu_a, A_view);
        auto cu_op = get_cuda_operation(type, opA);
        auto cu_type = get_cuda_value_type(type);
        auto cu_alg = get_cuda_spsv_alg(alg);
        auto cu_descr = spsv_descr->cu_descr;
        set_pointer_mode(cu_handle, is_alpha_host_accessible);
        auto status = cusparseSpSV_solve(cu_handle, cu_op, alpha, cu_a, cu_x, cu_y, cu_type, cu_alg,
                                         cu_descr);
        check_status(status, __func__);
        CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
    };
    auto event =
        dispatch_submit(__func__, queue, dependencies, functor, A_handle, x_handle, y_handle);
    return event;
}

} // namespace oneapi::mkl::sparse::cusparse

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

// Dense vector
template <typename fpType>
void init_dense_vector(sycl::queue & /*queue*/,
                       oneapi::mkl::sparse::dense_vector_handle_t *p_dvhandle, std::int64_t size,
                       sycl::buffer<fpType, 1> val) {
    *p_dvhandle = new oneapi::mkl::sparse::dense_vector_handle(val, size);
}

template <typename fpType>
void init_dense_vector(sycl::queue & /*queue*/,
                       oneapi::mkl::sparse::dense_vector_handle_t *p_dvhandle, std::int64_t size,
                       fpType *val) {
    *p_dvhandle = new oneapi::mkl::sparse::dense_vector_handle(val, size);
}

template <typename fpType>
void set_dense_vector_data(sycl::queue & /*queue*/,
                           oneapi::mkl::sparse::dense_vector_handle_t dvhandle, std::int64_t size,
                           sycl::buffer<fpType, 1> val) {
    dvhandle->size = size;
    dvhandle->set_buffer(val);
}

template <typename fpType>
void set_dense_vector_data(sycl::queue & /*queue*/,
                           oneapi::mkl::sparse::dense_vector_handle_t dvhandle, std::int64_t size,
                           fpType *val) {
    dvhandle->size = size;
    dvhandle->set_usm_ptr(val);
}

#define INSTANTIATE_DENSE_VECTOR_FUNCS(FP_TYPE, FP_SUFFIX)                                        \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> init_dense_vector<FP_TYPE>(     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t * p_dvhandle,             \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> val);                                         \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> init_dense_vector<FP_TYPE>(     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t * p_dvhandle,             \
        std::int64_t size, FP_TYPE *val);                                                         \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> set_dense_vector_data<FP_TYPE>( \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t dvhandle,                 \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> val);                                         \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> set_dense_vector_data<FP_TYPE>( \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t dvhandle,                 \
        std::int64_t size, FP_TYPE *val)
FOR_EACH_FP_TYPE(INSTANTIATE_DENSE_VECTOR_FUNCS);
#undef INSTANTIATE_DENSE_VECTOR_FUNCS

sycl::event release_dense_vector(sycl::queue &queue,
                                 oneapi::mkl::sparse::dense_vector_handle_t dvhandle,
                                 const std::vector<sycl::event> &dependencies) {
    return detail::submit_release(queue, dvhandle, dependencies);
}

// Dense matrix
template <typename fpType>
void init_dense_matrix(sycl::queue & /*queue*/,
                       oneapi::mkl::sparse::dense_matrix_handle_t *p_dmhandle,
                       std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                       oneapi::mkl::layout dense_layout, sycl::buffer<fpType, 1> val) {
    *p_dmhandle =
        new oneapi::mkl::sparse::dense_matrix_handle(val, num_rows, num_cols, ld, dense_layout);
}

template <typename fpType>
void init_dense_matrix(sycl::queue & /*queue*/,
                       oneapi::mkl::sparse::dense_matrix_handle_t *p_dmhandle,
                       std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                       oneapi::mkl::layout dense_layout, fpType *val) {
    *p_dmhandle =
        new oneapi::mkl::sparse::dense_matrix_handle(val, num_rows, num_cols, ld, dense_layout);
}

template <typename fpType>
void set_dense_matrix_data(sycl::queue & /*queue*/,
                           oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                           oneapi::mkl::layout dense_layout, sycl::buffer<fpType, 1> val) {
    dmhandle->num_rows = num_rows;
    dmhandle->num_cols = num_cols;
    dmhandle->ld = ld;
    dmhandle->dense_layout = dense_layout;
    dmhandle->set_buffer(val);
}

template <typename fpType>
void set_dense_matrix_data(sycl::queue & /*queue*/,
                           oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,
                           std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                           oneapi::mkl::layout dense_layout, fpType *val) {
    dmhandle->num_rows = num_rows;
    dmhandle->num_cols = num_cols;
    dmhandle->ld = ld;
    dmhandle->dense_layout = dense_layout;
    dmhandle->set_usm_ptr(val);
}

#define INSTANTIATE_DENSE_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX)                                        \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> init_dense_matrix<FP_TYPE>(     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t * p_dmhandle,             \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                            \
        oneapi::mkl::layout dense_layout, sycl::buffer<FP_TYPE, 1> val);                          \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> init_dense_matrix<FP_TYPE>(     \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t * p_dmhandle,             \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                            \
        oneapi::mkl::layout dense_layout, FP_TYPE *val);                                          \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> set_dense_matrix_data<FP_TYPE>( \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,                 \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                            \
        oneapi::mkl::layout dense_layout, sycl::buffer<FP_TYPE, 1> val);                          \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> set_dense_matrix_data<FP_TYPE>( \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,                 \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                            \
        oneapi::mkl::layout dense_layout, FP_TYPE *val)
FOR_EACH_FP_TYPE(INSTANTIATE_DENSE_MATRIX_FUNCS);
#undef INSTANTIATE_DENSE_MATRIX_FUNCS

sycl::event release_dense_matrix(sycl::queue &queue,
                                 oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,
                                 const std::vector<sycl::event> &dependencies) {
    return detail::submit_release(queue, dmhandle, dependencies);
}

// COO matrix
template <typename fpType, typename intType>
void init_coo_matrix(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                     oneapi::mkl::index_base index, sycl::buffer<intType, 1> row_ind,
                     sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    oneapi::mkl::sparse::matrix_handle_t mkl_handle;
    oneapi::mkl::sparse::init_matrix_handle(&mkl_handle);
    auto internal_smhandle = new detail::sparse_matrix_handle(mkl_handle, row_ind, col_ind, val);
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    oneapi::mkl::sparse::set_coo_data(queue, mkl_handle, static_cast<intType>(num_rows),
                                      static_cast<intType>(num_cols), static_cast<intType>(nnz),
                                      index, internal_smhandle->row_container.get_buffer<intType>(),
                                      internal_smhandle->col_container.get_buffer<intType>(),
                                      internal_smhandle->value_container.get_buffer<fpType>());
    *p_smhandle = reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void init_coo_matrix(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                     oneapi::mkl::index_base index, intType *row_ind, intType *col_ind,
                     fpType *val) {
    oneapi::mkl::sparse::matrix_handle_t mkl_handle;
    oneapi::mkl::sparse::init_matrix_handle(&mkl_handle);
    auto internal_smhandle = new detail::sparse_matrix_handle(mkl_handle, row_ind, col_ind, val);
    auto event = oneapi::mkl::sparse::set_coo_data(
        queue, mkl_handle, static_cast<intType>(num_rows), static_cast<intType>(num_cols),
        static_cast<intType>(nnz), index, row_ind, col_ind, val);
    event.wait_and_throw();
    *p_smhandle = reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void check_can_reset(const std::string &function_name,
                     detail::sparse_matrix_handle *internal_smhandle) {
    if (internal_smhandle->get_value_type() != detail::get_data_type<fpType>()) {
        throw oneapi::mkl::invalid_argument(
            "sparse_blas", function_name,
            "Incompatible data types expected " +
                data_type_to_str(internal_smhandle->get_value_type()) + " but got " +
                data_type_to_str(detail::get_data_type<fpType>()));
    }
    if (internal_smhandle->get_int_type() != detail::get_data_type<intType>()) {
        throw oneapi::mkl::invalid_argument(
            "sparse_blas", function_name,
            "Incompatible data types expected " +
                data_type_to_str(internal_smhandle->get_int_type()) + " but got " +
                data_type_to_str(detail::get_data_type<intType>()));
    }
    if (!internal_smhandle->can_be_reset) {
        throw mkl::unimplemented(
            "sparse_blas/mkl", function_name,
            "Reseting the matrix handle's data after it was used in a computation is not supported.");
    }
}

template <typename fpType, typename intType>
void set_coo_matrix_data(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                         oneapi::mkl::index_base index, sycl::buffer<intType, 1> row_ind,
                         sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    check_can_reset<fpType, intType>(__FUNCTION__, internal_smhandle);
    internal_smhandle->row_container.set_buffer(row_ind);
    internal_smhandle->col_container.set_buffer(col_ind);
    internal_smhandle->value_container.set_buffer(val);
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    oneapi::mkl::sparse::set_coo_data(queue, internal_smhandle->backend_handle,
                                      static_cast<intType>(num_rows),
                                      static_cast<intType>(num_cols), static_cast<intType>(nnz),
                                      index, internal_smhandle->row_container.get_buffer<intType>(),
                                      internal_smhandle->col_container.get_buffer<intType>(),
                                      internal_smhandle->value_container.get_buffer<fpType>());
}

template <typename fpType, typename intType>
void set_coo_matrix_data(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                         oneapi::mkl::index_base index, intType *row_ind, intType *col_ind,
                         fpType *val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    check_can_reset<fpType, intType>(__FUNCTION__, internal_smhandle);
    internal_smhandle->row_container.set_usm_ptr(row_ind);
    internal_smhandle->col_container.set_usm_ptr(col_ind);
    internal_smhandle->value_container.set_usm_ptr(val);
    auto event = oneapi::mkl::sparse::set_coo_data(
        queue, internal_smhandle->backend_handle, static_cast<intType>(num_rows),
        static_cast<intType>(num_cols), static_cast<intType>(nnz), index, row_ind, col_ind, val);
    event.wait_and_throw();
}

#define INSTANTIATE_COO_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                     \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    init_coo_matrix<FP_TYPE, INT_TYPE>(                                                            \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> row_ind,                          \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                          \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    init_coo_matrix<FP_TYPE, INT_TYPE>(                                                            \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, INT_TYPE *row_ind, INT_TYPE *col_ind, FP_TYPE *val);        \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    set_coo_matrix_data<FP_TYPE, INT_TYPE>(                                                        \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,                    \
        sycl::buffer<INT_TYPE, 1> row_ind, sycl::buffer<INT_TYPE, 1> col_ind,                      \
        sycl::buffer<FP_TYPE, 1> val);                                                             \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    set_coo_matrix_data<FP_TYPE, INT_TYPE>(                                                        \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index, INT_TYPE *row_ind, \
        INT_TYPE *col_ind, FP_TYPE *val)
FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_COO_MATRIX_FUNCS);
#undef INSTANTIATE_COO_MATRIX_FUNCS

// CSR matrix
template <typename fpType, typename intType>
void init_csr_matrix(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t /*nnz*/,
                     oneapi::mkl::index_base index, sycl::buffer<intType, 1> row_ptr,
                     sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    oneapi::mkl::sparse::matrix_handle_t mkl_handle;
    oneapi::mkl::sparse::init_matrix_handle(&mkl_handle);
    auto internal_smhandle = new detail::sparse_matrix_handle(mkl_handle, row_ptr, col_ind, val);
    // The backend deduces nnz from row_ptr.
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    oneapi::mkl::sparse::set_csr_data(queue, mkl_handle, static_cast<intType>(num_rows),
                                      static_cast<intType>(num_cols), index,
                                      internal_smhandle->row_container.get_buffer<intType>(),
                                      internal_smhandle->col_container.get_buffer<intType>(),
                                      internal_smhandle->value_container.get_buffer<fpType>());
    *p_smhandle = reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void init_csr_matrix(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *p_smhandle,
                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t /*nnz*/,
                     oneapi::mkl::index_base index, intType *row_ptr, intType *col_ind,
                     fpType *val) {
    oneapi::mkl::sparse::matrix_handle_t mkl_handle;
    oneapi::mkl::sparse::init_matrix_handle(&mkl_handle);
    auto internal_smhandle = new detail::sparse_matrix_handle(mkl_handle, row_ptr, col_ind, val);
    // The backend deduces nnz from row_ptr.
    auto event = oneapi::mkl::sparse::set_csr_data(
        queue, mkl_handle, static_cast<intType>(num_rows), static_cast<intType>(num_cols), index,
        row_ptr, col_ind, val);
    event.wait_and_throw();
    *p_smhandle = reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t>(internal_smhandle);
}

template <typename fpType, typename intType>
void set_csr_matrix_data(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t /*nnz*/,
                         oneapi::mkl::index_base index, sycl::buffer<intType, 1> row_ptr,
                         sycl::buffer<intType, 1> col_ind, sycl::buffer<fpType, 1> val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    check_can_reset<fpType, intType>(__FUNCTION__, internal_smhandle);
    internal_smhandle->row_container.set_buffer(row_ptr);
    internal_smhandle->col_container.set_buffer(col_ind);
    internal_smhandle->value_container.set_buffer(val);
    // The backend deduces nnz from row_ptr.
    // The backend handle must use the buffers from the internal handle as they will be kept alive until the handle is released.
    oneapi::mkl::sparse::set_csr_data(queue, internal_smhandle->backend_handle,
                                      static_cast<intType>(num_rows),
                                      static_cast<intType>(num_cols), index,
                                      internal_smhandle->row_container.get_buffer<intType>(),
                                      internal_smhandle->col_container.get_buffer<intType>(),
                                      internal_smhandle->value_container.get_buffer<fpType>());
}

template <typename fpType, typename intType>
void set_csr_matrix_data(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t smhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t /*nnz*/,
                         oneapi::mkl::index_base index, intType *row_ptr, intType *col_ind,
                         fpType *val) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    check_can_reset<fpType, intType>(__FUNCTION__, internal_smhandle);
    internal_smhandle->row_container.set_usm_ptr(row_ptr);
    internal_smhandle->col_container.set_usm_ptr(col_ind);
    internal_smhandle->value_container.set_usm_ptr(val);
    // The backend deduces nnz from row_ptr.
    auto event = oneapi::mkl::sparse::set_csr_data(
        queue, internal_smhandle->backend_handle, static_cast<intType>(num_rows),
        static_cast<intType>(num_cols), index, row_ptr, col_ind, val);
    event.wait_and_throw();
}

#define INSTANTIATE_CSR_MATRIX_FUNCS(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                     \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    init_csr_matrix<FP_TYPE, INT_TYPE>(                                                            \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> row_ptr,                          \
        sycl::buffer<INT_TYPE, 1> col_ind, sycl::buffer<FP_TYPE, 1> val);                          \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    init_csr_matrix<FP_TYPE, INT_TYPE>(                                                            \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,                    \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                            \
        oneapi::mkl::index_base index, INT_TYPE *row_ptr, INT_TYPE *col_ind, FP_TYPE *val);        \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    set_csr_matrix_data<FP_TYPE, INT_TYPE>(                                                        \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index,                    \
        sycl::buffer<INT_TYPE, 1> row_ptr, sycl::buffer<INT_TYPE, 1> col_ind,                      \
        sycl::buffer<FP_TYPE, 1> val);                                                             \
    template std::enable_if_t<detail::are_fp_int_supported_v<FP_TYPE, INT_TYPE>>                   \
    set_csr_matrix_data<FP_TYPE, INT_TYPE>(                                                        \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t smhandle, std::int64_t num_rows, \
        std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index, INT_TYPE *row_ptr, \
        INT_TYPE *col_ind, FP_TYPE *val)
FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_CSR_MATRIX_FUNCS);
#undef INSTANTIATE_CSR_MATRIX_FUNCS

// Common sparse matrix functions
sycl::event release_sparse_matrix(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t smhandle,
                                  const std::vector<sycl::event> &dependencies) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    // Asynchronously release the backend's handle followed by the internal handle.
    auto event = oneapi::mkl::sparse::release_matrix_handle(
        queue, &internal_smhandle->backend_handle, dependencies);
    return detail::submit_release(queue, internal_smhandle, event);
}

bool set_matrix_property(sycl::queue & /*queue*/, oneapi::mkl::sparse::matrix_handle_t smhandle,
                         oneapi::mkl::sparse::matrix_property property) {
    auto internal_smhandle = detail::get_internal_handle(smhandle);
    // Store the matrix property internally for better error checking
    internal_smhandle->set_matrix_property(property);
    // Set the matrix property on the backend handle
    // Backend and oneMKL interface types for the property don't match
    switch (property) {
        case oneapi::mkl::sparse::matrix_property::symmetric:
            oneapi::mkl::sparse::set_matrix_property(internal_smhandle->backend_handle,
                                                     oneapi::mkl::sparse::property::symmetric);
            return true;
        case oneapi::mkl::sparse::matrix_property::sorted:
            oneapi::mkl::sparse::set_matrix_property(internal_smhandle->backend_handle,
                                                     oneapi::mkl::sparse::property::sorted);
            return true;
        default: return false;
    }
}

/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/*
*
*  Content:
*       This example demonstrates use of DPCPP API oneapi::mkl::sparse::spmv
*       using unified shared memory to perform general sparse matrix-vector
*       multiplication on a INTEL CPU SYCL device.
*
*       y = alpha * op(A) * x + beta * y
*
*       where op() is defined by one of
*
*           oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       This example demonstrates only single precision (float) data type for
*       spmv matrix data
*
*
*******************************************************************************/

// stl includes
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"

#include "example_helper.hpp"

//
// Main example for Sparse Matrix-Vector Multiply consisting of
// initialization of A matrix, x and y vectors as well as
// scalars alpha and beta.  Then the product
//
// y = alpha * op(A) * x + beta * y
//
// is performed and finally the results are post processed.
//
template <typename fpType, typename intType, typename selectorType>
int run_sparse_matrix_vector_multiply_example(const selectorType &selector) {
    auto queue = selector.get_queue();

    // Matrix data size
    intType size = 4;
    intType nrows = size * size * size;

    // Set scalar fpType values
    fpType alpha = set_fp_value(fpType(1.0));
    fpType beta = set_fp_value(fpType(0.0));

    intType *ia, *ja;
    fpType *a, *x, *y, *z;
    std::size_t sizea = static_cast<std::size_t>(27 * nrows);
    std::size_t sizeja = static_cast<std::size_t>(27 * nrows);
    std::size_t sizeia = static_cast<std::size_t>(nrows + 1);
    std::size_t sizevec = static_cast<std::size_t>(nrows);

    ia = (intType *)sycl::malloc_shared(sizeia * sizeof(intType), queue);
    ja = (intType *)sycl::malloc_shared(sizeja * sizeof(intType), queue);
    a = (fpType *)sycl::malloc_shared(sizea * sizeof(fpType), queue);
    x = (fpType *)sycl::malloc_shared(sizevec * sizeof(fpType), queue);
    y = (fpType *)sycl::malloc_shared(sizevec * sizeof(fpType), queue);
    z = (fpType *)sycl::malloc_shared(sizevec * sizeof(fpType), queue);

    if (!ia || !ja || !a || !x || !y || !z) {
        throw std::runtime_error("Failed to allocate USM memory");
    }

    intType nnz = generate_sparse_matrix<fpType, intType>(size, ia, ja, a);

    // Init vectors x and y
    for (int i = 0; i < nrows; i++) {
        x[i] = set_fp_value(fpType(1.0));
        y[i] = set_fp_value(fpType(0.0));
        z[i] = set_fp_value(fpType(0.0));
    }

    std::vector<intType *> int_ptr_vec;
    int_ptr_vec.push_back(ia);
    int_ptr_vec.push_back(ja);
    std::vector<fpType *> fp_ptr_vec;
    fp_ptr_vec.push_back(a);
    fp_ptr_vec.push_back(x);
    fp_ptr_vec.push_back(y);
    fp_ptr_vec.push_back(z);

    //
    // Execute Matrix Multiply
    //

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::sparse::spmv_alg alg = oneapi::mkl::sparse::spmv_alg::default_alg;
    oneapi::mkl::sparse::matrix_view A_view;

    std::cout << "\n\t\tsparse::spmv parameters:\n";
    std::cout << "\t\t\ttransA = "
              << (transA == oneapi::mkl::transpose::nontrans
                      ? "nontrans"
                      : (transA == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tnrows = " << nrows << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;

    // Create and initialize handle for a Sparse Matrix in CSR format
    oneapi::mkl::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::mkl::sparse::init_csr_matrix(selector, &A_handle, nrows, nrows, nnz,
                                         oneapi::mkl::index_base::zero, ia, ja, a);

    // Create and initialize dense vector handles
    oneapi::mkl::sparse::dense_vector_handle_t x_handle = nullptr;
    oneapi::mkl::sparse::dense_vector_handle_t y_handle = nullptr;
    oneapi::mkl::sparse::init_dense_vector(selector, &x_handle, sizevec, x);
    oneapi::mkl::sparse::init_dense_vector(selector, &y_handle, sizevec, y);

    // Create operation descriptor
    oneapi::mkl::sparse::spmv_descr_t descr = nullptr;
    oneapi::mkl::sparse::init_spmv_descr(selector, &descr);

    // Allocate external workspace
    std::size_t workspace_size = 0;
    oneapi::mkl::sparse::spmv_buffer_size(selector, transA, &alpha, A_view, A_handle, x_handle,
                                          &beta, y_handle, alg, descr, workspace_size);
    void *workspace = sycl::malloc_device(workspace_size, queue);

    // Optimize spmv
    auto ev_opt =
        oneapi::mkl::sparse::spmv_optimize(selector, transA, &alpha, A_view, A_handle, x_handle,
                                           &beta, y_handle, alg, descr, workspace);

    // Run spmv
    auto ev_spmv = oneapi::mkl::sparse::spmv(selector, transA, &alpha, A_view, A_handle, x_handle,
                                             &beta, y_handle, alg, descr, { ev_opt });

    // Release handles and descriptor
    std::vector<sycl::event> release_events;
    release_events.push_back(
        oneapi::mkl::sparse::release_dense_vector(selector, x_handle, { ev_spmv }));
    release_events.push_back(
        oneapi::mkl::sparse::release_dense_vector(selector, y_handle, { ev_spmv }));
    release_events.push_back(
        oneapi::mkl::sparse::release_sparse_matrix(selector, A_handle, { ev_spmv }));
    release_events.push_back(oneapi::mkl::sparse::release_spmv_descr(selector, descr, { ev_spmv }));
    for (auto event : release_events) {
        event.wait_and_throw();
    }

    //
    // Post Processing
    //

    fpType *res = y;
    const bool isConj = (transA == oneapi::mkl::transpose::conjtrans);
    for (intType row = 0; row < nrows; row++) {
        z[row] *= beta;
    }
    for (intType row = 0; row < nrows; row++) {
        fpType tmp = alpha * x[row];
        for (intType i = ia[row]; i < ia[row + 1]; i++) {
            if constexpr (is_complex<fpType>()) {
                z[ja[i]] += tmp * (isConj ? std::conj(a[i]) : a[i]);
            }
            else {
                z[ja[i]] += tmp * a[i];
            }
        }
    }

    bool good = true;
    for (intType row = 0; row < nrows; row++) {
        good &= check_result(res[row], z[row], nrows, row);
    }

    std::cout << "\n\t\t sparse::spmv example " << (good ? "passed" : "failed") << "\n\tFinished"
              << std::endl;

    free_vec(fp_ptr_vec, queue);
    free_vec(int_ptr_vec, queue);

    if (!good)
        return 1;

    return 0;
}

//
// Description of example setup, apis used and supported floating point type
// precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# Sparse Matrix-Vector Multiply Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# y = alpha * op(A) * x + beta * y" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A is a sparse matrix in CSR format, x and y are "
                 "dense vectors"
              << std::endl;
    std::cout << "# and alpha, beta are floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   sparse::spmv" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Running on both Intel CPU and Nvidia GPU devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example
//
int main(int /*argc*/, char ** /*argv*/) {
    print_example_banner();

    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL "
                             "exception during sparse::spmv:\n"
                          << e.what() << std::endl;
            }
        }
    };

    try {
        sycl::queue cpu_queue(sycl::cpu_selector_v, exception_handler);
        sycl::queue gpu_queue(sycl::gpu_selector_v, exception_handler);
        unsigned int vendor_id = gpu_queue.get_info<sycl::info::device::vendor_id>();
        if (vendor_id != NVIDIA_ID) {
            std::cerr << "FAILED: NVIDIA GPU device not found" << std::endl;
            return 1;
        }
        oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> cpu_selector{ cpu_queue };
        oneapi::mkl::backend_selector<oneapi::mkl::backend::cusparse> gpu_selector{ gpu_queue };

        std::cout << "Running Sparse BLAS SPMV USM example on:" << std::endl;
        std::cout << "\tCPU device: " << cpu_queue.get_info<sycl::info::device::name>()
                  << std::endl;
        std::cout << "\tGPU device: " << gpu_queue.get_info<sycl::info::device::name>()
                  << std::endl;
        std::cout << "Running with single precision real data type:" << std::endl;

        run_sparse_matrix_vector_multiply_example<float, std::int32_t>(cpu_selector);
        run_sparse_matrix_vector_multiply_example<float, std::int32_t>(gpu_selector);
        std::cout << "Sparse BLAS SPMV USM example ran OK on MKLCPU and CUSPARSE." << std::endl;
    }
    catch (sycl::exception const &e) {
        std::cerr << "Caught synchronous SYCL exception during Sparse SPMV:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const &e) {
        std::cerr << "Caught std::exception during Sparse SPMV:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}

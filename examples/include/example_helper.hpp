/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef __EXAMPLE_HELPER_HPP__
#define __EXAMPLE_HELPER_HPP__

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <vector>

//
// helpers for initializing templated scalar data type values.
//
template <typename fp>
fp set_fp_value(fp arg1, fp /*arg2*/ = fp(0.0)) {
    return arg1;
}

//
// print a 2x2 block of data from matrix M using the sycl accessor
//
// M = [ M_00, M_01 ...
//     [ M_10, M_11 ...
//     [ ...
//
template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name) {
    std::cout << std::endl;
    std::cout << "\t\t\t" << M_name << " = [ " << M[0 * ldM + 0] << ", " << M[1 * ldM + 0]
              << ", ...\n";
    std::cout << "\t\t\t    [ " << M[0 * ldM + 1] << ", " << M[1 * ldM + 1] << ", ...\n";
    std::cout << "\t\t\t    [ "
              << "...\n";
    std::cout << std::endl;
}

template <typename fp>
fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}

template <typename vec>
void rand_matrix(vec &M, oneapi::mkl::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M.at(i + j * ld) = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M.at(j + i * ld) = rand_scalar<fp>();
    }
}

template <typename fp, typename intType>
intType generate_sparse_matrix(const intType nx, intType *ia, intType *ja, fp *a,
                               const intType index = 0) {
    intType nz = nx, ny = nx;
    intType nnz = 0;
    intType current_row;

    ia[0] = index;

    for (intType iz = 0; iz < nz; iz++) {
        for (intType iy = 0; iy < ny; iy++) {
            for (intType ix = 0; ix < nx; ix++) {
                current_row = iz * nx * ny + iy * nx + ix;

                for (intType sz = -1; sz <= 1; sz++) {
                    if (iz + sz > -1 && iz + sz < nz) {
                        for (intType sy = -1; sy <= 1; sy++) {
                            if (iy + sy > -1 && iy + sy < ny) {
                                for (intType sx = -1; sx <= 1; sx++) {
                                    if (ix + sx > -1 && ix + sx < nx) {
                                        intType current_column =
                                            current_row + sz * nx * ny + sy * nx + sx;
                                        ja[nnz] = current_column + index;
                                        if (current_column == current_row) {
                                            a[nnz++] = set_fp_value(fp(26.0));
                                        }
                                        else {
                                            a[nnz++] = set_fp_value(fp(-1.0));
                                        }
                                    } // end
                                    // x
                                    // bounds
                                    // test
                                } // end sx loop
                            } // end y bounds test
                        } // end sy loop
                    } // end z bounds test
                } // end sz loop
                ia[current_row + 1] = nnz + index;

            } // end ix loop
        } // end iy loop
    } // end iz loop
    return nnz;
}

template <typename T>
void free_vec(std::vector<T *> &ptr_vec, sycl::queue queue) {
    for (auto ptr : ptr_vec) {
        sycl::free(ptr, queue);
    }
    ptr_vec.clear();
}

#endif //__EXAMPLE_HELPER_HPP__

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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

void commit_f(descriptor<precision::SINGLE, domain::REAL> &desc, sycl::queue &queue) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void commit_c(descriptor<precision::SINGLE, domain::COMPLEX> &desc, sycl::queue &queue) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void commit_d(descriptor<precision::DOUBLE, domain::REAL> &desc, sycl::queue &queue) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void commit_z(descriptor<precision::DOUBLE, domain::COMPLEX> &desc, sycl::queue &queue) {
    throw std::runtime_error("Not implemented for mklcpu");
}

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

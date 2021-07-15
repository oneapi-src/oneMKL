/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <CL/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"
#include "mklcpu_common.hpp"
#include "fp16.hpp"
#include "oneapi/mkl/blas/detail/mklcpu/onemkl_blas_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace mklcpu {
namespace column_major {

#define CBLASMAJOR CblasColMajor
#define MKLMAJOR   MKL_COL_MAJOR
#include "mklcpu_level3.cxx"
#undef CBLASMAJOR
#undef MKLMAJOR

} // namespace column_major
namespace row_major {

#define CBLASMAJOR CblasRowMajor
#define MKLMAJOR   MKL_ROW_MAJOR
#include "mklcpu_level3.cxx"
#undef CBLASMAJOR
#undef MKLMAJOR

} // namespace row_major
} // namespace mklcpu
} // namespace blas
} // namespace mkl
} // namespace oneapi

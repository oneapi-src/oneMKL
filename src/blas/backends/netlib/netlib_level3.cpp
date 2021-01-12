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

#include "netlib_common.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/netlib/onemkl_blas_netlib.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace netlib {
namespace column_major {

#define MAJOR CblasColMajor
#define COLUMN_MAJOR
#include "netlib_level3.cxx"
#undef MAJOR
#undef COLUMN_MAJOR

} // namespace column_major
namespace row_major {

#define MAJOR CblasRowMajor
#define ROW_MAJOR
#include "netlib_level3.cxx"
#undef MAJOR
#undef ROW_MAJOR

} // namespace row_major
} // namespace netlib
} // namespace blas
} // namespace mkl
} // namespace oneapi

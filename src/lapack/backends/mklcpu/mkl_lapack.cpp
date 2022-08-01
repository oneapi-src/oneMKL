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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/lapack/types.hpp"
#include "oneapi/mkl/lapack/detail/mklcpu/onemkl_lapack_mklcpu.hpp"
#include "../mkl_common/mkl_lapack_backend.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace mklcpu {

#include "../mkl_common/mkl_lapack.cxx"

} // namespace mklcpu
} // namespace lapack
} // namespace mkl
} // namespace oneapi

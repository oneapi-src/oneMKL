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

#ifndef _ONEMKL_SPARSE_BLAS_TYPES_HPP_
#define _ONEMKL_SPARSE_BLAS_TYPES_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <vector>

#include "oneapi/mkl/types.hpp"
#include "detail/helper_types.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

using matrix_handle_t = detail::matrix_handle*;

} // namespace sparse
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_SPARSE_BLAS_TYPES_HPP_
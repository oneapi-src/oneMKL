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

#ifndef _ONEMATH_BLAS_NETLIB_HPP_
#define _ONEMATH_BLAS_NETLIB_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <complex>
#include <cstdint>

#include "oneapi/math/types.hpp"

#include "oneapi/math/detail/export.hpp"

namespace oneapi {
namespace mkl {

using oneapi::mkl::transpose;
using oneapi::mkl::uplo;
using oneapi::mkl::side;
using oneapi::mkl::diag;
using oneapi::mkl::offset;

namespace blas {
namespace netlib {
namespace column_major {

#include "oneapi/math/blas/detail/onemath_blas_backends.hxx"

} //namespace column_major
namespace row_major {

#include "oneapi/math/blas/detail/onemath_blas_backends.hxx"

} //namespace row_major
} //namespace netlib
} //namespace blas
} //namespace mkl
} //namespace oneapi

#endif //_ONEMATH_BLAS_NETLIB_HPP_

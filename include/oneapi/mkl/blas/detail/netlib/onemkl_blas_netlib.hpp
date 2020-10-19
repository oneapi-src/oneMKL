/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef _ONEMKL_BLAS_NETLIB_HPP_
#define _ONEMKL_BLAS_NETLIB_HPP_

#include <CL/sycl.hpp>

#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/detail/export.hpp"

namespace oneapi {
namespace mkl {

using oneapi::mkl::transpose;
using oneapi::mkl::uplo;
using oneapi::mkl::side;
using oneapi::mkl::diag;
using oneapi::mkl::offset;

namespace netlib {
namespace column_major {

#include "onemkl_blas_netlib.hxx"

} //namespace column_major
namespace row_major {

#include "onemkl_blas_netlib.hxx"

} //namespace row_major
} //namespace netlib
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BLAS_NETLIB_HPP_

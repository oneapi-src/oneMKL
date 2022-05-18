/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
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
#ifndef _ONEMKL_BLAS_ROCBLAS_HPP_
#define _ONEMKL_BLAS_ROCBLAS_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>
#include <string>
#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/detail/config.hpp"

namespace oneapi {
namespace mkl {
using oneapi::mkl::diag;
using oneapi::mkl::offset;
using oneapi::mkl::side;
using oneapi::mkl::transpose;
using oneapi::mkl::uplo;
namespace blas {
namespace rocblas {
namespace column_major {

#include "onemkl_blas_rocblas.hxx"

} //namespace column_major
namespace row_major {

#include "onemkl_blas_rocblas.hxx"

} //namespace row_major
} //namespace rocblas
} //namespace blas
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BLAS_ROCBLAS_HPP_

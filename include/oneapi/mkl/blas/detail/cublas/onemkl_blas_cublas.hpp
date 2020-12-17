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
#ifndef _ONEMKL_BLAS_CUBLAS_HPP_
#define _ONEMKL_BLAS_CUBLAS_HPP_
#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>
#include <string>
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
using oneapi::mkl::diag;
using oneapi::mkl::offset;
using oneapi::mkl::side;
using oneapi::mkl::transpose;
using oneapi::mkl::uplo;
namespace blas {
namespace cublas {
namespace column_major {

#include "onemkl_blas_cublas.hxx"

} //namespace column_major
namespace row_major {

#include "onemkl_blas_cublas.hxx"

} //namespace row_major
} //namespace cublas
} //namespace blas
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BLAS_CUBLAS_HPP_

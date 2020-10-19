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

#ifndef _DETAIL_CUBLAS_BLAS_CT_HPP_
#define _DETAIL_CUBLAS_BLAS_CT_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"
#include "oneapi/mkl/blas/detail/blas_ct_backends.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace column_major {

#define MAJOR column_major
#include "blas_ct.hxx"
#undef MAJOR

} //namespace column_major
namespace row_major {

#define MAJOR row_major
#include "blas_ct.hxx"
#undef MAJOR

} //namespace row_major
} //namespace blas
} //namespace mkl
} //namespace oneapi

#endif //_DETAIL_CUBLAS_BLAS_CT_HPP_

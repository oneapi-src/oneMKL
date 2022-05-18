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
#ifndef _ONEMKL_LAPACK_CUSOLVER_HPP_
#define _ONEMKL_LAPACK_CUSOLVER_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>
#include <string>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/detail/export.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace cusolver {

#include "onemkl_lapack_cusolver.hxx"

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_LAPACK_CUSOLVER_HPP_

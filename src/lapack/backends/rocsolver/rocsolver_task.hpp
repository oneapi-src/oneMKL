/***************************************************************************
*  Copyright 2020-2022 Intel Corporation 
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

#ifndef _MKL_LAPACK_ROCSOLVER_TASK_HPP_
#define _MKL_LAPACK_ROCSOLVER_TASK_HPP_
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <rocsolver.h>
#include <complex>
#include <CL/sycl.hpp>
#include "oneapi/mkl/types.hpp"
#include "rocsolver_scope_handle.hpp"
#include <CL/sycl/detail/pi.hpp>
namespace oneapi {
namespace mkl {
namespace lapack {
namespace rocsolver {

template <typename H, typename F>
static inline void host_task_internal(H &cgh, sycl::queue queue, F f) {
    cgh.host_task([f, queue](cl::sycl::interop_handle ih) {
        auto sc = RocsolverScopedContextHandler(queue, ih);
        f(sc);
    });
}

template <typename H, typename F>
static inline void onemkl_rocsolver_host_task(H &cgh, sycl::queue queue, F f) {
    (void)host_task_internal(cgh, queue, f);
}

} // namespace rocsolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
#endif // _MKL_LAPACK_ROCSOLVER_TASK_HPP_

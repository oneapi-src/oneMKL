/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "oneapi/mkl/detail/backends.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"

#include "dft/backends/mklcpu/mklcpu_helpers.hpp"
#include "mkl_service.h"
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

template <dft::detail::precision prec, dft::detail::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                        sycl::queue& sycl_queue) {
    return new detail::commit_derived_impl<prec, dom>(sycl_queue, desc.get_values());
}

template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi

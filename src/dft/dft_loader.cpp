/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/mkl/dft/detail/dft_loader.hpp"

#include "function_table_initializer.hpp"
#include "dft/function_table.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

static oneapi::mkl::detail::table_initializer<mkl::domain::dft, dft_function_table_t>
    function_tables;

commit_impl* create_commit(oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                                        oneapi::mkl::dft::domain::COMPLEX>& desc) {
    auto libkey = get_device_id(desc.get_queue());
    return function_tables[libkey].create_commit_sycl_fz(desc);
}

commit_impl* create_commit(oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                                        oneapi::mkl::dft::domain::COMPLEX>& desc) {
    auto libkey = get_device_id(desc.get_queue());
    return function_tables[libkey].create_commit_sycl_dz(desc);
}

commit_impl* create_commit(oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                                        oneapi::mkl::dft::domain::REAL>& desc) {
    auto libkey = get_device_id(desc.get_queue());
    return function_tables[libkey].create_commit_sycl_fr(desc);
}

commit_impl* create_commit(oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                                        oneapi::mkl::dft::domain::REAL>& desc) {
    auto libkey = get_device_id(desc.get_queue());
    return function_tables[libkey].create_commit_sycl_dr(desc);
}

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

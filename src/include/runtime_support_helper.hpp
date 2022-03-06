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

#ifndef _ONEMKL_RUNTIME_SUPPORT_HELPER_HPP_
#define _ONEMKL_RUNTIME_SUPPORT_HELPER_HPP_

#include <CL/sycl.hpp>
#include <type_traits>

// Utility function to verify that a given set of types is supported by the
// device compiler combination
template <typename verify_type, typename T, typename... Ts>
bool verify_support(sycl::queue q, sycl::aspect aspect) {
    bool has_aspect = q.get_device().has(aspect);
    if constexpr (sizeof...(Ts) > 0) {
        if constexpr (std::is_same_v<verify_type, T>) {
            return has_aspect && verify_support<verify_type, Ts...>(q, aspect);
        }
        else {
            return true && verify_support<verify_type, Ts...>(q, aspect);
        }
    }
    else {
        if constexpr (std::is_same_v<verify_type, T>) {
            return has_aspect;
        }
        else {
            return true;
        }
    }
}

#endif //_ONEMKL_RUNTIME_SUPPORT_HELPER_HPP_
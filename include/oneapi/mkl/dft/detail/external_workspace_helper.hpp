/*******************************************************************************
* Copyright Codeplay Software Ltd
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

#ifndef _ONEMKL_DFT_EXTERNAL_WORKSPACE_HELPER_HPP_
#define _ONEMKL_DFT_EXTERNAL_WORKSPACE_HELPER_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/detail/commit_impl.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

template <precision prec, domain dom>
class external_workspace_helper {
public:
    using commit_impl_t = commit_impl<prec, dom>;
    using scalar_t = typename commit_impl_t::scalar_type;

private:
    // Enum to represent whatever the workspace was set as.
    enum class ext_workspace_type {
        not_set,
        usm,
        buffer,
    };

    // Is an external workspace required?
    bool m_ext_workspace_rqd;

    // Set workspace type, with optional workspaces.
    ext_workspace_type m_workspace_type;

    // Minimum size of workspace in bytes. -1 indicates not set.
    std::int64_t m_workspace_bytes_rqd;

    std::optional<sycl::buffer<scalar_t>> m_workspace_buffer;

public:
    /** Constructor.
     *  @param ext_workspace_rqd True if WORKSPACE_PLACEMENT is set to WORKSPACE_EXTERNAL.
    */
    constexpr external_workspace_helper(bool ext_workspace_rqd)
            : m_ext_workspace_rqd(ext_workspace_rqd),
              m_workspace_type(ext_workspace_type::not_set),
              m_workspace_bytes_rqd(-1) {}

    /** Get the required workspace bytes for the backend's external workspace.
     *  @param committed_desc The backend's native descriptor.
    */
    std::int64_t get_rqd_workspace_bytes(commit_impl_t& committed_desc) {
        if (m_workspace_bytes_rqd == -1) {
            m_workspace_bytes_rqd = committed_desc.get_workspace_external_bytes_impl();
        }
        return m_workspace_bytes_rqd;
    }

    /** Throw according to spec for setting the workspace. USM version.
     *  @param committed_desc The backend's native descriptor.
     *  @param usm_workspace A USM allocation for the workspace. Assumed to be sufficeintly large.
    */
    void set_workspace_throw(commit_impl_t& committed_desc, scalar_t* usm_workspace) {
        if (get_rqd_workspace_bytes(committed_desc) > 0 && usm_workspace == nullptr) {
            throw mkl::invalid_argument("DFT", "set_workspace",
                                        "Backend expected a non-null workspace pointer.");
        }
        m_ext_workspace_rqd = true;
        m_workspace_type = ext_workspace_type::usm;
    }

    /** Throw according to spec for setting the workspace. Buffer version.
     *  @param committed_desc The backend's native descriptor.
     *  @param buffer_workspace A buffer for the workspace
    */
    void set_workspace_throw(commit_impl_t& committed_desc,
                             sycl::buffer<scalar_t>& buffer_workspace) {
        if (static_cast<std::size_t>(get_rqd_workspace_bytes(committed_desc)) / sizeof(scalar_t) >
            buffer_workspace.size()) {
            throw mkl::invalid_argument("DFT", "set_workspace", "Provided workspace is too small");
            return;
        }
        if (buffer_workspace.is_sub_buffer()) {
            throw mkl::invalid_argument("DFT", "set_workspace",
                                        "Cannot use sub-buffers for workspace");
            return;
        }
        m_ext_workspace_rqd = true;
        m_workspace_type = ext_workspace_type::buffer;
        m_workspace_buffer = buffer_workspace;
    }

    template <typename FirstArgT, typename... ArgTs>
    void compute_call_throw(const char* function_name) const {
        constexpr bool is_pointer = std::is_pointer_v<std::remove_reference_t<FirstArgT>>;
        if constexpr (is_pointer) {
            usm_compute_call_throw(function_name);
        }
        else {
            buffer_compute_call_throw(function_name);
        }
    }

    void get_workspace_buffer_access_if_rqd(const char* function_name, sycl::handler& cgh) {
        if (m_ext_workspace_rqd) {
            if (m_workspace_buffer) {
                if (m_workspace_buffer->size()) {
                    m_workspace_buffer->template get_access<sycl::access::mode::read_write>(cgh);
                }
            }
            else {
                throw mkl::invalid_argument(
                    "DFT", function_name,
                    "Buffer external workspace must be used with buffer compute calls");
            }
        }
    }

private:
    /** When a compute function using USM arguments is called, throw an exception if an incorrect workspace has been set.
     *  @param function_name The name of the function to use in the error.
    */
    void usm_compute_call_throw(const char* function_name) const {
        if (m_ext_workspace_rqd && m_workspace_type != ext_workspace_type::usm) {
            throw mkl::invalid_argument(
                "DFT", function_name, "USM external workspace must be used with usm compute calls");
        }
    }

    /** When a compute function using buffer arguments is called, throw an exception if an incorrect workspace has been set.
     *  @param function_name The name of the function to use in the error.
    */
    void buffer_compute_call_throw(const char* function_name) const {
        if (m_ext_workspace_rqd && m_workspace_type != ext_workspace_type::buffer) {
            throw mkl::invalid_argument(
                "DFT", function_name,
                "Buffer external workspace must be used with buffer compute calls");
        }
    }
};

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_EXTERNAL_WORKSPACE_HELPER_HPP_

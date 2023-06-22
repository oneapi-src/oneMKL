/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#include <rocfft.h>
#include <hip/hip_runtime_api.h>

#include "execute_helper.hpp"
#include "oneapi/mkl/dft/backward.hpp"
#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "rocfft_handle.hpp"

namespace oneapi::mkl::dft::rocfft {
namespace detail {
template <dft::precision prec, dft::domain dom>
rocfft_plan get_bwd_plan(dft::detail::commit_impl<prec, dom> *commit) {
    return static_cast<rocfft_handle *>(commit->get_handle())[1].plan.value();
}

template <dft::precision prec, dft::domain dom>
rocfft_execution_info get_bwd_info(dft::detail::commit_impl<prec, dom> *commit) {
    return static_cast<rocfft_handle *>(commit->get_handle())[1].info.value();
}
} // namespace detail
// BUFFER version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout) {
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    queue.submit([&](sycl::handler &cgh) {
        auto inout_acc = inout.template get_access<sycl::access::mode::read_write>(cgh);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, inout)";
            auto stream = detail::setup_stream(func_name, ih, info);

            auto inout_native = detail::native_mem(ih, inout_acc);
            detail::execute_checked(func_name, plan, &inout_native, nullptr, info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<data_type, 1> &inout_re,
                                    sycl::buffer<data_type, 1> &inout_im) {
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    queue.submit([&](sycl::handler &cgh) {
        auto inout_re_acc = inout_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto inout_im_acc = inout_im.template get_access<sycl::access::mode::read_write>(cgh);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, inout_re, inout_im)";
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void *, 2> inout_native = { detail::native_mem(ih, inout_re_acc),
                                                   detail::native_mem(ih, inout_im_acc) };
            detail::execute_checked(func_name, plan, inout_native.data(), nullptr, info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<input_type, 1> &in,
                                    sycl::buffer<output_type, 1> &out) {
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    queue.submit([&](sycl::handler &cgh) {
        auto in_acc = in.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_acc = out.template get_access<sycl::access::mode::read_write>(cgh);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, in, out)";
            auto stream = detail::setup_stream(func_name, ih, info);

            auto in_native = detail::native_mem(ih, in_acc);
            auto out_native = detail::native_mem(ih, out_acc);
            detail::execute_checked(func_name, plan, &in_native, &out_native, info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT void compute_backward(descriptor_type &desc, sycl::buffer<input_type, 1> &in_re,
                                    sycl::buffer<input_type, 1> &in_im,
                                    sycl::buffer<output_type, 1> &out_re,
                                    sycl::buffer<output_type, 1> &out_im) {
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    queue.submit([&](sycl::handler &cgh) {
        auto in_re_acc = in_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto in_im_acc = in_im.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_re_acc = out_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_im_acc = out_im.template get_access<sycl::access::mode::read_write>(cgh);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, in_re, in_im, out_re, out_im)";
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void *, 2> in_native = { detail::native_mem(ih, in_re_acc),
                                                detail::native_mem(ih, in_im_acc) };
            std::array<void *, 2> out_native = { detail::native_mem(ih, out_re_acc),
                                                 detail::native_mem(ih, out_im_acc) };
            detail::execute_checked(func_name, plan, in_native.data(), out_native.data(), info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, data_type *inout,
                                           const std::vector<sycl::event> &deps) {
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, inout, deps)";
            auto stream = detail::setup_stream(func_name, ih, info);

            void *inout_ptr = inout;
            detail::execute_checked(func_name, plan, &inout_ptr, nullptr, info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, data_type *inout_re,
                                           data_type *inout_im,
                                           const std::vector<sycl::event> &deps) {
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, inout_re, inout_im, deps)";
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void *, 2> inout_native = { inout_re, inout_im };
            detail::execute_checked(func_name, plan, inout_native.data(), nullptr, info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, input_type *in, output_type *out,
                                           const std::vector<sycl::event> &deps) {
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, in, out, deps)";
            auto stream = detail::setup_stream(func_name, ih, info);

            void *in_ptr = in;
            void *out_ptr = out;
            detail::execute_checked(func_name, plan, &in_ptr, &out_ptr, info);
            detail::sync_checked(func_name, stream);
        });
    });
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type &desc, input_type *in_re,
                                           input_type *in_im, output_type *out_re,
                                           output_type *out_im,
                                           const std::vector<sycl::event> &deps) {
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);

    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.host_task([=](sycl::interop_handle ih) {
            const std::string func_name =
                "compute_backward(desc, in_re, in_im, out_re, out_im, deps)";
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void *, 2> in_native = { in_re, in_im };
            std::array<void *, 2> out_native = { out_re, out_im };
            detail::execute_checked(func_name, plan, in_native.data(), out_native.data(), info);
            detail::sync_checked(func_name, stream);
        });
    });
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace oneapi::mkl::dft::rocfft

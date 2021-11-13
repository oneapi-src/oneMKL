/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <CL/sycl.hpp>

#include "lapack_common.hpp"

namespace {

std::vector<int64_t> host_data(1024);
int64_t* device_data = nullptr;

} // namespace

sycl::event create_dependency(sycl::queue queue) {
    ::device_data = device_alloc<int64_t>(queue, ::host_data.size());
    return host_to_device_copy(queue, ::host_data.data(), ::device_data, ::host_data.size());
}

void print_status(const char* name, sycl::info::event_command_status status) {
    global::log << name << " command execution status: ";
    if (sycl::info::event_command_status::submitted == status)
        global::log << "submitted";
    else if (sycl::info::event_command_status::running == status)
        global::log << "running";
    else if (sycl::info::event_command_status::complete == status)
        global::log << "complete";
    else
        global::log << "status unknown";
    global::log << " (" << static_cast<int64_t>(status) << ")" << std::endl;
}

bool check_dependency(sycl::queue queue, sycl::event in_event, sycl::event func_event) {
    sycl::info::event_command_status in_status;
    sycl::info::event_command_status func_status;

    do {
        func_status = func_event.get_info<sycl::info::event::command_execution_status>();
    } while (func_status != sycl::info::event_command_status::running &&
             func_status != sycl::info::event_command_status::complete);
    in_status = in_event.get_info<sycl::info::event::command_execution_status>();

    /* Print results */
    auto result = (in_status == sycl::info::event_command_status::complete);
    if (!result) {
        print_status("in_event", in_status);
        print_status("func_event", func_status);
    }

    device_free(queue, ::device_data);
    return result;
}

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

#include <CL/sycl.hpp>
#include <chrono>
#include <thread>

#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"

sycl::event create_dependent_event(sycl::queue queue) {
    auto sleep_duration = std::chrono::milliseconds(10);
    return queue.submit([&](sycl::handler &cgh) {
        cgh.codeplay_host_task([=]() { std::this_thread::sleep_for(2 * sleep_duration); });
    });
}

Dependency_Result get_result(sycl::info::event_command_status in_status,
                             sycl::info::event_command_status func_status) {
    /*   in\func | submitted | running  | complete */
    /* submitted |   inc.    |   fail   |  fail    */
    /* running   |   pass    |   fail   |  fail    */
    /* complete  |   inc.    |   inc.   |  inc.    */
    if (in_status == sycl::info::event_command_status::submitted) {
        if (func_status == sycl::info::event_command_status::submitted)
            return Dependency_Result::inconclusive;
        else if (func_status == sycl::info::event_command_status::running)
            return Dependency_Result::fail;
        else if (func_status == sycl::info::event_command_status::complete)
            return Dependency_Result::fail;
    }
    else if (in_status == sycl::info::event_command_status::running) {
        if (func_status == sycl::info::event_command_status::submitted)
            return Dependency_Result::pass;
        else if (func_status == sycl::info::event_command_status::running)
            return Dependency_Result::fail;
        else if (func_status == sycl::info::event_command_status::complete)
            return Dependency_Result::fail;
    }
    else if (in_status == sycl::info::event_command_status::complete) {
        if (func_status == sycl::info::event_command_status::submitted)
            return Dependency_Result::inconclusive;
        else if (func_status == sycl::info::event_command_status::running)
            return Dependency_Result::inconclusive;
        else if (func_status == sycl::info::event_command_status::complete)
            return Dependency_Result::inconclusive;
    }

    return Dependency_Result::unknown;
}

bool check_dependency(sycl::event in_event, sycl::event func_event) {
    auto result = Dependency_Result::inconclusive;
    sycl::info::event_command_status in_status;
    sycl::info::event_command_status func_status;

    do {
        in_status = in_event.get_info<sycl::info::event::command_execution_status>();
        func_status = func_event.get_info<sycl::info::event::command_execution_status>();

        auto temp_result = get_result(in_status, func_status);
        if (temp_result == Dependency_Result::pass || temp_result == Dependency_Result::fail)
            result = temp_result;

    } while (in_status != sycl::info::event_command_status::complete &&
             result != Dependency_Result::fail);

    /* Print results */
    if (result == Dependency_Result::pass)
        global::log << "Dependency Test: Successful" << std::endl;
    if (result == Dependency_Result::inconclusive)
        global::log << "Dependency Test: Inconclusive" << std::endl;
    if (result == Dependency_Result::fail)
        global::log << "Dependency Test: Failed" << std::endl;

    global::log << "\tin_event command execution status: " << static_cast<int64_t>(in_status);
    if (sycl::info::event_command_status::submitted == in_status)
        global::log << " (submitted)" << std::endl;
    else if (sycl::info::event_command_status::running == in_status)
        global::log << " (running)" << std::endl;
    else if (sycl::info::event_command_status::complete == in_status)
        global::log << " (complete)" << std::endl;
    else
        global::log << " (status unknown)" << std::endl;

    global::log << "\tfunction command execution status: " << static_cast<int64_t>(func_status);
    if (sycl::info::event_command_status::submitted == func_status)
        global::log << " (submitted)" << std::endl;
    else if (sycl::info::event_command_status::running == func_status)
        global::log << " (running)" << std::endl;
    else if (sycl::info::event_command_status::complete == func_status)
        global::log << " (complete)" << std::endl;
    else
        global::log << " (status unknown)" << std::endl;

    return (result == Dependency_Result::pass || result == Dependency_Result::inconclusive) ? true
                                                                                            : false;
}

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

#include <array>
#include <iostream>
#include <sstream>

namespace test_log {

std::stringstream lout{};
std::array<char, 1024> buffer{};
std::string padding{};

void print() {
    std::cout.clear();
    if (lout.rdbuf()->in_avail()) { /* check if stream is non-empty */
        while (lout.good()) {
            std::string line;
            std::getline(lout, line);
            std::cout << padding << "\t" << line << std::endl;
        }
    }
    lout.str("");
    lout.clear();
}

} // namespace test_log

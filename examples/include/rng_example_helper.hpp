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


/*
*
*  Content:
*       File contains checkers for statistics of various rng distributions
*
*******************************************************************************/

// stl includes
#include <iostream>
#include <vector>
#include <cmath>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

// local includes for common example helper functions
#include "example_helper.hpp"

// function to compare theoretical moments and sample moments
template<typename Type>
bool compare_moments(Type* r, std::size_t size, double tM, double tD, double tQ) {
    // sample moments
    double sum = 0.0;
    double sum2 = 0.0;
    double cur = 0.0;
    for(int i = 0; i < size; i++) {
        cur = static_cast<double>(r[i]);
        sum += cur;
        sum2 += cur * cur;
    }
    double n_d = static_cast<double>(size);
    double sM = sum / n_d;
    double sD = sum2 / n_d - (sM * sM);

    // comparison of theoretical and sample moments
    double tD2 = tD * tD;
    double s = ((tQ-tD2) / n_d) - ( 2 * (tQ - 2 * tD2) / (n_d * n_d))+((tQ - 3 * tD2) /
                                                            (n_d * n_d * n_d));

    double DeltaM = (tM - sM) / std::sqrt(tD / n_d);
    double DeltaD = (tD - sD) / std::sqrt(s);
    if(std::fabs(DeltaM) > 3.0 || std::fabs(DeltaD) > 3.0) {
        std::cout << "Error: sample moments (mean=" << sM << ", variance=" << sD
            << ") disagree with theory (mean=" << tM << ", variance=" << tD <<
            ")" << std:: endl;
        return false;
    }
    std::cout << "Success: sample moments (mean=" << sM << ", variance=" << sD
        << ") agree with theory (mean=" << tM << ", variance=" << tD <<
        ")" << std:: endl;
    return true;
}


// it is used to calculate theoretical moments of particular distribution
// and compare them with sample moments
template<typename Type, typename Method>
std::enable_if_t<!std::is_integral_v<Type>, bool>
check_statistics(Type* r, std::size_t size, const oneapi::mkl::rng::uniform<Type, Method>& distr) {
    double tM, tD, tQ;
    Type a = distr.a();
    Type b = distr.b();

    // theoretical moments of uniform real type distribution
    tM = (b + a) / 2.0;
    tD = ((b - a) * (b - a)) / 12.0;
    tQ = ((b - a) * (b - a) * (b - a) * (b - a)) / 80.0;

    return compare_moments(r, size, tM, tD, tQ);
}

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

#ifndef _LAPACK_TEST_CONTROLLER_HPP_
#define _LAPACK_TEST_CONTROLLER_HPP_

#include <sstream>
#include <string>
#include <vector>

#include "lapack_common.hpp"
#include "oneapi/mkl/exceptions.hpp"

template <class T>
std::istream& operator >>(std::istream& is, T& t) {
    int64_t i;
    is >> i;
    t = static_cast<T>(i);
    return is;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::job& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::jobsvd& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::transpose& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::uplo& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::side& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::diag& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator <<(std::ostream& os, const oneapi::mkl::generate& t) {
    os << static_cast<int64_t>(t);
    return os;
}

struct test_result {
    enum class type {
        fail,
        pass,
        exception
    };

    type result_type;
    std::string what;

    test_result() : result_type{type::pass} {}
    test_result(bool b) : result_type{b? type::pass: type::fail} {}
    test_result(const std::exception& e) : result_type{type::exception}, what{e.what()} {}

    operator bool() const& {
        return result_type == type::pass;
    }
};

inline bool operator== (const test_result& lhs, const test_result& rhs) {
    return (lhs.result_type == rhs.result_type && lhs.what == rhs.what);
}
inline bool operator!= (const test_result& lhs, const test_result& rhs) {
    return !(lhs == rhs);
}

inline std::ostream& operator <<(std::ostream& os, test_result result) {
    switch (result.result_type) {
        case test_result::type::pass:
            os << "PASS " << result.what;
            break;
        case test_result::type::fail:
            os << "FAIL " << result.what;
            break;
        case test_result::type::exception:
            os << "EXCEPTION " << result.what;
            break;
    }
    return os;
}

template<typename T>
struct function_info;

template<typename ...Args>
struct function_info<bool(const sycl::device&, Args...)> {
    using arg_type = std::tuple<Args...>;
    static constexpr size_t arg_count = sizeof...(Args);

    template <size_t n>
    struct arg {
        using type = typename std::tuple_element<n, std::tuple<Args...>>::type;
    };
};

template<typename T>
struct InputTestController {
    using TestPointer = T;
    using ArgTuple_T = typename function_info<T>::arg_type;
    static constexpr size_t arg_count = function_info<T>::arg_count;
    std::vector<ArgTuple_T> vargs;

    InputTestController(const char* input = nullptr) {
        if constexpr( arg_count == 0 )  /* test does not take input */
            return;

        if( input ) {
            std::stringstream input_stream(input);
            if( input_stream.fail() )
                std::cout << "Failed to process input: \'" << input << "\'" << std::endl;
            else
                store_input(input_stream, std::make_index_sequence<arg_count>());
        } else { /* search for input file */
            const char* input = std::getenv("IN");
            std::ifstream input_stream(input);
            if( input_stream.fail() )
                std::cout << "Failed to open input file: \'" << input << "\'" << std::endl;
            else
                store_input(input_stream, std::make_index_sequence<arg_count>());
        }
    }

    template<size_t...I>
    void store_input(std::istream& input_stream, std::index_sequence<I...>) {
        if constexpr( arg_count == 0 )  /* test does not take input */
            return;
        else {
            ArgTuple_T args;
            while( (..., (input_stream >> std::get<I>(args))) ) {
                vargs.push_back(args);
                input_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
    }

    template<size_t...I>
    void log_result(size_t input_file_line, test_result result, const ArgTuple_T& args = {}, std::index_sequence<I...> = std::make_index_sequence<0>{}) {
        std::cout.clear();
        std::cout << "[" << input_file_line << "]: ";
        (..., (std::cout << std::get<I>(args) << " "));
        std::cout << "# " << result << std::endl;
        if (global::log.rdbuf()->in_avail()) /* check if stream is non-empty */
            std::cout << global::log.rdbuf();
        global::log.str("");
        global::log.clear();
    }

    test_result call_test(TestPointer tp, const sycl::device &dev, ArgTuple_T args) {
        auto tp_args = tuple_cat(std::make_tuple(dev), args);
        test_result result;
        try {
            result = std::apply(tp, tp_args);
        } catch (const oneapi::mkl::unsupported_device& e) {
            result = test_result{e};
            result.result_type = test_result::type::pass;
        } catch (const oneapi::mkl::unimplemented& e) {
            result = test_result{e};
            result.result_type = test_result::type::pass;
        } catch(const std::exception& e) {
            result = test_result{e};
        }
        return result;
    }

    test_result run(TestPointer tp, const sycl::device &dev) {
        if constexpr( arg_count == 0 ) { /* test does not take input */
            test_result result = call_test(tp, dev, {});
            log_result(0, result);
            return result;
        } else {
            if( !vargs.size() ) {
                global::log << arg_count << " inputs expected, found none" << std::endl;
                log_result(1, false);
            }
            test_result aggregate_result;
            size_t input_file_line = 1;
            for( auto& args : vargs ) {
                test_result result = call_test(tp, dev, args);
                log_result(input_file_line++, result, args, std::make_index_sequence<arg_count>());
                if (!result)
                    aggregate_result = result;
            }
            return aggregate_result;
        }
    }

    test_result run_print_on_fail(TestPointer tp, const sycl::device &dev) {
        if constexpr( arg_count == 0 ) { /* test does not take input */
            test_result result = call_test(tp, dev, {});
            if (!result)
                log_result(0, result);
            else {
                global::log.str("");
                global::log.clear();
            }
            return result;
        } else {
            if( !vargs.size() ) {
                global::log << arg_count << " inputs expected, found none" << std::endl;
                log_result(1, false);
            }
            test_result aggregate_result;
            size_t input_file_line = 1;
            for( auto& args : vargs ) {
                test_result result = call_test(tp, dev, args);
                if (!result)
                    log_result(input_file_line++, result, args, std::make_index_sequence<arg_count>());
                else {
                    global::log.str("");
                    global::log.clear();
                }
                if (!result)
                    aggregate_result = result;
            }
            return aggregate_result;
        }
    }
};

using RealSinglePrecisionBuffer = sycl::buffer<float,1>;
using RealDoublePrecisionBuffer = sycl::buffer<double,1>;
using ComplexSinglePrecisionBuffer = sycl::buffer<std::complex<float>,1>;
using ComplexDoublePrecisionBuffer = sycl::buffer<std::complex<double>,1>;
using RealSinglePrecisionUsm = float;
using RealDoublePrecisionUsm = double;
using ComplexSinglePrecisionUsm = std::complex<float>;
using ComplexDoublePrecisionUsm = std::complex<double>;

#define RUN_SUITE(SUITE) \
    RUN_SUITE_REAL(SUITE) \
    RUN_SUITE_COMPLEX(SUITE)

#define RUN_SUITE_REAL(SUITE) \
    RUN_SUITE_REAL_USM(SUITE) \
    RUN_SUITE_REAL_BUFFER(SUITE)

#define RUN_SUITE_COMPLEX(SUITE) \
    RUN_SUITE_COMPLEX_USM(SUITE) \
    RUN_SUITE_COMPLEX_BUFFER(SUITE)

#define RUN_SUITE_USM(SUITE) \
    RUN_SUITE_REAL_USM(SUITE) \
    RUN_SUITE_COMPLEX_USM(SUITE)

#define RUN_SUITE_REAL_USM(SUITE) \
TEST_P(SUITE##Tests, AccuracyRealSinglePrecisionUsm){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<RealSinglePrecisionUsm>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, AccuracyRealDoublePrecisionUsm){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<RealDoublePrecisionUsm>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, DependencyRealSinglePrecisionUsm){ \
    EXPECT_TRUE( dependency_controller.run_print_on_fail(::usm_dependency<RealSinglePrecisionUsm>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, DependencyRealDoublePrecisionUsm){ \
    EXPECT_TRUE( dependency_controller.run_print_on_fail(::usm_dependency<RealDoublePrecisionUsm>, *GetParam()) ); \
}

#define RUN_SUITE_REAL_BUFFER(SUITE) \
TEST_P(SUITE##Tests, AccuracyRealSinglePrecisionBuffer){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<RealSinglePrecisionBuffer>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, AccuracyRealDoublePrecisionBuffer){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<RealDoublePrecisionBuffer>, *GetParam()) ); \
}

#define RUN_SUITE_COMPLEX_USM(SUITE) \
TEST_P(SUITE##Tests, AccuracyComplexSinglePrecisionUsm){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<ComplexSinglePrecisionUsm>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, AccuracyComplexDoublePrecisionUsm){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<ComplexDoublePrecisionUsm>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, DependencyComplexSinglePrecisionUsm){ \
    EXPECT_TRUE( dependency_controller.run_print_on_fail(::usm_dependency<ComplexSinglePrecisionUsm>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, DependencyComplexDoublePrecisionUsm){ \
    EXPECT_TRUE( dependency_controller.run_print_on_fail(::usm_dependency<ComplexDoublePrecisionUsm>, *GetParam()) ); \
}

#define RUN_SUITE_COMPLEX_BUFFER(SUITE) \
TEST_P(SUITE##Tests, AccuracyComplexSinglePrecisionBuffer){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<ComplexSinglePrecisionBuffer>, *GetParam()) ); \
} \
TEST_P(SUITE##Tests, AccuracyComplexDoublePrecisionBuffer){ \
    EXPECT_TRUE( accuracy_controller.run_print_on_fail(::accuracy<ComplexDoublePrecisionBuffer>, *GetParam()) ); \
}

#endif // _LAPACK_TEST_CONTROLLER_HPP_

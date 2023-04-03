/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef ONEMKL_COMPUTE_INPLACE_HPP
#define ONEMKL_COMPUTE_INPLACE_HPP

#include "compute_tester.hpp"
#include <oneapi/mkl/exceptions.hpp>

inline std::int64_t row_elements_to_conjugate_even_components(std::int64_t last_dim) {
    return ((last_dim / 2) + 1) * 2;
}

std::vector<std::int64_t> get_conjugate_even_real_component_strides(
    const std::vector<std::int64_t>& sizes) {
    switch (sizes.size()) {
        case 1: return { 0, 1 };
        case 2: return { 0, 2 * (sizes[1] / 2 + 1), 1 };
        case 3: return { 0, 2 * sizes[1] * (sizes[2] / 2 + 1), 2 * (sizes[2] / 2 + 1), 1 };
        default:
            throw oneapi::mkl::unimplemented(
                "compute_inplace", __FUNCTION__,
                "not implemented for " + std::to_string(sizes.size()) + " dimensions");
            return {};
    }
}

template <typename fp>
std::vector<fp> get_conjugate_even_ref(const std::vector<std::int64_t>& sizes, std::int64_t batches,
                                       std::vector<std::complex<fp>> output_ref) {
    const std::int64_t conjugate_even_last_dim =
        row_elements_to_conjugate_even_components(sizes.back());
    const std::int64_t rows =
        std::accumulate(sizes.begin(), sizes.end() - 1, batches, std::multiplies<>{});
    std::vector<fp> conjugate_even_ref(rows * conjugate_even_last_dim);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < conjugate_even_last_dim; i += 2) {
            conjugate_even_ref[j * conjugate_even_last_dim + i] =
                output_ref[j * sizes.back() + i / 2].real();
            conjugate_even_ref[j * conjugate_even_last_dim + i + 1] =
                output_ref[j * sizes.back() + i / 2].imag();
        }
    }
    return conjugate_even_ref;
}

template <typename T, typename al>
void copy_strided(const std::vector<std::int64_t>& sizes, const std::vector<T>& input,
                  std::vector<T, al>& output) {
    auto in_iter = input.cbegin();
    auto out_iter = output.begin();
    const auto row_len = sizes.back();
    const auto conjugate_row_len = row_elements_to_conjugate_even_components(row_len);
    while (in_iter < input.cend()) {
        std::copy(in_iter, in_iter + row_len, out_iter);
        in_iter += row_len;
        out_iter += conjugate_row_len;
    }
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    const std::int64_t container_size_total =
        domain == oneapi::mkl::dft::domain::REAL
            ? (size_total / sizes.back()) *
                  (row_elements_to_conjugate_even_components(sizes.back()))
            : size_total;
    const std::int64_t container_size_per_transform = container_size_total / batches;
    const std::int64_t backward_elements = domain == oneapi::mkl::dft::domain::REAL
                                               ? container_size_per_transform / 2
                                               : container_size_per_transform;

    descriptor_t descriptor{ sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    descriptor.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / forward_elements));
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                         container_size_per_transform);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_elements);
    descriptor.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / forward_elements));

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto real_strides = get_conjugate_even_real_component_strides(sizes);
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                             complex_strides.data());
    }

    commit_descriptor(descriptor, sycl_queue);

    std::vector<FwdInputType> inout_host(container_size_total, 0);
    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        copy_strided(sizes, input, inout_host);
    }
    else {
        std::copy(input.begin(), input.end(), inout_host.begin());
    }

    {
        sycl::buffer<FwdInputType, 1> inout_buf{ inout_host };

        try {
            oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout_buf);
        }
        catch (oneapi::mkl::unimplemented& e) {
            std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
            return test_skipped;
        }

        {
            auto acc_host = inout_buf.template get_host_access();
            if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
                std::vector<FwdInputType> conjugate_even_ref =
                    get_conjugate_even_ref(sizes, batches, out_host_ref);
                EXPECT_TRUE(check_equal_vector(acc_host.get_pointer(), conjugate_even_ref.data(),
                                               inout_host.size(), abs_error_margin,
                                               rel_error_margin, std::cout));
            }
            else {
                EXPECT_TRUE(check_equal_vector(acc_host.get_pointer(), out_host_ref.data(),
                                               inout_host.size(), abs_error_margin,
                                               rel_error_margin, std::cout));
            }
        }

        if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
            const auto real_strides = get_conjugate_even_real_component_strides(sizes);
            const auto complex_strides = get_conjugate_even_complex_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                                 complex_strides.data());
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                                 real_strides.data());
            commit_descriptor(descriptor, sycl_queue);
        }

        try {
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                               FwdInputType>(descriptor, inout_buf);
        }
        catch (oneapi::mkl::unimplemented& e) {
            std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
            return test_skipped;
        }
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        for (int j = 0; j < size_total / sizes.back(); j++) {
            EXPECT_TRUE(check_equal_vector(
                inout_host.data() + j * row_elements_to_conjugate_even_components(sizes.back()),
                input.data() + j * sizes.back(), sizes.back(), abs_error_margin, rel_error_margin,
                std::cout));
        }
    }
    else {
        EXPECT_TRUE(check_equal_vector(inout_host.data(), input.data(), input.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }

    const int64_t container_size_total =
        domain == oneapi::mkl::dft::domain::REAL
            ? (size_total / sizes.back()) * row_elements_to_conjugate_even_components(sizes.back())
            : size_total;
    const int64_t container_size_per_transform = container_size_total / batches;
    const std::int64_t backward_elements = domain == oneapi::mkl::dft::domain::REAL
                                               ? container_size_per_transform / 2
                                               : container_size_per_transform;

    descriptor_t descriptor = { sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                         container_size_per_transform);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_elements);
    descriptor.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / forward_elements));

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto real_strides = get_conjugate_even_real_component_strides(sizes);
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                             complex_strides.data());
    }

    commit_descriptor(descriptor, sycl_queue);

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);
    std::vector<FwdInputType, decltype(ua_input)> inout(container_size_total, ua_input);

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        copy_strided(sizes, input, inout);
    }
    else {
        std::copy(input.begin(), input.end(), inout.begin());
    }

    try {
        std::vector<sycl::event> dependencies;
        oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout.data(),
                                                                      dependencies)
            .wait();
    }
    catch (oneapi::mkl::unimplemented& e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        std::vector<FwdInputType> conjugate_even_ref =
            get_conjugate_even_ref(sizes, batches, out_host_ref);
        EXPECT_TRUE(check_equal_vector(inout.data(), conjugate_even_ref.data(), inout.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }
    else {
        EXPECT_TRUE(check_equal_vector(inout.data(), out_host_ref.data(), inout.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto real_strides = get_conjugate_even_real_component_strides(sizes);
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, complex_strides.data());
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, real_strides.data());
        commit_descriptor(descriptor, sycl_queue);
    }

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done =
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                               FwdInputType>(descriptor, inout.data());
        done.wait();
    }
    catch (oneapi::mkl::unimplemented& e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        for (int j = 0; j < size_total / sizes.back(); j++) {
            EXPECT_TRUE(check_equal_vector(
                inout.data() + j * row_elements_to_conjugate_even_components(sizes.back()),
                input.data() + j * sizes.back(), sizes.back(), abs_error_margin, rel_error_margin,
                std::cout));
        }
    }
    else {
        EXPECT_TRUE(check_equal_vector(inout.data(), input.data(), input.size(), abs_error_margin,
                                       rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_INPLACE_HPP

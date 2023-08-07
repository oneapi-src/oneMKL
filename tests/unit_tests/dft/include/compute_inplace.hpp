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

inline std::size_t row_elements_to_conjugate_even_components(std::size_t last_dim) {
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
    const std::size_t last_dim_size = cast_unsigned(sizes.back());
    const std::size_t conjugate_even_last_dim =
        row_elements_to_conjugate_even_components(last_dim_size);
    const std::size_t rows = cast_unsigned(
        std::accumulate(sizes.begin(), sizes.end() - 1, batches, std::multiplies<>{}));
    std::vector<fp> conjugate_even_ref(rows * conjugate_even_last_dim);
    for (std::size_t j = 0; j < rows; j++) {
        for (std::size_t i = 0; i < conjugate_even_last_dim; i += 2) {
            conjugate_even_ref[j * conjugate_even_last_dim + i] =
                output_ref[j * last_dim_size + i / 2].real();
            conjugate_even_ref[j * conjugate_even_last_dim + i + 1] =
                output_ref[j * last_dim_size + i / 2].imag();
        }
    }
    return conjugate_even_ref;
}

template <typename T, typename al>
void copy_strided(const std::vector<std::int64_t>& sizes, const std::vector<T>& input,
                  std::vector<T, al>& output) {
    auto in_iter = input.cbegin();
    auto out_iter = output.begin();
    const auto row_len = static_cast<std::ptrdiff_t>(sizes.back());
    const auto conjugate_row_len = static_cast<std::ptrdiff_t>(
        row_elements_to_conjugate_even_components(cast_unsigned(row_len)));
    while (in_iter < input.cend()) {
        std::copy(in_iter, in_iter + row_len, out_iter);
        in_iter += row_len;
        out_iter += conjugate_row_len;
    }
}

template<typename T>
void print(std::vector<T> v){
    for(T a : v){
        std::cout << a << ", ";
    }
    std::cout << std::endl;
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    /*const std::size_t last_dim_size = cast_unsigned(sizes.back());
    const std::size_t real_first_dims = size_total / last_dim_size;
    const std::size_t real_last_dim = row_elements_to_conjugate_even_components(last_dim_size);
*/
    auto strides_fwd = this->strides_fwd;
    if(domain == oneapi::mkl::dft::domain::REAL){
        if(strides_fwd.size() == 0){
            auto strides_tmp = get_conjugate_even_complex_strides(sizes);
            strides_fwd = {strides_tmp[0]};
            //to be able to calculate in place each row must fit backward data
            for(size_t i=0;i<sizes.size()-1;i++){
                strides_fwd.push_back(strides_tmp[i+1]*2);
            }
            strides_fwd.push_back(1);
        }
    } else {
        // spec says strides_bwd is ignored and strides_fwd is reused for backward domain for in-place complex
        strides_fwd = strides_bwd;
    }

    auto [forward_distance, backward_distance] = get_default_distances<domain>(sizes, strides_fwd, strides_bwd);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    std::cout << "forward_distance: " << forward_distance << std::endl;
    std::cout << "fwd stride: ";
    print(strides_fwd);
    std::cout << "backward_distance: " << backward_distance << std::endl;
    std::cout << "bwd stride: ";
    print(strides_bwd);

    //const std::size_t container_size_total =
      //  domain == oneapi::mkl::dft::domain::REAL ? real_first_dims * real_last_dim : size_total;

    /*const std::int64_t container_size_per_transform =
        static_cast<std::int64_t>(container_size_total) / batches;
    const std::int64_t backward_elements = domain == oneapi::mkl::dft::domain::REAL
                                               ? container_size_per_transform / 2
                                               : container_size_per_transform;
                                               */

    descriptor_t descriptor{ sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                         forward_distance);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_distance);
    if(strides_fwd.size()){
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_fwd.data());
    }
    if(strides_bwd.size()){
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_bwd.data());
    } else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                             complex_strides.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    std::vector<FwdInputType> inout_host(strided_copy(input, sizes, strides_fwd, batches));
    inout_host.resize(cast_unsigned(std::max(forward_distance, (domain == oneapi::mkl::dft::domain::REAL ? 2 : 1) * backward_distance) * batches + getdefault(strides_bwd,0,0L)));

    std::cout << "input: ";
    print(input);
    std::cout << "inout_host.size: " << inout_host.size() << std::endl;
    std::cout << "inout_host: ";
    print(inout_host);
    /*if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        copy_strided(sizes, input, inout_host);
    }
    else {
        std::copy(input.begin(), input.end(), inout_host.begin());
    }*/

    {
        sycl::buffer<FwdInputType, 1> inout_buf{ inout_host };
        
        oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout_buf);

        {
            auto inout_buf_cplx = inout_buf/*.template reinterpret<FwdOutputType>()*/;
            auto acc_host = inout_buf_cplx.template get_host_access();
            auto ptr_host = reinterpret_cast<FwdOutputType*>(acc_host.get_pointer());
            std::cout << "backward_distance " << backward_distance << " batches " << batches << std::endl;
            std::cout << "out_host_ref: ";
            print(out_host_ref);
            std::cout << "ptr_host: " << std::endl;
            for(int i=0;i<backward_distance * batches;i++){
                std::cout << ptr_host[i] << ", ";
            }
            std::cout << std::endl;
            for(int64_t i=0;i<batches;i++){
                std::cout << i << std::endl;
                EXPECT_TRUE(check_equal_strided<domain == oneapi::mkl::dft::domain::REAL>(ptr_host + backward_distance * i, out_host_ref.data() + ref_distance * i, sizes, strides_bwd, abs_error_margin, rel_error_margin, std::cout));
            }
            /*if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
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
            }*/
        }

        if(strides_bwd.size()){
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_bwd.data());
        } else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
            const auto complex_strides = get_conjugate_even_complex_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                                complex_strides.data());
        } else{
            auto real_strides = get_default_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                                real_strides.data());
        }
        if(strides_fwd.size()){
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_fwd.data());
        } else{
            auto real_strides = get_default_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                                real_strides.data());
        }
        commit_descriptor(descriptor, sycl_queue);

        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                           FwdInputType>(descriptor, inout_buf);
    }

    std::vector<FwdInputType> fwd_data_ref = input;
    // account for scaling that occurs during DFT
    std::for_each(fwd_data_ref.begin(), fwd_data_ref.end(),
                  [this](auto& x) { x *= static_cast<PrecisionType>(forward_elements); });
    std::cout << "fwd_data_ref: ";
    print(fwd_data_ref);
    std::cout << "inout_host: ";
    print(inout_host);
    /*if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        for (std::size_t j = 0; j < real_first_dims; j++) {
            EXPECT_TRUE(check_equal_vector(
                inout_host.data() + j * row_elements_to_conjugate_even_components(last_dim_size),
                input.data() + j * last_dim_size, last_dim_size, abs_error_margin, rel_error_margin,
                std::cout));
        }
    }
    else {
        EXPECT_TRUE(check_equal_vector(inout_host.data(), input.data(), input.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }*/
    //EXPECT_TRUE(check_equal_vector(inout_host.data(), fwd_data_ref.data(), fwd_data_ref.size(), abs_error_margin,
      //                             rel_error_margin, std::cout));
    for(int64_t i=0;i<batches;i++){
        EXPECT_TRUE(check_equal_strided<false>(inout_host.data() + forward_distance * i, fwd_data_ref.data() + ref_distance * i, sizes, strides_fwd, abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }

    auto strides_fwd = this->strides_fwd;
    if(domain == oneapi::mkl::dft::domain::REAL){
        if(strides_fwd.size() == 0){
            auto strides_tmp = get_conjugate_even_complex_strides(sizes);
            strides_fwd = {strides_tmp[0]};
            //to be able to calculate in place each row must fit backward data
            for(size_t i=0;i<sizes.size()-1;i++){
                strides_fwd.push_back(strides_tmp[i+1]*2);
            }
            strides_fwd.push_back(1);
        }
    } else {
        // spec says strides_bwd is ignored and strides_fwd is reused for backward domain for in-place complex
        strides_fwd = strides_bwd;
    }

    auto [forward_distance, backward_distance] = get_default_distances<domain>(sizes, strides_fwd, strides_bwd);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    descriptor_t descriptor = { sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                         forward_distance);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_distance);
    if(strides_fwd.size()){
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_fwd.data());
    }
    if(strides_bwd.size()){
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_bwd.data());
    } else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                             complex_strides.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);
    std::vector<FwdInputType, decltype(ua_input)> inout(strided_copy(input, sizes, strides_fwd, batches, ua_input), ua_input);
    inout.resize(cast_unsigned(std::max(forward_distance, (domain == oneapi::mkl::dft::domain::REAL ? 2 : 1) * backward_distance) * batches + getdefault(strides_bwd,0,0L)));

    std::vector<sycl::event> no_dependencies;
    oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout.data(),
                                                                  no_dependencies)
        .wait_and_throw();

    for(int64_t i=0;i<batches;i++){
        EXPECT_TRUE(check_equal_strided<domain == oneapi::mkl::dft::domain::REAL>(reinterpret_cast<FwdOutputType*>(inout.data()) + backward_distance * i, out_host_ref.data() + ref_distance * i, sizes, strides_bwd, abs_error_margin, rel_error_margin, std::cout));
    }

    if(strides_bwd.size()){
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_bwd.data());
    } else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                            complex_strides.data());
    } else{
        auto real_strides = get_default_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                            real_strides.data());
    }
    if(strides_fwd.size()){
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_fwd.data());
    } else{
        auto real_strides = get_default_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                            real_strides.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    sycl::event done =
        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                           FwdInputType>(descriptor, inout.data(), no_dependencies);
    done.wait_and_throw();

    std::for_each(input.begin(), input.end(),
                  [this](auto& x) { x *= static_cast<PrecisionType>(forward_elements); });

    for(int64_t i=0;i<batches;i++){
        EXPECT_TRUE(check_equal_strided<false>(inout.data() + forward_distance * i, input.data() + ref_distance * i, sizes, strides_fwd, abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_INPLACE_HPP

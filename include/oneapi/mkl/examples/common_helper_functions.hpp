// common helper functions for examples
//
// helpers for getting error code
//
template <typename T, typename = void>
struct has_member_code_meta : std::false_type {};

template <typename T>
struct has_member_code_meta<T, std::void_t<decltype( std::declval<T>().code() )> > : std::true_type {};

template <typename T, typename std::enable_if<has_member_code_meta<T>::value>::type* = nullptr >
auto get_error_code (T x) {
    return x.code().value();
};

//
// helpers for initializing templated scalar data type values.
//
template <typename fp>
fp set_fp_value(fp arg1, fp arg2 = 0.0)
{
    return arg1;
}

//
// print a 2x2 block of data from matrix M using the sycl accessor
//
// M = [ M_00, M_01 ...
//     [ M_10, M_11 ...
//     [ ...
//
template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name)
{

    std::cout << std::endl;
    std::cout << "\t\t\t" << M_name << " = [ " << M[0*ldM + 0] << ", " << M[1*ldM + 0]         << ", ...\n";
    std::cout << "\t\t\t    [ "                << M[0*ldM + 1] << ", " << M[1*ldM + 1] << ", ...\n";
    std::cout << "\t\t\t    [ "                << "...\n";
    std::cout << std::endl;

}


template <typename fp> fp rand_scalar() { return fp(std::rand()) / fp(RAND_MAX) - fp(0.5); }

template <typename fp> void rand_matrix(fp *M, oneapi::mkl::transpose trans, int m, int n, int ld)
{

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    } else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}


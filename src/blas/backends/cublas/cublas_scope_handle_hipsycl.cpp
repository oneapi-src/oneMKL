#include "cublas_scope_handle_hipsycl.hpp"
#include "cublas_handle.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

thread_local cublas_handle<int> CublasScopedContextHandler::handle_helper = cublas_handle<int>{};

CublasScopedContextHandler::CublasScopedContextHandler(cl::sycl::queue queue,
                                                       cl::sycl::interop_handle &ih)
        : interop_h(ih) {}

cublasHandle_t CublasScopedContextHandler::get_handle(const cl::sycl::queue &queue) {
    cl::sycl::device device = queue.get_device();
    int current_device = interop_h.get_native_device<cl::sycl::backend::cuda>();
    CUstream streamId = get_stream(queue);
    cublasStatus_t err;
    auto it = handle_helper.cublas_handle_mapper_.find(current_device);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.cublas_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                cudaStream_t currentStreamId;
                CUBLAS_ERROR_FUNC(cublasGetStream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.cublas_handle_mapper_.erase(it);
            }
        }
    }
    cublasHandle_t handle;

    CUBLAS_ERROR_FUNC(cublasCreate, err, &handle);
    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);

    auto insert_iter = handle_helper.cublas_handle_mapper_.insert(
        std::make_pair(current_device, new std::atomic<cublasHandle_t>(handle)));
    return handle;
}

CUstream CublasScopedContextHandler::get_stream(const cl::sycl::queue &queue) {
    return interop_h.get_native_queue<cl::sycl::backend::cuda>();
}

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
.. _selecting_a_compiler:

Selecting a Compiler
====================

You must choose a compiler according to the required backend and the operating system of your
application.

* If your application requires Intel GPU, use
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_ ``icpx`` on Linux or ``icx`` on Windows.
* If your Linux application requires NVIDIA GPU, build ``clang++`` from the latest source of
  `oneAPI DPC++ Compiler <https://github.com/intel/llvm>`_ with `support for NVIDIA CUDA <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda>`_ or use ``hipSYCL`` from the `hipSYCL repository <https://github.com/illuhad/hipSYCL>`_ (except for LAPACK domain).
* If your Linux application requires AMD GPU, build ``clang++`` from the latest source of `oneAPI DPC++ Compiler <https://github.com/intel/llvm>`_ with `support for HIP AMD <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-hip-amd>`_ or use ``hipSYCL``.
* If no Intel GPU, NVIDIA GPU, or AMD GPU is required, on Linux you can use
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
  ``icpx``, `oneAPI DPC++ Compiler <https://github.com/intel/llvm/releases>`_ ``clang++``, or ``hipSYCL``,
  and on Windows you can use either
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
  ``icx`` or `oneAPI DPC++ Compiler <https://github.com/intel/llvm/releases>`_ ``clang-cl``.

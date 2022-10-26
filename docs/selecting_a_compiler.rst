.. _selecting_a_compiler:

Selecting a Compiler
====================

You must choose a compiler according to the required backend of your
application.

* If your application requires Intel GPU, use
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_ ``icpx`` on Linux or ``icx`` on Windows.
* If your application requires NVIDIA GPU, use the latest release of
  ``clang++`` from `Intel project for LLVM* technology <https://github.com/intel/llvm/releases>`_ or use ``hipSYCL`` from the `hipSYCL repository <https://github.com/illuhad/hipSYCL>`_ (except for LAPACK domain).
* If your application requires AMD GPU, use ``hipSYCL`` or use the latest release of ``clang++`` from `Intel project for LLVM* technology <https://github.com/intel/llvm/releases>`_.
* If no Intel GPU, NVIDIA GPU, or AMD GPU is required, on Linux you can use either
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
  ``icpx``, ``clang++``, or ``hipSYCL`` and on Windows you can use either
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
  ``icx``, or ``clang-cl`` from
  `Intel project for LLVM* technology <https://github.com/intel/llvm/releases>`_.

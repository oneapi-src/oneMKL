.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_lapack_getri_batch_scratchpad_size:

getri_batch_scratchpad_size
===========================

Computed size of scratchpad memory required for the :ref:`onemkl_lapack_getri_batch` function.

.. container:: section

  .. rubric:: Description

``getri_batch_scratchpad_size`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

**Group API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemkl_lapack_getri_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t getri_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t *n, std::int64_t *lda, std::int64_t group_count, std::int64_t *group_sizes)
    }

.. container:: section

   .. rubric:: Input Parameters

queue
	Device queue where calculations will be performed.

n
  Array of ``group_count`` :math:`n_g` parameters specifying the order of the matrices belonging to group :math:`g`.

lda
	Array of ``group_count`` :math:`\text{lda}_g` parameters specifying the leading dimensions of the matrices belonging to group :math:`g`.

group_count
  Number of groups of parameters. Must be at least 0.

group_sizes
	Array of ``group_count`` integers. Array element with index :math:`g` specifies the number of problems to solve for each of the groups of parameters :math:`g`. So the total number of problems to solve, ``batch_size``, is a sum of all parameter group sizes.

.. container:: section
   
   .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Group API of the :ref:`onemkl_lapack_getri_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Strided API**

Computes the number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemkl_lapack_getri_batch` function.

.. container:: section

  .. rubric:: Syntax

.. code-block:: cpp

    namespace oneapi::mkl::lapack {
      template <typename T>
      std::int64_t getri_batch_scratchpad_size(cl::sycl::queue &queue, std::int64_t n, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size)
    };

.. container:: section

  .. rubric:: Input Parameters

queue
	Device queue where calculations will be performed.

n
  The order of the matrices :math:`A_i` (0 \le n).

lda
	Leading dimension of :math:`A_i` (:math:`n \le \text{lda}`).

stride_a
	Stride between the beginnings of matrices :math:`A_i` inside the batch array ``a``.

stride_ipiv
	Stride between the beginnings of arrays :math:`ipiv_i` inside the array ipiv.

batch_size
	Specifies the number of problems in a batch.

.. container:: section
   
  .. rubric:: Return Values

Number of elements of type ``T`` the scratchpad memory should able to hold to be passed to the Strided API of the :ref:`onemkl_lapack_getri_batch` function.

.. container:: section

  .. rubric:: Throws

This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

:ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`

:ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`

:ref:`oneapi::mkl::lapack::invalid_argument<onemkl_lapack_exception_invalid_argument>`

   Exception is thrown in case of incorrect supplied argument value.
   Position of wrong argument can be determined by `info()` method of exception object.

**Parent topic:** :ref:`onemkl_lapack-like-extensions-routines`


.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_blas_nrm2:

nrm2
====

Computes the Euclidean norm of a vector.

.. _onemkl_blas_nrm2_description:

.. rubric:: Description

The ``nrm2`` routines computes Euclidean norm of a vector

.. math:: 
   
      result = \| x\|   

where:

``x`` is a vector of ``n`` elements.

``nrm2`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
        -  T_res 
      * -  ``float`` 
        -  ``float`` 
      * -  ``double`` 
        -  ``double`` 
      * -  ``std::complex<float>`` 
        -  ``float`` 
      * -  ``std::complex<double>`` 
        -  ``double`` 

.. _onemkl_blas_nrm2_buffer:

nrm2 (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void nrm2(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T_res,1> &result)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void nrm2(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T_res,1> &result)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

.. container:: section

   .. rubric:: Output Parameters

   result
      Buffer where the Euclidean norm of the vector ``x`` will be
      stored.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::mkl::invalid_argument<onemkl_exception_invalid_argument>`
       
   
   :ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`
       

   :ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`
       

   :ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`
       

   :ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`
      

.. _onemkl_blas_nrm2_usm:

nrm2 (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event nrm2(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        T_res *result,
                        const sycl::vector_class<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event nrm2(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        T_res *result,
                        const sycl::vector_class<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   result
      Pointer to where the Euclidean norm of the vector ``x`` will be
      stored.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::mkl::invalid_argument<onemkl_exception_invalid_argument>`
       
       
   
   :ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`
       

   :ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`
       

   :ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`
       

   :ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`
      

   **Parent topic:** :ref:`blas-level-1-routines`

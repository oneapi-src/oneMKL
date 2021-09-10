.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_blas_iamin:

iamin
=====

Finds the index of the element with the smallest absolute value.

.. _onemkl_blas_iamin_description:

.. rubric:: Description

The ``iamin`` routines return an index ``i`` such that ``x[i]`` has
the minimum absolute value of all elements in vector ``x`` (real
variants), or such that (\|Re(``x[i]``)\| + \|Im(``x[i]``)\|) is minimal
(complex variants).

If either ``n`` or ``incx`` are not positive, the routine returns
``0``.

If more than one vector element is found with the same smallest
absolute value, the index of the first one encountered is returned.

If the vector contains ``NaN`` values, then the routine returns the
index of the first ``NaN``.

``iamin`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. container:: Note

   .. rubric:: Note
      :class: NoteTipHead

   The index is zero-based.

.. _onemkl_blas_iamin_buffer:

iamin (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void iamin(sycl::queue &queue,
                  std::int64_t n,
                  sycl::buffer<T,1> &x,
                  std::int64_t incx,
                  sycl::buffer<std::int64_t,1> &result)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void iamin(sycl::queue &queue,
                  std::int64_t n,
                  sycl::buffer<T,1> &x,
                  std::int64_t incx,
                  sycl::buffer<std::int64_t,1> &result)
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
      Stride of vector x.

.. container:: section

   .. rubric:: Output Parameters

   result
      Buffer where the zero-based index ``i`` of the minimum element
      will be stored.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::mkl::invalid_argument<onemkl_exception_invalid_argument>`
       
   
   :ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`
       

   :ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`
       

   :ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`
       

   :ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`
      

.. _onemkl_blas_iamin_usm:

iamin (USM Version)
-------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event iamin(sycl::queue &queue,
                         std::int64_t n,
                         const T *x,
                         std::int64_t incx,
                         T_res *result,
                         const sycl::vector_class<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event iamin(sycl::queue &queue,
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
      The pointer to input vector ``x``. The array holding input
      vector ``x`` must be of size at least (1 + (``n`` -
      1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector x.

.. container:: section

   .. rubric:: Output Parameters

   result
      Pointer to where the zero-based index ``i`` of the minimum
      element will be stored.

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

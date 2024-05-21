.. _onemkl_blas_dotu:

dotu
====

Computes the dot product of two complex vectors.

.. _onemkl_blas_dotu_description:

.. rubric:: Description

The ``dotu`` routines perform a dot product between two complex vectors:

.. math::

   result = \sum_{i=1}^{n}X_iY_i 

``dotu`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_dotu_buffer:

dotu (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void dotu(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &result)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void dotu(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &result)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vectors ``x`` and ``y``.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Buffer holding input vector ``y``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.


.. container:: section

   .. rubric:: Output Parameters

   result
      Buffer where the result (a scalar) is stored.


.. _onemkl_blas_dotu_usm:

dotu (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event dotu(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        const T *y,
                        std::int64_t incy,
                        T *result,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event dotu(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        const T *y,
                        std::int64_t incy,
                        T *result,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vectors ``x`` and ``y``.

   x
      Pointer to the input vector ``x``. The array holding input
      vector ``x`` must be of size at least (1 + (``n`` -
      1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Pointer to input vector ``y``. The array holding input vector
      ``y`` must be of size at least (1 + (``n`` - 1)*abs(``incy``)).
      See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   result
      Pointer to where the result (a scalar) is stored.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-1-routines`

.. _onemkl_blas_asum:

asum
====

Computes the sum of magnitudes of the vector elements.

.. _onemkl_blas_asum_description:

.. rubric:: Description

The ``asum`` routine computes the sum of the magnitudes of elements of a
real vector, or the sum of magnitudes of the real and imaginary parts
of elements of a complex vector:

.. math::

   result = \sum_{i=1}^{n}(|Re(x_i)| + |Im(x_i)|) 
   
where ``x`` is a vector with ``n`` elements.

``asum`` supports the following precisions for data:

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

.. _onemkl_blas_asum_buffer:

asum (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void asum(sycl::queue &queue,
                 std::int64_t n,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T_res,1> &result)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void asum(sycl::queue &queue,
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
      Buffer where the scalar result is stored (the sum of magnitudes of
      the real and imaginary parts of all elements of the vector).


.. _onemkl_blas_asum_usm:

asum (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event asum(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        T_res *result,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event asum(sycl::queue &queue,
                        std::int64_t n,
                        const T *x,
                        std::int64_t incx,
                        T_res *result,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   x
      Pointer to input vector ``x``. The array holding the vector
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
      Pointer to the output matrix where the scalar result is stored
      (the sum of magnitudes of the real and imaginary parts of all
      elements of the vector).

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-1-routines`

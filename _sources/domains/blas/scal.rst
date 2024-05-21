.. _onemkl_blas_scal:

scal
====

Computes the product of a vector by a scalar.

.. _onemkl_blas_scal_description:

.. rubric:: Description

The ``scal`` routines computes a scalar-vector product:

.. math::

      x \leftarrow alpha*x

where:

``x`` is a vector of ``n`` elements,

``alpha`` is a scalar.

``scal`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
        -  T_scalar 
      * -  ``float`` 
        -  ``float`` 
      * -  ``double`` 
        -  ``double`` 
      * -  ``std::complex<float>`` 
        -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 
        -  ``std::complex<double>`` 
      * -  ``std::complex<float>`` 
        -  ``float`` 
      * -  ``std::complex<double>`` 
        -  ``double`` 

.. _onemkl_blas_scal_buffer:

scal (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void scal(sycl::queue &queue,
                 std::int64_t n,
                 T_scalar alpha,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void scal(sycl::queue &queue,
                 std::int64_t n,
                 T_scalar alpha,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   alpha
      Specifies the scalar ``alpha``.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

.. container:: section

   .. rubric:: Output Parameters

   x
      Buffer holding updated buffer ``x``.


.. _onemkl_blas_scal_usm:

scal (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event scal(sycl::queue &queue,
                        std::int64_t n,
                        T_scalar alpha,
                        T *x,
                        std::int64_t incx,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event scal(sycl::queue &queue,
                        std::int64_t n,
                        T_scalar alpha,
                        T *x,
                        std::int64_t incx,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   n
      Number of elements in vector ``x``.

   alpha
      Specifies the scalar ``alpha``.

   x
      Pointer to the input vector ``x``. The array must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

.. container:: section

   .. rubric:: Output Parameters

   x
      Pointer to the updated array ``x``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-1-routines`

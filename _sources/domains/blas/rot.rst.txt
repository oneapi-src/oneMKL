.. _onemkl_blas_rot:

rot
===

Performs rotation of points in the plane.

.. _onemkl_blas_rot_description:

.. rubric:: Description

Given two vectors ``x`` and ``y`` of ``n`` elements, the ``rot`` routines
compute four scalar-vector products and update the input vectors with
the sum of two of these scalar-vector products as follow:

.. math::
  
   \left[\begin{array}{c}
      x\\y
   \end{array}\right]
   \leftarrow
   \left[\begin{array}{c}
      \phantom{-}c*x + s*y\\
      -s*x + c*y
   \end{array}\right]

``rot`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
        -  T_scalar 
      * -  ``float`` 
        -  ``float`` 
      * -  ``double`` 
        -  ``double`` 
      * -  ``std::complex<float>`` 
        -  ``float`` 
      * -  ``std::complex<double>`` 
        -  ``double`` 

.. _onemkl_blas_rot_buffer:

rot (Buffer Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void rot(sycl::queue &queue,
                std::int64_t n,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &y,
                std::int64_t incy,
                T_scalar c,
                T_scalar s)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void rot(sycl::queue &queue,
                std::int64_t n,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &y,
                std::int64_t incy,
                T_scalar c,
                T_scalar s)
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

   y
      Buffer holding input vector ``y``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   c
      Scaling factor.

   s
      Scaling factor.

.. container:: section

   .. rubric:: Output Parameters

   x
      Buffer holding updated buffer ``x``.

   y
      Buffer holding updated buffer ``y``.

      

.. _onemkl_blas_rot_usm:

rot (USM Version)
-----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event rot(sycl::queue &queue,
                       std::int64_t n,
                       T *x,
                       std::int64_t incx,
                       T *y,
                       std::int64_t incy,
                       T_scalar c,
                       T_scalar s,
                       const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event rot(sycl::queue &queue,
                       std::int64_t n,
                       T *x,
                       std::int64_t incx,
                       T *y,
                       std::int64_t incy,
                       T_scalar c,
                       T_scalar s,
                       const std::vector<sycl::event> &dependencies = {})
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

   y
      Pointer to input vector ``y``. The array holding input vector
      ``y`` must be of size at least (1 + (``n`` - 1)*abs(``incy``)).
      See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   c
      Scaling factor.

   s
      Scaling factor.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   x
      Pointer to the updated matrix ``x``.

   y
      Pointer to the updated matrix ``y``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-1-routines`

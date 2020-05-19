.. _onemkl_blas_rot:

rot
===


.. container::


   Performs rotation of points in the plane.



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




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   Given two vectors ``x`` and ``y`` of ``n`` elements, the ``rot`` routines
   compute four scalar-vector products and update the input vectors with
   the sum of two of these scalar-vector products as follow:

  
   x <- c*x + s*y

   y <- c*y - s*x
  


rot (Buffer Version)
--------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::rot(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, T_scalar c, T_scalar s)

.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      Number of elements in vector ``x``.


   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   y
      Buffer holding input vector ``y``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incy
      Stride of vector ``y``.


   c
      Scaling factor.


   s
      Scaling factor.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   x
      Buffer holding updated buffer ``x``.


   y
      Buffer holding updated buffer ``y``.


rot (USM Version)
-----------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::rot(sycl::queue &queue, std::int64_t n, T *x, std::int64_t incx, T *y, std::int64_t incy, T_scalar c, T_scalar s, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vector ``x``.


      x
         Pointer to input vector ``x``. The array holding input vector
         ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
         See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      y
         Pointer to input vector ``y``. The array holding input vector
         ``y`` must be of size at least (1 + (``n`` - 1)*abs(``incy``)).
         See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
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
         :class: sectiontitle


      x
         Pointer to the updated matrix ``x``.


      y
         Pointer to the updated matrix ``y``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`

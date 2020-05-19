.. _onemkl_blas_copy:

copy
====


.. container::


   Copies a vector to another vector.



      ``copy`` supports the following precisions.


      .. list-table:: 
         :header-rows: 1

         * -  T 
         * -  ``float`` 
         * -  ``double`` 
         * -  ``std::complex<float>`` 
         * -  ``std::complex<double>`` 




.. container:: section


   .. rubric:: Description
      :class: sectiontitle


   The copy routines copy one vector to another:


  


      y ←x


   where ``x`` and ``y`` are vectors of n elements.


copy (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::copy(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy)
.. container:: section


   .. rubric:: Input Parameters
      :class: sectiontitle


   queue
      The queue where the routine should be executed.


   n
      Number of elements in vector ``x``.


   x
      Buffer holding input vector ``x``. The buffer must be of size at least
      ``(1 + (n – 1)*abs(incx))``. See `Matrix and Vector
      Storage <../matrix-storage.html>`__ for
      more details.


   incx
      Stride of vector ``x``.


   incy
      Stride of vector ``y``.


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   y
      Buffer holding the updated vector ``y``.


copy (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::copy(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, T *y, std::int64_t incy, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vector ``x``.


      x
         Pointer to the input vector ``x``. The array holding the vector
         ``x`` must be of size at least ``(1 + (n – 1)*abs(incx))``. See
         `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      incy
         Stride of vector ``y``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      y
         Pointer to the updated vector ``y``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`

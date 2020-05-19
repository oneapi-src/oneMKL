.. _onemkl_blas_swap:

swap
====


.. container::


   Swaps a vector with another vector.



      ``swap`` supports the following precisions.


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


   Given two vectors of ``n`` elements, ``x`` and ``y``, the ``swap``
   routines return vectors ``y`` and ``x`` swapped, each replacing the
   other.


      y <- x, x <- y


swap (Buffer Version)
---------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. cpp:function::  void onemkl::blas::swap(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy)

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


.. container:: section


   .. rubric:: Output Parameters
      :class: sectiontitle


   x
      Buffer holding updated buffer ``x``, that is, the input vector
      ``y``.


   y
      Buffer holding updated buffer ``y``, that is, the input vector
      ``x``.


swap (USM Version)
------------------

.. container::

   .. container:: section


      .. rubric:: Syntax
         :class: sectiontitle


      .. container:: dlsyntaxpara


         .. cpp:function::  sycl::event onemkl::blas::swap(sycl::queue &queue, std::int64_t n, T *x, std::int64_t incx, T *y, std::int64_t incy, const sycl::vector_class<sycl::event> &dependencies = {})
   .. container:: section


      .. rubric:: Input Parameters
         :class: sectiontitle


      queue
         The queue where the routine should be executed.


      n
         Number of elements in vector ``x``.


      x
         Pointer to the input vector ``x``. The array must be of size at
         least (1 + (``n`` - 1)*abs(``incx``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incx
         Stride of vector ``x``.


      y
         Pointer to the input vector ``y``. The array must be of size at
         least (1 + (``n`` - 1)*abs(``incy``)). See `Matrix and Vector
         Storage <../matrix-storage.html>`__ for
         more details.


      incy
         Stride of vector ``y``.


      dependencies
         List of events to wait for before starting computation, if any.
         If omitted, defaults to no dependencies.


   .. container:: section


      .. rubric:: Output Parameters
         :class: sectiontitle


      x
         Pointer to the updated array ``x``, that is, the input vector
         ``y``.


      y
         Pointer to the updated array ``y``, that is, the input vector
         ``x``.


   .. container:: section


      .. rubric:: Return Values
         :class: sectiontitle


      Output event to wait on to ensure computation is complete.


.. container:: familylinks


   .. container:: parentlink


      **Parent topic:** :ref:`blas-level-1-routines`
